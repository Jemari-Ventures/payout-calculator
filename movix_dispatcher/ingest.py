"""Ingest and normalize Movix IC Payment Sheet1 exports."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Optional, Union, Tuple

import pandas as pd

from .config import MovixConfig
from .schema import missing_required, rename_to_canonical, resolve_columns

Source = Union[str, Path, BinaryIO]


def load_sheet1(
    source: Source,
    config: Optional[MovixConfig] = None,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """Read Sheet1 (or configured sheet) from an Excel workbook."""
    cfg = config or MovixConfig.load()
    name = sheet_name or cfg.sheet_name
    df = pd.read_excel(source, sheet_name=name, engine="openpyxl")
    return normalize_frame(df, cfg)


def normalize_frame(df: pd.DataFrame, config: Optional[MovixConfig] = None) -> pd.DataFrame:
    """Validate headers, canonicalize columns, coerce types, filter billable rows."""
    cfg = config or MovixConfig.load()
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "service_date",
                "service_type",
                "lm_type",
                "shipment_id",
                "chargeable_weight",
                "driver_id",
                "driver_name",
                "tier",
            ]
        )

    resolved = resolve_columns(df)
    missing = missing_required(resolved)
    if missing:
        raise ValueError(
            "IC Payment Sheet1 is missing required columns: "
            + ", ".join(missing)
            + f". Found headers: {list(df.columns)}"
        )

    out = rename_to_canonical(df, resolved)
    out["service_date"] = pd.to_datetime(out["service_date"], errors="coerce").dt.normalize()
    out["service_type"] = out["service_type"].astype(str).str.strip()
    out["shipment_id"] = out["shipment_id"].astype(str).str.strip()
    out["chargeable_weight"] = pd.to_numeric(out["chargeable_weight"], errors="coerce")
    out["driver_id"] = out["driver_id"].apply(_clean_driver_id)
    if "driver_name" in out.columns:
        out["driver_name"] = out["driver_name"].astype(str).str.strip()
    else:
        out["driver_name"] = ""

    # Drop rows without identity / waybill / date
    out = out[
        out["service_date"].notna()
        & out["driver_id"].astype(bool)
        & out["shipment_id"].astype(bool)
        & (out["shipment_id"].str.lower() != "nan")
    ].copy()

    out = filter_billable(out, cfg)
    # Deduplicate identical shipment+driver+date rows (keep first)
    out = out.drop_duplicates(subset=["shipment_id", "driver_id", "service_date"], keep="first")
    out = out.sort_values(["driver_id", "service_date", "shipment_id"]).reset_index(drop=True)
    return out


def filter_billable(df: pd.DataFrame, config: MovixConfig) -> pd.DataFrame:
    """Keep forward-delivery rows only (Del - FWD*)."""
    if df.empty:
        return df
    prefixes = tuple(config.billable_service_prefixes)
    mask = df["service_type"].apply(lambda s: any(str(s).startswith(p) for p in prefixes))
    return df.loc[mask].copy()


def _clean_driver_id(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    return text


def driver_directory(df: pd.DataFrame) -> pd.DataFrame:
    """Unique driver_id → display name map from normalized rows."""
    if df.empty:
        return pd.DataFrame(columns=["driver_id", "driver_name", "parcels"])
    grouped = (
        df.groupby("driver_id", as_index=False)
        .agg(
            driver_name=("driver_name", _first_non_empty),
            parcels=("shipment_id", "count"),
        )
        .sort_values("driver_id")
        .reset_index(drop=True)
    )
    return grouped


def _first_non_empty(series: pd.Series) -> str:
    for value in series:
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def date_bounds(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df.empty or "service_date" not in df.columns:
        return None, None
    return df["service_date"].min(), df["service_date"].max()
