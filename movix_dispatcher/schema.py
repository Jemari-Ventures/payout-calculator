"""Column contract for Movix IC Payment Sheet1 exports."""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

# Canonical internal field -> accepted header aliases (case-insensitive match)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "service_date": ["service date", "service_date", "date"],
    "service_type": ["service type", "service_type"],
    "lm_type": ["lm type", "lm_type", "lm"],
    "shipment_id": ["shipment id", "shipment_id", "waybill", "waybill number", "awb"],
    "chargeable_weight": [
        "chargeable weight",
        "chargeable_weight",
        "billing weight",
        "billing_weight",
        "weight",
        "weight_kg",
    ],
    "driver_id": ["driver id", "driver_id", "dispatcher id", "dispatcher_id"],
    "driver_name": ["driver name", "driver_name", "dispatcher name", "courier"],
    "tier": ["tier", "weight tier"],
}

REQUIRED_FIELDS = (
    "service_date",
    "service_type",
    "shipment_id",
    "chargeable_weight",
    "driver_id",
)


def _normalize_header(value: object) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Map canonical field names to actual dataframe column names."""
    header_map = {_normalize_header(col): col for col in df.columns}
    resolved: Dict[str, str] = {}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalize_header(alias)
            if key in header_map:
                resolved[field] = header_map[key]
                break
    return resolved


def missing_required(resolved: Dict[str, str]) -> List[str]:
    return [field for field in REQUIRED_FIELDS if field not in resolved]


def rename_to_canonical(df: pd.DataFrame, resolved: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Return a copy with canonical column names for known fields."""
    resolved = resolved or resolve_columns(df)
    rename = {src: canon for canon, src in resolved.items()}
    out = df.rename(columns=rename).copy()
    return out
