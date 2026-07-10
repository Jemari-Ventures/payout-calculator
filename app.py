import warnings
import urllib3
# Suppress the NotOpenSSLWarning
not_openssl_warning = getattr(urllib3.exceptions, "NotOpenSSLWarning", None)
if not_openssl_warning is not None:
    warnings.filterwarnings("ignore", category=not_openssl_warning)

import io
from typing import List, Optional, Tuple, Dict
import re
from urllib.parse import urlparse, parse_qs, quote
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
import altair as alt
import requests

from penalty_common import (
    clean_penalty_dispatcher_id,
    collect_dispatcher_ids_from_penalty_sheets,
    dedupe_no_outbound_scan_by_awb,
    extract_waybill_list,
    filter_penalty_data_by_date,
    filter_penalty_sheet_by_date,
    find_column,
    find_penalty_dispatcher_column,
    find_penalty_waybill_column,
    format_waybills_display,
    penalty_cell_to_float,
    route_penalty_dispatcher_key,
    sanitize_no_outbound_scan_df,
    normalize_waybill,
    waybill_column_read_dtypes,
    filter_rows_for_penalty_dispatcher,
    filter_penalty_sheet_for_dispatcher,
    filter_bulky_for_dispatcher,
    find_bulky_date_column,
    find_penalty_amount_column,
    compute_pickup_commission_series,
    find_pickup_commission_column,
    find_amount_column,
    find_reward_dispatcher_name_column,
    find_reward_employee_column,
    sum_benefit_deduction_float,
    sum_dispatcher_amount_penalty_float,
)

# =============================================================================
# COLOR SCHEME CONSTANTS
# =============================================================================

class ColorScheme:
    """Consistent color scheme for the entire application."""
    PRIMARY = "#4f46e5"  # Indigo - main brand color
    PRIMARY_LIGHT = "#818cf8"  # Lighter indigo
    SECONDARY = "#10b981"  # Emerald green
    ACCENT = "#f59e0b"  # Amber
    BACKGROUND = "#f8fafc"  # Slate 50
    SURFACE = "#ffffff"  # White
    TEXT_PRIMARY = "#1e293b"  # Slate 800
    TEXT_SECONDARY = "#64748b"  # Slate 500
    BORDER = "#e2e8f0"  # Slate 200
    SUCCESS = "#10b981"  # Green
    WARNING = "#f59e0b"  # Amber
    ERROR = "#ef4444"  # Red

    # Chart colors
    CHART_COLORS = [
        "#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
        "#06b6d4", "#84cc16", "#f97316", "#ec4899", "#6366f1"
    ]


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

class Config:
    """Configuration management from JSON file."""

    CONFIG_FILE = "config.json"
    DEFAULT_CONFIG = {
        "data_source": {
            "type": "gsheet",
            "gsheet_url": "",
            "sheet_name": None
        },
        "tiers": [
            {"Tier": "Tier 3", "Min Parcels": 0, "Max Parcels": 60, "Rate (RM)": 0.95},
            {"Tier": "Tier 2", "Min Parcels": 61, "Max Parcels": 120, "Rate (RM)": 1.00},
            {"Tier": "Tier 1", "Min Parcels": 121, "Max Parcels": None, "Rate (RM)": 1.10},
        ],
        "kpi_incentives": [
            {"parcels": 3000, "bonus": 100.00, "description": "3000 Parcel/month"},
            {"parcels": 4000, "bonus": 150.00, "description": "4000 Parcel/month"}
        ],
        "attendance_incentive": {
            "enabled": True,
            "required_days": 26,
            "min_parcels_per_day": 30,
            "bonus": 200.00,
            "description": "26 Days Attendance Bonus"
        },
        "special_rates": [
            {
                "start_date": "2025-01-01",
                "end_date": "2025-01-01",
                "rate": 1.50,
                "description": "New Year Special",
                "min_parcels": 160
            }
        ],
        "currency_symbol": "RM",
        "advance_payout": {
            "enabled": True,
            "percentage": 40.0,
            "description": "Advance Payout (40% of Base Delivery)"
        },
        "designated_driver": {
            "dispatcher_id": "",
            "basic_amount": 1700.0,
            "basic_parcels": 700,
            "rate_after_basic": 1.0,
            "kpi_incentives": [
                {"parcels": 3500, "bonus": 100.0, "description": "3500 Parcel/month"},
                {"parcels": 4500, "bonus": 150.0, "description": "4500 Parcel/month"},
            ],
        },
    }

    @classmethod
    def load(cls):
        """Load configuration from file or create default."""
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    # Ensure new fields exist
                    if "kpi_incentives" not in config:
                        config["kpi_incentives"] = cls.DEFAULT_CONFIG["kpi_incentives"]
                    if "special_rates" not in config:
                        config["special_rates"] = cls.DEFAULT_CONFIG["special_rates"]
                    if "attendance_incentive" not in config:
                        config["attendance_incentive"] = cls.DEFAULT_CONFIG["attendance_incentive"]
                    if "advance_payout" not in config:
                        config["advance_payout"] = cls.DEFAULT_CONFIG["advance_payout"]
                    if "designated_driver" not in config:
                        config["designated_driver"] = cls.DEFAULT_CONFIG["designated_driver"]
                    excel_sheets = config.setdefault("data_source", {}).setdefault("excel_sheets", {})
                    for key, default in {
                        "dispatch": "Dispatch",
                        "pickup": "Pickup",
                        "duitnow": "DuitNow",
                        "ldr": "LDR",
                        "fake_attempt": "Fake Attempt",
                        "cod": "COD",
                        "binding": "Binding",
                        "qr_order": "QR Order",
                        "return": "Return",
                        "attendance": "Attendance",
                        "reward": "Reward",
                        "pending_parcel": "Pending Parcel",
                        "no_outbound_scan": "No Outbound Scan",
                        "parcel_lost": "Parcel lost",
                        "hub": "hub",
                        "socso": "Socso",
                        "overpaid": "Ovepaid",
                        "bulky": "Bulky",
                    }.items():
                        excel_sheets.setdefault(key, default)
                    config.setdefault("fake_attempt_penalty_per_parcel", 2.0)
                    config.setdefault("pending_parcel_penalty_per_parcel", 2.0)
                    config.setdefault("no_outbound_scan_penalty_per_parcel", 3.0)
                    config.setdefault("route_penalty_amount", 0.0)
                    config.setdefault("pickup_payout_per_parcel", 1.0)
                    config.setdefault("qr_order_payout_per_order", 1.8)
                    config.setdefault("return_payout_per_parcel", 0.5)
                    config.setdefault("bulky_rates", {
                        "under_50": 4.0,
                        "over_50": 5.0,
                        "weight_threshold": 50.0,
                    })
                    excel_sheets.setdefault("bulky", "Bulky")
                    return config
            except Exception as e:
                st.error(f"Error loading config: {e}")
                return cls.DEFAULT_CONFIG
        else:
            cls.save(cls.DEFAULT_CONFIG)
            return cls.DEFAULT_CONFIG

    @classmethod
    def save(cls, config):
        """Save configuration to file."""
        try:
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving config: {e}")
            return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

find_no_outbound_scan_dispatcher_column = find_penalty_dispatcher_column
find_no_outbound_scan_awb_column = find_penalty_waybill_column


DISPATCHER_PREFIXES = ["JMR", "ECP", "AF", "PEN", "KUL", "JHR"]


def normalize_dispatcher_id(value) -> str:
    """Normalize dispatcher/employee ID to a stable comparable token."""
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).strip()
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    text = text.upper()
    for prefix in DISPATCHER_PREFIXES:
        if text.startswith(prefix) and text[len(prefix) :].isdigit():
            text = text[len(prefix) :]
            break
    if text.isdigit():
        text = text.lstrip("0") or "0"
    return text


def find_dispatch_id_column(df: pd.DataFrame) -> Optional[str]:
    """Find dispatch dispatcher ID column."""
    col = find_column(df, ["Dispatcher ID", "dispatcher_id", "Dispatcher Id", "DISPATCHER ID"])
    if col is not None:
        return col
    for c in df.columns:
        if "dispatcher" in str(c).lower() and "id" in str(c).lower():
            return c
    return None


def exclude_dispatch_rows_by_dispatcher_sheet(
    dispatch_df: pd.DataFrame,
    sheet_df: pd.DataFrame,
    sheet_dispatcher_col: Optional[str],
    dispatch_disp_col: Optional[str],
    dispatch_wb_col: Optional[str],
    preserve_waybills: Optional[set] = None,
) -> pd.DataFrame:
    """Remove dispatch rows when the same dispatcher's waybill is on pickup/return."""
    if (
        dispatch_df is None
        or dispatch_df.empty
        or sheet_df is None
        or sheet_df.empty
        or not sheet_dispatcher_col
        or not dispatch_disp_col
        or not dispatch_wb_col
        or dispatch_disp_col not in dispatch_df.columns
        or dispatch_wb_col not in dispatch_df.columns
    ):
        return dispatch_df

    sheet_wb_col = find_waybill_column(sheet_df)
    if not sheet_wb_col:
        return dispatch_df

    sheet_map = build_unique_awb_map(sheet_df, sheet_dispatcher_col, sheet_wb_col)
    if not sheet_map:
        return dispatch_df

    preserve = preserve_waybills or set()

    def _is_on_sheet(row) -> bool:
        disp = normalize_dispatcher_id(row[dispatch_disp_col])
        wb = normalize_waybill(row[dispatch_wb_col])
        if not disp or not wb:
            return False
        if wb in preserve:
            return False
        return wb in sheet_map.get(disp, set())

    return dispatch_df.loc[~dispatch_df.apply(_is_on_sheet, axis=1)].copy()


def find_waybill_column(df: pd.DataFrame) -> Optional[str]:
    """Find waybill/AWB column using common header names."""
    col = find_column(
        df,
        [
            "Waybill Number", "waybill_number", "Waybill", "waybill",
            "Waybill No", "AWB", "No. AWB", "AWB No.", "awb_no",
        ],
    )
    if col is not None:
        return col
    for c in df.columns:
        cl = str(c).lower().strip()
        if "waybill" in cl or "awb" in cl:
            return c
    return None


def find_pickup_dispatcher_column(df: pd.DataFrame) -> Optional[str]:
    """Find pickup dispatcher ID column; prefer pickup-specific headers."""
    if df is None or df.empty:
        return None
    columns_by_lower = {str(c).strip().lower(): c for c in df.columns}
    for name in (
        "Pick Up Dispatcher ID",
        "Pick Up Dispatcher Id",
        "Pickup Dispatcher ID",
        "Pickup Dispatcher Id",
        "pickup_dispatcher_id",
        "pick_up_dispatcher_id",
    ):
        if name.lower() in columns_by_lower:
            return columns_by_lower[name.lower()]
    for col in df.columns:
        c_lower = str(col).strip().lower()
        if "pickup" in c_lower and "dispatcher" in c_lower:
            return col
    for name in ("Dispatcher ID", "dispatcher_id"):
        if name.lower() in columns_by_lower:
            return columns_by_lower[name.lower()]
    return None


def build_pickup_waybill_set(pickup_df: pd.DataFrame) -> set:
    """All normalized waybills listed on the pickup sheet."""
    if pickup_df is None or pickup_df.empty:
        return set()
    waybill_col = find_waybill_column(pickup_df)
    if not waybill_col:
        return set()
    return {
        wb
        for wb in pickup_df[waybill_col].apply(normalize_waybill)
        if wb
    }


def exclude_dispatch_rows_by_waybill_set(
    dispatch_df: pd.DataFrame,
    waybill_set: set,
    dispatch_wb_col: Optional[str],
) -> pd.DataFrame:
    """Remove dispatch rows whose waybill appears on the pickup sheet."""
    if (
        dispatch_df is None
        or dispatch_df.empty
        or not waybill_set
        or not dispatch_wb_col
        or dispatch_wb_col not in dispatch_df.columns
    ):
        return dispatch_df
    wbs = dispatch_df[dispatch_wb_col].apply(normalize_waybill)
    return dispatch_df.loc[~wbs.isin(waybill_set)].copy()


def is_return_sheet_dispatch_fallback(return_df: pd.DataFrame) -> bool:
    """True when Return sheet data is actually Dispatch fallback content."""
    if return_df is None or return_df.empty:
        return False
    return_cols_lower = {str(c).strip().lower() for c in return_df.columns}
    has_dispatch_signature = (
        "delivery signature" in return_cols_lower
        and ("waybill number" in return_cols_lower or "waybill" in return_cols_lower)
        and ("dispatcher id" in return_cols_lower or "dispatcher_id" in return_cols_lower)
    )
    has_return_markers = any("return" in col for col in return_cols_lower)
    return has_dispatch_signature and not has_return_markers


def build_unique_awb_map(
    df: Optional[pd.DataFrame],
    dispatcher_col: Optional[str],
    waybill_col: Optional[str],
) -> Dict[str, set]:
    """Map dispatcher ID -> set of normalized waybills for batch summaries."""
    if (
        df is None
        or df.empty
        or not dispatcher_col
        or not waybill_col
        or dispatcher_col not in df.columns
        or waybill_col not in df.columns
    ):
        return {}

    work = df[[dispatcher_col, waybill_col]].copy()
    work["_key"] = work[dispatcher_col].apply(normalize_dispatcher_id)
    work["_wb"] = work[waybill_col].apply(normalize_waybill)
    work = work[(work["_key"] != "") & (work["_wb"] != "")]
    if work.empty:
        return {}

    return work.groupby("_key")["_wb"].apply(lambda values: set(values.tolist())).to_dict()


def find_return_dispatcher_column(df: pd.DataFrame) -> Optional[str]:
    """Find return dispatcher ID column."""
    col = find_column(
        df,
        ["dispatcher_id", "Dispatcher ID", "dispatcher", "Dispatcher", "DISPATCHER_ID", "DISPATCHER ID"],
    )
    if col is not None:
        return col
    for c in df.columns:
        cl = str(c).lower().strip()
        if "dispatcher" in cl and "id" in cl:
            return c
    return None


def filter_sheet_by_date_range(
    sheet_df: Optional[pd.DataFrame],
    start_date,
    end_date,
    date_column_names: List[str],
) -> Optional[pd.DataFrame]:
    """Filter a sheet to the selected period using its configured date column."""
    return filter_penalty_sheet_by_date(sheet_df, start_date, end_date, date_column_names)


def split_route_penalty_pool(pool_total: float, dispatcher_ids) -> Dict[str, float]:
    """Split a currency pool into per-dispatcher shares (2 dp) that sum exactly to *pool_total*.

    Equal cents per person plus remainder assigned in sorted-ID order avoids
    ``round(pool / n, 2) * n`` drift (e.g. RM1000 ÷ 19 → RM999.97 total).
    """
    ids = sorted(
        {str(x).strip() for x in dispatcher_ids if pd.notna(x) and str(x).strip() and str(x).strip().lower() != "nan"}
    )
    n = len(ids)
    if n == 0 or pool_total <= 0:
        return {}
    pool_cents = int(round(float(pool_total) * 100))
    if pool_cents <= 0:
        return {i: 0.0 for i in ids}
    base = pool_cents // n
    rem = pool_cents % n
    return {did: (base + (1 if j < rem else 0)) / 100.0 for j, did in enumerate(ids)}


# =============================================================================
# DATA SOURCE MANAGEMENT
# =============================================================================

class DataSource:
    """Handle data loading from Google Sheets."""

    @staticmethod
    def _extract_gsheet_id_and_gid(url_or_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract spreadsheet ID and GID from Google Sheets URL or ID."""
        if not url_or_id:
            return None, None

        if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url_or_id):
            return url_or_id, None

        try:
            parsed = urlparse(url_or_id)
            path_parts = [p for p in parsed.path.split('/') if p]
            spreadsheet_id = None
            if 'spreadsheets' in path_parts and 'd' in path_parts:
                try:
                    idx = path_parts.index('d')
                    spreadsheet_id = path_parts[idx + 1]
                except Exception:
                    spreadsheet_id = None

            query_gid = parse_qs(parsed.query).get('gid', [None])[0]
            frag_gid_match = re.search(r"gid=(\d+)", parsed.fragment or "")
            frag_gid = frag_gid_match.group(1) if frag_gid_match else None
            gid = query_gid or frag_gid
            return spreadsheet_id, gid
        except Exception:
            return None, None

    @staticmethod
    def _build_gsheet_csv_url(spreadsheet_id: str, sheet_name: Optional[str], gid: Optional[str]) -> str:
        """Construct CSV export URL for Google Sheets."""
        base = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        if sheet_name and str(sheet_name).strip():
            encoded = quote(str(sheet_name).strip(), safe="")
            return f"{base}/gviz/tq?tqx=out:csv&sheet={encoded}"
        if gid:
            return f"{base}/export?format=csv&gid={gid}"
        return f"{base}/export?format=csv"

    @staticmethod
    @st.cache_data(ttl=300)
    def read_google_sheet(url_or_id: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Fetch a Google Sheet as CSV and return a DataFrame."""
        spreadsheet_id, gid = DataSource._extract_gsheet_id_and_gid(url_or_id)
        if not spreadsheet_id:
            raise ValueError("Invalid Google Sheet URL or ID.")
        csv_url = DataSource._build_gsheet_csv_url(spreadsheet_id, sheet_name, gid)
        try:
            resp = requests.get(csv_url, timeout=30)
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError:
                # Missing/non-existent worksheet often returns 400/404. Treat it as empty data
                # so downstream payout calculations evaluate to 0 instead of hard-failing.
                if sheet_name and str(sheet_name).strip() and resp.status_code in (400, 404):
                    error_sample = (resp.text or "")[:1000].lower()
                    if (
                        "worksheet" in error_sample
                        or "sheet" in error_sample
                        or "unable to parse range" in error_sample
                        or "not found" in error_sample
                        or "invalid" in error_sample
                    ):
                        return pd.DataFrame()
                raise

            resp_content = resp.content

            # If a specific sheet name was requested but doesn't exist, Sheets returns HTML/error text.
            # In that case, treat it as an empty sheet so downstream calculations are 0.
            if sheet_name and str(sheet_name).strip():
                content_sample = resp_content[:500].decode("utf-8", errors="ignore").lower().strip()
                if (
                    "<html" in content_sample
                    or "<!doctype html" in content_sample
                    or ("worksheet" in content_sample and "not found" in content_sample)
                    or ("sheet" in content_sample and "not found" in content_sample)
                    or ("invalid" in content_sample and "sheet" in content_sample)
                ):
                    return pd.DataFrame()

            # Read CSV with waybill columns as text.
            header_preview = pd.read_csv(
                io.BytesIO(resp_content),
                nrows=0,
                encoding="utf-8-sig",
            )
            raw_columns = list(header_preview.columns)
            clean_columns = [str(c).replace("\ufeff", "").strip() for c in raw_columns]
            df = pd.read_csv(
                io.BytesIO(resp_content),
                dtype=waybill_column_read_dtypes(raw_columns) or None,
                keep_default_na=False,
                na_values=[],
                encoding="utf-8-sig",
                low_memory=False,
            )
            df.columns = clean_columns

            return df
        except Exception as exc:
            st.error(f"Failed to fetch Google Sheet: {exc}")
            raise

    @staticmethod
    def _normalize_compare_value(value):
        """Normalize a cell value for cross-sheet fallback comparison."""
        if pd.isna(value):
            return ""
        value_str = str(value).strip()
        if not value_str:
            return ""
        # Normalize integer-like floats (e.g. 123.0 -> 123) for stable matching
        try:
            num = float(value_str)
            if num.is_integer():
                return str(int(num))
        except Exception:
            pass
        return value_str

    @staticmethod
    def _is_fallback_dispatch_sheet(candidate_df: Optional[pd.DataFrame], dispatch_df: Optional[pd.DataFrame]) -> bool:
        """
        Detect Google Sheets fallback behavior where requesting a missing tab returns Dispatch.
        Uses column set + first rows fingerprint to avoid false positives.
        """
        if candidate_df is None or dispatch_df is None or candidate_df.empty or dispatch_df.empty:
            return False

        candidate_cols = [str(c).strip().lower() for c in candidate_df.columns]
        dispatch_cols = [str(c).strip().lower() for c in dispatch_df.columns]
        if candidate_cols != dispatch_cols:
            return False

        sample_size = min(10, len(candidate_df), len(dispatch_df))
        if sample_size <= 0:
            return False

        candidate_sample = (
            candidate_df.head(sample_size)
            .fillna("")
            .map(DataSource._normalize_compare_value)
            .astype(str)
        )
        dispatch_sample = (
            dispatch_df.head(sample_size)
            .fillna("")
            .map(DataSource._normalize_compare_value)
            .astype(str)
        )
        return candidate_sample.equals(dispatch_sample)

    @staticmethod
    def _get_excel_sheet_name(config: dict, key: str, default: str) -> str:
        return config.get("data_source", {}).get("excel_sheets", {}).get(key, default)

    @staticmethod
    def _read_optional_sheet_without_dispatch_fallback(
        gsheet_url: str,
        sheet_name: str,
        dispatch_sheet_name: str = "Dispatch",
    ) -> pd.DataFrame:
        """
        Read an optional sheet and guard against Google Sheets fallback behavior
        where a missing tab may return Dispatch content instead.
        """
        sheet_df = DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)
        if sheet_df is None or sheet_df.empty:
            return pd.DataFrame()

        dispatch_df = DataSource.read_google_sheet(gsheet_url, sheet_name=dispatch_sheet_name)
        if DataSource._is_fallback_dispatch_sheet(sheet_df, dispatch_df):
            return pd.DataFrame()
        return sheet_df

    @staticmethod
    def _load_optional_sheet(config: dict, sheet_key: str, default_name: str) -> Optional[pd.DataFrame]:
        data_source = config["data_source"]
        if data_source["type"] != "gsheet" or not data_source.get("gsheet_url"):
            return None
        sheet_name = DataSource._get_excel_sheet_name(config, sheet_key, default_name)
        dispatch_sheet = DataSource._get_excel_sheet_name(config, "dispatch", "Dispatch")
        try:
            return DataSource._read_optional_sheet_without_dispatch_fallback(
                data_source["gsheet_url"],
                sheet_name=sheet_name,
                dispatch_sheet_name=dispatch_sheet,
            )
        except Exception as exc:
            st.warning(f"Could not load {sheet_key} data from '{sheet_name}': {exc}")
            return None

    @staticmethod
    def load_data(config: dict) -> Optional[pd.DataFrame]:
        """Load dispatcher delivery data from Dispatch sheet."""
        data_source = config["data_source"]

        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                sheet_name = DataSource._get_excel_sheet_name(config, "dispatch", "Dispatch")
                return DataSource.read_google_sheet(data_source["gsheet_url"], sheet_name)
            except Exception as exc:
                st.error(f"Error reading dispatcher data: {exc}")
                return None
        return None

    @staticmethod
    def load_pickup_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "pickup", "Pickup")

    @staticmethod
    def load_duitnow_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "duitnow", "DuitNow")

    @staticmethod
    def load_ldr_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "ldr", "LDR")

    @staticmethod
    def load_fake_attempt_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "fake_attempt", "Fake Attempt")

    @staticmethod
    def load_cod_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "cod", "COD")

    @staticmethod
    def load_binding_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "binding", "Binding")

    @staticmethod
    def load_hub_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "hub", "hub")

    @staticmethod
    def load_socso_deduction_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "socso", "Socso")

    @staticmethod
    def load_overpaid_deduction_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "overpaid", "Ovepaid")

    @staticmethod
    def load_pending_parcel_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "pending_parcel", "Pending Parcel")

    @staticmethod
    def load_no_outbound_scan_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "no_outbound_scan", "No Outbound Scan")

    @staticmethod
    def load_parcel_lost_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "parcel_lost", "Parcel lost")

    @staticmethod
    def load_reward_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "reward", "Reward")

    @staticmethod
    def load_return_data(config: dict) -> Optional[pd.DataFrame]:
        return DataSource._load_optional_sheet(config, "return", "Return")

    @staticmethod
    def load_bulky_data(config: dict) -> Optional[pd.DataFrame]:
        """Load bulky parcel data directly from the Bulky sheet (same as management.py)."""
        data_source = config["data_source"]
        gsheet_url = data_source.get("gsheet_url")
        if not gsheet_url:
            return None
        sheet_name = DataSource._get_excel_sheet_name(config, "bulky", "Bulky")
        try:
            return DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)
        except Exception as exc:
            st.warning(f"Could not load bulky data from sheet '{sheet_name}': {exc}")
            return None

    @staticmethod
    def load_attendance_data(config: dict) -> Optional[pd.DataFrame]:
        """Load attendance penalty rows from the tab named in config (data_source.excel_sheets.attendance)."""
        data_source = config["data_source"]
        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                sheet = (
                    config.get("data_source", {})
                    .get("excel_sheets", {})
                    .get("attendance", "Attendance")
                )
                sheet = str(sheet).strip() if sheet is not None else "Attendance"
                return DataSource._read_optional_sheet_without_dispatch_fallback(
                    data_source["gsheet_url"], sheet_name=sheet
                )
            except Exception as exc:
                st.warning(f"Could not load Attendance data: {exc}")
                return None
        return None


# =============================================================================
# PAYOUT CALCULATIONS
# =============================================================================

class PayoutCalculator:
    """Handle payout calculations."""

    @staticmethod
    def _convert_to_float(value):
        """Convert a value to float, handling comma decimal separators.

        Handles European decimal format (e.g., '84,3' -> 84.3) and standard format.
        """
        # Handle NaN, None, or empty string
        if pd.isna(value) or value == '' or str(value).strip() == '':
            return 0.0

        # Convert to string and strip whitespace
        str_value = str(value).strip()

        # Handle empty string after conversion
        if not str_value or str_value.lower() == 'nan':
            return 0.0

        # Replace comma with period for decimal separator (handles European format like '84,3')
        str_value = str_value.replace(',', '.')

        try:
            return float(str_value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def calculate_kpi_bonus(total_parcels: int, kpi_config: List) -> Tuple[float, str]:
        """Calculate KPI bonus based on total monthly parcels."""
        if not kpi_config:
            return 0.0, "No KPI achieved"

        sorted_kpis = sorted(kpi_config, key=lambda x: x.get("parcels", 0), reverse=True)

        for kpi in sorted_kpis:
            required_parcels = kpi.get("parcels", 0)
            bonus = kpi.get("bonus", 0.0)
            description = kpi.get("description", f"{required_parcels} parcels")

            if total_parcels >= required_parcels:
                return round(float(bonus), 2), description

        return 0.0, "No KPI achieved"

    @staticmethod
    def get_designated_driver_ids(config: dict) -> set:
        """Return configured designated-driver dispatcher IDs (empty if none)."""
        dd_config = config.get("designated_driver") or {}
        raw_ids = dd_config.get("dispatcher_ids") or dd_config.get("dispatcher_id")
        if not raw_ids:
            return set()
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        return {
            str(dispatcher_id).strip()
            for dispatcher_id in raw_ids
            if dispatcher_id is not None and str(dispatcher_id).strip()
        }

    @staticmethod
    def is_designated_driver(dispatcher_id: str, config: dict) -> bool:
        """Check whether a dispatcher uses designated-driver payout rules."""
        normalized_id = str(dispatcher_id).strip()
        return bool(normalized_id) and normalized_id in PayoutCalculator.get_designated_driver_ids(config)

    @staticmethod
    def calculate_designated_driver_base(
        total_parcels: int,
        designated_driver_config: dict,
    ) -> Tuple[float, int, float]:
        """Calculate designated-driver base payout and extra-parcel earnings."""
        basic_amount = float(designated_driver_config.get("basic_amount", 1700.0))
        basic_parcels = int(designated_driver_config.get("basic_parcels", 700))
        rate_after_basic = float(designated_driver_config.get("rate_after_basic", 1.0))
        extra_parcels = max(0, int(total_parcels) - basic_parcels)
        extra_payout = round(extra_parcels * rate_after_basic, 2)
        base_payout = round(basic_amount + extra_payout, 2)
        return base_payout, extra_parcels, rate_after_basic

    @staticmethod
    def build_designated_driver_breakdown(
        per_day_df: pd.DataFrame,
        designated_driver_config: dict,
        total_parcels: int,
        base_payout: float,
    ) -> dict:
        """Build invoice breakdown for designated-driver base delivery payout."""
        basic_amount = float(designated_driver_config.get("basic_amount", 1700.0))
        basic_parcels = int(designated_driver_config.get("basic_parcels", 700))
        rate_after_basic = float(designated_driver_config.get("rate_after_basic", 1.0))
        extra_parcels = max(0, int(total_parcels) - basic_parcels)

        if per_day_df is not None and not per_day_df.empty and "special_rate" in per_day_df.columns:
            special_mask = per_day_df["special_rate"].notna()
            special_payout = round(float(per_day_df.loc[special_mask, "payout_per_day"].sum()), 2)
            extra_payout = round(float(per_day_df.loc[~special_mask, "payout_per_day"].sum()), 2)
        else:
            special_payout = 0.0
            extra_payout = round(max(0.0, float(base_payout) - basic_amount), 2)

        return {
            "basic_amount": basic_amount,
            "basic_parcels": basic_parcels,
            "rate_after_basic": rate_after_basic,
            "extra_parcels": extra_parcels,
            "extra_payout": extra_payout,
            "special_payout": special_payout,
        }

    @staticmethod
    def apply_designated_driver_daily_payout(
        per_day: pd.DataFrame,
        designated_driver_config: dict,
        special_rates_config: Optional[List] = None,
    ) -> pd.DataFrame:
        """Annotate per-day rows for designated-driver display and charts."""
        basic_parcels = int(designated_driver_config.get("basic_parcels", 700))
        rate_after_basic = float(designated_driver_config.get("rate_after_basic", 1.0))
        per_day = per_day.sort_values("__date").copy()
        cumulative_parcels = 0
        payout_per_day = []
        rate_per_parcel = []
        tiers = []
        special_rates = []
        special_descs = []

        for _, row in per_day.iterrows():
            daily_parcels = int(row["daily_parcels"])
            day_date = row["__date"]
            previous_total = cumulative_parcels
            cumulative_parcels += daily_parcels

            special_rate, special_desc = PayoutCalculator.get_special_rate(
                day_date, daily_parcels, special_rates_config or []
            )

            if special_rate is not None:
                day_payout = round(daily_parcels * float(special_rate), 2)
                payout_per_day.append(day_payout)
                rate_per_parcel.append(float(special_rate))
                tiers.append(special_desc or "Special Rate")
                special_rates.append(float(special_rate))
                special_descs.append(special_desc or "")
            else:
                extra_on_day = max(0, cumulative_parcels - max(previous_total, basic_parcels))
                payout_per_day.append(round(extra_on_day * rate_after_basic, 2))
                rate_per_parcel.append(rate_after_basic if extra_on_day > 0 else 0.0)
                tiers.append("Designated Driver")
                special_rates.append(pd.NA)
                special_descs.append("")

        per_day["tier"] = tiers
        per_day["base_rate"] = rate_after_basic
        per_day["special_rate"] = special_rates
        per_day["special_desc"] = special_descs
        per_day["rate_per_parcel"] = rate_per_parcel
        per_day["payout_per_day"] = payout_per_day
        return per_day

    @staticmethod
    def calculate_attendance_bonus(per_day_df: pd.DataFrame, attendance_config: dict) -> Tuple[float, str, int]:
        """Calculate attendance bonus based on working days and minimum parcels."""
        if not attendance_config or not attendance_config.get("enabled", False):
            return 0.0, "Attendance bonus not enabled", 0

        required_days = attendance_config.get("required_days", 26)
        min_parcels = attendance_config.get("min_parcels_per_day", 30)
        bonus_amount = attendance_config.get("bonus", 0.0)

        qualified_days = len(per_day_df[per_day_df["daily_parcels"] >= min_parcels])

        if qualified_days >= required_days:
            description = f"{qualified_days}/{required_days} days qualified (≥{min_parcels} parcels/day)"
            return round(float(bonus_amount), 2), description, qualified_days
        else:
            description = f"{qualified_days}/{required_days} days qualified (need ≥{min_parcels} parcels/day)"
            return 0.0, description, qualified_days

    @staticmethod
    def get_special_rate(date, daily_parcels: int, special_rates_config: List) -> Tuple[Optional[float], str]:
        """Get special rate for a specific date if configured and parcel count meets minimum requirement."""
        if not special_rates_config:
            return None, ""

        date_obj = pd.to_datetime(date).date()

        for special in special_rates_config:
            min_parcels = special.get("min_parcels", 160)
            if daily_parcels <= min_parcels:
                continue

            if "date" in special:
                special_date = pd.to_datetime(special["date"]).date()
                if date_obj == special_date:
                    return float(special.get("rate", 0)), special.get("description", "Special Rate")

            elif "start_date" in special and "end_date" in special:
                start_date = pd.to_datetime(special["start_date"]).date()
                end_date = pd.to_datetime(special["end_date"]).date()
                if start_date <= date_obj <= end_date:
                    return float(special.get("rate", 0)), special.get("description", "Special Rate")

        return None, ""

    @staticmethod
    def count_total_awb(
        delivery_parcels: int,
        pickup_parcels: int,
        return_parcels: int,
    ) -> int:
        """Total AWB = delivery parcels + pickup parcels + return parcels."""
        return int(delivery_parcels) + int(pickup_parcels) + int(return_parcels)

    @staticmethod
    def map_tier_rate(daily_parcels: float, tiers_config: List) -> Tuple[str, float]:
        """Map daily parcel count to tier name and per-parcel rate."""
        tiers = []
        for tier in tiers_config or []:
            tmin = tier.get("Min Parcels")
            tmax = tier.get("Max Parcels")
            trate = tier.get("Rate (RM)")
            tname = tier.get("Tier")
            if pd.notna(trate):
                tiers.append((tmin, tmax, trate, tname))
        tiers.sort(key=lambda t: (t[0] or 0), reverse=True)
        for tmin, tmax, trate, tname in tiers:
            lower_ok = True if pd.isna(tmin) else daily_parcels >= tmin
            upper_ok = True if pd.isna(tmax) else daily_parcels <= tmax
            if lower_ok and upper_ok:
                return str(tname), float(trate)
        return "Unmatched", 0.0

    @staticmethod
    def _append_bulky_delivery_records(
        work: pd.DataFrame,
        bulky_df: Optional[pd.DataFrame],
        dispatcher_id: str,
    ) -> pd.DataFrame:
        """Add all bulky sheet rows for this dispatcher as delivery records."""
        if work is None:
            work = pd.DataFrame()
        if bulky_df is None or bulky_df.empty or not dispatcher_id:
            return work

        all_bulky = filter_bulky_for_dispatcher(bulky_df, dispatcher_id)
        if all_bulky.empty:
            return work

        bulky_wb_col = find_waybill_column(all_bulky)
        bulky_date_col = find_bulky_date_column(all_bulky)
        if not bulky_wb_col or not bulky_date_col:
            return work

        additions = all_bulky.copy()
        additions["__date"] = pd.to_datetime(
            additions[bulky_date_col], errors="coerce"
        ).dt.date
        additions["__date"] = additions["__date"].fillna(pd.Timestamp.today().date())
        additions["__waybill"] = additions[bulky_wb_col].apply(normalize_waybill)
        additions["__waybill_original"] = additions[bulky_wb_col]
        additions = additions[additions["__waybill"] != ""]

        if additions.empty:
            return work
        return pd.concat([work, additions], ignore_index=True)

    @staticmethod
    def calculate_gross_payout(
        base_payout: float,
        pickup_payout: float,
        return_payout: float,
        reward_payout: float,
        kpi_bonus: float,
        attendance_bonus: float,
        penalty_total: float,
        socso_deduction: float = 0.0,
        overpaid_deduction: float = 0.0,
    ) -> float:
        """Gross = earnings + bonuses − penalties − SOCSO − Overpaid (before Advance)."""
        earnings = (
            float(base_payout)
            + float(pickup_payout)
            + float(return_payout)
            + float(reward_payout)
            + float(kpi_bonus)
            + float(attendance_bonus)
            - float(penalty_total)
        )
        return round(earnings - float(socso_deduction) - float(overpaid_deduction), 2)

    @staticmethod
    def calculate_advance_payout(
        base_payout: float,
        percentage: float,
        enabled: bool = True,
    ) -> float:
        """Advance = percentage of base delivery payout only."""
        if not enabled or float(percentage) <= 0:
            return 0.0
        return round(float(base_payout) * float(percentage) / 100.0, 2)

    @staticmethod
    def calculate_final_payout(gross_payout: float, advance_payout: float) -> float:
        """Final = Gross Payout − Advance."""
        return round(float(gross_payout) - float(advance_payout), 2)

    @staticmethod
    def calculate_benefit_deduction(
        dispatcher_id: str,
        deduction_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, int]:
        """Sum benefit deduction amount for a dispatcher (SOCSO, Overpaid — not penalties)."""
        return sum_benefit_deduction_float(deduction_df, dispatcher_id)

    @staticmethod
    def calculate_penalty(dispatcher_id: str,
                         duitnow_df: Optional[pd.DataFrame] = None,
                         ldr_df: Optional[pd.DataFrame] = None,
                         fake_df: Optional[pd.DataFrame] = None,
                         cod_df: Optional[pd.DataFrame] = None,
                         binding_df: Optional[pd.DataFrame] = None,
                         hub_df: Optional[pd.DataFrame] = None,
                         pending_parcel_df: Optional[pd.DataFrame] = None,
                         no_outbound_scan_df: Optional[pd.DataFrame] = None,
                         parcel_lost_df: Optional[pd.DataFrame] = None,
                         attendance_df: Optional[pd.DataFrame] = None,
                         working_days: Optional[int] = None,
                         fake_attempt_penalty_per_parcel: float = 2.0,
                         pending_parcel_penalty_per_parcel: float = 2.0,
                         no_outbound_scan_penalty_per_parcel: float = 3.0,
                         route_penalty_per_dispatcher: float = 0.0,
                         route_penalty_dispatcher_count: int = 0,
                         route_penalty_pool_total: float = 0.0) -> Dict:
        """Calculate total penalty for a dispatcher from penalty sheets.

        Args:
            dispatcher_id: The dispatcher ID to check
            duitnow_df: DataFrame from Sheet3 (DuitNow penalty)
            ldr_df: DataFrame from Sheet4 (LD&R penalty)
            fake_df: DataFrame from Sheet5 (Fake attempt penalty)
            cod_df: DataFrame from COD sheet (COD penalty)
            binding_df: DataFrame from Binding sheet (Dispatcher ID, Penalty)
            attendance_df: DataFrame from Attendance sheet
            working_days: Total working days for the dispatcher

        Returns:
            Dictionary containing penalty breakdown by type
        """
        penalty_breakdown = {
            'duitnow': {'amount': 0.0, 'count': 0, 'waybills': []},
            'ldr': {'amount': 0.0, 'count': 0, 'waybills': []},
            'fake_attempt': {'amount': 0.0, 'count': 0, 'waybills': []},
            'cod': {'amount': 0.0, 'count': 0},
            'binding': {'amount': 0.0, 'count': 0},
            'hub': {'amount': 0.0, 'count': 0},
            'pending_parcel': {'amount': 0.0, 'count': 0, 'waybills': []},
            'no_outbound_scan': {'amount': 0.0, 'count': 0, 'waybills': []},
            'parcel_lost': {'amount': 0.0, 'count': 0, 'waybills': []},
            'route': {'amount': 0.0, 'count': 0, 'pool_total': 0.0},
            'attendance': {'amount': 0.0, 'count': 0},
            'total_amount': 0.0,
            'total_count': 0
        }

        # Normalize dispatcher_id for comparison
        dispatcher_id_normalized = clean_penalty_dispatcher_id(dispatcher_id)

        # 1. Process DuitNow penalty — dispatcher_id match + Achieve = FAIL only
        if duitnow_df is not None and not duitnow_df.empty:
            duitnow_rows = filter_penalty_sheet_for_dispatcher(duitnow_df, dispatcher_id)
            achieve_col = find_column(duitnow_df, ["Achieve", "achieve", "ACHIEVE"])
            if achieve_col is not None and not duitnow_rows.empty:
                duitnow_rows = duitnow_rows[
                    duitnow_rows[achieve_col].astype(str).str.strip().str.upper() == "FAIL"
                ]
            if not duitnow_rows.empty:
                penalty_col = find_penalty_amount_column(duitnow_df)
                if penalty_col is not None:
                    penalty_values = duitnow_rows[penalty_col].apply(penalty_cell_to_float)
                    penalty_amount = penalty_values.sum()
                    penalty_breakdown['duitnow']['amount'] = round(float(penalty_amount), 2)
                    if penalty_breakdown['duitnow']['amount'] > 0:
                        penalty_breakdown['duitnow']['count'] = 1

        # 2. Process LD&R penalty
        if ldr_df is not None and not ldr_df.empty:
            ldr_rows = filter_penalty_sheet_for_dispatcher(ldr_df, dispatcher_id)
            if not ldr_rows.empty:
                amount_col = find_column(ldr_df, ["Amount", "amount", "AMOUNT"])
                if amount_col is not None:
                    penalty_values = ldr_rows[amount_col].apply(penalty_cell_to_float)
                    penalty_breakdown['ldr']['amount'] = round(float(penalty_values.sum()), 2)
                    penalty_breakdown['ldr']['count'] = len(ldr_rows)
                    awb_col = find_penalty_waybill_column(ldr_df)
                    if awb_col is not None:
                        penalty_breakdown['ldr']['waybills'] = extract_waybill_list(ldr_rows[awb_col])

        # 3. Process Fake attempt penalty (Sheet5)
        if fake_df is not None and not fake_df.empty:
            fake_rows = filter_penalty_sheet_for_dispatcher(fake_df, dispatcher_id)

            if not fake_rows.empty:
                penalty_amount = len(fake_rows) * float(fake_attempt_penalty_per_parcel)
                penalty_breakdown['fake_attempt']['amount'] = round(float(penalty_amount), 2)
                penalty_breakdown['fake_attempt']['count'] = len(fake_rows)

                waybill_col = find_penalty_waybill_column(fake_df)
                if waybill_col is not None and waybill_col in fake_rows.columns:
                    penalty_breakdown['fake_attempt']['waybills'] = extract_waybill_list(
                        fake_rows[waybill_col]
                    )

        # 3b. Process No Outbound Scan penalty (RM3 per unique AWB — not sum of PENALTY column)
        if no_outbound_scan_df is not None and not no_outbound_scan_df.empty:
            nos_disp_col = find_no_outbound_scan_dispatcher_column(no_outbound_scan_df)
            if nos_disp_col is not None:
                nos_copy = no_outbound_scan_df.copy()
                nos_copy["_nos_disp_key"] = nos_copy[nos_disp_col].apply(clean_penalty_dispatcher_id)
                nos_rows = nos_copy[nos_copy["_nos_disp_key"] == dispatcher_id_normalized]

                if not nos_rows.empty:
                    deduped = dedupe_no_outbound_scan_by_awb(nos_rows, no_outbound_scan_df)
                    awb_col = find_no_outbound_scan_awb_column(no_outbound_scan_df)
                    if awb_col is None:
                        awb_col = find_no_outbound_scan_awb_column(nos_rows)
                    if not deduped.empty and awb_col is not None and awb_col in deduped.columns:
                        penalty_breakdown['no_outbound_scan']['waybills'] = extract_waybill_list(
                            deduped[awb_col]
                        )
                    parcel_count = len(deduped)
                    penalty_amount = parcel_count * float(no_outbound_scan_penalty_per_parcel)
                    penalty_breakdown['no_outbound_scan']['amount'] = round(float(penalty_amount), 2)
                    penalty_breakdown['no_outbound_scan']['count'] = int(parcel_count)

        # 4. Process COD penalty (COD sheet)
        if cod_df is not None and not cod_df.empty:
            dispatcher_id_col = find_penalty_dispatcher_column(cod_df)
            if dispatcher_id_col is not None:
                cod_rows = filter_rows_for_penalty_dispatcher(cod_df, dispatcher_id_col, dispatcher_id)
                if not cod_rows.empty:
                    penalty_col = find_penalty_amount_column(cod_df)
                    if penalty_col is not None:
                        penalty_values = cod_rows[penalty_col].apply(penalty_cell_to_float)
                        penalty_breakdown['cod']['amount'] = round(float(penalty_values.sum()), 2)
                        penalty_breakdown['cod']['count'] = len(cod_rows)

        # 5. Process Binding penalty
        if binding_df is not None and not binding_df.empty:
            binding_disp_col = find_penalty_dispatcher_column(binding_df)
            if binding_disp_col is not None:
                binding_rows = filter_rows_for_penalty_dispatcher(binding_df, binding_disp_col, dispatcher_id)
                if not binding_rows.empty:
                    penalty_col = find_penalty_amount_column(binding_df)
                    if penalty_col is not None:
                        penalty_values = binding_rows[penalty_col].apply(penalty_cell_to_float)
                        penalty_values = penalty_values[penalty_values > 0]
                        penalty_breakdown['binding']['amount'] = round(float(penalty_values.sum()), 2)
                        penalty_breakdown['binding']['count'] = int(len(penalty_values))

        # 5b. Hub penalty (Dispatcher ID, Amount)
        if hub_df is not None and not hub_df.empty:
            amount, count = sum_dispatcher_amount_penalty_float(hub_df, dispatcher_id)
            penalty_breakdown["hub"]["amount"] = amount
            penalty_breakdown["hub"]["count"] = count

        # 6. Pending Parcel penalty (sum Amount column by dispatcher)
        if pending_parcel_df is not None and not pending_parcel_df.empty:
            pp_disp_col = find_penalty_dispatcher_column(pending_parcel_df)
            if pp_disp_col is not None:
                pp_rows = filter_rows_for_penalty_dispatcher(pending_parcel_df, pp_disp_col, dispatcher_id)
                if not pp_rows.empty:
                    awb_col = find_penalty_waybill_column(pending_parcel_df)
                    amount_col = find_column(pending_parcel_df, ["Amount", "amount", "AMOUNT"])
                    if awb_col is not None:
                        waybills = extract_waybill_list(pp_rows[awb_col])
                        parcel_count = len(waybills) if waybills else len(pp_rows)
                        penalty_breakdown['pending_parcel']['waybills'] = waybills
                    else:
                        parcel_count = len(pp_rows)
                    if amount_col is not None:
                        amount_values = pp_rows[amount_col].apply(penalty_cell_to_float)
                        penalty_breakdown['pending_parcel']['amount'] = round(float(amount_values.sum()), 2)
                    else:
                        # Fallback for legacy sheets without Amount column
                        penalty_breakdown['pending_parcel']['amount'] = round(
                            float(parcel_count) * float(pending_parcel_penalty_per_parcel), 2
                        )
                    penalty_breakdown['pending_parcel']['count'] = int(parcel_count)

        # 7. Parcel Lost penalty (sum amount column by dispatcher_id)
        if parcel_lost_df is not None and not parcel_lost_df.empty:
            pl_disp_col = find_penalty_dispatcher_column(parcel_lost_df)
            if pl_disp_col:
                pl_rows = filter_rows_for_penalty_dispatcher(parcel_lost_df, pl_disp_col, dispatcher_id)
                if not pl_rows.empty:
                    pl_amount_col = find_column(pl_rows, ["amount", "Amount", "AMOUNT"])
                    pl_waybill_col = find_penalty_waybill_column(pl_rows)
                    if pl_amount_col:
                        penalty_values = pl_rows[pl_amount_col].apply(penalty_cell_to_float)
                        penalty_breakdown['parcel_lost']['amount'] = round(float(penalty_values.sum()), 2)
                        penalty_breakdown['parcel_lost']['count'] = int(len(pl_rows))
                    if pl_waybill_col:
                        penalty_breakdown['parcel_lost']['waybills'] = extract_waybill_list(pl_rows[pl_waybill_col])

        # 8. Route penalty: total from config divided equally among dispatchers (see main / calculate_tiered_daily)
        if route_penalty_per_dispatcher > 0:
            penalty_breakdown['route']['amount'] = round(float(route_penalty_per_dispatcher), 2)
            penalty_breakdown['route']['count'] = int(route_penalty_dispatcher_count) if route_penalty_dispatcher_count else 0
            penalty_breakdown['route']['pool_total'] = round(float(route_penalty_pool_total), 2)

        # 9. Process Attendance penalty (Attendance sheet) - direct penalty column sum by dispatcher_id
        if attendance_df is not None and not attendance_df.empty:
            attendance_rows = filter_penalty_sheet_for_dispatcher(attendance_df, dispatcher_id)

            if not attendance_rows.empty:
                penalty_col = find_column(attendance_rows, ["Penalty", "penalty", "ATTENDANCE PENALTY", "attendance_penalty", "attendance penalty"])
                if penalty_col:
                    penalty_values = attendance_rows[penalty_col].apply(
                        lambda x: PayoutCalculator._convert_to_float(x)
                    )
                    penalty_breakdown['attendance']['amount'] = round(float(penalty_values.sum()), 2)
                    penalty_breakdown['attendance']['count'] = int((penalty_values > 0).sum())

        # Calculate totals (round to 2 decimal places to avoid floating-point precision issues)
        penalty_breakdown['total_amount'] = round(
            penalty_breakdown['duitnow']['amount'] +
            penalty_breakdown['ldr']['amount'] +
            penalty_breakdown['fake_attempt']['amount'] +
            penalty_breakdown['cod']['amount'] +
            penalty_breakdown['binding']['amount'] +
            penalty_breakdown['hub']['amount'] +
            penalty_breakdown['pending_parcel']['amount'] +
            penalty_breakdown['no_outbound_scan']['amount'] +
            penalty_breakdown['parcel_lost']['amount'] +
            penalty_breakdown['route']['amount'] +
            penalty_breakdown['attendance']['amount'],
            2
        )
        penalty_breakdown['total_count'] = (
            penalty_breakdown['duitnow']['count'] +
            penalty_breakdown['ldr']['count'] +
            penalty_breakdown['fake_attempt']['count'] +
            penalty_breakdown['cod']['count'] +
            penalty_breakdown['binding']['count'] +
            penalty_breakdown['hub']['count'] +
            penalty_breakdown['pending_parcel']['count'] +
            penalty_breakdown['no_outbound_scan']['count'] +
            penalty_breakdown['parcel_lost']['count'] +
            penalty_breakdown['attendance']['count'] +
            (1 if penalty_breakdown['route']['amount'] > 0 else 0)
        )

        return penalty_breakdown

    @staticmethod
    def calculate_pickup(pickup_df: pd.DataFrame, dispatcher_id: str, rate: float = 1.00) -> Tuple[int, float, pd.DataFrame]:
        """
        Count pickup parcels for a dispatcher and compute payout from Order Source rules.

        Uses Order Source + Billing Weight when available; otherwise commission column
        or flat fallback rate per parcel.

        Returns:
            Tuple of (parcel_count, payout, filtered_pickup_df)
        """
        if pickup_df is None or pickup_df.empty or not dispatcher_id:
            return 0, 0.0, pd.DataFrame()

        # Find dispatcher column - handle multiple possible column name formats
        dispatcher_col = None
        possible_dispatcher_cols = [
            "Pick Up Dispatcher ID", "Pick Up Dispatcher Id", "Pickup Dispatcher ID",
            "pickup_dispatcher_id", "pickup_dispatcher", "Pickup Dispatcher",
            "dispatcher_id", "Dispatcher ID"
        ]
        for col_name in possible_dispatcher_cols:
            if col_name in pickup_df.columns:
                dispatcher_col = col_name
                break

        # If not found, try case-insensitive search
        if dispatcher_col is None:
            for col in pickup_df.columns:
                col_lower = str(col).lower().strip()
                if "pickup" in col_lower and "dispatcher" in col_lower:
                    dispatcher_col = col
                    break

        # Find waybill column - handle multiple possible column name formats
        waybill_col = None
        possible_waybill_cols = [
            "Waybill Number", "Waybill", "waybill_number", "waybill",
            "Waybill No", "AWB", "No. AWB"
        ]
        for col_name in possible_waybill_cols:
            if col_name in pickup_df.columns:
                waybill_col = col_name
                break

        # If not found, try case-insensitive search
        if waybill_col is None:
            for col in pickup_df.columns:
                col_lower = str(col).lower().strip()
                if "waybill" in col_lower or "awb" in col_lower:
                    waybill_col = col
                    break

        # Make sure required columns exist
        if dispatcher_col is None or waybill_col is None:
            return 0, 0.0, pd.DataFrame()

        # Convert dispatcher_id to string for comparison
        dispatcher_id_str = str(dispatcher_id).strip()

        # Clean and prepare the dispatcher ID column
        pickup_df = pickup_df.copy()
        pickup_df[dispatcher_col] = pickup_df[dispatcher_col].astype(str).str.strip()

        # Filter records for this dispatcher
        matched_records = pickup_df[pickup_df[dispatcher_col] == dispatcher_id_str]

        if matched_records.empty:
            return 0, 0.0, pd.DataFrame()

        # Count unique waybills - TREAT ALL WAYBILLS AS STRINGS
        # Convert to string to preserve format (including "-", letters, numbers, etc.)
        # Only filter out NaN values, keep everything else
        def safe_waybill_to_string(value):
            """Safely convert waybill to string, preserving original format.

            Handles numeric waybills (int/float) by converting to string without decimal notation.
            """
            if pd.isna(value):
                return None
            # Handle numeric types (int/float) - convert to string without decimal notation
            if isinstance(value, (int, float)):
                # If it's a float with no decimal part (e.g., 631891688390.0), convert to int first
                if isinstance(value, float) and value.is_integer():
                    return str(int(value))
                else:
                    return str(value)
            # For strings and other types, convert to string and strip
            waybill_str = str(value).strip()
            if waybill_str == "" or waybill_str.lower() == "nan":
                return None
            return waybill_str

        waybill_strings = matched_records[waybill_col].apply(safe_waybill_to_string)

        # Filter records to only include those with valid waybills
        valid_waybill_mask = waybill_strings.notna() & (waybill_strings != "")
        valid_records = matched_records[valid_waybill_mask].copy()

        if valid_records.empty:
            return 0, 0.0, pd.DataFrame()

        # Convert waybill column to string in the returned DataFrame
        # This ensures waybills are treated as strings and preserves format
        if waybill_col in valid_records.columns:
            valid_records[waybill_col] = valid_records[waybill_col].apply(safe_waybill_to_string)

        parcel_count = len(valid_records)

        commission_values = compute_pickup_commission_series(valid_records, fallback_rate=rate).round(2)
        commission_col = find_pickup_commission_column(valid_records)
        if commission_col is None:
            valid_records["Commission"] = commission_values
        else:
            valid_records[commission_col] = commission_values

        payout = round(float(commission_values.sum()), 2)

        return parcel_count, payout, valid_records

    @staticmethod
    def calculate_reward(
        reward_df: pd.DataFrame,
        dispatcher_id: str,
    ) -> Tuple[float, int, pd.DataFrame]:
        """Sum reward amount for dispatcher from Reward sheet (employee_id + amount)."""
        if reward_df is None or reward_df.empty or not dispatcher_id:
            return 0.0, 0, pd.DataFrame()

        disp_id_col = find_reward_employee_column(reward_df)
        amount_col = find_amount_column(reward_df)
        if disp_id_col is None or amount_col is None:
            return 0.0, 0, pd.DataFrame()

        dispatcher_key = clean_penalty_dispatcher_id(dispatcher_id)
        reward_copy = reward_df.copy()
        reward_copy["_dispatcher_id"] = reward_copy[disp_id_col].apply(clean_penalty_dispatcher_id)
        matched = reward_copy[reward_copy["_dispatcher_id"] == dispatcher_key]
        if matched.empty:
            return 0.0, 0, pd.DataFrame()

        amounts = pd.to_numeric(matched[amount_col], errors="coerce").fillna(0.0)
        amounts = amounts[amounts > 0]
        reward_amount = round(float(amounts.sum()), 2)
        return reward_amount, int(len(amounts)), matched

    @staticmethod
    def calculate_return(return_df: pd.DataFrame, dispatcher_id: str, rate: float = 0.50) -> Tuple[int, float, pd.DataFrame]:
        """
        Counts total return parcels for selected dispatcher based on dispatcher_id column matching the dispatcher_id.
        Filters out rows with empty or invalid waybill numbers.

        Returns:
            Tuple of (return_count, payout, filtered_return_df)
        """
        if return_df is None or return_df.empty or not dispatcher_id:
            return 0, 0.0, pd.DataFrame()

        # Guard: if requested Return sheet falls back to Dispatch data (when tab is missing),
        # do not count delivery rows as returns.
        return_cols_lower = {str(c).strip().lower() for c in return_df.columns}
        has_dispatch_signature = (
            "delivery signature" in return_cols_lower and
            ("waybill number" in return_cols_lower or "waybill" in return_cols_lower) and
            ("dispatcher id" in return_cols_lower or "dispatcher_id" in return_cols_lower)
        )
        has_return_markers = any("return" in col for col in return_cols_lower)
        if has_dispatch_signature and not has_return_markers:
            return 0, 0.0, pd.DataFrame()

        # Find dispatcher column - handle multiple possible column name formats
        dispatcher_col = None
        possible_dispatcher_cols = [
            "dispatcher_id", "Dispatcher ID", "dispatcher", "Dispatcher", "DISPATCHER_ID", "DISPATCHER ID"
        ]
        for col_name in possible_dispatcher_cols:
            if col_name in return_df.columns:
                dispatcher_col = col_name
                break

        # If not found, try case-insensitive search
        if dispatcher_col is None:
            for col in return_df.columns:
                col_lower = str(col).lower().strip()
                if "dispatcher" in col_lower and "id" in col_lower:
                    dispatcher_col = col
                    break

        # Find waybill column - handle multiple possible column name formats
        waybill_col = None
        possible_waybill_cols = [
            "waybill_number", "Waybill Number", "waybill", "Waybill", "WAYBILL_NUMBER", "WAYBILL NUMBER"
        ]
        for col_name in possible_waybill_cols:
            if col_name in return_df.columns:
                waybill_col = col_name
                break

        # If not found, try case-insensitive search
        if waybill_col is None:
            for col in return_df.columns:
                col_lower = str(col).lower().strip()
                if "waybill" in col_lower:
                    waybill_col = col
                    break

        # Make sure required columns exist
        if dispatcher_col is None or waybill_col is None:
            return 0, 0.0, pd.DataFrame()

        # Convert dispatcher_id to string for comparison
        dispatcher_id_str = str(dispatcher_id).strip()

        # Clean and prepare the dispatcher column
        return_df = return_df.copy()
        return_df[dispatcher_col] = return_df[dispatcher_col].astype(str).str.strip()

        # Filter records for this dispatcher
        matched_records = return_df[return_df[dispatcher_col] == dispatcher_id_str]

        if matched_records.empty:
            return 0, 0.0, pd.DataFrame()

        # Count unique waybills - TREAT ALL WAYBILLS AS STRINGS
        # Convert to string to preserve format
        # Only filter out NaN values, keep everything else
        def safe_waybill_to_string(value):
            """Safely convert waybill to string, preserving original format."""
            if pd.isna(value):
                return None
            # Handle numeric types (int/float) - convert to string without decimal notation
            if isinstance(value, (int, float)):
                # If it's a float with no decimal part, convert to int first
                if isinstance(value, float) and value.is_integer():
                    return str(int(value))
                else:
                    return str(value)
            # For strings and other types, convert to string and strip
            waybill_str = str(value).strip()
            if waybill_str == "" or waybill_str.lower() == "nan":
                return None
            return waybill_str

        waybill_strings = matched_records[waybill_col].apply(safe_waybill_to_string)

        # Filter records to only include those with valid waybills
        valid_waybill_mask = waybill_strings.notna() & (waybill_strings != "")
        valid_records = matched_records[valid_waybill_mask].copy()

        if valid_records.empty:
            return 0, 0.0, pd.DataFrame()

        # Convert waybill column to string in the returned DataFrame
        # This ensures waybills are treated as strings and preserves format
        if waybill_col in valid_records.columns:
            valid_records[waybill_col] = valid_records[waybill_col].apply(safe_waybill_to_string)

        # Count total return parcels (rows) - each row represents one return parcel
        return_count = len(valid_records)

        payout = round(return_count * rate, 2)

        return return_count, payout, valid_records

    @staticmethod
    def calculate_tiered_daily(filtered_df: pd.DataFrame, tiers_config: List,
                              kpi_config: List, special_rates_config: List,
                              attendance_config: dict, currency_symbol: str,
                              duitnow_df: Optional[pd.DataFrame] = None,
                              ldr_df: Optional[pd.DataFrame] = None,
                              fake_df: Optional[pd.DataFrame] = None,
                              cod_df: Optional[pd.DataFrame] = None,
                              binding_df: Optional[pd.DataFrame] = None,
                              hub_df: Optional[pd.DataFrame] = None,
                              pending_parcel_df: Optional[pd.DataFrame] = None,
                              no_outbound_scan_df: Optional[pd.DataFrame] = None,
                              parcel_lost_df: Optional[pd.DataFrame] = None,
                              attendance_df: Optional[pd.DataFrame] = None,
                              fake_attempt_penalty_per_parcel: float = 2.0,
                              pending_parcel_penalty_per_parcel: float = 2.0,
                              no_outbound_scan_penalty_per_parcel: float = 3.0,
                              return_df: Optional[pd.DataFrame] = None,
                              route_penalty_per_dispatcher: float = 0.0,
                              route_penalty_dispatcher_count: int = 0,
                              route_penalty_pool_total: float = 0.0,
                              pickup_df: Optional[pd.DataFrame] = None,
                              bulky_df: Optional[pd.DataFrame] = None,
                              designated_driver_config: Optional[dict] = None,
                              fallback_dispatcher_id: Optional[str] = None) -> Tuple:
        """Calculate tiered base delivery payout from dispatch and bulky sheet rows.

        Dispatch AWBs on Return or Pickup are excluded. Dispatch rows that also appear on
        the Bulky sheet are excluded from dispatch counting and counted via Bulky instead.
        """
        penalty_id = clean_penalty_dispatcher_id(fallback_dispatcher_id or "")
        dispatcher_id = str(fallback_dispatcher_id or "").strip()
        work = filtered_df.copy() if filtered_df is not None and not filtered_df.empty else pd.DataFrame()

        if not work.empty:
            dispatcher_id_col = find_dispatch_id_column(work)
            if dispatcher_id_col and dispatcher_id_col in work.columns:
                dispatcher_id = str(work[dispatcher_id_col].iloc[0]).strip() or dispatcher_id
                penalty_id = clean_penalty_dispatcher_id(dispatcher_id) or penalty_id

        def map_rate(daily_parcels: float) -> Tuple[str, float]:
            return PayoutCalculator.map_tier_rate(daily_parcels, tiers_config)

        delivery_sig_col = (
            find_column(work, ["Delivery Signature", "delivery_signature", "Delivery Signature", "delivery_sig"])
            if not work.empty
            else find_bulky_date_column(bulky_df)
        )
        waybill_col = find_waybill_column(work) if not work.empty else find_waybill_column(bulky_df)
        dispatcher_id_col = find_dispatch_id_column(work) if not work.empty else find_dispatch_id_column(bulky_df)

        if delivery_sig_col is None or waybill_col is None:
            if bulky_df is None or bulky_df.empty:
                per_day = pd.DataFrame(
                    columns=["__date", "daily_parcels", "tier", "rate_per_parcel", "payout_per_day"]
                )
                penalty_breakdown = PayoutCalculator.calculate_penalty(
                    penalty_id,
                    duitnow_df,
                    ldr_df,
                    fake_df,
                    cod_df,
                    binding_df,
                    hub_df,
                    pending_parcel_df,
                    no_outbound_scan_df,
                    parcel_lost_df,
                    attendance_df,
                    0,
                    fake_attempt_penalty_per_parcel,
                    pending_parcel_penalty_per_parcel,
                    no_outbound_scan_penalty_per_parcel,
                    route_penalty_per_dispatcher,
                    route_penalty_dispatcher_count,
                    route_penalty_pool_total,
                ) if penalty_id else {
                    'duitnow': {'amount': 0.0, 'count': 0, 'waybills': []},
                    'ldr': {'amount': 0.0, 'count': 0, 'waybills': []},
                    'fake_attempt': {'amount': 0.0, 'count': 0, 'waybills': []},
                    'cod': {'amount': 0.0, 'count': 0},
                    'binding': {'amount': 0.0, 'count': 0},
                    'hub': {'amount': 0.0, 'count': 0},
                    'pending_parcel': {'amount': 0.0, 'count': 0, 'waybills': []},
                    'no_outbound_scan': {'amount': 0.0, 'count': 0, 'waybills': []},
                    'parcel_lost': {'amount': 0.0, 'count': 0, 'waybills': []},
                    'route': {'amount': 0.0, 'count': 0, 'pool_total': 0.0},
                    'attendance': {'amount': 0.0, 'count': 0},
                    'total_amount': 0.0,
                    'total_count': 0,
                }
                display_df = pd.DataFrame(
                    columns=["Date", "Total Parcel", "Tier", "Payout Rate", "Payout"]
                )
                return (
                    display_df, 0.0, round(-penalty_breakdown['total_amount'], 2), 0.0, "",
                    0.0, "No delivery records in selected period", 0, per_day, penalty_breakdown,
                )
            raise ValueError("Required columns (Delivery Signature and Waybill Number) not found in data")

        if not work.empty:
            if (
                return_df is not None
                and not return_df.empty
                and not is_return_sheet_dispatch_fallback(return_df)
            ):
                return_disp_col = find_return_dispatcher_column(return_df)
                work = exclude_dispatch_rows_by_dispatcher_sheet(
                    work,
                    return_df,
                    return_disp_col,
                    dispatcher_id_col,
                    waybill_col,
                )

            if pickup_df is not None and not pickup_df.empty:
                pickup_waybills = build_pickup_waybill_set(pickup_df)
                work = exclude_dispatch_rows_by_waybill_set(
                    work, pickup_waybills, waybill_col
                )

            if bulky_df is not None and not bulky_df.empty:
                bulky_disp_col = find_dispatch_id_column(bulky_df)
                if bulky_disp_col is None:
                    bulky_disp_col = find_column(
                        bulky_df, ["Dispatcher ID", "dispatcher_id", "Dispatcher Id"]
                    )
                work = exclude_dispatch_rows_by_dispatcher_sheet(
                    work,
                    bulky_df,
                    bulky_disp_col,
                    dispatcher_id_col,
                    waybill_col,
                )

            work["__date"] = pd.to_datetime(work[delivery_sig_col], errors="coerce").dt.date
            work["__waybill_original"] = work[waybill_col].copy()
            work["__waybill"] = work[waybill_col].apply(normalize_waybill)
            work.loc[work["__waybill"] == "", "__waybill"] = pd.NA
            work["__date"] = work["__date"].fillna(pd.Timestamp.today().date())

        if not dispatcher_id and fallback_dispatcher_id:
            dispatcher_id = str(fallback_dispatcher_id).strip()

        work = PayoutCalculator._append_bulky_delivery_records(
            work,
            bulky_df,
            dispatcher_id or "",
        )

        if work.empty:
            penalty_breakdown = PayoutCalculator.calculate_penalty(
                penalty_id,
                duitnow_df,
                ldr_df,
                fake_df,
                cod_df,
                binding_df,
                hub_df,
                pending_parcel_df,
                no_outbound_scan_df,
                parcel_lost_df,
                attendance_df,
                0,
                fake_attempt_penalty_per_parcel,
                pending_parcel_penalty_per_parcel,
                no_outbound_scan_penalty_per_parcel,
                route_penalty_per_dispatcher,
                route_penalty_dispatcher_count,
                route_penalty_pool_total,
            ) if penalty_id else {
                'duitnow': {'amount': 0.0, 'count': 0, 'waybills': []},
                'ldr': {'amount': 0.0, 'count': 0, 'waybills': []},
                'fake_attempt': {'amount': 0.0, 'count': 0, 'waybills': []},
                'cod': {'amount': 0.0, 'count': 0},
                'binding': {'amount': 0.0, 'count': 0},
                'hub': {'amount': 0.0, 'count': 0},
                'pending_parcel': {'amount': 0.0, 'count': 0, 'waybills': []},
                'no_outbound_scan': {'amount': 0.0, 'count': 0, 'waybills': []},
                'parcel_lost': {'amount': 0.0, 'count': 0, 'waybills': []},
                'route': {'amount': 0.0, 'count': 0, 'pool_total': 0.0},
                'attendance': {'amount': 0.0, 'count': 0},
                'total_amount': 0.0,
                'total_count': 0,
            }
            per_day = pd.DataFrame(
                columns=["__date", "daily_parcels", "tier", "rate_per_parcel", "payout_per_day"]
            )
            display_df = pd.DataFrame(
                columns=["Date", "Total Parcel", "Tier", "Payout Rate", "Payout"]
            )
            return (
                display_df, 0.0, round(-penalty_breakdown['total_amount'], 2), 0.0, "",
                0.0, "No delivery records in selected period", 0, per_day, penalty_breakdown,
            )

        sort_cols = ["__date", "__waybill"]
        if delivery_sig_col in work.columns:
            sort_cols.append(delivery_sig_col)
        work = work.sort_values(by=sort_cols)

        per_day = (
            work.groupby(["__date"], dropna=False)
            .size()
            .reset_index(name="daily_parcels")
        )

        total_parcels = int(per_day["daily_parcels"].sum())

        if designated_driver_config:
            basic_amount = float(designated_driver_config.get("basic_amount", 1700.0))
            per_day = PayoutCalculator.apply_designated_driver_daily_payout(
                per_day, designated_driver_config, special_rates_config
            )
            variable_payout = round(float(per_day["payout_per_day"].sum()), 2)
            base_payout = round(basic_amount + variable_payout, 2)
            dd_kpi_config = designated_driver_config.get("kpi_incentives", [])
            kpi_bonus, kpi_description = PayoutCalculator.calculate_kpi_bonus(
                total_parcels, dd_kpi_config
            )
        else:
            per_day[["tier", "base_rate"]] = per_day["daily_parcels"].apply(
                lambda x: pd.Series(map_rate(float(x)))
            )

            per_day["special_rate"] = per_day.apply(
                lambda row: PayoutCalculator.get_special_rate(row["__date"], int(row["daily_parcels"]), special_rates_config)[0],
                axis=1
            )
            per_day["special_desc"] = per_day.apply(
                lambda row: PayoutCalculator.get_special_rate(row["__date"], int(row["daily_parcels"]), special_rates_config)[1],
                axis=1
            )
            per_day["rate_per_parcel"] = per_day.apply(
                lambda row: row["special_rate"] if pd.notna(row["special_rate"]) else row["base_rate"],
                axis=1
            )
            per_day["tier"] = per_day.apply(
                lambda row: row["special_desc"] if pd.notna(row["special_rate"]) else row["tier"],
                axis=1
            )

            per_day["payout_per_day"] = per_day["daily_parcels"] * per_day["rate_per_parcel"]
            base_payout = round(float(per_day["payout_per_day"].sum()), 2)
            kpi_bonus, kpi_description = PayoutCalculator.calculate_kpi_bonus(total_parcels, kpi_config)

        attendance_bonus, attendance_desc, qualified_days = PayoutCalculator.calculate_attendance_bonus(
            per_day, attendance_config
        )

        # Calculate penalty breakdown
        penalty_breakdown = {'duitnow': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'ldr': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'fake_attempt': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'cod': {'amount': 0.0, 'count': 0},
                           'binding': {'amount': 0.0, 'count': 0},
                           'hub': {'amount': 0.0, 'count': 0},
                           'pending_parcel': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'no_outbound_scan': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'parcel_lost': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'route': {'amount': 0.0, 'count': 0, 'pool_total': 0.0},
                           'attendance': {'amount': 0.0, 'count': 0},
                           'total_amount': 0.0, 'total_count': 0}

        # Extract dispatcher_id for penalty lookup.
        penalty_id = clean_penalty_dispatcher_id(dispatcher_id or fallback_dispatcher_id or "")
        if penalty_id:
            working_days = len(per_day)
            penalty_breakdown = PayoutCalculator.calculate_penalty(
                str(penalty_id),
                duitnow_df,
                ldr_df,
                fake_df,
                cod_df,
                binding_df,
                hub_df,
                pending_parcel_df,
                no_outbound_scan_df,
                parcel_lost_df,
                attendance_df,
                working_days,
                fake_attempt_penalty_per_parcel,
                pending_parcel_penalty_per_parcel,
                no_outbound_scan_penalty_per_parcel,
                route_penalty_per_dispatcher,
                route_penalty_dispatcher_count,
                route_penalty_pool_total,
            )

        # Calculate gross payout (base + bonuses - penalties) - round to 2 decimal places
        gross_payout = round(
            base_payout + kpi_bonus + attendance_bonus - penalty_breakdown['total_amount'],
            2
        )

        display_df = per_day[["__date", "daily_parcels", "tier", "rate_per_parcel", "payout_per_day"]].copy()
        display_df = display_df.rename(columns={
            "__date": "Date",
            "daily_parcels": "Total Parcel",
            "tier": "Tier",
            "rate_per_parcel": "Payout Rate",
            "payout_per_day": "Payout",
        })

        display_df["Payout Rate"] = display_df["Payout Rate"].apply(
            lambda x: "Basic" if designated_driver_config and (x == 0 or x == 0.0) else f"{currency_symbol}{x:.2f}"
        )
        display_df["Payout"] = display_df["Payout"].apply(lambda x: f"{currency_symbol}{x:.2f}")

        return (display_df, base_payout, gross_payout, kpi_bonus, kpi_description, attendance_bonus,
                attendance_desc, qualified_days, per_day, penalty_breakdown)

    @staticmethod
    def calculate_designated_driver(
        filtered_df: pd.DataFrame,
        designated_driver_config: dict,
        attendance_config: dict,
        currency_symbol: str,
        duitnow_df: Optional[pd.DataFrame] = None,
        ldr_df: Optional[pd.DataFrame] = None,
        fake_df: Optional[pd.DataFrame] = None,
        cod_df: Optional[pd.DataFrame] = None,
        binding_df: Optional[pd.DataFrame] = None,
        hub_df: Optional[pd.DataFrame] = None,
        pending_parcel_df: Optional[pd.DataFrame] = None,
        no_outbound_scan_df: Optional[pd.DataFrame] = None,
        parcel_lost_df: Optional[pd.DataFrame] = None,
        attendance_df: Optional[pd.DataFrame] = None,
        fake_attempt_penalty_per_parcel: float = 2.0,
        pending_parcel_penalty_per_parcel: float = 2.0,
        no_outbound_scan_penalty_per_parcel: float = 3.0,
        return_df: Optional[pd.DataFrame] = None,
        route_penalty_per_dispatcher: float = 0.0,
        route_penalty_dispatcher_count: int = 0,
        route_penalty_pool_total: float = 0.0,
        pickup_df: Optional[pd.DataFrame] = None,
        bulky_df: Optional[pd.DataFrame] = None,
        special_rates_config: Optional[List] = None,
        fallback_dispatcher_id: Optional[str] = None,
    ) -> Tuple:
        """Calculate payout for designated-driver position.

        Basic RM1700 covers the first 700 parcels, then RM1.00 per parcel after that.
        KPI rewards: RM100 at 3500 parcels/month, RM150 at 4500 parcels/month.
        Qualifying special-rate days use the configured special rate for all parcels that day.
        """
        return PayoutCalculator.calculate_tiered_daily(
            filtered_df,
            tiers_config=[],
            kpi_config=[],
            special_rates_config=special_rates_config or [],
            attendance_config=attendance_config,
            currency_symbol=currency_symbol,
            duitnow_df=duitnow_df,
            ldr_df=ldr_df,
            fake_df=fake_df,
            cod_df=cod_df,
            binding_df=binding_df,
            hub_df=hub_df,
            pending_parcel_df=pending_parcel_df,
            no_outbound_scan_df=no_outbound_scan_df,
            parcel_lost_df=parcel_lost_df,
            attendance_df=attendance_df,
            fake_attempt_penalty_per_parcel=fake_attempt_penalty_per_parcel,
            pending_parcel_penalty_per_parcel=pending_parcel_penalty_per_parcel,
            no_outbound_scan_penalty_per_parcel=no_outbound_scan_penalty_per_parcel,
            return_df=return_df,
            route_penalty_per_dispatcher=route_penalty_per_dispatcher,
            route_penalty_dispatcher_count=route_penalty_dispatcher_count,
            route_penalty_pool_total=route_penalty_pool_total,
            pickup_df=pickup_df,
            bulky_df=bulky_df,
            designated_driver_config=designated_driver_config,
            fallback_dispatcher_id=fallback_dispatcher_id,
        )


# =============================================================================
# DATA VISUALIZATION
# =============================================================================

class DataVisualizer:
    """Create performance charts and graphs."""

    @staticmethod
    def create_performance_charts(per_day_df: pd.DataFrame, currency_symbol: str):
        return DataVisualizer._create_tiered_daily_charts(per_day_df, currency_symbol)

    @staticmethod
    def _create_tiered_daily_charts(per_day_df: pd.DataFrame, currency_symbol: str):
        charts = {}
        if per_day_df.empty:
            return charts

        chart_data = per_day_df.copy()
        chart_data['date'] = pd.to_datetime(chart_data['__date'])

        base = alt.Chart(chart_data).encode(
            x=alt.X('date:T', title='Date', axis=alt.Axis(format='%b %d'))
        )

        parcels_bar = base.mark_bar(color=ColorScheme.PRIMARY, opacity=0.7).encode(
            y=alt.Y('daily_parcels:Q', title='Parcels', axis=alt.Axis(grid=False)),
            tooltip=['date:T', 'daily_parcels:Q', 'tier:N']
        )

        payout_line = base.mark_line(stroke=ColorScheme.SECONDARY, strokeWidth=3).encode(
            y=alt.Y('payout_per_day:Q', title=f'Payout ({currency_symbol})', axis=alt.Axis(gridColor=ColorScheme.BORDER)),
            tooltip=['date:T', alt.Tooltip('payout_per_day:Q', format='.2f', title=f'Payout ({currency_symbol})')]
        )

        parcels_chart = alt.layer(parcels_bar, payout_line).resolve_scale(
            y='independent'
        ).properties(
            title='Daily Parcels and Payout',
            width=400,
            height=300
        )
        charts['parcels_payout'] = parcels_chart

        performance_chart = alt.Chart(chart_data).mark_circle(size=60).encode(
            x=alt.X('daily_parcels:Q', title='Parcels Delivered', scale=alt.Scale(zero=False)),
            y=alt.Y('payout_per_day:Q', title=f'Daily Payout ({currency_symbol})', scale=alt.Scale(zero=False)),
            color=alt.Color('tier:N', scale=alt.Scale(range=ColorScheme.CHART_COLORS), legend=alt.Legend(title='Tier')),
            size=alt.Size('daily_parcels:Q', legend=None),
            tooltip=['date:T', 'daily_parcels:Q', alt.Tooltip('payout_per_day:Q', format='.2f'), 'tier:N']
        ).properties(
            title='Payout vs Parcels Performance',
            width=400,
            height=300
        )
        charts['performance_scatter'] = performance_chart

        payout_trend = alt.Chart(chart_data).mark_area(
            color=ColorScheme.SECONDARY,
            opacity=0.3,
            line={'color': ColorScheme.SECONDARY, 'width': 2}
        ).encode(
            x=alt.X('date:T', title='Date', axis=alt.Axis(format='%b %d')),
            y=alt.Y('payout_per_day:Q', title=f'Daily Payout ({currency_symbol})'),
            tooltip=['date:T', alt.Tooltip('payout_per_day:Q', format='.2f', title=f'Payout ({currency_symbol})'), 'tier:N']
        ).properties(
            title='Daily Payout Trend',
            width=400,
            height=300
        )
        charts['payout_trend'] = payout_trend

        return charts


# =============================================================================
# NAME CLEANING
# =============================================================================

def clean_dispatcher_name(name: str) -> str:
    prefixes = ['JMR', 'ECP', 'AF', 'PEN', 'KUL', 'JHR']
    cleaned_name = str(name).strip()
    for prefix in prefixes:
        if cleaned_name.startswith(prefix):
            cleaned_name = cleaned_name[len(prefix):].strip()
            cleaned_name = cleaned_name.lstrip(' -')
            break
    return cleaned_name


# =============================================================================
# INVOICE GENERATION
# =============================================================================

class InvoiceGenerator:
    """Generate professional invoices."""

    @staticmethod
    def build_invoice_html(
        df_disp: pd.DataFrame, base_payout: float, gross_payout: float, kpi_bonus: float,
        attendance_bonus: float, advance_payout: float, advance_payout_desc: str,
        penalty_breakdown: Dict, name: str, dpid: str, currency_symbol: str,
        socso_deduction: float = 0.0,
        socso_deduction_count: int = 0,
        overpaid_deduction: float = 0.0,
        overpaid_deduction_count: int = 0,
        reward_payout: float = 0.0,
        reward_count: int = 0,
        pickup_payout: float = 0.0,
        pickup_parcels: int = 0,
        return_payout: float = 0.0,
        return_count: int = 0,
        kpi_description: str = "",
        attendance_description: str = "",
        total_awb: int = 0,
        designated_driver_breakdown: Optional[dict] = None,
        pickup_rate: float = 1.0,
        return_rate: float = 0.5,
        advance_percentage: float = 40.0,
        advance_enabled: bool = True,
    ) -> str:
        total_parcels = df_disp['Total Parcel'].sum() if 'Total Parcel' in df_disp.columns else 0
        total_days = len(df_disp) if 'Date' in df_disp.columns else 0
        cleaned_name = clean_dispatcher_name(name)

        penalty_total = float(penalty_breakdown.get("total_amount", 0.0) or 0.0)
        gross_payout_amount = PayoutCalculator.calculate_gross_payout(
            base_payout,
            pickup_payout,
            return_payout,
            reward_payout,
            kpi_bonus,
            attendance_bonus,
            penalty_total,
            socso_deduction,
            overpaid_deduction,
        )
        advance_line_amount = PayoutCalculator.calculate_advance_payout(
            base_payout,
            advance_percentage,
            advance_enabled,
        )
        final_payout = PayoutCalculator.calculate_final_payout(
            gross_payout_amount,
            advance_line_amount,
        )

        gross_tooltip = (
            "Base delivery + Commission Delivery + Return + Reward "
            "+ KPI bonus + Attendance bonus − Total penalty − SOCSO − Overpaid."
        )
        advance_tooltip = (
            f"{advance_percentage:g}% of base delivery payout "
            f"({currency_symbol}{base_payout:,.2f})."
        )
        final_tooltip = "Gross Payout − Advance payout."

        html_content = f"""
        <html>
        <head>
            <style>
            :root {{
              --primary: {ColorScheme.PRIMARY};
              --primary-light: {ColorScheme.PRIMARY_LIGHT};
              --secondary: {ColorScheme.SECONDARY};
              --accent: {ColorScheme.ACCENT};
              --background: {ColorScheme.BACKGROUND};
              --surface: {ColorScheme.SURFACE};
              --text-primary: {ColorScheme.TEXT_PRIMARY};
              --text-secondary: {ColorScheme.TEXT_SECONDARY};
              --border: {ColorScheme.BORDER};
              --success: {ColorScheme.SUCCESS};
              --warning: {ColorScheme.WARNING};
              --error: {ColorScheme.ERROR};
            }}
            html, body {{ margin: 0; padding: 0; background: var(--background); }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif; color: var(--text-primary); line-height: 1.5; }}
            .container {{ max-width: 960px; margin: 24px auto; padding: 0 16px; }}
            .header {{
              display: grid; grid-template-columns: 1fr auto; gap: 16px;
              border: 1px solid var(--border); border-radius: 12px; padding: 24px; align-items: center;
              background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
              color: white;
              box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            }}
            .brand {{ font-weight: 700; font-size: 24px; letter-spacing: -0.5px; }}
            .idline {{ opacity: 0.9; font-size: 14px; margin-top: 4px; }}
            .summary {{
              margin-top: 24px; display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
            }}
            .chip {{
              border: 1px solid var(--border); border-radius: 12px;
              padding: 16px; background: var(--surface); min-width: 160px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1);
              transition: transform 0.2s, box-shadow 0.2s;
              text-align: center;
            }}
            .chip:hover {{
              transform: translateY(-2px);
              box-shadow: 0 4px 12px rgba(0,0,0,0.15);
              border-color: var(--primary-light);
            }}
            .chip .label {{ color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
            .chip .value {{ font-size: 18px; font-weight: 700; margin-top: 6px; color: var(--primary); }}
            .bonus-section {{
              margin-top: 24px;
              display: grid;
              grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
              gap: 12px;
            }}
            .kpi-bonus {{
              padding: 16px;
              background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
              border-radius: 12px;
              color: white;
              box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            .attendance-bonus {{
              padding: 16px;
              background: linear-gradient(135deg, var(--accent) 0%, #d97706 100%);
              border-radius: 12px;
              color: white;
              box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            .penalty-section {{
              padding: 16px;
              background: linear-gradient(135deg, var(--error) 0%, #dc2626 100%);
              border-radius: 12px;
              color: white;
              box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            }}
            .penalty-detail {{
              margin-top: 12px;
              padding: 12px;
              background: rgba(255, 255, 255, 0.1);
              border-radius: 8px;
              font-size: 13px;
            }}
            .penalty-item {{
              display: flex;
              justify-content: space-between;
              padding: 6px 0;
              border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .penalty-item:last-child {{
              border-bottom: none;
            }}
            .bonus-title {{ font-size: 14px; opacity: 0.9; margin-bottom: 8px; }}
            .bonus-amount {{ font-size: 28px; font-weight: 800; }}
            .bonus-description {{ font-size: 12px; opacity: 0.9; margin-top: 4px; }}
            table {{
              border-collapse: collapse; width: 100%; margin-top: 24px;
              box-shadow: 0 1px 3px rgba(0,0,0,0.1);
              border-radius: 8px; overflow: hidden;
              border: 1px solid var(--border);
            }}
            th, td {{ border: none; padding: 12px 16px; text-align: left; font-size: 14px; }}
            th {{
              background: var(--primary);
              color: white;
              font-weight: 600;
              text-transform: uppercase;
              letter-spacing: 0.5px;
              font-size: 12px;
            }}
            tbody tr:nth-child(even) {{ background: var(--background); }}
            tbody tr:hover {{ background: rgba(79, 70, 229, 0.05); }}
            .payout-summary {{
              margin-top: 24px;
              padding: 20px;
              background: var(--surface);
              border: 2px solid var(--border);
              border-radius: 12px;
            }}
            .payout-row {{
              display: flex;
              justify-content: space-between;
              padding: 8px 0;
              font-size: 16px;
            }}
            .payout-row.total {{
              border-top: 2px solid var(--border);
              margin-top: 12px;
              padding-top: 12px;
              font-weight: 800;
              font-size: 20px;
              color: var(--primary);
            }}
            .payout-row.penalty-detail-row {{
              font-size: 14px;
              color: var(--error);
              padding-left: 20px;
            }}
            .payout-row.payout-detail-row {{
              font-size: 14px;
              color: var(--text-secondary);
              padding-left: 20px;
            }}
            .note {{
              margin-top: 8px;
              color: var(--text-secondary);
              font-size: 12px;
              text-align: center;
              padding: 16px;
            }}
            .total-badge {{
              background: rgba(255,255,255,0.2);
              padding: 8px 16px;
              border-radius: 20px;
              text-align: center;
              border: 1px solid rgba(255,255,255,0.3);
            }}
            .total-badge .label {{
              font-size: 12px;
              opacity: 0.9;
              margin-bottom: 4px;
            }}
            .total-badge .value {{
              font-size: 28px;
              font-weight: 800;
            }}
            .tooltip-wrap {{
              display: inline-flex;
              align-items: center;
              gap: 6px;
            }}
            .tooltip-icon {{
              display: inline-flex;
              align-items: center;
              justify-content: center;
              width: 16px;
              height: 16px;
              border-radius: 50%;
              background: var(--text-secondary);
              color: white;
              font-size: 11px;
              font-weight: 700;
              cursor: help;
              position: relative;
            }}
            .tooltip-icon .tooltiptext {{
              visibility: hidden;
              opacity: 0;
              width: 280px;
              background: #1f2937;
              color: #fff;
              text-align: left;
              border-radius: 8px;
              padding: 10px 12px;
              position: absolute;
              z-index: 10;
              bottom: 125%;
              left: 50%;
              margin-left: -140px;
              font-size: 12px;
              font-weight: 400;
              line-height: 1.45;
              box-shadow: 0 4px 12px rgba(0,0,0,0.15);
              transition: opacity 0.2s;
            }}
            .tooltip-icon:hover .tooltiptext {{
              visibility: visible;
              opacity: 1;
            }}
            @media (max-width: 768px) {{
              .header {{ grid-template-columns: 1fr; text-align: center; }}
              .summary {{ flex-direction: column; }}
              .chip {{ min-width: auto; }}
              .bonus-section {{ grid-template-columns: 1fr; }}
            }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div>
                        <div class="brand">Invoice</div>
                        <div class="idline">From: JEMARI VENTURES &nbsp;&nbsp;|&nbsp;&nbsp; To: {cleaned_name}</div>
                    </div>
                    <div class="total-badge">
                        <div class="label">Gross Payout</div>
                        <div class="value">{currency_symbol} {gross_payout_amount:,.2f}</div>
                    </div>
                </div>

                <div class="summary">
                    <div class="chip">
                        <div class="label">Dispatcher ID</div>
                        <div class="value">{dpid}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Total AWB</div>
                        <div class="value">{total_awb:,}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Total Delivery Parcels</div>
                        <div class="value">{total_parcels:,}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Pickup Parcels</div>
                        <div class="value">{pickup_parcels:,}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Return Parcels</div>
                        <div class="value">{return_count:,}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Working Days</div>
                        <div class="value">{total_days}</div>
                    </div>
                </div>
        """

        # Bonus and Penalty Section
        has_bonuses_or_penalties = (kpi_bonus > 0 or attendance_bonus > 0 or penalty_breakdown['total_amount'] > 0)

        if has_bonuses_or_penalties:
            html_content += '<div class="bonus-section">'

            if kpi_bonus > 0:
                html_content += f"""
                    <div class="kpi-bonus">
                        <div class="bonus-title">KPI Incentive Achieved!</div>
                        <div class="bonus-amount">+ {currency_symbol} {kpi_bonus:,.2f}</div>
                        <div class="bonus-description">{kpi_description}</div>
                    </div>
                """

            if attendance_bonus > 0:
                html_content += f"""
                    <div class="attendance-bonus">
                        <div class="bonus-title">Attendance Incentive Achieved!</div>
                        <div class="bonus-amount">+ {currency_symbol} {attendance_bonus:,.2f}</div>
                        <div class="bonus-description">{attendance_description}</div>
                    </div>
                """

            if penalty_breakdown['total_amount'] > 0:
                html_content += f"""
                    <div class="penalty-section">
                        <div class="bonus-title">Penalty Applied</div>
                        <div class="bonus-amount">- {currency_symbol} {penalty_breakdown['total_amount']:,.2f}</div>
                        <div class="bonus-description">{penalty_breakdown['total_count']} record(s) affected</div>
                        <div class="penalty-detail">
                """

                # DuitNow penalties
                if penalty_breakdown['duitnow']['amount'] > 0:
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>DuitNow:</strong></span>
                            <span>- {currency_symbol} {penalty_breakdown['duitnow']['amount']:,.2f}</span>
                        </div>
                    """

                # LD&R penalties
                if penalty_breakdown['ldr']['count'] > 0:
                    waybills_ldr = ", ".join(penalty_breakdown['ldr']['waybills'][:3])
                    if len(penalty_breakdown['ldr']['waybills']) > 3:
                        waybills_ldr += f" (+{len(penalty_breakdown['ldr']['waybills']) - 3} more)"

                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>LD&R:</strong> {penalty_breakdown['ldr']['count']} parcel(s)</span>
                            <span>- {currency_symbol} {penalty_breakdown['ldr']['amount']:,.2f}</span>
                        </div>
                    """
                    if penalty_breakdown['ldr']['waybills']:
                        html_content += f"""
                        <div style="font-size: 11px; opacity: 0.8; padding: 4px 0;">
                            Waybills: {waybills_ldr}
                        </div>
                        """

                # Fake Attempt penalties
                if penalty_breakdown['fake_attempt']['count'] > 0:
                    waybills_fake = format_waybills_display(penalty_breakdown['fake_attempt']['waybills'], limit=3)

                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>Fake Attempt:</strong> {penalty_breakdown['fake_attempt']['count']} parcel(s)</span>
                            <span>- {currency_symbol} {penalty_breakdown['fake_attempt']['amount']:,.2f}</span>
                        </div>
                    """
                    if penalty_breakdown['fake_attempt']['waybills']:
                        html_content += f"""
                        <div style="font-size: 11px; opacity: 0.8; padding: 4px 0;">
                            Waybills: {waybills_fake}
                        </div>
                        """

                if penalty_breakdown['no_outbound_scan']['count'] > 0:
                    waybills_nos = format_waybills_display(penalty_breakdown['no_outbound_scan']['waybills'], limit=3)
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>No Outbound Scan:</strong> {penalty_breakdown['no_outbound_scan']['count']} AWB(s)</span>
                            <span>- {currency_symbol} {penalty_breakdown['no_outbound_scan']['amount']:,.2f}</span>
                        </div>
                    """
                    if penalty_breakdown['no_outbound_scan']['waybills']:
                        html_content += f"""
                        <div style="font-size: 11px; opacity: 0.8; padding: 4px 0;">
                            Waybills: {waybills_nos}
                        </div>
                        """

                # COD penalties
                if penalty_breakdown['cod']['count'] > 0:
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>COD:</strong> {penalty_breakdown['cod']['count']} record(s)</span>
                            <span>- {currency_symbol} {penalty_breakdown['cod']['amount']:,.2f}</span>
                        </div>
                    """

                if penalty_breakdown['binding']['count'] > 0:
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>Binding:</strong> {penalty_breakdown['binding']['count']} record(s)</span>
                            <span>- {currency_symbol} {penalty_breakdown['binding']['amount']:,.2f}</span>
                        </div>
                    """

                if penalty_breakdown['hub']['count'] > 0:
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>Hub:</strong> {penalty_breakdown['hub']['count']} record(s)</span>
                            <span>- {currency_symbol} {penalty_breakdown['hub']['amount']:,.2f}</span>
                        </div>
                    """

                if penalty_breakdown['pending_parcel']['count'] > 0:
                    waybills_pp = ", ".join(penalty_breakdown['pending_parcel']['waybills'][:3])
                    if len(penalty_breakdown['pending_parcel']['waybills']) > 3:
                        waybills_pp += f" (+{len(penalty_breakdown['pending_parcel']['waybills']) - 3} more)"
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>Pending Parcel:</strong> {penalty_breakdown['pending_parcel']['count']} parcel(s)</span>
                            <span>- {currency_symbol} {penalty_breakdown['pending_parcel']['amount']:,.2f}</span>
                        </div>
                    """
                    if penalty_breakdown['pending_parcel']['waybills']:
                        html_content += f"""
                        <div style="font-size: 11px; opacity: 0.8; padding: 4px 0;">
                            Waybills: {waybills_pp}
                        </div>
                        """

                if penalty_breakdown['parcel_lost']['count'] > 0:
                    waybills_pl = ", ".join(penalty_breakdown['parcel_lost']['waybills'][:3])
                    if len(penalty_breakdown['parcel_lost']['waybills']) > 3:
                        waybills_pl += f" (+{len(penalty_breakdown['parcel_lost']['waybills']) - 3} more)"
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>Parcel Lost:</strong> {penalty_breakdown['parcel_lost']['count']} parcel(s)</span>
                            <span>- {currency_symbol} {penalty_breakdown['parcel_lost']['amount']:,.2f}</span>
                        </div>
                    """
                    if penalty_breakdown['parcel_lost']['waybills']:
                        html_content += f"""
                        <div style="font-size: 11px; opacity: 0.8; padding: 4px 0;">
                            Waybills: {waybills_pl}
                        </div>
                        """

                if penalty_breakdown.get('route', {}).get('amount', 0) > 0:
                    rc = penalty_breakdown['route'].get('count', 0) or 0
                    pool = float(penalty_breakdown['route'].get('pool_total', 0) or 0)
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>Route:</strong> {currency_symbol}{pool:,.2f} ÷ {rc} dispatchers</span>
                            <span>- {currency_symbol} {penalty_breakdown['route']['amount']:,.2f}</span>
                        </div>
                    """

                # Attendance penalties
                if penalty_breakdown['attendance']['amount'] > 0:
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>Attendance:</strong> Missing Clock-in</span>
                            <span>- {currency_symbol} {penalty_breakdown['attendance']['amount']:,.2f}</span>
                        </div>
                    """

                html_content += """
                        </div>
                    </div>
                """

            html_content += '</div>'

        # Daily breakdown table
        if len(df_disp) > 0:
            html_content += "<table>"
            html_content += "<thead><tr>"
            for col in df_disp.columns:
                html_content += f"<th>{col}</th>"
            html_content += "</tr></thead>"
            html_content += "<tbody>"
            for _, row in df_disp.iterrows():
                html_content += "<tr>"
                for col in df_disp.columns:
                    html_content += f"<td>{row[col]}</td>"
                html_content += "</tr>"
            html_content += "</tbody></table>"

        # Payout breakdown
        html_content += f"""
            <div class="payout-summary">
                <div class="payout-row">
                    <span>Base Delivery Payout:</span>
                    <span>{currency_symbol} {base_payout:,.2f}</span>
                </div>"""

        if designated_driver_breakdown:
            dd_basic_amount = designated_driver_breakdown.get("basic_amount", 0.0)
            dd_basic_parcels = designated_driver_breakdown.get("basic_parcels", 0)
            dd_rate_after_basic = designated_driver_breakdown.get("rate_after_basic", 0.0)
            dd_extra_parcels = designated_driver_breakdown.get("extra_parcels", 0)
            dd_extra_payout = designated_driver_breakdown.get("extra_payout", 0.0)
            dd_special_payout = designated_driver_breakdown.get("special_payout", 0.0)

            html_content += f"""
                <div class="payout-row payout-detail-row">
                    <span>↳ Basic Amount (first {dd_basic_parcels:,} parcels):</span>
                    <span>{currency_symbol} {dd_basic_amount:,.2f}</span>
                </div>"""

            if dd_extra_payout > 0:
                html_content += f"""
                <div class="payout-row payout-detail-row">
                    <span>↳ Extra Parcels ({dd_extra_parcels:,} × {currency_symbol}{dd_rate_after_basic:,.2f}):</span>
                    <span>+ {currency_symbol} {dd_extra_payout:,.2f}</span>
                </div>"""

            if dd_special_payout > 0:
                html_content += f"""
                <div class="payout-row payout-detail-row">
                    <span>↳ Special Rate Incentive:</span>
                    <span>+ {currency_symbol} {dd_special_payout:,.2f}</span>
                </div>"""

        html_content += f"""
                <div class="payout-row">
                    <span>Commission Delivery ({pickup_parcels} parcel(s)):</span>
                    <span>+ {currency_symbol} {pickup_payout:,.2f}</span>
                </div>
                <div class="payout-row">
                    <span>Return Payout ({return_count} parcel(s) × {currency_symbol}{return_rate:.2f}):</span>
                    <span>+ {currency_symbol} {return_payout:,.2f}</span>
                </div>"""

        if reward_payout > 0 or reward_count > 0:
            html_content += f"""
                <div class="payout-row">
                    <span>Reward ({reward_count} record(s)):</span>
                    <span>+ {currency_symbol} {reward_payout:,.2f}</span>
                </div>"""

        html_content += f"""
                <div class="payout-row">
                    <span>KPI Incentive Bonus:</span>
                    <span>+ {currency_symbol} {kpi_bonus:,.2f}</span>
                </div>
                <div class="payout-row">
                    <span>Attendance Incentive Bonus:</span>
                    <span>+ {currency_symbol} {attendance_bonus:,.2f}</span>
                </div>"""

        if penalty_breakdown['total_amount'] > 0:
            html_content += f"""
                <div class="payout-row" style="color: var(--error);">
                    <span>Total Penalty:</span>
                    <span>- {currency_symbol} {penalty_breakdown['total_amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['duitnow']['amount'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ DuitNow:</span>
                    <span>- {currency_symbol} {penalty_breakdown['duitnow']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['ldr']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ LD&amp;R ({penalty_breakdown['ldr']['count']} parcel(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['ldr']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['fake_attempt']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ Fake Attempt ({penalty_breakdown['fake_attempt']['count']} parcel(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['fake_attempt']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['no_outbound_scan']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ No Outbound Scan ({penalty_breakdown['no_outbound_scan']['count']} AWB(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['no_outbound_scan']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['cod']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ COD ({penalty_breakdown['cod']['count']} record(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['cod']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['binding']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ Binding ({penalty_breakdown['binding']['count']} record(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['binding']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['hub']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ Hub ({penalty_breakdown['hub']['count']} record(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['hub']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['pending_parcel']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ Pending Parcel ({penalty_breakdown['pending_parcel']['count']} parcel(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['pending_parcel']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['parcel_lost']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ Parcel Lost ({penalty_breakdown['parcel_lost']['count']} parcel(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['parcel_lost']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown.get('route', {}).get('amount', 0) > 0:
                rc = penalty_breakdown['route'].get('count', 0) or 0
                pool = float(penalty_breakdown['route'].get('pool_total', 0) or 0)
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ Route ({currency_symbol}{pool:,.2f} ÷ {rc} dispatchers):</span>
                    <span>- {currency_symbol} {penalty_breakdown['route']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['attendance']['amount'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ Attendance (Missing Clock-in):</span>
                    <span>- {currency_symbol} {penalty_breakdown['attendance']['amount']:,.2f}</span>
                </div>"""

        if socso_deduction > 0:
            html_content += f"""
                <div class="payout-row">
                    <span>SOCSO (Insurance{f' · {socso_deduction_count} record(s)' if socso_deduction_count else ''}):</span>
                    <span>- {currency_symbol} {socso_deduction:,.2f}</span>
                </div>"""

        if overpaid_deduction > 0:
            html_content += f"""
                <div class="payout-row">
                    <span>Overpaid{f' · {overpaid_deduction_count} record(s)' if overpaid_deduction_count else ''}:</span>
                    <span>- {currency_symbol} {overpaid_deduction:,.2f}</span>
                </div>"""

        html_content += f"""
                <div class="payout-row" style="border-top: 1px dashed var(--border); margin-top: 8px; padding-top: 8px;">
                    <span class="tooltip-wrap">
                        <strong>Gross Payout:</strong>
                        <span class="tooltip-icon" title="{gross_tooltip}">?
                            <span class="tooltiptext">{gross_tooltip}</span>
                        </span>
                    </span>
                    <span><strong>{currency_symbol} {gross_payout_amount:,.2f}</strong></span>
                </div>"""

        html_content += f"""
                <div class="payout-row">
                    <span class="tooltip-wrap">
                        <span>{advance_payout_desc or 'Advance Payout'}:</span>
                        <span class="tooltip-icon" title="{advance_tooltip}">?
                            <span class="tooltiptext">{advance_tooltip}</span>
                        </span>
                    </span>
                    <span>- {currency_symbol} {advance_line_amount:,.2f}</span>
                </div>
                <div class="payout-row total">
                    <span class="tooltip-wrap">
                        <strong>Final Payout:</strong>
                        <span class="tooltip-icon" title="{final_tooltip}">?
                            <span class="tooltiptext">{final_tooltip}</span>
                        </span>
                    </span>
                    <span>{currency_symbol} {final_payout:,.2f}</span>
                </div>
            </div>
        """

        html_content += f"""
                <div class="note">
                    Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} • JMR Dispatcher Payout System
                </div>
            </div>
        </body>
        </html>
        """

        return html_content


# =============================================================================
# STREAMLIT UI ENHANCEMENTS
# =============================================================================

def apply_custom_styles():
    st.markdown(f"""
    <style>
        .stApp {{
            background-color: {ColorScheme.BACKGROUND};
        }}
        .stMarkdown, .stText, .stWrite {{
            color: {ColorScheme.TEXT_PRIMARY} !important;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: {ColorScheme.TEXT_PRIMARY} !important;
        }}
        .stMarkdown div h1 {{
            color: white !important;
        }}
        .css-1d391kg, .css-1lcbmhc {{
            background-color: {ColorScheme.SURFACE};
        }}
        .stButton>button {{
            background-color: {ColorScheme.PRIMARY};
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: 600;
        }}
        .stButton>button:hover {{
            background-color: {ColorScheme.PRIMARY_LIGHT};
            color: white;
        }}
        .streamlit-expanderHeader {{
            background-color: {ColorScheme.SURFACE};
            border: 1px solid {ColorScheme.BORDER};
            border-radius: 8px;
            color: {ColorScheme.TEXT_PRIMARY} !important;
        }}
        .streamlit-expanderContent {{
            color: {ColorScheme.TEXT_PRIMARY} !important;
        }}
        .dataframe {{
            border: 1px solid {ColorScheme.BORDER};
            border-radius: 8px;
        }}
        [data-testid="stMetric"] {{
            background-color: {ColorScheme.SURFACE};
            border: 1px solid {ColorScheme.BORDER};
            padding: 1rem;
            border-radius: 8px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {ColorScheme.TEXT_SECONDARY};
        }}
        [data-testid="stMetricValue"] {{
            color: {ColorScheme.PRIMARY};
        }}
        .stSelectbox>div>div {{
            border: 1px solid {ColorScheme.BORDER};
            border-radius: 8px;
        }}
        .stAlert {{
            border-radius: 8px;
        }}
        .stCaption {{
            color: {ColorScheme.TEXT_SECONDARY} !important;
        }}
        .footer {{
            position: relative;
            bottom: 0;
            width: 100%;
            margin-top: 3rem;
            padding: 1.5rem 0;
            background: linear-gradient(135deg, {ColorScheme.PRIMARY} 0%, {ColorScheme.PRIMARY_LIGHT} 100%);
            color: white !important;
            text-align: center;
            border-radius: 12px;
        }}
        .footer-content {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.5rem;
            color: white !important;
        }}
        .footer-logo {{
            font-weight: 700;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
            color: white !important;
        }}
        .footer-links {{
            display: flex;
            gap: 1.5rem;
            margin: 0.5rem 0;
            flex-wrap: wrap;
            justify-content: center;
            color: white !important;
        }}
        .footer-link {{
            color: rgba(255,255,255,0.8) !important;
            text-decoration: none;
            font-size: 0.9rem;
            transition: color 0.2s;
        }}
        .footer-link:hover {{
            color: white !important;
        }}
        .footer-copyright {{
            color: rgba(255,255,255,0.9) !important;
            font-size: 0.9rem;
            margin-top: 0.5rem;
        }}
        .footer * {{
            color: white !important;
        }}
        @media (max-width: 768px) {{
            .footer-links {{
                flex-direction: column;
                gap: 0.5rem;
            }}
        }}
    </style>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown(f"""
    <div class="footer">
        <div class="footer-content">
            <div class="footer-copyright" style="color: white !important;">
                © 2025 Jemari Ventures. All rights reserved. | JMR Dispatcher Payout System v1.0
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    st.set_page_config(
        page_title="JMR Dispatcher Payout System",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    apply_custom_styles()
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {ColorScheme.PRIMARY} 0%, {ColorScheme.PRIMARY_LIGHT} 100%);
                padding: 2rem;
                border-radius: 12px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
        <h1 style="color: white !important; margin: 0; padding: 0;">🚚 JMR Dispatcher Payout System</h1>
        <p style="color: rgba(255,255,255,0.9) !important; margin: 0.5rem 0 0 0; padding: 0;">Calculate dispatcher payout online</p>
    </div>
    """, unsafe_allow_html=True)

    config = Config.load()
    config["data_source"]["type"] = "gsheet"

    with st.spinner("Loading data from Google Sheets..."):
        df = DataSource.load_data(config)

    if df is None:
        st.error("❌ Failed to load data from Google Sheets.")
        st.info("Please check the configuration file or ensure the Google Sheet is accessible.")
        add_footer()
        return

    if df.empty:
        st.warning("No data found in the Google Sheet.")
        add_footer()
        return

    df = df.rename(columns={c: str(c).strip() for c in df.columns})

    # Find required columns with flexible matching
    dispatcher_id_col = find_column(df, ["Dispatcher ID", "dispatcher_id", "Dispatcher Id", "DISPATCHER ID"])
    waybill_col = find_waybill_column(df)
    delivery_sig_col = find_column(df, ["Delivery Signature", "delivery_signature", "Delivery Signature", "delivery_sig"])

    missing_columns = []
    if dispatcher_id_col is None:
        missing_columns.append("Dispatcher ID")
    if waybill_col is None:
        missing_columns.append("Waybill Number")
    if delivery_sig_col is None:
        missing_columns.append("Delivery Signature")

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Available columns: {', '.join(df.columns.tolist())}")
        add_footer()
        return

    # Store column names for later use
    df.attrs['dispatcher_id_col'] = dispatcher_id_col
    df.attrs['waybill_col'] = waybill_col
    df.attrs['delivery_sig_col'] = delivery_sig_col

    # Convert Delivery Signature to datetime
    df[delivery_sig_col] = pd.to_datetime(df[delivery_sig_col], errors="coerce")

    valid_dates = df[delivery_sig_col].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()
    else:
        today = datetime.now().date()
        min_date = today - pd.Timedelta(days=365)
        max_date = today

    default_start = max(min_date, max_date.replace(day=1))
    st.sidebar.header("📅 Date Range")
    st.sidebar.caption(f"Data available: {min_date.strftime('%d %b %Y')} – {max_date.strftime('%d %b %Y')}")
    selected_range = st.sidebar.date_input(
        "Select period",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Dispatch (Delivery Signature), pickup (date_pick_up), and return (Delivery Signature).",
    )

    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_date, end_date = selected_range
    else:
        start_date = end_date = selected_range

    if start_date > end_date:
        start_date, end_date = end_date, start_date

    df["__delivery_date"] = pd.to_datetime(df[delivery_sig_col], errors="coerce").dt.date
    df = df[
        (df["__delivery_date"] >= start_date) & (df["__delivery_date"] <= end_date)
    ].copy()
    df = df.drop(columns=["__delivery_date"], errors="ignore")

    if df.empty:
        st.warning(
            f"No dispatch records found between {start_date.strftime('%B %d, %Y')} "
            f"and {end_date.strftime('%B %d, %Y')}. Penalty-only dispatchers may still be selectable."
        )
    else:
        st.caption(
            f"Showing deliveries from {start_date.strftime('%B %d, %Y')} "
            f"to {end_date.strftime('%B %d, %Y')}."
        )

    # Load supplemental sheets before dispatcher selection
    pickup_df = DataSource.load_pickup_data(config)
    reward_df = DataSource.load_reward_data(config)
    return_df = DataSource.load_return_data(config)
    bulky_df = DataSource.load_bulky_data(config)
    attendance_df = DataSource.load_attendance_data(config)
    duitnow_df = DataSource.load_duitnow_penalty_data(config)
    ldr_df = DataSource.load_ldr_penalty_data(config)
    fake_attempt_df = DataSource.load_fake_attempt_penalty_data(config)
    cod_df = DataSource.load_cod_penalty_data(config)
    binding_df = DataSource.load_binding_penalty_data(config)
    hub_df = DataSource.load_hub_penalty_data(config)
    socso_df = DataSource.load_socso_deduction_data(config)
    overpaid_df = DataSource.load_overpaid_deduction_data(config)
    pending_parcel_df = DataSource.load_pending_parcel_penalty_data(config)
    no_outbound_scan_df = DataSource.load_no_outbound_scan_penalty_data(config)
    parcel_lost_df = DataSource.load_parcel_lost_penalty_data(config)

    pickup_df = filter_sheet_by_date_range(
        pickup_df, start_date, end_date, ["date_pick_up", "Date Pick Up", "date pick up"]
    )
    return_df = filter_sheet_by_date_range(
        return_df, start_date, end_date, ["Delivery Signature", "delivery_signature"]
    )
    bulky_df = filter_sheet_by_date_range(
        bulky_df, start_date, end_date, ["Delivery Signature", "delivery_signature"]
    )

    penalty_filtered = filter_penalty_data_by_date(
        {
            "duitnow": duitnow_df,
            "ldr": ldr_df,
            "fake_attempt": fake_attempt_df,
            "cod": cod_df,
            "binding": binding_df,
            "hub": hub_df,
            "socso": socso_df,
            "overpaid": overpaid_df,
            "pending_parcel": pending_parcel_df,
            "no_outbound_scan": no_outbound_scan_df,
            "parcel_lost": parcel_lost_df,
        },
        start_date,
        end_date,
    )
    duitnow_df = penalty_filtered.get("duitnow", duitnow_df if duitnow_df is not None else pd.DataFrame())
    ldr_df = penalty_filtered.get("ldr", ldr_df if ldr_df is not None else pd.DataFrame())
    fake_attempt_df = penalty_filtered.get("fake_attempt", fake_attempt_df if fake_attempt_df is not None else pd.DataFrame())
    cod_df = penalty_filtered.get("cod", cod_df if cod_df is not None else pd.DataFrame())
    binding_df = penalty_filtered.get("binding", binding_df if binding_df is not None else pd.DataFrame())
    hub_df = penalty_filtered.get("hub", hub_df if hub_df is not None else pd.DataFrame())
    socso_df = penalty_filtered.get("socso", socso_df if socso_df is not None else pd.DataFrame())
    overpaid_df = penalty_filtered.get("overpaid", overpaid_df if overpaid_df is not None else pd.DataFrame())
    pending_parcel_df = penalty_filtered.get("pending_parcel", pending_parcel_df if pending_parcel_df is not None else pd.DataFrame())
    no_outbound_scan_df = penalty_filtered.get("no_outbound_scan", no_outbound_scan_df if no_outbound_scan_df is not None else pd.DataFrame())
    parcel_lost_df = penalty_filtered.get("parcel_lost", parcel_lost_df if parcel_lost_df is not None else pd.DataFrame())

    st.subheader("👤 Dispatcher Selection")
    dispatcher_mapping = {}
    dispatcher_name_col = find_column(df, ["Dispatcher Name", "dispatcher_name", "Rider Name", "rider_name", "Name", "name"])

    if not df.empty and dispatcher_name_col and dispatcher_id_col:
        temp_mapping = df[[dispatcher_id_col, dispatcher_name_col]].dropna()
        temp_mapping[dispatcher_id_col] = temp_mapping[dispatcher_id_col].astype(str)
        temp_mapping[dispatcher_name_col] = temp_mapping[dispatcher_name_col].astype(str)
        for _, row in temp_mapping.iterrows():
            dispatcher_id = clean_penalty_dispatcher_id(row[dispatcher_id_col])
            dispatcher_name = clean_dispatcher_name(row[dispatcher_name_col])
            if dispatcher_id and dispatcher_id not in dispatcher_mapping:
                dispatcher_mapping[dispatcher_id] = dispatcher_name

    for penalty_id in collect_dispatcher_ids_from_penalty_sheets(
        {
            "duitnow": duitnow_df,
            "ldr": ldr_df,
            "fake_attempt": fake_attempt_df,
            "cod": cod_df,
            "binding": binding_df,
            "hub": hub_df,
            "socso": socso_df,
            "overpaid": overpaid_df,
            "pending_parcel": pending_parcel_df,
            "no_outbound_scan": no_outbound_scan_df,
            "parcel_lost": parcel_lost_df,
            "attendance": attendance_df,
        }
    ):
        dispatcher_mapping.setdefault(penalty_id, "")

    if reward_df is not None and not reward_df.empty:
        reward_id_col = find_reward_employee_column(reward_df)
        reward_name_col = find_reward_dispatcher_name_column(reward_df)
        if reward_id_col:
            reward_cols = [reward_id_col] + ([reward_name_col] if reward_name_col else [])
            for _, row in reward_df[reward_cols].dropna(subset=[reward_id_col]).iterrows():
                reward_id = clean_penalty_dispatcher_id(row[reward_id_col])
                if not reward_id:
                    continue
                reward_name = ""
                if reward_name_col:
                    reward_name = clean_dispatcher_name(row[reward_name_col])
                dispatcher_mapping.setdefault(reward_id, reward_name or dispatcher_mapping.get(reward_id, ""))

    if bulky_df is not None and not bulky_df.empty:
        bulky_id_col = find_dispatch_id_column(bulky_df)
        bulky_name_col = find_column(
            bulky_df, ["Dispatcher Name", "dispatcher_name", "Rider Name", "rider_name", "Name", "name"]
        )
        if bulky_id_col:
            bulky_cols = [bulky_id_col] + ([bulky_name_col] if bulky_name_col else [])
            for _, row in bulky_df[bulky_cols].dropna(subset=[bulky_id_col]).iterrows():
                bulky_id = clean_penalty_dispatcher_id(row[bulky_id_col])
                if not bulky_id:
                    continue
                bulky_name = ""
                if bulky_name_col:
                    bulky_name = clean_dispatcher_name(row[bulky_name_col])
                dispatcher_mapping.setdefault(bulky_id, bulky_name or dispatcher_mapping.get(bulky_id, ""))

    dispatcher_options = []
    for dispatcher_id, dispatcher_name in dispatcher_mapping.items():
        display_name = f"{dispatcher_id} - {dispatcher_name}" if dispatcher_name else dispatcher_id
        dispatcher_options.append((dispatcher_id, display_name))

    dispatcher_options.sort(key=lambda x: (dispatcher_mapping.get(x[0], "").lower() or x[0].lower()))

    if not dispatcher_options:
        st.warning("No dispatchers found in dispatch or penalty sheets for this period.")
        add_footer()
        return

    selected_display = st.selectbox(
        "Select Dispatcher",
        options=[opt[1] for opt in dispatcher_options],
        index=0,
        help="Choose a dispatcher to view their payout details",
    )

    selected_dispatcher_id = None
    selected_dispatcher_name = None
    for dispatcher_id, display_name in dispatcher_options:
        if display_name == selected_display:
            selected_dispatcher_id = dispatcher_id
            selected_dispatcher_name = dispatcher_mapping.get(dispatcher_id, "")
            break

    if not selected_dispatcher_id:
        st.warning("No dispatcher selected.")
        add_footer()
        return

    if not df.empty and dispatcher_id_col:
        dispatcher_df = df[
            df[dispatcher_id_col].apply(clean_penalty_dispatcher_id) == selected_dispatcher_id
        ].copy()
    else:
        dispatcher_df = pd.DataFrame()

    if dispatcher_df.empty:
        st.info(
            f"No delivery records for {selected_dispatcher_name or selected_dispatcher_id} "
            f"in the selected period. Showing penalties and other earnings only."
        )

    route_penalty_total = float(config.get("route_penalty_amount", 0.0))
    if not df.empty and dispatcher_id_col:
        route_id_list = df[dispatcher_id_col].dropna().astype(str).str.strip().unique().tolist()
    else:
        route_id_list = list(dispatcher_mapping.keys())
    route_penalty_dispatcher_count = len(route_id_list)
    route_penalty_split = split_route_penalty_pool(route_penalty_total, route_id_list)
    route_penalty_per_dispatcher = float(
        route_penalty_split.get(route_penalty_dispatcher_key(selected_dispatcher_id), 0.0)
    )

    # Visibility: show which optional sheets are missing/empty and therefore treated as 0.
    optional_sheets = {
        "Pickup": pickup_df,
        "DuitNow": duitnow_df,
        "LDR": ldr_df,
        "Fake Attempt": fake_attempt_df,
        "COD": cod_df,
        "Binding": binding_df,
        "Hub": hub_df,
        "SOCSO": socso_df,
        "Overpaid": overpaid_df,
        "Pending Parcel": pending_parcel_df,
        "No Outbound Scan": no_outbound_scan_df,
        "Parcel Lost": parcel_lost_df,
        "Return": return_df,
        "Bulky": bulky_df,
        "Reward": reward_df,
        "Attendance": attendance_df,
    }
    missing_or_empty_sheets = [
        sheet_name for sheet_name, sheet_df in optional_sheets.items()
        if sheet_df is None or sheet_df.empty
    ]
    if missing_or_empty_sheets:
        sheet_list = ", ".join(missing_or_empty_sheets)
        attendance_tab = (
            config.get("data_source", {})
            .get("excel_sheets", {})
            .get("attendance", "Attendance")
        )
        st.warning(
            f"Missing or empty optional sheet(s): {sheet_list}. "
            f"Related pickup/penalty calculations are treated as 0."
        )
        if "Attendance" in missing_or_empty_sheets:
            st.caption(
                f"Attendance: the workbook tab name must match config "
                f"data_source.excel_sheets.attendance (currently \"{attendance_tab}\"), "
                f"the tab needs at least one data row (not only headers), and the CSV export must not be "
                f"identical to Dispatch (a wrong tab name can make Google return Dispatch data, which this app ignores)."
            )

    # Calculate payout
    st.subheader(f"💰 Payout Calculation for {selected_dispatcher_name or selected_dispatcher_id}")

    is_designated_driver = PayoutCalculator.is_designated_driver(selected_dispatcher_id, config)
    designated_driver_config = config.get("designated_driver", {})
    if is_designated_driver:
        basic_amount = float(designated_driver_config.get("basic_amount", 1700.0))
        basic_parcels = int(designated_driver_config.get("basic_parcels", 700))
        rate_after_basic = float(designated_driver_config.get("rate_after_basic", 1.0))
        st.info(
            f"**Designated Driver position:** "
            f"{config['currency_symbol']}{basic_amount:,.2f} basic for first {basic_parcels:,} parcels, "
            f"then {config['currency_symbol']}{rate_after_basic:,.2f} per parcel. "
            f"KPI: {config['currency_symbol']}100 at 3,500 parcels/month, "
            f"{config['currency_symbol']}150 at 4,500 parcels/month. "
            f"Special rate incentives from config also apply on qualifying days."
        )

    # Get advance payout configuration
    advance_config = config.get("advance_payout", {"enabled": False, "percentage": 0.0, "description": "Advance Payout"})
    advance_enabled = advance_config.get("enabled", False)
    advance_percentage = advance_config.get("percentage", 0.0)
    advance_description = advance_config.get("description", "Advance Payout (40% of Base Delivery)")

    # Initialize variables that might be used in other tabs
    advance_payout = 0.0
    base_delivery_payout = 0.0
    gross_delivery_payout = 0.0
    gross_total_payout = 0.0
    final_payout = 0.0
    kpi_bonus = 0.0
    attendance_bonus = 0.0
    pickup_payout = 0.0
    pickup_parcels = 0
    return_payout = 0.0
    return_count = 0
    reward_payout = 0.0
    reward_count = 0
    socso_deduction = 0.0
    socso_deduction_count = 0
    overpaid_deduction = 0.0
    overpaid_deduction_count = 0
    kpi_description = ""
    attendance_desc = ""
    qualified_days = 0
    display_df = pd.DataFrame()
    penalty_breakdown = {'duitnow': {'amount': 0.0, 'count': 0, 'waybills': []},
                        'ldr': {'amount': 0.0, 'count': 0, 'waybills': []},
                        'fake_attempt': {'amount': 0.0, 'count': 0, 'waybills': []},
                        'cod': {'amount': 0.0, 'count': 0},
                        'binding': {'amount': 0.0, 'count': 0},
                'hub': {'amount': 0.0, 'count': 0},
                        'pending_parcel': {'amount': 0.0, 'count': 0, 'waybills': []},
                        'no_outbound_scan': {'amount': 0.0, 'count': 0, 'waybills': []},
                        'parcel_lost': {'amount': 0.0, 'count': 0, 'waybills': []},
                        'route': {'amount': 0.0, 'count': 0, 'pool_total': 0.0},
                        'attendance': {'amount': 0.0, 'count': 0},
                        'total_amount': 0.0, 'total_count': 0}
    per_day_df = pd.DataFrame()
    pickup_filtered_df = pd.DataFrame()
    return_filtered_df = pd.DataFrame()
    reward_filtered_df = pd.DataFrame()
    designated_driver_breakdown = None

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Payout Details", "📈 Performance Charts", "🧾 Invoice"])

    with tab1:
        # Calculate delivery payout (tiered or designated driver)
        try:
            payout_kwargs = dict(
                filtered_df=dispatcher_df,
                attendance_config=config.get("attendance_incentive", {}),
                currency_symbol=config["currency_symbol"],
                duitnow_df=duitnow_df,
                ldr_df=ldr_df,
                fake_df=fake_attempt_df,
                cod_df=cod_df,
                binding_df=binding_df,
                hub_df=hub_df,
                pending_parcel_df=pending_parcel_df,
                no_outbound_scan_df=no_outbound_scan_df,
                parcel_lost_df=parcel_lost_df,
                attendance_df=attendance_df,
                fake_attempt_penalty_per_parcel=config.get("fake_attempt_penalty_per_parcel", 2.0),
                pending_parcel_penalty_per_parcel=config.get("pending_parcel_penalty_per_parcel", 2.0),
                no_outbound_scan_penalty_per_parcel=config.get("no_outbound_scan_penalty_per_parcel", 3.0),
                return_df=return_df,
                route_penalty_per_dispatcher=route_penalty_per_dispatcher,
                route_penalty_dispatcher_count=route_penalty_dispatcher_count,
                route_penalty_pool_total=route_penalty_total,
                pickup_df=pickup_df,
                bulky_df=bulky_df,
                special_rates_config=config.get("special_rates", []),
                fallback_dispatcher_id=selected_dispatcher_id,
            )

            if is_designated_driver:
                (display_df, base_delivery_payout, gross_delivery_payout, kpi_bonus, kpi_description,
                 attendance_bonus, attendance_desc, qualified_days,
                 per_day_df, penalty_breakdown) = PayoutCalculator.calculate_designated_driver(
                    designated_driver_config=designated_driver_config,
                    **payout_kwargs,
                )
            else:
                (display_df, base_delivery_payout, gross_delivery_payout, kpi_bonus, kpi_description,
                 attendance_bonus, attendance_desc, qualified_days,
                 per_day_df, penalty_breakdown) = PayoutCalculator.calculate_tiered_daily(
                    tiers_config=config["tiers"],
                    kpi_config=config.get("kpi_incentives", []),
                    **payout_kwargs,
                )

            if is_designated_driver:
                total_delivery_parcels = int(display_df['Total Parcel'].sum()) if 'Total Parcel' in display_df.columns else 0
                designated_driver_breakdown = PayoutCalculator.build_designated_driver_breakdown(
                    per_day_df,
                    designated_driver_config,
                    total_delivery_parcels,
                    base_delivery_payout,
                )

            # Calculate pickup payout (Order Source rules; fallback rate from config)
            pickup_parcels, pickup_payout, pickup_filtered_df = PayoutCalculator.calculate_pickup(
                pickup_df,
                selected_dispatcher_id,
                rate=float(config.get("pickup_payout_per_parcel", 1.0)),
            )

            return_count, return_payout, return_filtered_df = PayoutCalculator.calculate_return(
                return_df,
                selected_dispatcher_id,
                rate=float(config.get("return_payout_per_parcel", 0.5)),
            )

            reward_payout, reward_count, reward_filtered_df = PayoutCalculator.calculate_reward(
                reward_df,
                selected_dispatcher_id,
            )

            socso_deduction, socso_deduction_count = PayoutCalculator.calculate_benefit_deduction(
                selected_dispatcher_id,
                socso_df,
            )
            overpaid_deduction, overpaid_deduction_count = PayoutCalculator.calculate_benefit_deduction(
                selected_dispatcher_id,
                overpaid_df,
            )

            gross_total_payout = PayoutCalculator.calculate_gross_payout(
                base_delivery_payout,
                pickup_payout,
                return_payout,
                reward_payout,
                kpi_bonus,
                attendance_bonus,
                penalty_breakdown["total_amount"],
                socso_deduction,
                overpaid_deduction,
            )

            advance_payout = PayoutCalculator.calculate_advance_payout(
                base_delivery_payout,
                advance_percentage,
                advance_enabled,
            )

            final_payout = PayoutCalculator.calculate_final_payout(
                gross_total_payout,
                advance_payout,
            )

            # Total AWB = delivery parcels + pickup parcels + return parcels.
            total_delivery_parcels = (
                int(display_df['Total Parcel'].sum())
                if 'Total Parcel' in display_df.columns
                else 0
            )
            total_awb = PayoutCalculator.count_total_awb(
                total_delivery_parcels,
                pickup_parcels,
                return_count,
            )

            summary_row1_col1, summary_row1_col2, summary_row1_col3 = st.columns(3)
            with summary_row1_col1:
                st.metric(
                    "Total AWB",
                    f"{total_awb:,}",
                    help=(
                        "Total parcels across Delivery, Pickup, and Return for this dispatcher "
                        f"({total_delivery_parcels:,} delivery + {pickup_parcels:,} pickup + "
                        f"{return_count:,} return)."
                    ),
                )
            with summary_row1_col2:
                st.metric(
                    "Parcels Delivered",
                    f"{display_df['Total Parcel'].sum():,}",
                    help=(
                        "Dispatch and bulky parcels for tier base rate "
                        "(excludes return and pickup AWBs; overlap counted via Bulky only)."
                    ),
                )
            with summary_row1_col3:
                st.metric(
                    "Working Days",
                    f"{len(display_df)}",
                    help="Number of days worked",
                )

            payout_row1_col1, payout_row1_col2, payout_row1_col3 = st.columns(3)
            with payout_row1_col1:
                if pickup_parcels > 0:
                    st.metric(
                        "Commission Delivery",
                        f"{config['currency_symbol']}{pickup_payout:,.2f}",
                        delta=f"{pickup_parcels} parcels",
                        delta_color="normal",
                    )
                else:
                    st.metric(
                        "Commission Delivery",
                        f"{config['currency_symbol']}0.00",
                        delta="No pickups",
                        delta_color="off",
                    )
            with payout_row1_col2:
                if return_count > 0:
                    st.metric(
                        "Return Payout",
                        f"{config['currency_symbol']}{return_payout:,.2f}",
                        delta=f"{return_count} parcels",
                        delta_color="normal",
                    )
                else:
                    st.metric(
                        "Return Payout",
                        f"{config['currency_symbol']}0.00",
                        delta="No returns",
                        delta_color="off",
                    )
            with payout_row1_col3:
                st.metric(
                    "Base Delivery Payout",
                    f"{config['currency_symbol']}{base_delivery_payout:,.2f}",
                    help="Total base payout from daily deliveries",
                )

            payout_row2_col1, payout_row2_col2, payout_row2_col3 = st.columns(3)
            with payout_row2_col1:
                st.metric(
                    "Gross Payout",
                    f"{config['currency_symbol']}{gross_total_payout:,.2f}",
                    help=(
                        "Base delivery + Commission Delivery + Return + Reward "
                        "+ KPI bonus + Attendance bonus − Total penalty − SOCSO − Overpaid."
                    ),
                )
            with payout_row2_col2:
                st.metric(
                    "Advance Payout",
                    f"{config['currency_symbol']}{advance_payout:,.2f}",
                    help=f"{advance_percentage:g}% of base delivery payout",
                )
            with payout_row2_col3:
                st.metric(
                    "Final Payout",
                    f"{config['currency_symbol']}{final_payout:,.2f}",
                    help="Gross Payout − Advance payout",
                )

            # Display daily breakdown
            st.subheader("📅 Daily Delivery Breakdown")
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                    "Total Parcel": st.column_config.NumberColumn("Parcels", format="%d"),
                    "Tier": "Tier",
                    "Payout Rate": "Rate",
                    "Payout": st.column_config.TextColumn("Payout")
                }
            )

            # Display bonuses and penalties
            st.subheader("🎯 Incentives & Penalties")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "KPI Bonus",
                    f"{config['currency_symbol']}{kpi_bonus:,.2f}",
                    delta="Achieved" if kpi_bonus > 0 else "Not achieved",
                    delta_color="normal" if kpi_bonus > 0 else "off"
                )

            with col2:
                st.metric(
                    "Attendance Bonus",
                    f"{config['currency_symbol']}{attendance_bonus:,.2f}",
                    delta=f"{qualified_days} days",
                    delta_color="normal" if attendance_bonus > 0 else "off"
                )

            with col3:
                if reward_payout > 0:
                    st.metric(
                        "Reward",
                        f"{config['currency_symbol']}{reward_payout:,.2f}",
                        delta=f"{reward_count} record(s)",
                        delta_color="normal",
                    )
                else:
                    st.metric(
                        "Reward",
                        f"{config['currency_symbol']}0.00",
                        delta="No rewards",
                        delta_color="off",
                    )

            with col4:
                if penalty_breakdown['total_amount'] > 0:
                    st.metric(
                        "Total Penalty",
                        f"-{config['currency_symbol']}{penalty_breakdown['total_amount']:,.2f}",
                        delta=f"{penalty_breakdown['total_count']} records",
                        delta_color="inverse",
                    )
                else:
                    st.metric(
                        "Total Penalty",
                        f"{config['currency_symbol']}0.00",
                        delta="No penalties",
                        delta_color="off",
                    )

            # Display pickup details if available
            if pickup_parcels > 0 and not pickup_filtered_df.empty:
                with st.expander("📦 Pickup Details", expanded=False):
                    st.info(
                        f"**Total Pickup Parcels: {pickup_parcels}** | "
                        f"**Payout: {config['currency_symbol']}{pickup_payout:,.2f}** "
                        f"(Order Source commission rules)"
                    )

                    # Clean up column names for display and remove unnamed columns
                    display_pickup_df = pickup_filtered_df.copy()

                    # Ensure Waybill Number is treated as string BEFORE filtering columns
                    # This prevents scientific notation or number formatting
                    # Use safe conversion to handle numeric waybills properly
                    if "Waybill Number" in display_pickup_df.columns:
                        def safe_waybill_display(value):
                            """Safely convert waybill to string for display, preserving format."""
                            if pd.isna(value):
                                return ""
                            # Handle numeric types - remove .0 from floats
                            if isinstance(value, (int, float)):
                                if isinstance(value, float) and value.is_integer():
                                    return str(int(value))
                                return str(value)
                            # For strings, return as-is
                            return str(value).strip()

                        display_pickup_df["Waybill Number"] = display_pickup_df["Waybill Number"].apply(safe_waybill_display)

                    # Filter out unnamed columns (columns that start with "Unnamed" or are empty)
                    valid_columns = [
                        col for col in display_pickup_df.columns
                        if not (str(col).startswith('Unnamed') or str(col).strip() == '' or str(col).lower() == 'nan')
                    ]
                    display_pickup_df = display_pickup_df[valid_columns]

                    # Store original waybill column name before renaming
                    waybill_col_original = None
                    if "Waybill Number" in display_pickup_df.columns:
                        waybill_col_original = "Waybill Number"

                    # Clean up column names for display
                    display_pickup_df.columns = [str(col).title() for col in display_pickup_df.columns]

                    # Configure column display - ensure waybill is shown as text (not number)
                    column_config = {}
                    # Find waybill column after title transformation
                    waybill_col_display = None
                    for col in display_pickup_df.columns:
                        if "waybill" in col.lower():
                            waybill_col_display = col
                            break

                    if waybill_col_display:
                        column_config[waybill_col_display] = st.column_config.TextColumn(waybill_col_display)

                    # Display the filtered pickup dataframe
                    st.dataframe(
                        display_pickup_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=column_config if column_config else None
                    )
            else:
                with st.expander("📦 Pickup Details", expanded=False):
                    st.info("No pickup records found for this dispatcher.")

            # Display return details if available
            if return_count > 0 and not return_filtered_df.empty:
                with st.expander("📦 Return Details", expanded=False):
                    st.info(f"**Total Return Parcels: {return_count}** | **Payout: {config['currency_symbol']}{return_payout:,.2f}** (RM0.50 per parcel)")

                    # Clean up column names for display and remove unnamed columns
                    display_return_df = return_filtered_df.copy()

                    # Ensure Waybill Number is treated as string BEFORE filtering columns
                    # This prevents scientific notation or number formatting
                    # Use safe conversion to handle numeric waybills properly
                    waybill_col_return = None
                    for col in display_return_df.columns:
                        if "waybill" in str(col).lower():
                            waybill_col_return = col
                            break

                    if waybill_col_return:
                        def safe_waybill_display(value):
                            """Safely convert waybill to string for display, preserving format."""
                            if pd.isna(value):
                                return ""
                            # Handle numeric types - remove .0 from floats
                            if isinstance(value, (int, float)):
                                if isinstance(value, float) and value.is_integer():
                                    return str(int(value))
                                return str(value)
                            # For strings, return as-is
                            return str(value).strip()

                        display_return_df[waybill_col_return] = display_return_df[waybill_col_return].apply(safe_waybill_display)

                    # Filter out unnamed columns (columns that start with "Unnamed" or are empty)
                    valid_columns = [
                        col for col in display_return_df.columns
                        if not (str(col).startswith('Unnamed') or str(col).strip() == '' or str(col).lower() == 'nan')
                    ]
                    display_return_df = display_return_df[valid_columns]

                    # Clean up column names for display
                    display_return_df.columns = [str(col).title() for col in display_return_df.columns]

                    # Configure column display - ensure waybill is shown as text (not number)
                    column_config = {}
                    # Find waybill column after title transformation
                    waybill_col_display = None
                    for col in display_return_df.columns:
                        if "waybill" in col.lower():
                            waybill_col_display = col
                            break

                    if waybill_col_display:
                        column_config[waybill_col_display] = st.column_config.TextColumn(waybill_col_display)

                    # Display the filtered return dataframe
                    st.dataframe(
                        display_return_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config=column_config if column_config else None
                    )
            else:
                with st.expander("📦 Return Details", expanded=False):
                    st.info("No return records found for this dispatcher.")

            # Display penalty details if any
            if penalty_breakdown['total_amount'] > 0:
                with st.expander("⚠️ Penalty Details"):
                    penalty_col1, penalty_col2, penalty_col3, penalty_col4, penalty_col5, penalty_col6, penalty_col7, penalty_col8, penalty_col9, penalty_col10 = st.columns(10)

                    with penalty_col1:
                        if penalty_breakdown['duitnow']['amount'] > 0:
                            st.error(f"**DuitNow:** - {config['currency_symbol']}{penalty_breakdown['duitnow']['amount']:,.2f}")

                    with penalty_col2:
                        if penalty_breakdown['ldr']['count'] > 0:
                            st.error(f"**LD&R:** {penalty_breakdown['ldr']['count']} parcel(s) - {config['currency_symbol']}{penalty_breakdown['ldr']['amount']:,.2f}")
                            if penalty_breakdown['ldr']['waybills']:
                                st.caption(f"Waybills: {', '.join(penalty_breakdown['ldr']['waybills'][:5])}")

                    with penalty_col3:
                        if penalty_breakdown['fake_attempt']['count'] > 0:
                            st.error(f"**Fake Attempt:** {penalty_breakdown['fake_attempt']['count']} parcel(s) - {config['currency_symbol']}{penalty_breakdown['fake_attempt']['amount']:,.2f}")
                            if penalty_breakdown['fake_attempt']['waybills']:
                                st.caption(f"Waybills: {', '.join(penalty_breakdown['fake_attempt']['waybills'][:5])}")

                    with penalty_col4:
                        if penalty_breakdown['cod']['count'] > 0:
                            st.error(f"**COD:** {penalty_breakdown['cod']['count']} record(s) - {config['currency_symbol']}{penalty_breakdown['cod']['amount']:,.2f}")

                    with penalty_col5:
                        if penalty_breakdown['binding']['count'] > 0:
                            st.error(f"**Binding:** {penalty_breakdown['binding']['count']} record(s) - {config['currency_symbol']}{penalty_breakdown['binding']['amount']:,.2f}")
                        if penalty_breakdown['hub']['count'] > 0:
                            st.error(f"**Hub:** {penalty_breakdown['hub']['count']} record(s) - {config['currency_symbol']}{penalty_breakdown['hub']['amount']:,.2f}")

                    with penalty_col6:
                        if penalty_breakdown['pending_parcel']['count'] > 0:
                            st.error(f"**Pending Parcel:** {penalty_breakdown['pending_parcel']['count']} parcel(s) - {config['currency_symbol']}{penalty_breakdown['pending_parcel']['amount']:,.2f}")
                            if penalty_breakdown['pending_parcel']['waybills']:
                                st.caption(f"Waybills: {', '.join(penalty_breakdown['pending_parcel']['waybills'][:5])}")

                    with penalty_col7:
                        if penalty_breakdown['no_outbound_scan']['count'] > 0:
                            st.error(f"**No Outbound Scan:** {penalty_breakdown['no_outbound_scan']['count']} AWB(s) - {config['currency_symbol']}{penalty_breakdown['no_outbound_scan']['amount']:,.2f}")
                            if penalty_breakdown['no_outbound_scan']['waybills']:
                                st.caption("Waybills:")
                                for wb in penalty_breakdown['no_outbound_scan']['waybills']:
                                    st.markdown(f"- `{wb}`")

                    with penalty_col8:
                        if penalty_breakdown['parcel_lost']['count'] > 0:
                            st.error(f"**Parcel Lost:** {penalty_breakdown['parcel_lost']['count']} parcel(s) - {config['currency_symbol']}{penalty_breakdown['parcel_lost']['amount']:,.2f}")
                            if penalty_breakdown['parcel_lost']['waybills']:
                                st.caption(f"Waybills: {', '.join(penalty_breakdown['parcel_lost']['waybills'][:5])}")

                    with penalty_col9:
                        if penalty_breakdown.get('route', {}).get('amount', 0) > 0:
                            rc = penalty_breakdown['route'].get('count', 0) or 0
                            st.error(
                                f"**Route:** {rc} dispatcher(s) - "
                                f"{config['currency_symbol']}{penalty_breakdown['route']['amount']:,.2f}"
                            )

                    with penalty_col10:
                        if penalty_breakdown['attendance']['amount'] > 0:
                            st.error(f"**Attendance:** Missing Clock-in - {config['currency_symbol']}{penalty_breakdown['attendance']['amount']:,.2f}")

            if socso_deduction > 0 or overpaid_deduction > 0:
                with st.expander("🏛️ Benefit Deductions", expanded=False):
                    if socso_deduction > 0:
                        st.info(
                            f"**SOCSO (Insurance):** {socso_deduction_count} record(s) — "
                            f"-{config['currency_symbol']}{socso_deduction:,.2f}"
                        )
                    if overpaid_deduction > 0:
                        st.info(
                            f"**Overpaid:** {overpaid_deduction_count} record(s) — "
                            f"-{config['currency_symbol']}{overpaid_deduction:,.2f}"
                        )

            # Display payout breakdown
            st.subheader("💰 Payout Breakdown")

            breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)

            with breakdown_col1:
                st.info(f"""
                **Delivery Earnings:**
                - Base: {config['currency_symbol']}{base_delivery_payout:,.2f}
                - KPI Bonus: +{config['currency_symbol']}{kpi_bonus:,.2f}
                - Attendance Bonus: +{config['currency_symbol']}{attendance_bonus:,.2f}
                - Penalties: -{config['currency_symbol']}{penalty_breakdown['total_amount']:,.2f}
                """)

            with breakdown_col2:
                st.info(f"""
                **Additional Earnings:**
                - Pickup ({pickup_parcels} parcels): +{config['currency_symbol']}{pickup_payout:,.2f}
                - Return ({return_count} parcels): +{config['currency_symbol']}{return_payout:,.2f}
                - Reward ({reward_count} record(s)): +{config['currency_symbol']}{reward_payout:,.2f}
                - Gross Total: {config['currency_symbol']}{gross_total_payout:,.2f}
                """)

            with breakdown_col3:
                if advance_enabled:
                    st.warning(f"""
                    **Advance & Final:**
                    - Advance ({advance_percentage:g}% of Base Delivery): -{config['currency_symbol']}{advance_payout:,.2f}
                    - **Final Payout:** {config['currency_symbol']}{final_payout:,.2f}
                    """)
                else:
                    st.success(f"""
                    **Final Payout:**
                    - **Gross Payout:** {config['currency_symbol']}{gross_total_payout:,.2f}
                    - **Final Payout:** {config['currency_symbol']}{final_payout:,.2f}
                    """)

        except Exception as e:
            st.error(f"Error calculating payout: {str(e)}")
            st.exception(e)

    with tab2:
        # Display performance charts
        if 'per_day_df' in locals() and not per_day_df.empty:
            charts = DataVisualizer.create_performance_charts(per_day_df, config["currency_symbol"])

            if charts:
                col1, col2 = st.columns(2)

                with col1:
                    st.altair_chart(charts.get('parcels_payout'), use_container_width=True)

                with col2:
                    st.altair_chart(charts.get('performance_scatter'), use_container_width=True)

                st.altair_chart(charts.get('payout_trend'), use_container_width=True)

                # Additional statistics
                st.subheader("📊 Performance Statistics")

                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)

                with stats_col1:
                    avg_parcels = per_day_df['daily_parcels'].mean()
                    st.metric("Average Daily Parcels", f"{avg_parcels:.1f}")

                with stats_col2:
                    max_parcels = per_day_df['daily_parcels'].max()
                    st.metric("Maximum Daily Parcels", f"{max_parcels:.0f}")

                with stats_col3:
                    avg_payout = per_day_df['payout_per_day'].mean()
                    st.metric("Average Daily Payout", f"{config['currency_symbol']}{avg_payout:.2f}")

                with stats_col4:
                    best_day = per_day_df.loc[per_day_df['payout_per_day'].idxmax()]
                    st.metric("Best Day Payout", f"{config['currency_symbol']}{best_day['payout_per_day']:.2f}")
            else:
                st.info("No chart data available for this dispatcher.")
        else:
            st.info("Performance data not available. Complete the calculation in the Payout Details tab first.")

    with tab3:
        # Generate and display invoice
        if 'display_df' in locals() and not display_df.empty:
            # Generate HTML invoice
            invoice_html = InvoiceGenerator.build_invoice_html(
                df_disp=display_df,
                base_payout=base_delivery_payout,
                gross_payout=gross_delivery_payout,
                kpi_bonus=kpi_bonus,
                attendance_bonus=attendance_bonus,
                advance_payout=advance_payout,
                advance_payout_desc=advance_description,
                penalty_breakdown=penalty_breakdown,
                socso_deduction=socso_deduction,
                socso_deduction_count=socso_deduction_count,
                overpaid_deduction=overpaid_deduction,
                overpaid_deduction_count=overpaid_deduction_count,
                reward_payout=reward_payout,
                reward_count=reward_count,
                name=selected_dispatcher_name or selected_dispatcher_id,
                dpid=selected_dispatcher_id,
                currency_symbol=config["currency_symbol"],
                pickup_payout=pickup_payout,
                pickup_parcels=pickup_parcels,
                return_payout=return_payout,
                return_count=return_count,
                kpi_description=kpi_description,
                attendance_description=attendance_desc,
                total_awb=total_awb,
                designated_driver_breakdown=designated_driver_breakdown,
                pickup_rate=float(config.get("pickup_payout_per_parcel", 1.0)),
                return_rate=float(config.get("return_payout_per_parcel", 0.5)),
                advance_percentage=float(advance_percentage),
                advance_enabled=bool(advance_enabled),
            )

            # Display invoice
            st.components.v1.html(invoice_html, height=1200, scrolling=True)

            # Generate date range string for file names
            date_range_str = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            period_display = f"{start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}"

            # Download buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                # Download as HTML
                st.download_button(
                    label="📥 Download Invoice (HTML)",
                    data=invoice_html,
                    file_name=f"invoice_{selected_dispatcher_id}_{date_range_str}.html",
                    mime="text/html",
                    use_container_width=True
                )

            with col2:
                # Download as CSV (daily breakdown)
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Daily Breakdown (CSV)",
                    data=csv_data,
                    file_name=f"breakdown_{selected_dispatcher_id}_{date_range_str}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col3:
                # Download summary as text
                base_delivery_lines = f"Base Delivery Payout: {config['currency_symbol']}{base_delivery_payout:,.2f}\n"
                if designated_driver_breakdown:
                    dd = designated_driver_breakdown
                    base_delivery_lines += (
                        f"  Basic Amount (first {dd['basic_parcels']:,} parcels): "
                        f"{config['currency_symbol']}{dd['basic_amount']:,.2f}\n"
                    )
                    if dd.get("extra_payout", 0) > 0:
                        base_delivery_lines += (
                            f"  Extra Parcels ({dd['extra_parcels']:,} × "
                            f"{config['currency_symbol']}{dd['rate_after_basic']:,.2f}): "
                            f"+{config['currency_symbol']}{dd['extra_payout']:,.2f}\n"
                        )
                    if dd.get("special_payout", 0) > 0:
                        base_delivery_lines += (
                            f"  Special Rate Incentive: "
                            f"+{config['currency_symbol']}{dd['special_payout']:,.2f}\n"
                        )

                summary_text = f"""
INVOICE SUMMARY
===============
Dispatcher: {selected_dispatcher_name or selected_dispatcher_id}
Dispatcher ID: {selected_dispatcher_id}
Period: {period_display}

SUMMARY
-------
Total AWB: {total_awb:,}
Total Delivery Parcels: {display_df['Total Parcel'].sum():,}
Working Days: {len(display_df)}
Pickup Parcels: {pickup_parcels:,}
Return Parcels: {return_count:,}

PAYOUT BREAKDOWN
----------------
{base_delivery_lines}Commission Delivery: +{config['currency_symbol']}{pickup_payout:,.2f}
Return Payout: +{config['currency_symbol']}{return_payout:,.2f}
Reward: +{config['currency_symbol']}{reward_payout:,.2f}
KPI Bonus: +{config['currency_symbol']}{kpi_bonus:,.2f}
Attendance Bonus: +{config['currency_symbol']}{attendance_bonus:,.2f}
Total Penalties: -{config['currency_symbol']}{penalty_breakdown['total_amount']:,.2f}
SOCSO: -{config['currency_symbol']}{socso_deduction:,.2f}
Overpaid: -{config['currency_symbol']}{overpaid_deduction:,.2f}

GROSS PAYOUT
------------
Gross Payout: {config['currency_symbol']}{gross_total_payout:,.2f}

ADVANCE PAYOUT
--------------
Advance ({advance_percentage:g}% of Base Delivery): -{config['currency_symbol']}{advance_payout:,.2f}

FINAL PAYOUT
------------
Final Payout: {config['currency_symbol']}{final_payout:,.2f}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """
                st.download_button(
                    label="📝 Download Summary (TXT)",
                    data=summary_text,
                    file_name=f"summary_{selected_dispatcher_id}_{date_range_str}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            st.info("No invoice data available. Complete the calculation in the Payout Details tab first.")

    # # Configuration sidebar
    # st.sidebar.title("⚙️ Configuration")

    # if st.sidebar.button("🔄 Reload Data", use_container_width=True):
    #     st.cache_data.clear()
    #     st.rerun()

    # # Display current configuration
    # st.sidebar.subheader("Current Settings")

    # with st.sidebar.expander("📋 View Configuration"):
    #     st.json(config, expanded=False)

    # # Configuration editor
    # st.sidebar.subheader("⚙️ Edit Configuration")

    # if st.sidebar.button("Edit Configuration", use_container_width=True):
    #     # Open configuration in a modal or new page
    #     st.session_state.edit_config = True

    # if 'edit_config' in st.session_state and st.session_state.edit_config:
    #     with st.sidebar.expander("✏️ Edit Config", expanded=True):
    #         # Google Sheet URL
    #         new_gsheet_url = st.text_input(
    #             "Google Sheet URL",
    #             value=config["data_source"]["gsheet_url"],
    #             help="URL of the Google Sheet containing dispatcher data"
    #         )

    #         # Tiers configuration
    #         st.write("### Tier Configuration")
    #         for i, tier in enumerate(config["tiers"]):
    #             col1, col2, col3 = st.columns(3)
    #             with col1:
    #                 tier_name = st.text_input(f"Tier {i+1} Name", value=tier["Tier"], key=f"tier_name_{i}")
    #             with col2:
    #                 min_parcels = st.number_input(f"Min Parcels", value=tier["Min Parcels"] or 0, key=f"tier_min_{i}")
    #             with col3:
    #                 max_parcels = st.number_input(f"Max Parcels", value=tier["Max Parcels"] or 0, key=f"tier_max_{i}")
    #             rate = st.number_input(f"Rate (RM)", value=tier["Rate (RM)"], min_value=0.0, step=0.05, key=f"tier_rate_{i}")
    #             config["tiers"][i] = {"Tier": tier_name, "Min Parcels": min_parcels, "Max Parcels": max_parcels, "Rate (RM)": rate}

    #         # KPI incentives
    #         st.write("### KPI Incentives")
    #         for i, kpi in enumerate(config.get("kpi_incentives", [])):
    #             col1, col2 = st.columns(2)
    #             with col1:
    #                 parcels = st.number_input(f"Parcels Required", value=kpi["parcels"], step=100, key=f"kpi_parcels_{i}")
    #             with col2:
    #                 bonus = st.number_input(f"Bonus (RM)", value=kpi["bonus"], step=50.0, key=f"kpi_bonus_{i}")
    #             description = st.text_input(f"Description", value=kpi["description"], key=f"kpi_desc_{i}")
    #             config["kpi_incentives"][i] = {"parcels": parcels, "bonus": bonus, "description": description}

    #         # Save configuration
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             if st.button("💾 Save", use_container_width=True):
    #                 config["data_source"]["gsheet_url"] = new_gsheet_url
    #                 if Config.save(config):
    #                     st.success("Configuration saved!")
    #                     st.cache_data.clear()
    #                     st.rerun()
    #         with col2:
    #             if st.button("❌ Cancel", use_container_width=True):
    #                 st.session_state.edit_config = False
    #                 st.rerun()

    # Add footer
    add_footer()

if __name__ == "__main__":
    main()
