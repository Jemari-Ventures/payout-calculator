import warnings
import urllib3
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)

import io
import re
import json
import os
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
from decimal import Decimal

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import requests
from sqlalchemy import create_engine
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# =============================================================================
# CONSTANTS
# =============================================================================
PROPHET_AVAILABLE = True
class ColorScheme:
    """Consistent color scheme for the entire application."""
    PRIMARY = "#4f46e5"
    PRIMARY_LIGHT = "#818cf8"
    SECONDARY = "#10b981"
    ACCENT = "#f59e0b"
    BACKGROUND = "#f8fafc"
    SURFACE = "#ffffff"
    TEXT_PRIMARY = "#1e293b"
    TEXT_SECONDARY = "#64748b"
    BORDER = "#e2e8f0"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    ERROR = "#ef4444"
    CHART_COLORS = [
        "#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
        "#06b6d4", "#84cc16", "#f97316", "#ec4899", "#6366f1"
    ]


# Centralized column mappings
COLUMN_MAPPINGS = {
    'waybill': ['Waybill Number', 'Waybill', 'waybill_number', 'waybill'],
    'date': [
        'Delivery Signature', 'Delivery Date', 'Date',
        'signature_date', 'delivery_date', 'delivery_signature',
        'delivery_signature_date'
    ],
    'dispatcher_id': ['Dispatcher ID', 'Dispatcher Id', 'dispatcher_id'],
    'dispatcher_name': ['Dispatcher Name', 'Dispatcher', 'dispatcher_name'],
    'weight': ['Billing Weight', 'Weight', 'weight', 'billing_weight']
}

DISPATCHER_PREFIXES = ['JMR', 'ECP', 'AF', 'PEN', 'KUL', 'JHR']

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration management with caching."""

    CONFIG_FILE = "config.json"
    DEFAULT_CONFIG = {
        "data_source": {
            "type": "postgres",
            "postgres_table": "dispatch",
            "pickup_table": "pickup",
            "excel_file": "",
            "excel_sheets": {
                "dispatch": "Dispatch",
                "pickup": "Pickup",
                "duitnow": "DuitNow",
                "ldr": "LDR",
                "fake_attempt": "FakeAttempt",
                "cod": "COD",
                "qr_order": "QR Order",
                "return": "Return",
                "attendance": "Attendance",
                "reward": "Reward"
            }
        },
        "database": {"table_name": "dispatch"},
        "weight_tiers": [
            {"min": 0, "max": 5, "rate": 1.50},
            {"min": 5, "max": 10, "rate": 1.60},
            {"min": 10, "max": 30, "rate": 2.70},
            {"min": 30, "max": float('inf'), "rate": 4.00}
        ],
        "pickup_fee": 151.00,
        "currency_symbol": "RM",
        "forecast_days": 30,
        "qr_order_payout_per_order": 1.80,
        "return_payout_per_parcel": 1.50,
        "fake_attempt_penalty_per_parcel": 1.00
    }

    _cache = None

    @classmethod
    def load(cls) -> dict:
        """Load configuration with caching."""
        if cls._cache is not None:
            return cls._cache

        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    # Ensure new fields exist
                    if "postgres_table" not in config.get("data_source", {}):
                        config["data_source"]["postgres_table"] = "dispatch"
                    if "pickup_table" not in config.get("data_source", {}):
                        config["data_source"]["pickup_table"] = "pickup"
                    if "excel_file" not in config.get("data_source", {}):
                        config["data_source"]["excel_file"] = ""
                    if "excel_sheets" not in config.get("data_source", {}):
                        config["data_source"]["excel_sheets"] = Config.DEFAULT_CONFIG["data_source"]["excel_sheets"]
                    elif "reward" not in config["data_source"]["excel_sheets"]:
                        config["data_source"]["excel_sheets"]["reward"] = "Reward"
                    if "pickup_payout_per_parcel" not in config:
                        config["pickup_payout_per_parcel"] = 1.50
                    if "forecast_days" not in config:
                        config["forecast_days"] = 30
                    if "qr_order_payout_per_order" not in config:
                        config["qr_order_payout_per_order"] = 1.80
                    if "return_payout_per_parcel" not in config:
                        config["return_payout_per_parcel"] = 1.50
                    if "fake_attempt_penalty_per_parcel" not in config:
                        config["fake_attempt_penalty_per_parcel"] = 1.00
                    cls._cache = config
                    return cls._cache
            except Exception as e:
                st.error(f"Error loading config: {e}")

        cls.save(cls.DEFAULT_CONFIG)
        cls._cache = cls.DEFAULT_CONFIG
        return cls._cache

    @classmethod
    def save(cls, config: dict) -> bool:
        """Save configuration."""
        try:
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            cls._cache = config
            return True
        except Exception as e:
            st.error(f"Error saving config: {e}")
            return False

# =============================================================================
# UTILITIES
# =============================================================================

def find_column(df: pd.DataFrame, column_type: str) -> Optional[str]:
    """Find first matching column name from mappings."""
    for col in COLUMN_MAPPINGS.get(column_type, []):
        if col in df.columns:
            return col
    return None


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
    # Strip known prefixes (e.g., JMR001 -> 001) when followed by digits
    for prefix in DISPATCHER_PREFIXES:
        if text.startswith(prefix) and text[len(prefix):].isdigit():
            text = text[len(prefix):]
            break
    # Normalize purely numeric IDs by trimming leading zeros
    if text.isdigit():
        text = text.lstrip("0") or "0"
    return text


def clean_dispatcher_name(name: str) -> str:
    """Remove prefixes from dispatcher names."""
    cleaned = str(name).strip()
    for prefix in DISPATCHER_PREFIXES:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].lstrip(' -')
    return cleaned


def normalize_weight(series: pd.Series) -> pd.Series:
    """Normalize weight column to numeric."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace(r'[^0-9\.-]', '', regex=True),
        errors='coerce'
    )

# =============================================================================
# DATA SOURCE
# =============================================================================

class DataSource:
    """Handle data loading from PostgreSQL."""

    @staticmethod
    def _extract_gsheet_id_and_gid(url_or_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract spreadsheet ID and gid from a URL or ID."""
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
        if sheet_name:
            return f"{base}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
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
        resp = requests.get(csv_url, timeout=30)
        resp.raise_for_status()

        resp_content = resp.content
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

        df = pd.read_csv(
            io.BytesIO(resp_content),
            keep_default_na=False,
            na_values=[],
            encoding='utf-8',
            low_memory=False
        )

        waybill_col = find_column(df, "waybill")
        if waybill_col:
            df[waybill_col] = df[waybill_col].astype(str)

        return df

    @staticmethod
    def _get_postgres_connection():
        """Get PostgreSQL connection from Streamlit secrets."""
        try:
            if "postgres" in st.secrets and "connection_string" in st.secrets["postgres"]:
                return st.secrets["postgres"]["connection_string"]
            else:
                st.error("PostgreSQL connection string not found in secrets.toml")
                return None
        except Exception as e:
            st.error(f"Error reading PostgreSQL secrets: {e}")
            return None

    @staticmethod
    @st.cache_resource
    def get_postgres_engine():
        """Create and cache PostgreSQL engine."""
        conn_str = DataSource._get_postgres_connection()
        if conn_str:
            try:
                engine = create_engine(conn_str)
                return engine
            except Exception as e:
                st.error(f"Error creating PostgreSQL engine: {e}")
                return None
        return None

    @staticmethod
    @st.cache_data(ttl=300)
    def read_postgres_table(_engine, table_name: str) -> pd.DataFrame:
        """Read data from PostgreSQL table with column mapping."""
        try:
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, _engine)

            # Define column mappings for different tables
            if table_name == 'dispatch' or table_name.endswith('dispatch'):
                column_mapping = {
                    'waybill_number': 'Waybill Number',  # Primary column name
                    'waybill': 'Waybill Number',  # Fallback
                    'delivery_signature_date': 'Delivery Signature',  # Primary column name
                    'delivery_signature': 'Delivery Signature',  # Fallback
                    'dispatcher_id': 'Dispatcher ID',
                    'rider_name': 'Dispatcher Name',  # Primary column name (from dispatch table)
                    'dispatcher_name': 'Dispatcher Name',  # Fallback
                    'weight_kg': 'Billing Weight',  # Primary column name
                    'billing_weight': 'Billing Weight',  # Fallback
                    'date_pick_up': 'Pick Up Date',  # Primary column name
                    'date_|_pick_up': 'Pick Up Date',  # Fallback
                    'pickup_delivery_point': 'Pick Up DP',  # Primary column name
                    'pick_up_dp': 'Pick Up DP',  # Fallback
                    'pickup_dispatcher_id': 'Pick Up Dispatcher ID',  # Primary column name
                    'pick_up_dispatcher_id': 'Pick Up Dispatcher ID',  # Fallback
                    'pickup_rider_name': 'Pick Up Dispatcher Name',  # Primary column name
                    'pick_up_dispatcher_name': 'Pick Up Dispatcher Name'  # Fallback
                }
            elif table_name == 'pickup' or table_name.endswith('pickup'):
                column_mapping = {
                    'waybill_number': 'Waybill Number',
                    'date_|_pick_up': 'Pick Up Date',
                    'pickup_dp': 'Pickup DP',
                    'responsible_department': 'Responsible Department',
                    'pickup_dispatcher_id': 'Pickup Dispatcher ID',
                    'pickup_dispatcher_name': 'Pickup Dispatcher Name',
                    'vip_code': 'VIP Code',
                    'domestic___international': 'Domestic/International',
                    'order_source': 'Order Source',
                    'product_type': 'Product Type',
                    'dp_|_signing': 'DP Signing',
                    'delivery_signature': 'Delivery Signature',
                    'dispatcher_id': 'Dispatcher ID',
                    'dispatcher_name': 'Dispatcher Name',
                    'arrival_time': 'Arrival Time',
                    'arriving_dp': 'Arriving DP',
                    'billing_weight': 'Billing Weight',
                    'item_type': 'Item Type',
                    'cod_amount': 'COD Amount'
                }
            elif table_name == 'duitnow_penalty' or table_name.endswith('duitnow_penalty'):
                column_mapping = {}  # No mapping needed, use original column names
            elif table_name == 'ldr_penalty' or table_name.endswith('ldr_penalty'):
                column_mapping = {}  # No mapping needed, use original column names
            elif table_name == 'fake_attempt_penalty' or table_name.endswith('fake_attempt_penalty'):
                column_mapping = {}  # No mapping needed, use original column names
            elif table_name == 'cod_penalty' or table_name.endswith('cod_penalty'):
                column_mapping = {}  # No mapping needed, use original column names
            elif table_name == 'qr_order' or table_name.endswith('qr_order'):
                column_mapping = {}  # No mapping needed, use original column names
            elif table_name == 'return' or table_name.endswith('return'):
                column_mapping = {
                    'waybill_number': 'Waybill Number',
                    'dispatcher_id': 'Dispatcher ID',
                    'rider_name': 'Dispatcher Name',
                    'sender_name': 'Sender Name',
                    'delivered_signature_id': 'Delivered Signature ID',
                    'created_at': 'Created At',
                    'updated_at': 'Updated At'
                }
            elif table_name == 'attendance' or table_name.endswith('attendance'):
                column_mapping = {
                    'employee_id': 'Employee ID',
                    'employee_name': 'Employee name',
                    'attendance_record_date': 'Attendance Record Date'
                }
            else:
                column_mapping = {}

            # Rename columns that exist in the dataframe
            # Handle priority: if multiple columns map to same target, use the first one that exists
            rename_dict = {}
            seen_targets = set()
            for old, new in column_mapping.items():
                if old in df.columns and new not in seen_targets:
                    rename_dict[old] = new
                    seen_targets.add(new)
            df = df.rename(columns=rename_dict)

            # For dispatch table, ensure required columns exist even if original columns were missing
            if table_name == 'dispatch' or table_name.endswith('dispatch'):
                # Ensure Waybill Number exists
                if 'Waybill Number' not in df.columns:
                    # Try to find waybill_number or waybill column (case-insensitive)
                    waybill_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['waybill_number', 'waybill']:
                            waybill_col = col
                            break
                    if waybill_col:
                        df['Waybill Number'] = df[waybill_col]

                # Ensure Dispatcher ID exists
                if 'Dispatcher ID' not in df.columns:
                    # Try to find dispatcher_id column (case-insensitive)
                    dispatcher_id_col = None
                    for col in df.columns:
                        if col.lower() == 'dispatcher_id':
                            dispatcher_id_col = col
                            break
                    if dispatcher_id_col:
                        df['Dispatcher ID'] = df[dispatcher_id_col]
                    else:
                        df['Dispatcher ID'] = 'Unknown'

                # Ensure Dispatcher Name exists
                if 'Dispatcher Name' not in df.columns:
                    # Try to find rider_name or dispatcher_name column (case-insensitive)
                    dispatcher_name_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['rider_name', 'dispatcher_name']:
                            dispatcher_name_col = col
                            break
                    if dispatcher_name_col:
                        df['Dispatcher Name'] = df[dispatcher_name_col]
                    else:
                        df['Dispatcher Name'] = 'Unknown'

                # Ensure Delivery Signature exists
                if 'Delivery Signature' not in df.columns:
                    # Try to find delivery_signature_date or delivery_signature column (case-insensitive)
                    date_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['delivery_signature_date', 'delivery_signature']:
                            date_col = col
                            break
                    if date_col:
                        df['Delivery Signature'] = df[date_col]

                # Ensure Billing Weight exists
                if 'Billing Weight' not in df.columns:
                    # Try to find weight_kg or billing_weight column (case-insensitive)
                    weight_col = None
                    for col in df.columns:
                        col_lower = col.lower()
                        if col_lower in ['weight_kg', 'billing_weight']:
                            weight_col = col
                            break
                    if weight_col:
                        df['Billing Weight'] = df[weight_col]

            return df
        except Exception as e:
            st.error(f"Error reading from PostgreSQL table '{table_name}': {e}")
            raise

    @staticmethod
    def _get_excel_sheet_name(config: dict, key: str, fallback: str) -> str:
        sheets = config.get("data_source", {}).get("excel_sheets", {})
        return sheets.get(key, fallback)

    @staticmethod
    @st.cache_data(ttl=300)
    def read_excel_sheet(excel_source, sheet_name: str) -> pd.DataFrame:
        """Read data from an Excel sheet (path or uploaded bytes)."""
        if excel_source is None:
            raise ValueError("Excel source not provided")

        if isinstance(excel_source, (bytes, bytearray)):
            return pd.read_excel(io.BytesIO(excel_source), sheet_name=sheet_name)

        if hasattr(excel_source, "read"):
            return pd.read_excel(excel_source, sheet_name=sheet_name)

        return pd.read_excel(excel_source, sheet_name=sheet_name)

    @staticmethod
    def load_data(config: dict, excel_source=None) -> Optional[pd.DataFrame]:
        """Load data based on configuration."""
        data_source = config["data_source"]
        source_type = data_source.get("type", "postgres")

        if source_type == "excel":
            try:
                sheet_name = DataSource._get_excel_sheet_name(config, "dispatch", "Dispatch")
                gsheet_url = data_source.get("gsheet_url")
                if not gsheet_url:
                    st.error("Google Sheet URL not provided")
                    return None
                df = DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)

                # Normalize dispatch columns (Excel/gsheet may use snake_case)
                if 'Waybill Number' not in df.columns:
                    waybill_col = next((col for col in df.columns if str(col).lower() in ['waybill_number', 'waybill']), None)
                    if waybill_col:
                        df['Waybill Number'] = df[waybill_col]

                if 'Dispatcher ID' not in df.columns:
                    dispatcher_id_col = next((col for col in df.columns if str(col).lower() == 'dispatcher_id'), None)
                    if dispatcher_id_col:
                        df['Dispatcher ID'] = df[dispatcher_id_col]

                if 'Dispatcher Name' not in df.columns:
                    dispatcher_name_col = next((col for col in df.columns if str(col).lower() in ['rider_name', 'dispatcher_name']), None)
                    if dispatcher_name_col:
                        df['Dispatcher Name'] = df[dispatcher_name_col]

                if 'Delivery Signature' not in df.columns:
                    date_col = next((col for col in df.columns if str(col).lower() in ['delivery_signature_date', 'delivery_signature', 'delivery_point_signing']), None)
                    if date_col:
                        df['Delivery Signature'] = df[date_col]

                if 'Billing Weight' not in df.columns:
                    weight_col = next((col for col in df.columns if str(col).lower() in ['weight_kg', 'billing_weight']), None)
                    if weight_col:
                        df['Billing Weight'] = df[weight_col]

                # Verify required columns exist
                required = ["Dispatcher ID", "Waybill Number", "Delivery Signature"]
                missing = [col for col in required if col not in df.columns]
                if missing:
                    st.error(f"Missing required columns in Excel sheet '{sheet_name}': {', '.join(missing)}")
                    st.info("Available columns: " + ", ".join(df.columns.tolist()))
                    return None

                return df
            except Exception as exc:
                st.error(f"Error reading from Excel: {exc}")
                return None

        engine = DataSource.get_postgres_engine()
        if not engine:
            st.error("PostgreSQL engine not available")
            return None

        try:
            table_name = data_source.get("postgres_table", "dispatch")
            df = DataSource.read_postgres_table(engine, table_name)

            # Verify required columns exist after mapping
            required = ["Dispatcher ID", "Waybill Number", "Delivery Signature"]
            missing = [col for col in required if col not in df.columns]
            if missing:
                st.error(f"Missing required columns after mapping: {', '.join(missing)}")
                st.info("Available columns: " + ", ".join(df.columns.tolist()))
                return None

            return df
        except Exception as exc:
            st.error(f"Error reading from PostgreSQL: {exc}")
            return None

    @staticmethod
    def load_penalty_data(config: dict, excel_source=None) -> Optional[Dict[str, pd.DataFrame]]:
        """Load penalty data from all penalty tables.

        Returns:
            Dictionary with keys: 'duitnow', 'ldr', 'fake_attempt', 'cod'
        """
        data_source = config.get("data_source", {})
        source_type = data_source.get("type", "postgres")

        if source_type == "excel":
            penalty_data = {}
            sheet_map = {
                "duitnow": DataSource._get_excel_sheet_name(config, "duitnow", "DuitNow"),
                "ldr": DataSource._get_excel_sheet_name(config, "ldr", "LDR"),
                "fake_attempt": DataSource._get_excel_sheet_name(config, "fake_attempt", "FakeAttempt"),
                "cod": DataSource._get_excel_sheet_name(config, "cod", "COD"),
            }

            gsheet_url = data_source.get("gsheet_url")
            if not gsheet_url:
                st.warning("Google Sheet URL not provided for penalty data")
                return None
            for key, sheet_name in sheet_map.items():
                try:
                    df = DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)
                    if not df.empty:
                        penalty_data[key] = df
                except Exception as exc:
                    st.warning(f"Could not load {key} penalty data from Excel sheet '{sheet_name}': {exc}")

            return penalty_data if penalty_data else None

        engine = DataSource.get_postgres_engine()
        if not engine:
            return None

        penalty_data = {}

        # Load DuitNow penalty
        try:
            duitnow_df = DataSource.read_postgres_table(engine, 'duitnow_penalty')
            if not duitnow_df.empty:
                penalty_data['duitnow'] = duitnow_df
        except Exception as exc:
            st.warning(f"Could not load DuitNow penalty data: {exc}")

        # Load LDR penalty
        try:
            ldr_df = DataSource.read_postgres_table(engine, 'ldr_penalty')
            if not ldr_df.empty:
                penalty_data['ldr'] = ldr_df
        except Exception as exc:
            st.warning(f"Could not load LDR penalty data: {exc}")

        # Load Fake Attempt penalty
        try:
            fake_attempt_df = DataSource.read_postgres_table(engine, 'fake_attempt_penalty')
            if not fake_attempt_df.empty:
                penalty_data['fake_attempt'] = fake_attempt_df
        except Exception as exc:
            st.warning(f"Could not load Fake Attempt penalty data: {exc}")

        # Load COD penalty
        try:
            cod_df = DataSource.read_postgres_table(engine, 'cod_penalty')
            if not cod_df.empty:
                penalty_data['cod'] = cod_df
        except Exception as exc:
            st.warning(f"Could not load COD penalty data: {exc}")

        return penalty_data if penalty_data else None

    @staticmethod
    def load_pickup_data(config: dict, excel_source=None) -> Optional[pd.DataFrame]:
        """Load pickup data from pickup table."""
        data_source = config["data_source"]
        source_type = data_source.get("type", "postgres")

        if source_type == "excel":
            try:
                sheet_name = DataSource._get_excel_sheet_name(config, "pickup", "Pickup")
                gsheet_url = data_source.get("gsheet_url")
                if not gsheet_url:
                    return None
                return DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)
            except Exception as exc:
                st.warning(f"Could not load pickup data from Excel sheet '{sheet_name}': {exc}")
                return None

        engine = DataSource.get_postgres_engine()

        if not engine:
            return None

        try:
            pickup_table = data_source.get("pickup_table", "pickup")
            df = DataSource.read_postgres_table(engine, pickup_table)
            return df
        except Exception as exc:
            st.warning(f"Could not load pickup data from PostgreSQL table '{pickup_table}': {exc}")
            return None

    @staticmethod
    def load_qr_order_data(config: dict, excel_source=None) -> Optional[pd.DataFrame]:
        """Load QR order data from qr_order table."""
        data_source = config.get("data_source", {})
        source_type = data_source.get("type", "postgres")

        if source_type == "excel":
            try:
                sheet_name = DataSource._get_excel_sheet_name(config, "qr_order", "QR Order")
                gsheet_url = data_source.get("gsheet_url")
                if not gsheet_url:
                    return None
                return DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)
            except Exception as exc:
                st.warning(f"Could not load QR order data from Excel sheet '{sheet_name}': {exc}")
                return None

        engine = DataSource.get_postgres_engine()

        if not engine:
            return None

        try:
            df = DataSource.read_postgres_table(engine, 'qr_order')
            return df
        except Exception as exc:
            st.warning(f"Could not load QR order data from PostgreSQL table 'qr_order': {exc}")
            return None

    @staticmethod
    def load_return_data(config: dict, excel_source=None) -> Optional[pd.DataFrame]:
        """Load return data from return table."""
        data_source = config.get("data_source", {})
        source_type = data_source.get("type", "postgres")

        if source_type == "excel":
            try:
                sheet_name = DataSource._get_excel_sheet_name(config, "return", "Return")
                gsheet_url = data_source.get("gsheet_url")
                if not gsheet_url:
                    return None
                return DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)
            except Exception as exc:
                st.warning(f"Could not load return data from Excel sheet '{sheet_name}': {exc}")
                return None

        engine = DataSource.get_postgres_engine()

        if not engine:
            return None

        try:
            df = DataSource.read_postgres_table(engine, 'return')
            return df
        except Exception as exc:
            st.warning(f"Could not load return data from PostgreSQL table 'return': {exc}")
            return None

    @staticmethod
    def load_attendance_data(config: dict, excel_source=None) -> Optional[pd.DataFrame]:
        """Load attendance data from attendance table or Excel sheet."""
        data_source = config.get("data_source", {})
        source_type = data_source.get("type", "postgres")

        if source_type == "excel":
            try:
                sheet_name = DataSource._get_excel_sheet_name(config, "attendance", "Attendance")
                gsheet_url = data_source.get("gsheet_url")
                if not gsheet_url:
                    return None
                df = DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)
                if df.empty and sheet_name != sheet_name.upper():
                    df = DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name.upper())
                return df
            except Exception as exc:
                st.warning(f"Could not load attendance data from Excel sheet '{sheet_name}': {exc}")
                return None

        engine = DataSource.get_postgres_engine()
        if not engine:
            return None

        try:
            df = DataSource.read_postgres_table(engine, 'attendance')
            return df
        except Exception as exc:
            st.warning(f"Could not load attendance data from PostgreSQL table 'attendance': {exc}")
            return None

    @staticmethod
    def load_reward_data(config: dict, excel_source=None) -> Optional[pd.DataFrame]:
        """Load reward data from reward table or Excel sheet."""
        data_source = config.get("data_source", {})
        source_type = data_source.get("type", "postgres")

        if source_type == "excel":
            try:
                sheet_name = DataSource._get_excel_sheet_name(config, "reward", "Reward")
                gsheet_url = data_source.get("gsheet_url")
                if not gsheet_url:
                    return None
                df = DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name)
                if df.empty and sheet_name != sheet_name.upper():
                    df = DataSource.read_google_sheet(gsheet_url, sheet_name=sheet_name.upper())
                return df
            except Exception as exc:
                st.warning(f"Could not load reward data from Excel sheet '{sheet_name}': {exc}")
                return None

        engine = DataSource.get_postgres_engine()
        if not engine:
            return None

        try:
            df = DataSource.read_postgres_table(engine, 'reward')
            return df
        except Exception as exc:
            st.warning(f"Could not load reward data from PostgreSQL table 'reward': {exc}")
            return None

# =============================================================================
# DATA PROCESSING
# =============================================================================

class DataProcessor:
    """Handle data cleaning and preparation."""

    @staticmethod
    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe columns."""
        df_clean = df.copy()
        df_clean = df_clean.rename(columns={c: str(c).strip() for c in df_clean.columns})

        # Standardize column names
        for standard_name, possible_names in COLUMN_MAPPINGS.items():
            col = find_column(df_clean, standard_name)
            if col and col != standard_name:
                df_clean[standard_name] = df_clean[col]

        # Clean dispatcher names
        if 'dispatcher_name' in df_clean.columns:
            df_clean['dispatcher_name'] = df_clean['dispatcher_name'].apply(clean_dispatcher_name)
        else:
            df_clean['dispatcher_name'] = 'Unknown'

        # Ensure dispatcher_id exists
        if 'dispatcher_id' not in df_clean.columns:
            df_clean['dispatcher_id'] = 'Unknown'

        # Normalize weight
        if 'weight' in df_clean.columns:
            df_clean['weight'] = normalize_weight(df_clean['weight'])

        return df_clean

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """Keep duplicate waybills (no filtering)."""
        return df

# =============================================================================
# PAYOUT CALCULATIONS
# =============================================================================

class PayoutCalculator:
    """Handle payout calculations."""

    @staticmethod
    def get_rate_by_weight(weight: float, tiers: Optional[List[dict]] = None) -> float:
        """Get per-parcel rate based on weight tiers.

        Tier boundaries (as specified):
        - 0kg < x < 5kg → 1.50 (0.01 to 4.99, or exactly 5.0 falls here)
        - 5.01kg < x < 10kg → 1.60 (5.01 to 9.99, or exactly 10.0 falls here)
        - 10.01kg < x < 30kg → 2.70 (10.01 to 29.99, or exactly 30.0 falls here)
        - x > 30kg → 4.00 (30.01+)
        """
        if tiers is None:
            tiers = Config.load().get("weight_tiers", Config.DEFAULT_CONFIG["weight_tiers"])

        w = 0.0 if pd.isna(weight) else float(weight)

        # Handle zero or negative weights - use first tier
        if w <= 0:
            return tiers[0]['rate']

        # Sort tiers by min value to ensure correct order
        sorted_tiers = sorted(tiers, key=lambda t: t['min'])

        # Tier 4: x > 30kg (30.01+)
        if w > 30:
            return sorted_tiers[3]['rate'] if len(sorted_tiers) > 3 else sorted_tiers[-1]['rate']

        # Tier 3: 10.01kg <= x <= 30kg (10.01 to 30.0)
        # Note: 30.0 falls into tier 3, not tier 4
        if w >= 10.01:
            return sorted_tiers[2]['rate'] if len(sorted_tiers) > 2 else sorted_tiers[-1]['rate']

        # Tier 2: 5.01kg <= x < 10.01kg (5.01 to 10.0)
        # Note: 10.0 falls into tier 2, not tier 3
        if w >= 5.01:
            return sorted_tiers[1]['rate'] if len(sorted_tiers) > 1 else sorted_tiers[0]['rate']

        # Tier 1: 0kg < x < 5.01kg (0.01 to 5.0)
        # Note: 5.0 falls into tier 1, not tier 2
        return sorted_tiers[0]['rate']

    @staticmethod
    def _preprocess_penalty_data(penalty_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Pre-process penalty dataframes to cache column lookups and Decimal conversions.
        This improves performance by doing the work once instead of for each dispatcher.
        """
        processed = {}

        for penalty_type, df in penalty_data.items():
            if df is None or df.empty:
                continue

            df_processed = df.copy()

            # Pre-process DuitNow penalty data
            if penalty_type == 'duitnow':
                # Find and cache column names
                rider_col = next((col for col in df.columns if col.lower() == 'rider'), None)
                penalty_col = next((col for col in df.columns if col.lower() == 'penalty'), None)

                if rider_col and penalty_col:
                    # Pre-convert penalty column to Decimal once
                    if 'penalty_numeric' not in df_processed.columns:
                        df_processed['penalty_numeric'] = df_processed[penalty_col].apply(
                            lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0')
                        )
                    # Pre-filter positive penalties
                    df_processed = df_processed[df_processed['penalty_numeric'] > 0].copy()
                    # Pre-normalize rider IDs
                    df_processed['_rider_normalized'] = df_processed[rider_col].apply(normalize_dispatcher_id)

            # Pre-process LDR penalty data
            elif penalty_type == 'ldr':
                employee_id_col = next((col for col in df.columns if col.lower() == 'employee_id'), None)
                if employee_id_col:
                    df_processed['_employee_id_normalized'] = df_processed[employee_id_col].apply(normalize_dispatcher_id)
                ldr_penalty_col = next((col for col in df.columns if col.lower() in ['penalty', 'amount']), None)
                if ldr_penalty_col:
                    # Use explicit penalty column when provided by source sheet
                    if 'penalty_numeric' not in df_processed.columns:
                        df_processed['penalty_numeric'] = df_processed[ldr_penalty_col].apply(
                            lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0')
                        )
                    df_processed = df_processed[df_processed['penalty_numeric'] > 0].copy()

            # Pre-process Fake Attempt penalty data
            elif penalty_type == 'fake_attempt':
                dispatcher_id_col = next((col for col in df.columns if col.lower() == 'dispatcher_id'), None)
                if dispatcher_id_col:
                    df_processed['_dispatcher_id_normalized'] = df_processed[dispatcher_id_col].apply(normalize_dispatcher_id)

            # Pre-process COD penalty data
            elif penalty_type == 'cod':
                dispatcher_id_col = next((col for col in df.columns if col.lower() == 'dispatcher_id'), None)
                penalty_col = next((col for col in df.columns if col.lower() == 'penalty'), None)

                if dispatcher_id_col:
                    df_processed['_dispatcher_id_normalized'] = df_processed[dispatcher_id_col].apply(normalize_dispatcher_id)

                if penalty_col:
                    # Pre-convert penalty column to Decimal once (COD penalty is bigint, convert to Decimal for precision)
                    if 'penalty_numeric' not in df_processed.columns:
                        df_processed['penalty_numeric'] = df_processed[penalty_col].apply(
                            lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0')
                        )
                    # Pre-filter positive penalties
                    df_processed = df_processed[df_processed['penalty_numeric'] > 0].copy()

            processed[penalty_type] = df_processed

        return processed

    @staticmethod
    def calculate_penalty(dispatcher_id: str, penalty_data: Optional[Dict[str, pd.DataFrame]],
                          attendance_penalty_amount: float = 0.0) -> Tuple[float, int, List[str], float]:
        """
        Calculate total penalty for a dispatcher from all penalty types.

        Args:
            dispatcher_id: Dispatcher ID to calculate penalty for
            penalty_data: Dictionary containing penalty dataframes from all penalty tables
            attendance_penalty_amount: Attendance penalty amount for dispatcher

        Returns:
            (total_penalty_amount, total_penalty_count, waybill_numbers, attendance_penalty)
        """
        if penalty_data is None or not penalty_data:
            attendance_penalty = float(attendance_penalty_amount)
            return attendance_penalty, 0, [], attendance_penalty

        total_penalty = 0.0
        total_count = 0
        waybill_numbers = []
        dispatcher_id_clean = normalize_dispatcher_id(dispatcher_id)

        # 1. DuitNow Penalty: rider column = dispatcher_id, penalty amount from penalty column (only positive amounts)
        if 'duitnow' in penalty_data:
            duitnow_df = penalty_data['duitnow']
            if not duitnow_df.empty:
                # Use pre-processed data if available
                if '_rider_normalized' in duitnow_df.columns:
                    duitnow_records = duitnow_df[duitnow_df['_rider_normalized'] == dispatcher_id_clean]
                else:
                    # Fallback to original logic if not pre-processed
                    rider_col = next((col for col in duitnow_df.columns if col.lower() == 'rider'), None)
                    if rider_col:
                        rider_series = duitnow_df[rider_col].apply(normalize_dispatcher_id)
                        duitnow_records = duitnow_df[rider_series == dispatcher_id_clean]
                    else:
                        duitnow_records = pd.DataFrame()

                if not duitnow_records.empty and 'penalty_numeric' in duitnow_records.columns:
                    # Round each penalty value first, then sum (matching SQL: SUM((FLOOR((penalty * 100) + 0.5) / 100)))
                    # This ensures individual dispatcher totals match SQL calculation
                    rounded_penalties = [
                        penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                        for penalty in duitnow_records['penalty_numeric'].tolist()
                    ]
                    duitnow_penalty_rounded = sum(rounded_penalties).quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                    total_penalty += float(duitnow_penalty_rounded)  # Convert to float only at the end
                    total_count += len(duitnow_records)

        # 2. LDR Penalty: employee_id column = dispatcher_id, penalty from penalty/amount column
        if 'ldr' in penalty_data:
            ldr_df = penalty_data['ldr']
            if not ldr_df.empty:
                # Use pre-processed data if available
                if '_employee_id_normalized' in ldr_df.columns:
                    ldr_records = ldr_df[ldr_df['_employee_id_normalized'] == dispatcher_id_clean]
                else:
                    # Fallback to original logic
                    employee_id_col = next((col for col in ldr_df.columns if col.lower() == 'employee_id'), None)
                    if employee_id_col:
                        employee_series = ldr_df[employee_id_col].apply(normalize_dispatcher_id)
                        ldr_records = ldr_df[employee_series == dispatcher_id_clean]
                    else:
                        ldr_records = pd.DataFrame()

                if not ldr_records.empty:
                    waybill_col = next((col for col in ldr_df.columns if col.lower() in ['ticket_no', 'no_awb', 'waybill_number']), None)
                    if waybill_col:
                        waybill_list = ldr_records[waybill_col].dropna().astype(str).unique().tolist()
                        waybill_numbers.extend([wb for wb in waybill_list if wb and wb.lower() != 'nan'])

                    if 'penalty_numeric' in ldr_records.columns:
                        rounded_penalties = [
                            penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                            for penalty in ldr_records['penalty_numeric'].tolist()
                        ]
                        ldr_penalty = float(
                            sum(rounded_penalties).quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                        )
                    else:
                        # Fallback for legacy datasets without penalty column
                        if waybill_col:
                            waybill_count = ldr_records[waybill_col].nunique()
                        else:
                            waybill_count = len(ldr_records)
                        ldr_penalty = waybill_count * 100.0

                    total_penalty += ldr_penalty
                    total_count += len(ldr_records)

        # 3. Fake Attempt Penalty: dispatcher_id column = dispatcher_id, penalty = waybill count * RM 1.00
        if 'fake_attempt' in penalty_data:
            fake_attempt_df = penalty_data['fake_attempt']
            if not fake_attempt_df.empty:
                # Use pre-processed data if available
                if '_dispatcher_id_normalized' in fake_attempt_df.columns:
                    fake_attempt_records = fake_attempt_df[fake_attempt_df['_dispatcher_id_normalized'] == dispatcher_id_clean]
                else:
                    # Fallback to original logic
                    dispatcher_id_col = next((col for col in fake_attempt_df.columns if col.lower() == 'dispatcher_id'), None)
                    if dispatcher_id_col:
                        dispatcher_series = fake_attempt_df[dispatcher_id_col].apply(normalize_dispatcher_id)
                        fake_attempt_records = fake_attempt_df[dispatcher_series == dispatcher_id_clean]
                    else:
                        fake_attempt_records = pd.DataFrame()

                if not fake_attempt_records.empty:
                    # Count unique waybills
                    waybill_col = next((col for col in fake_attempt_df.columns if col.lower() in ['waybill_number', 'waybill']), None)
                    if waybill_col:
                        waybill_count = fake_attempt_records[waybill_col].nunique()
                        waybill_list = fake_attempt_records[waybill_col].dropna().astype(str).unique().tolist()
                        waybill_numbers.extend([wb for wb in waybill_list if wb and wb.lower() != 'nan'])
                    else:
                        waybill_count = len(fake_attempt_records)

                    fake_attempt_rate = Config.load().get("fake_attempt_penalty_per_parcel", 1.00)
                    fake_attempt_penalty = waybill_count * float(fake_attempt_rate)
                    total_penalty += fake_attempt_penalty
                    total_count += waybill_count

        # 4. COD Penalty: dispatcher_id column = dispatcher_id, penalty amount from penalty column (only positive amounts)
        if 'cod' in penalty_data:
            cod_df = penalty_data['cod']
            if not cod_df.empty:
                # Use pre-processed data if available
                if '_dispatcher_id_normalized' in cod_df.columns:
                    cod_records = cod_df[cod_df['_dispatcher_id_normalized'] == dispatcher_id_clean]
                else:
                    # Fallback to original logic if not pre-processed
                    dispatcher_id_col = next((col for col in cod_df.columns if col.lower() == 'dispatcher_id'), None)
                    if dispatcher_id_col:
                        dispatcher_series = cod_df[dispatcher_id_col].apply(normalize_dispatcher_id)
                        cod_records = cod_df[dispatcher_series == dispatcher_id_clean]
                    else:
                        cod_records = pd.DataFrame()

                if not cod_records.empty and 'penalty_numeric' in cod_records.columns:
                    # Round each penalty value first, then sum (matching SQL: SUM((FLOOR((penalty * 100) + 0.5) / 100)))
                    # This ensures individual dispatcher totals match SQL calculation
                    rounded_penalties = [
                        penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                        for penalty in cod_records['penalty_numeric'].tolist()
                    ]
                    cod_penalty_rounded = sum(rounded_penalties)
                    total_penalty += float(cod_penalty_rounded)  # Convert to float only at the end
                    total_count += len(cod_records)

        attendance_penalty = float(attendance_penalty_amount)
        total_penalty += attendance_penalty

        return float(total_penalty), total_count, waybill_numbers, attendance_penalty

    @staticmethod
    def calculate_penalty_breakdown(dispatcher_id: str, penalty_data: Optional[Dict[str, pd.DataFrame]],
                                    attendance_penalty_amount: float = 0.0) -> Dict[str, float]:
        """
        Calculate penalty breakdown by type for a dispatcher.

        Args:
            dispatcher_id: Dispatcher ID to calculate penalty for
            penalty_data: Dictionary containing penalty dataframes from all penalty tables

        Returns:
            Dictionary with keys: 'duitnow', 'ldr', 'fake_attempt', 'cod', 'attendance' and their penalty amounts
        """
        breakdown = {
            'duitnow': 0.0,
            'ldr': 0.0,
            'fake_attempt': 0.0,
            'cod': 0.0,
            'attendance': 0.0
        }

        breakdown['attendance'] = float(attendance_penalty_amount)

        if penalty_data is None or not penalty_data:
            return breakdown

        dispatcher_id_clean = normalize_dispatcher_id(dispatcher_id)

        # 1. DuitNow Penalty: rider column = dispatcher_id, penalty amount from penalty column (only positive amounts)
        if 'duitnow' in penalty_data:
            duitnow_df = penalty_data['duitnow']
            if not duitnow_df.empty:
                # Use pre-processed data if available
                if '_rider_normalized' in duitnow_df.columns:
                    duitnow_records = duitnow_df[duitnow_df['_rider_normalized'] == dispatcher_id_clean]
                else:
                    # Fallback to original logic if not pre-processed
                    rider_col = next((col for col in duitnow_df.columns if col.lower() == 'rider'), None)
                    if rider_col:
                        rider_series = duitnow_df[rider_col].apply(normalize_dispatcher_id)
                        duitnow_records = duitnow_df[rider_series == dispatcher_id_clean]
                    else:
                        duitnow_records = pd.DataFrame()

                if not duitnow_records.empty and 'penalty_numeric' in duitnow_records.columns:
                    # Round each penalty value first, then sum (matching SQL: SUM((FLOOR((penalty * 100) + 0.5) / 100)))
                    # This ensures individual dispatcher totals match SQL calculation
                    rounded_penalties = [
                        penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                        for penalty in duitnow_records['penalty_numeric'].tolist()
                    ]
                    breakdown['duitnow'] = float(
                        sum(rounded_penalties).quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                    )

        # 2. LDR Penalty: employee_id column = dispatcher_id, penalty from penalty/amount column
        if 'ldr' in penalty_data:
            ldr_df = penalty_data['ldr']
            if not ldr_df.empty:
                # Use pre-processed data if available
                if '_employee_id_normalized' in ldr_df.columns:
                    ldr_records = ldr_df[ldr_df['_employee_id_normalized'] == dispatcher_id_clean]
                else:
                    # Fallback to original logic
                    employee_id_col = next((col for col in ldr_df.columns if col.lower() == 'employee_id'), None)
                    if employee_id_col:
                        employee_series = ldr_df[employee_id_col].apply(normalize_dispatcher_id)
                        ldr_records = ldr_df[employee_series == dispatcher_id_clean]
                    else:
                        ldr_records = pd.DataFrame()

                if not ldr_records.empty:
                    if 'penalty_numeric' in ldr_records.columns:
                        rounded_penalties = [
                            penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                            for penalty in ldr_records['penalty_numeric'].tolist()
                        ]
                        breakdown['ldr'] = float(
                            sum(rounded_penalties).quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                        )
                    else:
                        # Fallback for legacy datasets without penalty column
                        waybill_col = next((col for col in ldr_df.columns if col.lower() in ['ticket_no', 'no_awb', 'waybill_number']), None)
                        if waybill_col:
                            waybill_count = ldr_records[waybill_col].nunique()
                        else:
                            waybill_count = len(ldr_records)
                        breakdown['ldr'] = waybill_count * 100.0

        # 3. Fake Attempt Penalty: dispatcher_id column = dispatcher_id, penalty = waybill count * RM 1.00
        if 'fake_attempt' in penalty_data:
            fake_attempt_df = penalty_data['fake_attempt']
            if not fake_attempt_df.empty:
                # Use pre-processed data if available
                if '_dispatcher_id_normalized' in fake_attempt_df.columns:
                    fake_attempt_records = fake_attempt_df[fake_attempt_df['_dispatcher_id_normalized'] == dispatcher_id_clean]
                else:
                    # Fallback to original logic
                    dispatcher_id_col = next((col for col in fake_attempt_df.columns if col.lower() == 'dispatcher_id'), None)
                    if dispatcher_id_col:
                        dispatcher_series = fake_attempt_df[dispatcher_id_col].apply(normalize_dispatcher_id)
                        fake_attempt_records = fake_attempt_df[dispatcher_series == dispatcher_id_clean]
                    else:
                        fake_attempt_records = pd.DataFrame()

                if not fake_attempt_records.empty:
                    waybill_col = next((col for col in fake_attempt_df.columns if col.lower() in ['waybill_number', 'waybill']), None)
                    if waybill_col:
                        waybill_count = fake_attempt_records[waybill_col].nunique()
                    else:
                        waybill_count = len(fake_attempt_records)

                    fake_attempt_rate = Config.load().get("fake_attempt_penalty_per_parcel", 1.00)
                    breakdown['fake_attempt'] = waybill_count * float(fake_attempt_rate)

        # 4. COD Penalty: dispatcher_id column = dispatcher_id, penalty amount from penalty column (only positive amounts)
        if 'cod' in penalty_data:
            cod_df = penalty_data['cod']
            if not cod_df.empty:
                # Use pre-processed data if available
                if '_dispatcher_id_normalized' in cod_df.columns:
                    cod_records = cod_df[cod_df['_dispatcher_id_normalized'] == dispatcher_id_clean]
                else:
                    # Fallback to original logic if not pre-processed
                    dispatcher_id_col = next((col for col in cod_df.columns if col.lower() == 'dispatcher_id'), None)
                    if dispatcher_id_col:
                        dispatcher_series = cod_df[dispatcher_id_col].apply(normalize_dispatcher_id)
                        cod_records = cod_df[dispatcher_series == dispatcher_id_clean]
                    else:
                        cod_records = pd.DataFrame()

                if not cod_records.empty and 'penalty_numeric' in cod_records.columns:
                    # Round each penalty value first, then sum (matching SQL: SUM((FLOOR((penalty * 100) + 0.5) / 100)))
                    # This ensures individual dispatcher totals match SQL calculation
                    rounded_penalties = [
                        penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                        for penalty in cod_records['penalty_numeric'].tolist()
                    ]
                    breakdown['cod'] = float(sum(rounded_penalties))

        return breakdown

    @staticmethod
    def calculate_penalty_by_type(penalty_data: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, float]:
        """
        Calculate total penalty amounts by type for all dispatchers.

        Args:
            penalty_data: Dictionary containing penalty dataframes from all penalty tables

        Returns:
            Dictionary with keys: 'duitnow', 'ldr', 'fake_attempt', 'cod', 'attendance' and their total amounts
        """
        penalty_totals = {
            'duitnow': 0.0,
            'ldr': 0.0,
            'fake_attempt': 0.0,
            'cod': 0.0,
            'attendance': 0.0
        }

        if penalty_data is None or not penalty_data:
            return penalty_totals

        # 1. DuitNow Penalty: sum all penalty amounts (only positive amounts)
        if 'duitnow' in penalty_data:
            duitnow_df = penalty_data['duitnow']

            # Check if dataframe is empty (might have been filtered out by date range)
            if duitnow_df is None or duitnow_df.empty:
                st.warning("⚠️ DuitNow penalty dataframe is empty (may have been filtered out by date range)")
                penalty_totals['duitnow'] = 0.0
            else:
                penalty_col = None
                for col in duitnow_df.columns:
                    if col.lower() == 'penalty':
                        penalty_col = col
                        break

                if penalty_col:
                    # Filter to only include records with positive penalty amounts
                    # Use Decimal to preserve exact precision
                    duitnow_df['penalty_numeric'] = duitnow_df[penalty_col].apply(
                        lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0')
                    )
                    duitnow_filtered = duitnow_df[duitnow_df['penalty_numeric'] > 0]

                    # Round each penalty value first, then sum (matching SQL: SUM((FLOOR((penalty * 100) + 0.5) / 100)))
                    # This ensures the total matches SQL exactly
                    if len(duitnow_filtered) > 0:
                        rounded_penalties = [
                            penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                            for penalty in duitnow_filtered['penalty_numeric'].tolist()
                        ]
                        penalty_totals['duitnow'] = float(sum(rounded_penalties))
                    else:
                        # Show diagnostic info if no positive penalties found
                        total_records = len(duitnow_df)
                        sample_penalties = duitnow_df[penalty_col].head(5).tolist()
                        st.warning(f"⚠️ No positive penalty amounts found in DuitNow data. Total records: {total_records}, Sample values: {sample_penalties}")
                        penalty_totals['duitnow'] = 0.0
                else:
                    st.error(f"❌ 'penalty' column not found in DuitNow data. Available columns: {list(duitnow_df.columns)}")
                    penalty_totals['duitnow'] = 0.0

        # 2. LDR Penalty: sum penalty column (fallback to legacy count-based)
        if 'ldr' in penalty_data:
            ldr_df = penalty_data['ldr']
            ldr_penalty_col = next((col for col in ldr_df.columns if col.lower() in ['penalty', 'amount']), None)
            if ldr_penalty_col:
                ldr_df = ldr_df.copy()
                ldr_df['penalty_numeric'] = ldr_df[ldr_penalty_col].apply(
                    lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0')
                )
                ldr_filtered = ldr_df[ldr_df['penalty_numeric'] > 0]
                if len(ldr_filtered) > 0:
                    rounded_penalties = [
                        penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                        for penalty in ldr_filtered['penalty_numeric'].tolist()
                    ]
                    penalty_totals['ldr'] = float(
                        sum(rounded_penalties).quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                    )
                else:
                    penalty_totals['ldr'] = 0.0
            else:
                # Fallback for legacy datasets without penalty column
                waybill_col = None
                for col in ldr_df.columns:
                    if col.lower() in ['ticket_no', 'no_awb', 'waybill_number']:
                        waybill_col = col
                        break
                if waybill_col:
                    waybill_count = ldr_df[waybill_col].nunique()
                else:
                    waybill_count = len(ldr_df)
                penalty_totals['ldr'] = waybill_count * 100.0

        # 3. Fake Attempt Penalty: count waybills * RM 1.00
        if 'fake_attempt' in penalty_data:
            fake_attempt_df = penalty_data['fake_attempt']
            waybill_col = None
            for col in fake_attempt_df.columns:
                if col.lower() in ['waybill_number', 'waybill']:
                    waybill_col = col
                    break

            if waybill_col:
                waybill_count = fake_attempt_df[waybill_col].nunique()
            else:
                waybill_count = len(fake_attempt_df)

            fake_attempt_rate = Config.load().get("fake_attempt_penalty_per_parcel", 1.00)
            penalty_totals['fake_attempt'] = waybill_count * float(fake_attempt_rate)

        # 4. COD Penalty: sum all penalty amounts (only positive amounts)
        if 'cod' in penalty_data:
            cod_df = penalty_data['cod']

            # Check if dataframe is empty (might have been filtered out by date range)
            if cod_df is None or cod_df.empty:
                st.warning("⚠️ COD penalty dataframe is empty (may have been filtered out by date range)")
                penalty_totals['cod'] = 0.0
            else:
                penalty_col = None
                for col in cod_df.columns:
                    if col.lower() == 'penalty':
                        penalty_col = col
                        break

                if penalty_col:
                    # Filter to only include records with positive penalty amounts
                    # Use Decimal to preserve exact precision
                    cod_df['penalty_numeric'] = cod_df[penalty_col].apply(
                        lambda x: Decimal(str(x)) if pd.notna(x) else Decimal('0')
                    )
                    cod_filtered = cod_df[cod_df['penalty_numeric'] > 0]

                    # Round each penalty value first, then sum (matching SQL: SUM((FLOOR((penalty * 100) + 0.5) / 100)))
                    # This ensures the total matches SQL exactly
                    if len(cod_filtered) > 0:
                        rounded_penalties = [
                            penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                            for penalty in cod_filtered['penalty_numeric'].tolist()
                        ]
                        penalty_totals['cod'] = float(sum(rounded_penalties))
                    else:
                        # Show diagnostic info if no positive penalties found
                        total_records = len(cod_df)
                        sample_penalties = cod_df[penalty_col].head(5).tolist()
                        st.warning(f"⚠️ No positive penalty amounts found in COD data. Total records: {total_records}, Sample values: {sample_penalties}")
                        penalty_totals['cod'] = 0.0
                else:
                    st.error(f"❌ 'penalty' column not found in COD data. Available columns: {list(cod_df.columns)}")
                    penalty_totals['cod'] = 0.0

        return penalty_totals

    @staticmethod
    def calculate_pickup_payout(pickup_df: pd.DataFrame, dispatcher_summary_df: pd.DataFrame, pickup_payout_per_parcel: float = 1.50) -> pd.DataFrame:
        """
        Calculate pickup payout based on pickup data.

        Args:
            pickup_df: DataFrame containing pickup data
            dispatcher_summary_df: DataFrame containing dispatcher summary (must have 'Dispatcher ID' column)
            pickup_payout_per_parcel: Payout per pickup parcel

        Returns:
            DataFrame with pickup payout added to dispatcher summary
        """
        if pickup_df is None or pickup_df.empty:
            dispatcher_summary_df['pickup_parcels'] = 0
            dispatcher_summary_df['pickup_payout'] = 0.0
            return dispatcher_summary_df

        # Find pickup dispatcher ID column
        pickup_dispatcher_col = None
        for col in pickup_df.columns:
            if 'pickup_dispatcher_id' in col.lower() or 'Pickup Dispatcher ID' in col:
                pickup_dispatcher_col = col
                break

        if not pickup_dispatcher_col:
            st.warning("⚠️ No pickup dispatcher ID column found in pickup data")
            dispatcher_summary_df['pickup_parcels'] = 0
            dispatcher_summary_df['pickup_payout'] = 0.0
            return dispatcher_summary_df

        # Find waybill column in pickup data
        waybill_col = None
        for col in pickup_df.columns:
            if 'waybill' in col.lower() or 'Waybill Number' in col:
                waybill_col = col
                break

        if not waybill_col:
            st.warning("⚠️ No waybill column found in pickup data")
            dispatcher_summary_df['pickup_parcels'] = 0
            dispatcher_summary_df['pickup_payout'] = 0.0
            return dispatcher_summary_df

        # Clean pickup dispatcher IDs
        pickup_df['clean_pickup_dispatcher_id'] = pickup_df[pickup_dispatcher_col].astype(str).str.strip()

        # Group by pickup dispatcher ID to count unique waybills
        pickup_summary = pickup_df.groupby('clean_pickup_dispatcher_id').agg(
            pickup_parcels=(waybill_col, 'nunique')
        ).reset_index()

        # Calculate pickup payout
        pickup_summary['pickup_payout'] = pickup_summary['pickup_parcels'] * pickup_payout_per_parcel

        # Rename column for merging
        pickup_summary = pickup_summary.rename(columns={'clean_pickup_dispatcher_id': 'dispatcher_id'})

        # Merge with dispatcher summary
        dispatcher_summary_df = dispatcher_summary_df.merge(
            pickup_summary[['dispatcher_id', 'pickup_parcels', 'pickup_payout']],
            on='dispatcher_id',
            how='left'
        )

        # Fill NaN values
        dispatcher_summary_df['pickup_parcels'] = dispatcher_summary_df['pickup_parcels'].fillna(0)
        dispatcher_summary_df['pickup_payout'] = dispatcher_summary_df['pickup_payout'].fillna(0.0)

        return dispatcher_summary_df

    @staticmethod
    def calculate_qr_order_payout(qr_order_df: pd.DataFrame, dispatcher_summary_df: pd.DataFrame, qr_order_payout_per_order: float = 1.80) -> pd.DataFrame:
        """
        Calculate QR order payout based on QR order data.

        Args:
            qr_order_df: DataFrame containing QR order data
            dispatcher_summary_df: DataFrame containing dispatcher summary (must have 'Dispatcher ID' column)
            qr_order_payout_per_order: Payout per QR order (RM 1.80 per order_no)

        Returns:
            DataFrame with QR order payout added to dispatcher summary
        """
        if qr_order_df is None or qr_order_df.empty:
            dispatcher_summary_df['qr_order_count'] = 0
            dispatcher_summary_df['qr_order_payout'] = 0.0
            return dispatcher_summary_df

        # Find dispatcher column in QR order data
        dispatcher_col = None
        for col in qr_order_df.columns:
            col_lower = str(col).lower()
            if col_lower in ('pickup_dispatcher_id', 'dispatcher_id', 'dispatcher'):
                dispatcher_col = col
                break

        if not dispatcher_col:
            st.warning("⚠️ No dispatcher column found in QR order data")
            dispatcher_summary_df['qr_order_count'] = 0
            dispatcher_summary_df['qr_order_payout'] = 0.0
            return dispatcher_summary_df

        # Find waybill_number column (unique identifier for each order)
        order_no_col = None
        for col in qr_order_df.columns:
            col_lower = str(col).lower()
            if col_lower in ('waybill_number', 'no_awb'):
                order_no_col = col
                break

        if not order_no_col:
            st.warning("⚠️ No waybill_number column found in QR order data")
            dispatcher_summary_df['qr_order_count'] = 0
            dispatcher_summary_df['qr_order_payout'] = 0.0
            return dispatcher_summary_df

        # Clean dispatcher IDs
        qr_order_df['clean_dispatcher_id'] = qr_order_df[dispatcher_col].astype(str).str.strip()

        # Group by dispatcher ID to count unique orders
        qr_order_summary = qr_order_df.groupby('clean_dispatcher_id').agg(
            qr_order_count=(order_no_col, 'nunique')
        ).reset_index()

        # Calculate QR order payout
        qr_order_summary['qr_order_payout'] = qr_order_summary['qr_order_count'] * qr_order_payout_per_order

        # Rename column for merging
        qr_order_summary = qr_order_summary.rename(columns={'clean_dispatcher_id': 'dispatcher_id'})

        # Merge with dispatcher summary
        dispatcher_summary_df = dispatcher_summary_df.merge(
            qr_order_summary[['dispatcher_id', 'qr_order_count', 'qr_order_payout']],
            on='dispatcher_id',
            how='left'
        )

        # Fill NaN values
        dispatcher_summary_df['qr_order_count'] = dispatcher_summary_df['qr_order_count'].fillna(0)
        dispatcher_summary_df['qr_order_payout'] = dispatcher_summary_df['qr_order_payout'].fillna(0.0)

        return dispatcher_summary_df

    @staticmethod
    def calculate_return_payout(return_df: pd.DataFrame, dispatcher_summary_df: pd.DataFrame, return_payout_per_parcel: float = 1.50) -> pd.DataFrame:
        """
        Calculate return payout based on return data.

        Args:
            return_df: DataFrame containing return data
            dispatcher_summary_df: DataFrame containing dispatcher summary (must have 'Dispatcher ID' column)
            return_payout_per_parcel: Payout per return parcel

        Returns:
            DataFrame with return payout added to dispatcher summary
        """
        if return_df is None or return_df.empty:
            dispatcher_summary_df['return_parcels'] = 0
            dispatcher_summary_df['return_payout'] = 0.0
            return dispatcher_summary_df

        # Find dispatcher ID column in return data
        return_dispatcher_col = None
        for col in return_df.columns:
            col_lower = str(col).lower()
            if 'dispatcher_id' in col_lower or 'Dispatcher ID' in col:
                return_dispatcher_col = col
                break

        if not return_dispatcher_col:
            st.warning("⚠️ No dispatcher ID column found in return data")
            dispatcher_summary_df['return_parcels'] = 0
            dispatcher_summary_df['return_payout'] = 0.0
            return dispatcher_summary_df

        # Find waybill column in return data
        waybill_col = None
        for col in return_df.columns:
            col_lower = str(col).lower()
            if 'waybill' in col_lower or 'Waybill Number' in col:
                waybill_col = col
                break

        if not waybill_col:
            st.warning("⚠️ No waybill column found in return data")
            dispatcher_summary_df['return_parcels'] = 0
            dispatcher_summary_df['return_payout'] = 0.0
            return dispatcher_summary_df

        # Clean dispatcher IDs
        return_df['clean_dispatcher_id'] = return_df[return_dispatcher_col].astype(str).str.strip()

        # Group by dispatcher ID to count unique waybills
        return_summary = return_df.groupby('clean_dispatcher_id').agg(
            return_parcels=(waybill_col, 'nunique')
        ).reset_index()

        # Calculate return payout
        return_summary['return_payout'] = return_summary['return_parcels'] * return_payout_per_parcel

        # Rename column for merging
        return_summary = return_summary.rename(columns={'clean_dispatcher_id': 'dispatcher_id'})

        # Merge with dispatcher summary
        dispatcher_summary_df = dispatcher_summary_df.merge(
            return_summary[['dispatcher_id', 'return_parcels', 'return_payout']],
            on='dispatcher_id',
            how='left'
        )

        # Fill NaN values
        dispatcher_summary_df['return_parcels'] = dispatcher_summary_df['return_parcels'].fillna(0)
        dispatcher_summary_df['return_payout'] = dispatcher_summary_df['return_payout'].fillna(0.0)

        return dispatcher_summary_df

    @staticmethod
    def calculate_payout(df: pd.DataFrame, currency_symbol: str, penalty_data: Optional[Dict[str, pd.DataFrame]] = None,
                        pickup_df: Optional[pd.DataFrame] = None,
                        pickup_payout_per_parcel: float = 1.50,
                        qr_order_df: Optional[pd.DataFrame] = None,
                        qr_order_payout_per_order: float = 1.80,
                        return_df: Optional[pd.DataFrame] = None,
                        return_payout_per_parcel: float = 1.50,
                        reward_df: Optional[pd.DataFrame] = None,
                        attendance_df: Optional[pd.DataFrame] = None,
                        processing_warnings: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Calculate payout using tier-based weight calculation."""
        # Prepare data
        df_clean = DataProcessor.prepare_dataframe(df)
        df_clean = DataProcessor.remove_duplicates(df_clean)

        if 'weight' not in df_clean.columns:
            st.error("❌ Missing weight column in data")
            return pd.DataFrame(), pd.DataFrame(), 0.0

        # Calculate payout per parcel
        tiers = Config.load().get("weight_tiers")
        df_clean['payout_rate'] = df_clean['weight'].apply(
            lambda w: PayoutCalculator.get_rate_by_weight(w, tiers)
        )
        df_clean['payout'] = df_clean['payout_rate']

        # Keep all waybills (no deduplication)
        waybill_col = find_column(df_clean, 'waybill') or 'waybill'
        delivery_sig_col = None
        for col in df.columns:
            col_norm = str(col).strip().lower()
            if col_norm in [
                "delivery signature", "delivery_signature", "delivery_sig",
                "delivery signature date", "delivery_signature_date"
            ]:
                delivery_sig_col = col
                break
        date_col = delivery_sig_col or ('date' if 'date' in df_clean.columns else find_column(df_clean, 'date'))
        df_unique = df_clean.copy()

        # Attendance penalty map per dispatcher (sum of attendance penalty column)
        attendance_penalty_map = {}
        if attendance_df is not None and not attendance_df.empty:
            emp_id_col = next((col for col in attendance_df.columns if str(col).strip().lower() in ['employee id', 'employee_id', 'employee id.']), None)
            penalty_col = next((col for col in attendance_df.columns if str(col).strip().lower() in ['penalty', 'attendance_penalty', 'attendance penalty']), None)
            if emp_id_col and penalty_col:
                attendance_copy = attendance_df.copy()
                attendance_copy['_emp_id'] = attendance_copy[emp_id_col].apply(normalize_dispatcher_id)
                attendance_copy['_penalty_amount'] = pd.to_numeric(attendance_copy[penalty_col], errors='coerce').fillna(0.0)
                attendance_copy = attendance_copy[attendance_copy['_penalty_amount'] > 0]
                attendance_penalty_map = (
                    attendance_copy.groupby('_emp_id')['_penalty_amount']
                    .sum()
                    .round(2)
                    .to_dict()
                )

        reward_map = {}
        if reward_df is not None and not reward_df.empty:
            reward_emp_id_col = next((col for col in reward_df.columns if str(col).strip().lower() in ['employee id', 'employee_id', 'employee id.']), None)
            reward_amount_col = next((col for col in reward_df.columns if str(col).strip().lower() in ['amount', 'reward', 'reward_amount']), None)
            if reward_emp_id_col and reward_amount_col:
                reward_copy = reward_df.copy()
                reward_copy['_emp_id'] = reward_copy[reward_emp_id_col].apply(normalize_dispatcher_id)
                reward_copy['_reward_amount'] = pd.to_numeric(reward_copy[reward_amount_col], errors='coerce').fillna(0.0)
                reward_map = (
                    reward_copy.groupby('_emp_id')['_reward_amount']
                    .sum()
                    .round(2)
                    .to_dict()
                )

        # Diagnostics
        raw_weight = df_clean['weight'].sum()

        # Add tier columns
        tiers = Config.load().get("weight_tiers", Config.DEFAULT_CONFIG["weight_tiers"])
        rate_tier1 = next(t['rate'] for t in tiers if t['min'] == 0)
        rate_tier2 = next(t['rate'] for t in tiers if t['min'] == 5)
        rate_tier3 = next(t['rate'] for t in tiers if t['min'] == 10)
        rate_tier4 = next(t['rate'] for t in tiers if t['min'] == 30)

        df_unique['tier1_count'] = (df_unique['payout_rate'] == rate_tier1).astype(int)
        df_unique['tier2_count'] = (df_unique['payout_rate'] == rate_tier2).astype(int)
        df_unique['tier3_count'] = (df_unique['payout_rate'] == rate_tier3).astype(int)
        df_unique['tier4_count'] = (df_unique['payout_rate'] == rate_tier4).astype(int)

        # Group by dispatcher
        grouped = df_unique.groupby('dispatcher_id').agg(
            dispatcher_name=('dispatcher_name', 'first'),
            parcel_count=(waybill_col, 'size'),
            total_weight=('weight', 'sum'),
            avg_weight=('weight', 'mean'),
            total_payout=('payout', 'sum'),
            tier1_parcels=('tier1_count', 'sum'),
            tier2_parcels=('tier2_count', 'sum'),
            tier3_parcels=('tier3_count', 'sum'),
            tier4_parcels=('tier4_count', 'sum')
        ).reset_index()

        # Recompute total payout from tier counts
        tier_rates = [rate_tier1, rate_tier2, rate_tier3, rate_tier4]
        grouped['dispatch_payout'] = (
            grouped['tier1_parcels'] * tier_rates[0]
            + grouped['tier2_parcels'] * tier_rates[1]
            + grouped['tier3_parcels'] * tier_rates[2]
            + grouped['tier4_parcels'] * tier_rates[3]
        )

        # Calculate avg_rate with division by zero protection
        grouped['avg_rate'] = grouped.apply(
            lambda row: row['dispatch_payout'] / row['parcel_count'] if row['parcel_count'] > 0 else 0.0,
            axis=1
        )

        # Calculate penalties
        grouped['penalty_amount'] = 0.0
        grouped['penalty_count'] = 0
        grouped['penalty_waybills'] = ''
        grouped['duitnow_penalty'] = 0.0
        grouped['ldr_penalty'] = 0.0
        grouped['fake_attempt_penalty'] = 0.0
        grouped['cod_penalty'] = 0.0
        grouped['attendance_penalty'] = 0.0

        # Pre-process penalty dataframes once for better performance
        penalty_data_processed = PayoutCalculator._preprocess_penalty_data(penalty_data) if penalty_data else None

        # Use apply instead of iterrows for better performance
        def calculate_penalties(row):
            dispatcher_id = str(row['dispatcher_id'])
            dispatcher_key = normalize_dispatcher_id(dispatcher_id)
            attendance_penalty = float(attendance_penalty_map.get(dispatcher_key, 0.0))

            penalty_amount, penalty_count, penalty_waybills, attendance_penalty = PayoutCalculator.calculate_penalty(
                dispatcher_id, penalty_data_processed, attendance_penalty
            )
            penalty_breakdown = PayoutCalculator.calculate_penalty_breakdown(
                dispatcher_id, penalty_data_processed, attendance_penalty
            )
            return pd.Series({
                'penalty_amount': penalty_amount,
                'penalty_count': penalty_count,
                'penalty_waybills': ', '.join(penalty_waybills) if penalty_waybills else '',
                'duitnow_penalty': penalty_breakdown['duitnow'],
                'ldr_penalty': penalty_breakdown['ldr'],
                'fake_attempt_penalty': penalty_breakdown['fake_attempt'],
                'cod_penalty': penalty_breakdown['cod'],
                'attendance_penalty': attendance_penalty
            })

        penalty_results = grouped.apply(calculate_penalties, axis=1)
        grouped[['penalty_amount', 'penalty_count', 'penalty_waybills',
                 'duitnow_penalty', 'ldr_penalty', 'fake_attempt_penalty', 'cod_penalty',
                 'attendance_penalty']] = penalty_results

        grouped['reward'] = grouped['dispatcher_id'].apply(
            lambda x: float(reward_map.get(normalize_dispatcher_id(str(x)), 0.0))
        )

        # Calculate pickup payout
        grouped = PayoutCalculator.calculate_pickup_payout(pickup_df, grouped, pickup_payout_per_parcel)

        # Calculate QR order payout
        grouped = PayoutCalculator.calculate_qr_order_payout(qr_order_df, grouped, qr_order_payout_per_order)

        # Calculate return payout
        grouped = PayoutCalculator.calculate_return_payout(return_df, grouped, return_payout_per_parcel)

        # Calculate total payout: dispatch payout - penalty + pickup payout + QR order payout + return payout + reward
        grouped['total_payout'] = (
            grouped['dispatch_payout']
            - grouped['penalty_amount']
            + grouped['pickup_payout']
            + grouped['qr_order_payout']
            + grouped['return_payout']
            + grouped['reward']
        )

        # Create display and numeric dataframes
        numeric_df = grouped.rename(columns={
            "dispatcher_id": "Dispatcher ID",
            "dispatcher_name": "Dispatcher Name",
            "parcel_count": "Parcels Delivered",
            "total_weight": "Total Weight (kg)",
            "avg_weight": "Avg Weight (kg)",
            "avg_rate": "Avg Rate per Parcel",
            "dispatch_payout": "Delivery Parcels Payout",
            "total_payout": "Total Payout",
            "penalty_amount": "Penalty",
            "penalty_count": "Penalty Parcels",
            "penalty_waybills": "Penalty Waybills",
            "duitnow_penalty": "DuitNow Penalty",
            "ldr_penalty": "LDR Penalty",
            "fake_attempt_penalty": "Fake Attempt Penalty",
            "cod_penalty": "COD Penalty",
            "attendance_penalty": "Attendance Penalty",
            "pickup_parcels": "Pickup Parcels",
            "pickup_payout": "Pickup Parcels Payout",
            "qr_order_count": "QR Orders",
            "qr_order_payout": "QR Orders Payout",
            "return_parcels": "Return Parcels",
            "return_payout": "Return Parcels Payout",
            "reward": "Reward",
            "tier1_parcels": "Parcels 0-5kg",
            "tier2_parcels": "Parcels 5.01-10kg",
            "tier3_parcels": "Parcels 10.01-30kg",
            "tier4_parcels": "Parcels 30+kg"
        }).sort_values(by="Dispatcher Name", ascending=True)

        display_df = numeric_df.copy()
        display_df["Total Weight (kg)"] = display_df["Total Weight (kg)"].apply(lambda x: f"{x:.2f}")
        display_df["Avg Weight (kg)"] = display_df["Avg Weight (kg)"].apply(lambda x: f"{x:.2f}")
        display_df["Avg Rate per Parcel"] = display_df["Avg Rate per Parcel"].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_df["Delivery Parcels Payout"] = display_df["Delivery Parcels Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        display_df["Total Payout"] = display_df["Total Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        display_df["Penalty"] = display_df["Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        if "DuitNow Penalty" in display_df.columns:
            display_df["DuitNow Penalty"] = display_df["DuitNow Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        if "LDR Penalty" in display_df.columns:
            display_df["LDR Penalty"] = display_df["LDR Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        if "Fake Attempt Penalty" in display_df.columns:
            display_df["Fake Attempt Penalty"] = display_df["Fake Attempt Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        if "COD Penalty" in display_df.columns:
            display_df["COD Penalty"] = display_df["COD Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        if "Attendance Penalty" in display_df.columns:
            display_df["Attendance Penalty"] = display_df["Attendance Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        display_df["Pickup Parcels Payout"] = display_df["Pickup Parcels Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        if "QR Orders Payout" in display_df.columns:
            display_df["QR Orders Payout"] = display_df["QR Orders Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        if "Return Parcels Payout" in display_df.columns:
            display_df["Return Parcels Payout"] = display_df["Return Parcels Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        if "Reward" in display_df.columns:
            display_df["Reward"] = display_df["Reward"].apply(lambda x: f"{currency_symbol}{x:,.2f}")

        # Keep Penalty Waybills and Penalty Parcels in numeric_df but remove from display_df
        if "Penalty Waybills" in display_df.columns:
            display_df = display_df.drop(columns=["Penalty Waybills"])
        if "Penalty Parcels" in display_df.columns:
            display_df = display_df.drop(columns=["Penalty Parcels"])

        total_payout = numeric_df["Total Payout"].sum()

        with st.expander("📊 Processing Details", expanded=False):
            # Display any processing warnings
            if processing_warnings:
                for warning in processing_warnings:
                    st.warning(warning)

            st.info(f"Weight totals – Raw: {raw_weight:,.2f} kg")
            st.success(f"✅ Processed {len(df_unique)} parcels from {len(grouped)} dispatchers")
            if attendance_df is not None:
                total_att_rows = len(attendance_df)
                dispatch_ids = {normalize_dispatcher_id(x) for x in grouped['dispatcher_id'].tolist()}
                matched_dispatchers = len(set(attendance_penalty_map.keys()) & dispatch_ids)
                st.info(f"Attendance rows loaded: {total_att_rows:,} | Dispatchers with attendance penalty: {matched_dispatchers:,}")
                st.info(
                    f"Attendance penalty IDs: {len(attendance_penalty_map):,} | "
                    f"Dispatch IDs: {len(dispatch_ids):,} | "
                    f"Sample attendance IDs: {list(attendance_penalty_map.keys())[:10]} | "
                    f"Sample dispatch IDs: {list(dispatch_ids)[:10]}"
                )
                if date_col:
                    if delivery_sig_col and delivery_sig_col in df.columns:
                        raw_col = delivery_sig_col
                        raw_valid = df[raw_col].notna().sum()
                    elif date_col in df_clean.columns:
                        raw_col = date_col
                        raw_valid = df_clean[raw_col].notna().sum()
                    else:
                        raw_col = date_col
                        raw_valid = 0

                    sample_ids = df_clean['dispatcher_id'].dropna().astype(str).head(10).tolist() if 'dispatcher_id' in df_clean.columns else []
                    st.info(f"Dispatch date column used: {raw_col} | Raw non-null: {raw_valid:,}")
                    st.info(f"Sample dispatch IDs (raw): {sample_ids}")

        # Calculate breakdown for info message
        total_dispatch_payout = numeric_df["Delivery Parcels Payout"].sum()
        total_pickup_payout = numeric_df["Pickup Parcels Payout"].sum()
        total_qr_order_payout = numeric_df["QR Orders Payout"].sum() if "QR Orders Payout" in numeric_df.columns else 0.0
        total_return_payout = numeric_df["Return Parcels Payout"].sum() if "Return Parcels Payout" in numeric_df.columns else 0.0
        total_reward = numeric_df["Reward"].sum() if "Reward" in numeric_df.columns else 0.0
        total_penalty = numeric_df["Penalty"].sum()

        st.info(f"""
        💰 **Payout Breakdown:**
        - Delivery Parcels Payout: {currency_symbol} {total_dispatch_payout:,.2f}
        + Pickup Parcels Payout: {currency_symbol} {total_pickup_payout:,.2f}
        + QR Orders Payout: {currency_symbol} {total_qr_order_payout:,.2f}
        + Return Parcels Payout: {currency_symbol} {total_return_payout:,.2f}
        + Reward: {currency_symbol} {total_reward:,.2f}
        - Penalties: {currency_symbol} {total_penalty:,.2f}
        **Total Payout: {currency_symbol} {total_payout:,.2f}**
        """)

        return display_df, numeric_df, total_payout

    @staticmethod
    def get_daily_trend(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
        """Get daily parcel delivery trend."""
        # If date_col is provided, use it; otherwise try to find it
        if date_col is None:
            date_col = find_column(df, 'date')

        # If still not found, try to find any date-like column
        if date_col is None or date_col not in df.columns:
            # Try common date column names
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ["date", "signature", "delivery_signature", "delivery_signature_date"]):
                    # Check if it has date-like values
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        try:
                            pd.to_datetime(sample, errors='raise')
                            date_col = col
                            break
                        except:
                            continue

        if date_col and date_col in df.columns:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            # Filter out invalid dates
            df_copy = df_copy[df_copy[date_col].notna()]
            if not df_copy.empty:
                daily_df = df_copy.groupby(df_copy[date_col].dt.date).size().reset_index(name='total_parcels')
                daily_df = daily_df.rename(columns={date_col: 'signature_date'})
                return daily_df.sort_values('signature_date')
        return pd.DataFrame()

    @staticmethod
    def get_daily_payout_trend(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
        """Get daily payout trend."""
        # If date_col is provided, use it; otherwise try to find it
        if date_col is None:
            date_col = find_column(df, 'date')

        # If still not found, try to find any date-like column
        if date_col is None or date_col not in df.columns:
            # Try common date column names
            for col in df.columns:
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ["date", "signature", "delivery_signature", "delivery_signature_date"]):
                    # Check if it has date-like values
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        try:
                            pd.to_datetime(sample, errors='raise')
                            date_col = col
                            break
                        except:
                            continue

        if date_col and date_col in df.columns:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            # Filter out invalid dates
            df_copy = df_copy[df_copy[date_col].notna()]
            if not df_copy.empty and 'payout' in df_copy.columns:
                # Calculate total payout per day (sum of all payouts)
                daily_df = df_copy.groupby(df_copy[date_col].dt.date).agg(
                    total_payout=('payout', 'sum'),
                    parcel_count=(date_col, 'size')
                ).reset_index()
                daily_df = daily_df.rename(columns={date_col: 'signature_date'})
                return daily_df.sort_values('signature_date')
        return pd.DataFrame()

# =============================================================================
# FORECASTING
# =============================================================================

class ForecastGenerator:
    """Generate forecasts using Prophet."""

    @staticmethod
    def prepare_forecast_data(daily_df: pd.DataFrame, value_column: str = 'total_parcels') -> pd.DataFrame:
        """Prepare daily data for forecasting."""
        if daily_df.empty:
            return pd.DataFrame()

        # Ensure we have a proper datetime column
        forecast_df = daily_df.copy()
        forecast_df['ds'] = pd.to_datetime(forecast_df['signature_date'])
        forecast_df['y'] = forecast_df[value_column]

        return forecast_df[['ds', 'y']]

    @staticmethod
    def generate_forecast(daily_df: pd.DataFrame, periods: int = 30, value_column: str = 'total_parcels') -> Optional[tuple]:
        """Generate forecast using Prophet."""
        if not PROPHET_AVAILABLE or daily_df.empty or len(daily_df) < 7:
            return None

        try:
            # Prepare data
            df = ForecastGenerator.prepare_forecast_data(daily_df, value_column)

            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                changepoint_prior_scale=0.05
            )
            model.fit(df)

            # Make future dataframe
            future = model.make_future_dataframe(periods=periods)

            # Make predictions
            forecast = model.predict(future)

            # Get forecast components
            forecast_summary = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

            return model, forecast, forecast_summary

        except Exception as e:
            st.warning(f"Forecast generation failed: {e}")
            return None

    @staticmethod
    def calculate_forecast_metrics(historical: pd.DataFrame, forecast_summary: pd.DataFrame, value_column: str = 'total_parcels') -> Dict:
        """Calculate forecast accuracy metrics."""
        if historical.empty or forecast_summary.empty:
            return {}

        # Get last N days of historical data for comparison
        recent_history = historical.tail(min(7, len(historical)))
        avg_historical = recent_history[value_column].mean()

        # Forecast metrics
        avg_forecast = forecast_summary['yhat'].mean()
        forecast_std = forecast_summary['yhat'].std()
        max_forecast = forecast_summary['yhat'].max()
        min_forecast = forecast_summary['yhat'].min()

        # Calculate change percentage
        if avg_historical > 0:
            change_pct = ((avg_forecast - avg_historical) / avg_historical) * 100
        else:
            change_pct = 0

        return {
            'avg_historical': avg_historical,
            'avg_forecast': avg_forecast,
            'forecast_std': forecast_std,
            'max_forecast': max_forecast,
            'min_forecast': min_forecast,
            'change_pct': change_pct,
            'total_forecast': forecast_summary['yhat'].sum()
        }

    @staticmethod
    def create_forecast_chart(historical: pd.DataFrame, forecast_summary: pd.DataFrame, title: str = '30-Day Parcel Delivery Forecast',
                             value_column: str = 'total_parcels', y_title: str = 'Parcels') -> alt.Chart:
        """Create forecast visualization using Altair."""
        # Prepare historical data
        hist_df = historical.copy()
        hist_df['type'] = 'Historical'
        hist_df = hist_df.rename(columns={'signature_date': 'ds', value_column: 'y'})

        # Prepare forecast data
        forecast_plot = forecast_summary.copy()
        forecast_plot['type'] = 'Forecast'

        # Combine data
        combined = pd.concat([
            hist_df[['ds', 'y', 'type']],
            forecast_plot[['ds', 'yhat', 'type']].rename(columns={'yhat': 'y'})
        ])

        # Create chart
        base = alt.Chart(combined).encode(
            x=alt.X('ds:T', title='Date'),
            y=alt.Y('y:Q', title=y_title),
            color=alt.Color('type:N', scale=alt.Scale(
                domain=['Historical', 'Forecast'],
                range=[ColorScheme.PRIMARY, ColorScheme.ACCENT]
            ), legend=alt.Legend(title="Data Type"))
        )

        line = base.mark_line(point=True).encode(
            opacity=alt.value(0.8)
        )

        # Add confidence interval for forecast
        if 'yhat_lower' in forecast_summary.columns and 'yhat_upper' in forecast_summary.columns:
            forecast_area = alt.Chart(forecast_plot).mark_area(
                opacity=0.3,
                color=ColorScheme.ACCENT
            ).encode(
                x='ds:T',
                y='yhat_lower:Q',
                y2='yhat_upper:Q'
            )

            return (line + forecast_area).properties(
                title=title,
                height=400
            )

        return line.properties(
            title=title,
            height=400
        )

# =============================================================================
# VISUALIZATION
# =============================================================================

class DataVisualizer:
    """Create charts and visualizations."""

    @staticmethod
    def create_daily_trend_chart(daily_df: pd.DataFrame) -> alt.Chart:
        """Create daily trend area chart."""
        return alt.Chart(daily_df).mark_area(
            line={'color': ColorScheme.PRIMARY, 'width': 2},
            color=ColorScheme.PRIMARY_LIGHT,
            opacity=0.6
        ).encode(
            x=alt.X('signature_date:T', title='Date', axis=alt.Axis(format='%b %d')),
            y=alt.Y('total_parcels:Q', title='Parcels Delivered'),
            tooltip=['signature_date:T', 'total_parcels:Q']
        ).properties(title='Daily Parcel Delivery Trend', height=300)

    @staticmethod
    def create_performers_chart(numeric_df: pd.DataFrame) -> alt.Chart:
        """Create top performers bar chart."""
        # Sort by Total Payout descending and take top 10
        top_10 = numeric_df.sort_values('Total Payout', ascending=False).head(10)
        return alt.Chart(top_10).mark_bar(color=ColorScheme.PRIMARY).encode(
            y=alt.Y('Dispatcher Name:N', title='Dispatcher', sort='-x'),
            x=alt.X('Total Payout:Q', title='Total Payout'),
            color=alt.Color('Total Payout:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=[
                'Dispatcher Name:N',
                'Parcels Delivered:Q',
                alt.Tooltip('Total Weight (kg):Q', format=',.2f'),
                alt.Tooltip('Total Payout:Q', format=',.2f')
            ]
        ).properties(title='Payout Performance', height=300, width=400)

    @staticmethod
    def create_payout_distribution(numeric_df: pd.DataFrame) -> alt.Chart:
        """Create payout distribution donut chart."""
        # Sort by Total Payout descending for consistent ordering
        sorted_df = numeric_df.sort_values('Total Payout', ascending=False)
        return alt.Chart(sorted_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Total Payout", type="quantitative"),
            color=alt.Color(field="Dispatcher Name", type="nominal",
                          scale=alt.Scale(range=ColorScheme.CHART_COLORS),
                          legend=alt.Legend(title="Dispatchers", orient="right")),
            order=alt.Order(field="Total Payout", sort="descending"),
            tooltip=['Dispatcher Name:N', 'Total Payout:Q']
        ).properties(title='Payout Distribution', height=300, width=400)

    @staticmethod
    def create_all_charts(daily_df: pd.DataFrame, numeric_df: pd.DataFrame) -> Dict[str, alt.Chart]:
        """Create all charts for dashboard."""
        charts = {}

        if not daily_df.empty:
            charts['daily_trend'] = DataVisualizer.create_daily_trend_chart(daily_df)

        if not numeric_df.empty:
            charts['performers'] = DataVisualizer.create_performers_chart(numeric_df)
            charts['payout_dist'] = DataVisualizer.create_payout_distribution(numeric_df)

        return charts

# =============================================================================
# INVOICE GENERATION
# =============================================================================

class InvoiceGenerator:
    """Generate invoices."""

    @staticmethod
    def build_invoice_html(display_df: pd.DataFrame, numeric_df: pd.DataFrame,
                          total_payout: float, currency_symbol: str,
                          pickup_payout_per_parcel: float = 1.50) -> str:
        """Build management summary invoice HTML with original layout."""
        try:
            total_parcels = int(numeric_df["Parcels Delivered"].sum())
            total_dispatchers = len(numeric_df)
            total_weight = numeric_df["Total Weight (kg)"].sum()
            total_dispatch_payout = numeric_df["Delivery Parcels Payout"].sum() if "Delivery Parcels Payout" in numeric_df.columns else 0.0
            total_pickup_payout = numeric_df["Pickup Parcels Payout"].sum() if "Pickup Parcels Payout" in numeric_df.columns else 0.0
            total_pickup_parcels = int(numeric_df["Pickup Parcels"].sum()) if "Pickup Parcels" in numeric_df.columns else 0
            total_return_payout = numeric_df["Return Parcels Payout"].sum() if "Return Parcels Payout" in numeric_df.columns else 0.0
            total_return_parcels = int(numeric_df["Return Parcels"].sum()) if "Return Parcels" in numeric_df.columns else 0
            total_qr_orders = int(numeric_df["QR Orders"].sum()) if "QR Orders" in numeric_df.columns else 0
            total_qr_order_payout = numeric_df["QR Orders Payout"].sum() if "QR Orders Payout" in numeric_df.columns else 0.0
            total_reward = numeric_df["Reward"].sum() if "Reward" in numeric_df.columns else 0.0
            total_penalty = numeric_df["Penalty"].sum() if "Penalty" in numeric_df.columns else 0.0
            penalty_by_type = {
                'duitnow': numeric_df["DuitNow Penalty"].sum() if "DuitNow Penalty" in numeric_df.columns else 0.0,
                'ldr': numeric_df["LDR Penalty"].sum() if "LDR Penalty" in numeric_df.columns else 0.0,
                'fake_attempt': numeric_df["Fake Attempt Penalty"].sum() if "Fake Attempt Penalty" in numeric_df.columns else 0.0,
                'cod': numeric_df["COD Penalty"].sum() if "COD Penalty" in numeric_df.columns else 0.0,
                'attendance': numeric_df["Attendance Penalty"].sum() if "Attendance Penalty" in numeric_df.columns else 0.0
            }
            top_3 = display_df.head(3)

            table_columns = ["Dispatcher ID", "Dispatcher Name", "Parcels Delivered",
                           "Delivery Parcels Payout", "Pickup Parcels", "Pickup Parcels Payout",
                           "Return Parcels", "Return Parcels Payout",
                           "Reward", "Penalty", "Total Payout"]

            html = f"""
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
                }}
                html, body {{ margin: 0; padding: 0; background: var(--background); }}
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
                        color: var(--text-primary); line-height: 1.5; }}
                .container {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}
                .header {{
                  display: grid; grid-template-columns: 1fr auto; gap: 16px;
                  border: 1px solid var(--border); border-radius: 12px; padding: 24px; align-items: center;
                  background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
                  color: white;
                  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                }}
                .brand {{ font-weight: 700; font-size: 24px; letter-spacing: -0.5px; }}
                .subbrand {{ font-size: 16px; color: #fff; opacity:0.85; margin-top:4px; font-weight:500; }}
                .idline {{ opacity: 0.9; font-size: 14px; margin-top: 8px; }}
                .total-badge {{
                  background: rgba(255,255,255,0.2);
                  padding: 8px 16px;
                  border-radius: 20px;
                  text-align: center;
                  border: 1px solid rgba(255,255,255,0.3);
                }}
                .total-badge .label {{ font-size: 12px; opacity: 0.9; margin-bottom: 4px; }}
                .total-badge .value {{ font-size: 28px; font-weight: 800; }}
                .summary {{
                  margin-top: 24px;
                }}
                .summary-row {{
                  display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; margin-bottom: 12px;
                }}
                .chip {{
                  border: 1px solid var(--border); border-radius: 12px;
                  padding: 16px; background: var(--surface); min-width: 180px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                  transition: transform 0.2s, box-shadow 0.2s;
                  text-align: center;
                }}
                .chip:hover {{
                  transform: translateY(-2px);
                  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                  border-color: var(--primary-light);
                }}
                .chip .label {{ color: var(--text-secondary); font-size: 12px; text-transform: uppercase;
                               letter-spacing: 0.5px; font-weight: 600; }}
                .chip .value {{ font-size: 18px; font-weight: 700; margin-top: 6px; color: var(--primary); }}
                .chip .subtext {{ margin-top: 8px; font-size: 12px; color: var(--text-secondary); line-height: 1.4; }}
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
                .note {{
                  margin-top: 24px;
                  color: var(--text-secondary);
                  font-size: 12px;
                  text-align: center;
                  padding: 16px;
                }}
                @media (max-width: 768px) {{
                  .header {{ grid-template-columns: 1fr; text-align: center; }}
                  .summary {{ flex-direction: column; align-items: center; }}
                }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div>
                            <div class="brand">Invoice</div>
                            <div class="subbrand">From: Jemari Ventures</div>
                            <div class="subbrand">To: Niagamatic Sdn Bhd</div>
                            <div class="idline">Invoice No: JMR{datetime.now().strftime('%Y%m')}</div>
                            <div class="idline">Period: {(datetime.now().replace(day=1) - pd.Timedelta(days=1)).strftime('%B %Y')}</div>
                        </div>
                        <div class="total-badge">
                            <div class="label">Total Payout (Inclusive)</div>
                            <div class="value">{currency_symbol} {total_payout:,.2f}</div>
                        </div>
                    </div>

                    <div class="summary">
                        <div class="summary-row">
                        <div class="chip">
                            <div class="label">Total Dispatchers</div>
                            <div class="value">{total_dispatchers}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Total Parcels</div>
                            <div class="value">{total_parcels:,}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Pickup Parcels</div>
                            <div class="value">{total_pickup_parcels:,}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Return Parcels</div>
                            <div class="value">{total_return_parcels:,}</div>
                        </div>
                        <div class="chip">
                            <div class="label">QR Orders</div>
                            <div class="value">{total_qr_orders:,}</div>
                        </div>
                        </div>
                        <div class="summary-row">
                            <div class="chip">
                                <div class="label">Delivery Parcels Payout</div>
                                <div class="value">{currency_symbol} {total_dispatch_payout:,.2f}</div>
                                <div class="subtext">
                                    Pickup: {currency_symbol} {total_pickup_payout:,.2f} ·
                                    Return: {currency_symbol} {total_return_payout:,.2f} ·
                                    QR Orders: {currency_symbol} {total_qr_order_payout:,.2f} ·
                                    Reward: {currency_symbol} {total_reward:,.2f}
                                </div>
                            </div>
                            <div class="chip">
                                <div class="label">Total Penalty</div>
                                <div class="value">-{currency_symbol} {total_penalty:,.2f}</div>
                                <div class="subtext">
                                    DuitNow: -{currency_symbol} {penalty_by_type['duitnow']:,.2f} ·
                                    LDR: -{currency_symbol} {penalty_by_type['ldr']:,.2f} ·
                                    Fake: -{currency_symbol} {penalty_by_type['fake_attempt']:,.2f} ·
                                    COD: -{currency_symbol} {penalty_by_type['cod']:,.2f} ·
                                    Attendance: -{currency_symbol} {penalty_by_type['attendance']:,.2f}
                                </div>
                            </div>
                        </div>
                    </div>

                    <table>
                        <thead><tr>"""
            for col in table_columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"

            for _, row in display_df.iterrows():
                html += "<tr>"
                for col in table_columns:
                    html += f"<td>{row.get(col, '')}</td>"
                html += "</tr>"

            html += "</tbody></table>"

            html += f"""
                    <div style="margin-top:3rem">
                        <table style="width:100%; background:var(--surface); border-radius:8px; margin-top:2rem; border:1px solid var(--border)">
                            <tr><th style="background:var(--primary);color:white;text-align:left;">Summary</th><th style="background:var(--primary);color:white;text-align:right;">Amount</th></tr>
                            <tr><td>Total Delivery Parcels Payout</td><td style="text-align:right;">{currency_symbol} {total_dispatch_payout:,.2f}</td></tr>
                            <tr><td>Pickup Parcels Payout</td><td style="text-align:right;">{currency_symbol} {total_pickup_payout:,.2f}</td></tr>
                            <tr><td>Return Parcels Payout</td><td style="text-align:right;">{currency_symbol} {total_return_payout:,.2f}</td></tr>
                            <tr><td>Total Reward</td><td style="text-align:right;">{currency_symbol} {total_reward:,.2f}</td></tr>
                            <tr><td>Total Penalty</td><td style="text-align:right;">-{currency_symbol} {total_penalty:,.2f}</td></tr>
                            <tr><td><strong>Total Payout</strong></td><td style="text-align:right;"><strong>{currency_symbol} {total_payout:,.2f}</strong></td></tr>
                        </table>
                    </div>
                    <div class="note">
                        Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} • JMR Management Dashboard v2.2<br>
                        <em>Payout calculated using tier-based weight system + pickup parcel count</em>
                    </div>
                </div>
            </body>
            </html>
            """
            return html
        except Exception as e:
            return f"<html><body><h1>Error: {str(e)}</h1></body></html>"

# =============================================================================
# UI COMPONENTS
# =============================================================================

def apply_custom_styles():
    """Apply custom CSS styling."""
    st.markdown(f"""
    <style>
        .stApp {{ background-color: {ColorScheme.BACKGROUND}; }}
        .stButton>button {{ background-color: {ColorScheme.PRIMARY}; color: white;
                           border-radius: 8px; padding: 0.5rem 1rem; font-weight: 600; border: none; }}
        .stButton>button:hover {{ background-color: {ColorScheme.PRIMARY_LIGHT}; }}
        [data-testid="stMetric"] {{ background: white; border: 1px solid {ColorScheme.BORDER};
                                    padding: 1rem; border-radius: 8px; }}
        [data-testid="stMetricValue"] {{ color: {ColorScheme.PRIMARY}; }}
        .forecast-metric {{ background: linear-gradient(135deg, {ColorScheme.ACCENT}20, {ColorScheme.SECONDARY}20);
                           border: 1px solid {ColorScheme.ACCENT}50; padding: 1rem; border-radius: 8px; }}
    </style>
    """, unsafe_allow_html=True)


def add_footer():
    """Add footer to page."""
    st.markdown(f"""
    <div style="margin-top: 3rem; padding: 1.5rem; background: linear-gradient(135deg, {ColorScheme.PRIMARY}, {ColorScheme.PRIMARY_LIGHT});
                color: white; text-align: center; border-radius: 12px;">
        © 2026 Jemari Ventures. All rights reserved. | JMR Management Dashboard v2.2
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    st.set_page_config(page_title="JMR Management Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
    apply_custom_styles()

    # Header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {ColorScheme.PRIMARY}, {ColorScheme.PRIMARY_LIGHT});
                padding: 2rem; border-radius: 12px; color: white; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white; margin: 0;">📊 JMR Management Dashboard</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;">
            Overview of dispatcher performance and payouts including pickup calculations
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    config = Config.load()

    # Data source selection
    source_type = config.get("data_source", {}).get("type", "postgres")
    selected_source = st.sidebar.selectbox(
        "Data Source",
        options=["postgres", "excel"],
        index=0 if source_type == "postgres" else 1
    )

    if selected_source != source_type:
        config["data_source"]["type"] = selected_source
        Config.save(config)

    excel_source = None

    if selected_source == "excel":
        st.sidebar.success("✅ Using Google Sheet Data Source")
        gsheet_url = st.sidebar.text_input(
            "Google Sheet URL",
            value=config.get("data_source", {}).get("gsheet_url", ""),
            help="Use the Google Sheet URL for data loading"
        )

        if gsheet_url != config.get("data_source", {}).get("gsheet_url", ""):
            config["data_source"]["gsheet_url"] = gsheet_url
            Config.save(config)

        if not gsheet_url:
            st.warning("Please set a Google Sheet URL.")
            add_footer()
            return

        with st.spinner("📄 Loading data from Google Sheets..."):
            df = DataSource.load_data(config, excel_source=None)
    else:
        # Show current database configuration
        st.sidebar.success("✅ Using PostgreSQL Database")

        # Database connection status
        engine = DataSource.get_postgres_engine()
        if engine:
            st.sidebar.success("✅ Database connection established")
        else:
            st.sidebar.error("❌ Database connection failed")
            st.error("Please check your PostgreSQL connection configuration in secrets.toml")
            add_footer()
            return

        with st.spinner("📄 Loading data from PostgreSQL..."):
            df = DataSource.load_data(config)

    if df is None or df.empty:
        st.error("❌ No data loaded from PostgreSQL. Check your configuration.")
        add_footer()
        return

    st.sidebar.header("⚙️ Payout Settings")

    # Add configuration for pickup payout per parcel
    pickup_payout_per_parcel = st.sidebar.number_input(
        "Pickup Payout per Parcel",
        min_value=0.0,
        max_value=100.0,
        value=config.get("pickup_payout_per_parcel", 1.50),
        step=0.10,
        help="Payout amount per pickup parcel"
    )

    # Update config with new value
    if pickup_payout_per_parcel != config.get("pickup_payout_per_parcel", 1.50):
        config["pickup_payout_per_parcel"] = pickup_payout_per_parcel
        Config.save(config)

    # Add configuration for QR order payout per order
    qr_order_payout_per_order = st.sidebar.number_input(
        "QR Order Payout per Order",
        min_value=0.0,
        max_value=100.0,
        value=config.get("qr_order_payout_per_order", 1.80),
        step=0.10,
        help="Payout amount per QR order"
    )

    # Update config with new value
    if qr_order_payout_per_order != config.get("qr_order_payout_per_order", 1.80):
        config["qr_order_payout_per_order"] = qr_order_payout_per_order
        Config.save(config)

    # Add configuration for return payout per parcel
    return_payout_per_parcel = st.sidebar.number_input(
        "Return Payout per Parcel",
        min_value=0.0,
        max_value=100.0,
        value=config.get("return_payout_per_parcel", 1.50),
        step=0.10,
        help="Payout amount per return parcel"
    )

    # Update config with new value
    if return_payout_per_parcel != config.get("return_payout_per_parcel", 1.50):
        config["return_payout_per_parcel"] = return_payout_per_parcel
        Config.save(config)

    # Add configuration for fake attempt penalty per parcel
    fake_attempt_penalty_per_parcel = st.sidebar.number_input(
        "Fake Attempt Penalty per Parcel",
        min_value=0.0,
        max_value=100.0,
        value=config.get("fake_attempt_penalty_per_parcel", 1.00),
        step=0.10,
        help="Penalty amount per fake attempt waybill"
    )

    # Update config with new value
    if fake_attempt_penalty_per_parcel != config.get("fake_attempt_penalty_per_parcel", 1.00):
        config["fake_attempt_penalty_per_parcel"] = fake_attempt_penalty_per_parcel
        Config.save(config)

    # Add configuration for forecast days
    forecast_days = st.sidebar.number_input(
        "Forecast Period (days)",
        min_value=7,
        max_value=90,
        value=config.get("forecast_days", 30),
        step=7,
        help="Number of days to forecast"
    )

    # Update config with new value
    if forecast_days != config.get("forecast_days", 30):
        config["forecast_days"] = forecast_days
        Config.save(config)

    st.sidebar.info(f"""
    **💰 Weight-Based Payout:**
    - 0-5kg: RM1.50
    - 5-10kg: RM1.60
    - 10-30kg: RM2.70
    - 30kg+: RM4.00

    **📦 Pickup Parcels Payout:**
    - RM{pickup_payout_per_parcel:.2f} per parcel

    **📱 QR Orders Payout:**
    - RM{qr_order_payout_per_order:.2f} per order

    **↩️ Return Parcels Payout:**
    - RM{return_payout_per_parcel:.2f} per parcel

    **🚫 Fake Attempt Penalty:**
    - RM{fake_attempt_penalty_per_parcel:.2f} per parcel

    **📈 Forecast Period:**
    - {forecast_days} days
    """)

    # Date filter
    auto_date_col = find_column(df, 'date')
    date_candidates = {
        col for col in df.columns
        if any(keyword in str(col).lower() for keyword in ["date", "signature", "scan"])
    }
    if auto_date_col:
        date_candidates.add(auto_date_col)
    date_candidates = sorted(date_candidates)
    date_options = ["-- None --"] + date_candidates
    default_index = date_options.index(auto_date_col) if auto_date_col and auto_date_col in date_options else 0
    selected_date_col = st.sidebar.selectbox(
        "Date Column",
        date_options,
        index=default_index,
        help="Choose which column should be used for date filtering."
    )

    # Initialize date range variables
    start_date = None
    end_date = None
    date_col_to_use = None  # Track the date column used for filtering
    processing_warnings = []  # Store warnings to display in Processing Details

    if selected_date_col != "-- None --":
        # Track dispatchers before date filtering
        dispatchers_before_filter = set()
        if 'Dispatcher ID' in df.columns:
            dispatchers_before_filter = set(df['Dispatcher ID'].dropna().astype(str).str.strip().unique())
            dispatchers_before_filter = {d for d in dispatchers_before_filter if d and d.lower() not in ['unknown', 'nan', 'none', 'null', '']}

        # If "Delivery Signature" is selected, check for "delivery_signature" column (original database column)
        # The correct database column name is "delivery_signature" (lowercase with underscore)
        date_col_to_use = selected_date_col
        if selected_date_col == "Delivery Signature":
            # Prefer the original database column name if it exists
            if "delivery_signature" in df.columns:
                date_col_to_use = "delivery_signature"
            elif "delivery_signature_date" in df.columns:
                date_col_to_use = "delivery_signature_date"

        df[date_col_to_use] = pd.to_datetime(df[date_col_to_use], errors="coerce")
        valid_dates = df[date_col_to_use].dropna()
        if not valid_dates.empty:
            min_date, max_date = valid_dates.min().date(), valid_dates.max().date()
            default_start = max(min_date, max_date.replace(day=1))
            selected_range = st.sidebar.date_input(
                "Select Date Range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if isinstance(selected_range, tuple) and len(selected_range) == 2:
                start_date, end_date = selected_range
            else:
                start_date = end_date = selected_range

            # Filter by date range
            df_filtered = df[
                (df[date_col_to_use].dt.date >= start_date) &
                (df[date_col_to_use].dt.date <= end_date)
            ]

            # Track dispatchers after filtering
            dispatchers_after_filter = set()
            if 'Dispatcher ID' in df_filtered.columns:
                dispatchers_after_filter = set(df_filtered['Dispatcher ID'].dropna().astype(str).str.strip().unique())
                dispatchers_after_filter = {d for d in dispatchers_after_filter if d and d.lower() not in ['unknown', 'nan', 'none', 'null', '']}

            # Check for dispatchers lost due to invalid dates
            lost_dispatchers = dispatchers_before_filter - dispatchers_after_filter
            if lost_dispatchers:
                # These dispatchers have no valid dates in the selected date column
                # Include rows with NULL/invalid dates for these dispatchers
                dispatchers_with_invalid_dates = df[
                    (df['Dispatcher ID'].isin(lost_dispatchers)) &
                    (df[date_col_to_use].isna())
                ]

                if not dispatchers_with_invalid_dates.empty:
                    st.info(f"📋 Including {len(lost_dispatchers)} dispatcher(s) with invalid dates in '{selected_date_col}': {sorted(lost_dispatchers)}")
                    # Combine filtered data with dispatchers that have invalid dates
                    df = pd.concat([df_filtered, dispatchers_with_invalid_dates], ignore_index=True)
                else:
                    # Store warning message to be displayed in Processing Details
                    processing_warnings.append(f"⚠️ {len(lost_dispatchers)} dispatcher(s) have no valid dates in '{selected_date_col}' and will show 0 parcels: {sorted(lost_dispatchers)}")
                    df = df_filtered
            else:
                df = df_filtered

            if df.empty:
                st.warning("No records found for the selected date range.")
                add_footer()
                return
        else:
            st.sidebar.warning("Selected date column has no valid date values; showing all data.")

    # Load penalty, pickup, QR order, return, reward, and attendance data
    penalty_data = DataSource.load_penalty_data(config, excel_source)
    pickup_df = DataSource.load_pickup_data(config, excel_source)
    qr_order_df = DataSource.load_qr_order_data(config, excel_source)
    return_df = DataSource.load_return_data(config, excel_source)
    reward_df = DataSource.load_reward_data(config, excel_source)
    attendance_df = DataSource.load_attendance_data(config, excel_source)

    # Filter all data by selected date range if a date column is selected
    if selected_date_col != "-- None --" and start_date is not None and end_date is not None:
        with st.expander("📋 Data Filtering Details", expanded=False):
            # Filter pickup_df by selected month/date range
            if pickup_df is not None and not pickup_df.empty:
                pickup_date_col = None
                for col in pickup_df.columns:
                    if any(k in str(col).lower() for k in ["date", "pick_up", "pickup", "signature"]):
                        pickup_date_col = col
                        break

                if pickup_date_col is not None:
                    pickup_df[pickup_date_col] = pd.to_datetime(pickup_df[pickup_date_col], errors="coerce")
                    initial_pickup_count = len(pickup_df)
                    pickup_df = pickup_df[
                        (pickup_df[pickup_date_col].dt.date >= start_date) &
                        (pickup_df[pickup_date_col].dt.date <= end_date)
                    ]
                    filtered_pickup_count = len(pickup_df)
                    if initial_pickup_count != filtered_pickup_count:
                        st.info(f"📦 Filtered pickup data: {initial_pickup_count:,} → {filtered_pickup_count:,} records")
                else:
                    st.warning("⚠️ Pickup table has no detectable date column; pickup parcels are not filtered by date range.")

            # Filter penalty data by selected date range
            if penalty_data is not None:
                # Filter DuitNow penalty - use created_at column for date filtering
                if 'duitnow' in penalty_data and penalty_data['duitnow'] is not None and not penalty_data['duitnow'].empty:
                    duitnow_df = penalty_data['duitnow']
                    duitnow_date_col = None

                    # Explicitly look for created_at column for DuitNow penalty
                    if 'created_at' in duitnow_df.columns:
                        duitnow_date_col = 'created_at'
                    else:
                        # Try case-insensitive search
                        for col in duitnow_df.columns:
                            if str(col).lower() == "created_at":
                                duitnow_date_col = col
                                break

                    # If created_at not found, try other date columns as fallback
                    if duitnow_date_col is None:
                        for col in duitnow_df.columns:
                            col_lower = str(col).lower()
                            if any(k in col_lower for k in ["date", "time", "delivery", "signature", "updated_at"]):
                                duitnow_date_col = col
                                break

                    if duitnow_date_col is not None:
                        duitnow_df[duitnow_date_col] = pd.to_datetime(duitnow_df[duitnow_date_col], errors="coerce")
                        initial_duitnow_count = len(duitnow_df)

                        # Check for valid dates before filtering
                        valid_dates = duitnow_df[duitnow_date_col].notna()
                        invalid_date_count = (~valid_dates).sum()

                        if invalid_date_count > 0:
                            st.warning(f"⚠️ DuitNow: {invalid_date_count:,} records have invalid/null dates in '{duitnow_date_col}' column")

                        # Show date range of data before filtering
                        if valid_dates.any():
                            min_date_in_data = duitnow_df[valid_dates][duitnow_date_col].min().date()
                            max_date_in_data = duitnow_df[valid_dates][duitnow_date_col].max().date()
                            st.info(f"📅 DuitNow date range in data: {min_date_in_data} to {max_date_in_data} (filtering: {start_date} to {end_date})")

                        # Filter by date range using created_at minus 1 month
                        adjusted_duitnow_dates = (duitnow_df[duitnow_date_col] - pd.DateOffset(months=1))
                        duitnow_df = duitnow_df[
                            (adjusted_duitnow_dates.dt.date >= start_date) &
                            (adjusted_duitnow_dates.dt.date <= end_date)
                        ]
                        filtered_duitnow_count = len(duitnow_df)
                        penalty_data['duitnow'] = duitnow_df

                        if filtered_duitnow_count == 0:
                            st.error(f"❌ All DuitNow penalty records filtered out! Initial: {initial_duitnow_count:,} records, after date filter: 0 records. Date range might not match data dates.")
                        elif initial_duitnow_count != filtered_duitnow_count:
                            st.info(f"⚠️ Filtered DuitNow penalty using '{duitnow_date_col}': {initial_duitnow_count:,} → {filtered_duitnow_count:,} records")
                        else:
                            st.info(f"ℹ️ DuitNow penalty filtered using '{duitnow_date_col}' column (all {initial_duitnow_count:,} records within date range)")
                    else:
                        st.warning("⚠️ DuitNow penalty table has no 'created_at' column or other detectable date column; penalties are not filtered by date range.")

                # Filter LDR penalty
                if 'ldr' in penalty_data and penalty_data['ldr'] is not None and not penalty_data['ldr'].empty:
                    ldr_df = penalty_data['ldr']
                    ldr_date_col = None
                    # Prefer audit timestamps if present
                    if 'created_at' in ldr_df.columns:
                        ldr_date_col = 'created_at'
                    elif 'updated_at' in ldr_df.columns:
                        ldr_date_col = 'updated_at'
                    else:
                        # Fallback: find any date/time-like column
                        for col in ldr_df.columns:
                            col_lower = str(col).lower()
                            if any(k in col_lower for k in ["date", "time", "delivery", "signature"]):
                                ldr_date_col = col
                                break

                    if ldr_date_col is not None:
                        ldr_parsed = pd.to_datetime(ldr_df[ldr_date_col], errors="coerce")
                        # If parsing fails and column is numeric, try Excel serial date
                        if ldr_parsed.notna().sum() == 0:
                            numeric_series = pd.to_numeric(ldr_df[ldr_date_col], errors="coerce")
                            if numeric_series.notna().sum() > 0:
                                ldr_parsed = pd.to_datetime(
                                    numeric_series,
                                    unit='D',
                                    origin='1899-12-30',
                                    errors='coerce'
                                )
                        ldr_df[ldr_date_col] = ldr_parsed
                        initial_ldr_count = len(ldr_df)
                        ldr_df = ldr_df[
                            (ldr_df[ldr_date_col].dt.date >= start_date) &
                            (ldr_df[ldr_date_col].dt.date <= end_date)
                        ]
                        filtered_ldr_count = len(ldr_df)
                        penalty_data['ldr'] = ldr_df
                        if initial_ldr_count != filtered_ldr_count:
                            st.info(f"⚠️ Filtered LDR penalty: {initial_ldr_count:,} → {filtered_ldr_count:,} records")
                    else:
                        st.warning("⚠️ LDR penalty table has no detectable date column; penalties are not filtered by date range.")

                # Filter Fake Attempt penalty
                if 'fake_attempt' in penalty_data and penalty_data['fake_attempt'] is not None and not penalty_data['fake_attempt'].empty:
                    fake_attempt_df = penalty_data['fake_attempt']
                    fake_attempt_date_col = None
                    # Try to find time_delivery or other date columns
                    for col in fake_attempt_df.columns:
                        col_lower = str(col).lower()
                        if any(k in col_lower for k in ["time_delivery", "date", "time", "delivery", "signature", "created_at", "updated_at"]):
                            fake_attempt_date_col = col
                            # Prefer time_delivery if it exists
                            if col_lower == "time_delivery":
                                break

                    if fake_attempt_date_col is not None:
                        fake_attempt_df[fake_attempt_date_col] = pd.to_datetime(fake_attempt_df[fake_attempt_date_col], errors="coerce")
                        initial_fake_count = len(fake_attempt_df)
                        fake_attempt_df = fake_attempt_df[
                            (fake_attempt_df[fake_attempt_date_col].dt.date >= start_date) &
                            (fake_attempt_df[fake_attempt_date_col].dt.date <= end_date)
                        ]
                        filtered_fake_count = len(fake_attempt_df)
                        penalty_data['fake_attempt'] = fake_attempt_df
                        if initial_fake_count != filtered_fake_count:
                            st.info(f"⚠️ Filtered Fake Attempt penalty: {initial_fake_count:,} → {filtered_fake_count:,} records")
                    else:
                        st.warning("⚠️ Fake Attempt penalty table has no detectable date column; penalties are not filtered by date range.")

                # Filter COD penalty
                if 'cod' in penalty_data and penalty_data['cod'] is not None and not penalty_data['cod'].empty:
                    cod_df = penalty_data['cod']
                    cod_date_col = None
                    # Try to find date column (prefer 'date' column, fallback to 'created_at')
                    if 'date' in cod_df.columns:
                        cod_date_col = 'date'
                    elif 'created_at' in cod_df.columns:
                        cod_date_col = 'created_at'
                    else:
                        # Try case-insensitive search
                        for col in cod_df.columns:
                            col_lower = str(col).lower()
                            if col_lower in ['date', 'created_at']:
                                cod_date_col = col
                                break
                            elif any(k in col_lower for k in ["date", "time", "delivery", "signature", "updated_at"]):
                                cod_date_col = col
                                break

                    if cod_date_col is not None:
                        cod_df[cod_date_col] = pd.to_datetime(cod_df[cod_date_col], errors="coerce")
                        initial_cod_count = len(cod_df)
                        cod_df = cod_df[
                            (cod_df[cod_date_col].dt.date >= start_date) &
                            (cod_df[cod_date_col].dt.date <= end_date)
                        ]
                        filtered_cod_count = len(cod_df)
                        penalty_data['cod'] = cod_df
                        if initial_cod_count != filtered_cod_count:
                            st.info(f"⚠️ Filtered COD penalty: {initial_cod_count:,} → {filtered_cod_count:,} records")
                    else:
                        st.warning("⚠️ COD penalty table has no detectable date column; penalties are not filtered by date range.")

                # Filter QR order data by selected date range
                if qr_order_df is not None and not qr_order_df.empty:
                    qr_order_date_col = None
                    # Prefer date_pick_up for QR orders, fallback to created_at
                    if 'date_pick_up' in qr_order_df.columns:
                        qr_order_date_col = 'date_pick_up'
                    elif 'created_at' in qr_order_df.columns:
                        qr_order_date_col = 'created_at'
                    else:
                        # Try case-insensitive search
                        for col in qr_order_df.columns:
                            col_lower = str(col).lower()
                            if col_lower in ("date_pick_up", "created_at"):
                                qr_order_date_col = col
                                break

                    if qr_order_date_col is not None:
                        qr_order_df[qr_order_date_col] = pd.to_datetime(qr_order_df[qr_order_date_col], errors="coerce")
                        initial_qr_count = len(qr_order_df)

                        # Check for valid dates before filtering
                        valid_dates = qr_order_df[qr_order_date_col].notna()
                        invalid_date_count = (~valid_dates).sum()

                        if invalid_date_count > 0:
                            st.warning(f"⚠️ QR Orders: {invalid_date_count:,} records have invalid/null dates in '{qr_order_date_col}' column")

                        # Filter by date range
                        qr_order_df = qr_order_df[
                            (qr_order_df[qr_order_date_col].dt.date >= start_date) &
                            (qr_order_df[qr_order_date_col].dt.date <= end_date)
                        ]
                        filtered_qr_count = len(qr_order_df)

                        if filtered_qr_count == 0:
                            st.error(f"❌ All QR order records filtered out! Initial: {initial_qr_count:,} records, after date filter: 0 records.")
                        elif initial_qr_count != filtered_qr_count:
                            st.info(f"📦 Filtered QR orders using '{qr_order_date_col}': {initial_qr_count:,} → {filtered_qr_count:,} records")
                    else:
                        st.warning("⚠️ QR order table has no 'date_pick_up' column; QR orders are not filtered by date range.")

                # Filter return data by selected date range
                if return_df is not None and not return_df.empty:
                    return_date_col = None
                    # Check for mapped column name first (Created At), then original (created_at)
                    if 'Created At' in return_df.columns:
                        return_date_col = 'Created At'
                    elif 'created_at' in return_df.columns:
                        return_date_col = 'created_at'
                    else:
                        # Try case-insensitive search
                        for col in return_df.columns:
                            col_lower = str(col).lower()
                            if col_lower == "created_at" or col_lower == "created at":
                                return_date_col = col
                                break

                    if return_date_col is not None:
                        return_df[return_date_col] = pd.to_datetime(return_df[return_date_col], errors="coerce")
                        initial_return_count = len(return_df)

                        # Check for valid dates before filtering
                        valid_dates = return_df[return_date_col].notna()
                        invalid_date_count = (~valid_dates).sum()

                        if invalid_date_count > 0:
                            st.warning(f"⚠️ Returns: {invalid_date_count:,} records have invalid/null dates in '{return_date_col}' column")

                        # Filter by date range
                        return_df = return_df[
                            (return_df[return_date_col].dt.date >= start_date) &
                            (return_df[return_date_col].dt.date <= end_date)
                        ]
                        filtered_return_count = len(return_df)

                        if filtered_return_count == 0:
                            st.error(f"❌ All return records filtered out! Initial: {initial_return_count:,} records, after date filter: 0 records.")
                        elif initial_return_count != filtered_return_count:
                            st.info(f"↩️ Filtered returns using '{return_date_col}': {initial_return_count:,} → {filtered_return_count:,} records")
                    else:
                        st.warning("⚠️ Return table has no 'created_at' column; returns are not filtered by date range.")

                # Filter attendance data by selected date range
                if attendance_df is not None and not attendance_df.empty:
                    attendance_date_col = None
                    for col in attendance_df.columns:
                        col_lower = str(col).lower().strip()
                        if col_lower in ["attendance record date", "attendance_record_date", "attendance date", "attendance_date"]:
                            attendance_date_col = col
                            break

                    if attendance_date_col is not None:
                        attendance_df[attendance_date_col] = pd.to_datetime(attendance_df[attendance_date_col], errors="coerce")
                        initial_attendance_count = len(attendance_df)
                        attendance_df = attendance_df[
                            (attendance_df[attendance_date_col].dt.date >= start_date) &
                            (attendance_df[attendance_date_col].dt.date <= end_date)
                        ]
                        filtered_attendance_count = len(attendance_df)
                        if initial_attendance_count != filtered_attendance_count:
                            st.info(f"🕒 Filtered attendance data: {initial_attendance_count:,} → {filtered_attendance_count:,} records")
                    else:
                        st.warning("⚠️ Attendance table has no 'Attendance Record Date' column; attendance is not filtered by date range.")

            # Show summary of date filtering
            if selected_date_col != "-- None --" and start_date is not None and end_date is not None:
                st.success(f"✅ All data filtered by date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Calculate payouts
    currency = config.get("currency_symbol", "RM")
    qr_order_payout_per_order = config.get("qr_order_payout_per_order", 1.80)
    return_payout_per_parcel = config.get("return_payout_per_parcel", 1.50)

    display_df, numeric_df, total_payout = PayoutCalculator.calculate_payout(
        df, currency, penalty_data, pickup_df, pickup_payout_per_parcel,
        qr_order_df, qr_order_payout_per_order, return_df, return_payout_per_parcel,
        reward_df=reward_df,
        attendance_df=attendance_df,
        processing_warnings=processing_warnings
    )

    if numeric_df.empty:
        st.warning("No data after filtering.")
        add_footer()
        return

    # Get daily trends for forecasting
    # Use the date column that was used for filtering, or try to find it
    forecast_date_col = date_col_to_use if date_col_to_use and date_col_to_use in df.columns else None
    daily_parcel_df = PayoutCalculator.get_daily_trend(df, forecast_date_col)

    # Debug info
    if daily_parcel_df.empty:
        st.warning(f"⚠️ No daily trend data available. Date column used: {forecast_date_col if forecast_date_col else 'auto-detected'}. Total rows in df: {len(df)}")
    elif len(daily_parcel_df) < 7:
        st.warning(f"⚠️ Only {len(daily_parcel_df)} unique dates found in data. Need at least 7 days for forecasting.")

    # Calculate daily payout trend (requires additional processing)
    df_clean = DataProcessor.prepare_dataframe(df)
    df_clean = DataProcessor.remove_duplicates(df_clean)

    if 'weight' in df_clean.columns:
        tiers = Config.load().get("weight_tiers")
        df_clean['payout_rate'] = df_clean['weight'].apply(
            lambda w: PayoutCalculator.get_rate_by_weight(w, tiers)
        )
        df_clean['payout'] = df_clean['payout_rate']

        daily_payout_df = PayoutCalculator.get_daily_payout_trend(df_clean, forecast_date_col)

    # Tabs layout
    tab_overview, tab_analytics, tab_details, tab_invoice = st.tabs(["📊 Overview", "📈 Analytics", "🧾 Details", "📄 Invoice"])

    # Metrics - Updated to include Pickup Payout per Parcel
    with tab_overview:
        st.subheader("📈 Performance Overview")

        # Calculate totals
        total_pickup_parcels = int(numeric_df["Pickup Parcels"].sum()) if "Pickup Parcels" in numeric_df.columns else 0
        total_pickup_payout = numeric_df["Pickup Parcels Payout"].sum() if "Pickup Parcels Payout" in numeric_df.columns else 0.0
        total_qr_orders = int(numeric_df["QR Orders"].sum()) if "QR Orders" in numeric_df.columns else 0
        total_qr_order_payout = numeric_df["QR Orders Payout"].sum() if "QR Orders Payout" in numeric_df.columns else 0.0
        total_return_parcels = int(numeric_df["Return Parcels"].sum()) if "Return Parcels" in numeric_df.columns else 0
        total_return_payout = numeric_df["Return Parcels Payout"].sum() if "Return Parcels Payout" in numeric_df.columns else 0.0
        total_dispatch_payout = numeric_df["Delivery Parcels Payout"].sum() if "Delivery Parcels Payout" in numeric_df.columns else 0.0
        total_reward = numeric_df["Reward"].sum() if "Reward" in numeric_df.columns else 0.0

        total_penalty = numeric_df["Penalty"].sum() if "Penalty" in numeric_df.columns else 0.0

        # Calculate penalty breakdown by type
        penalty_by_type = {
            'duitnow': numeric_df["DuitNow Penalty"].sum() if "DuitNow Penalty" in numeric_df.columns else 0.0,
            'ldr': numeric_df["LDR Penalty"].sum() if "LDR Penalty" in numeric_df.columns else 0.0,
            'fake_attempt': numeric_df["Fake Attempt Penalty"].sum() if "Fake Attempt Penalty" in numeric_df.columns else 0.0,
            'cod': numeric_df["COD Penalty"].sum() if "COD Penalty" in numeric_df.columns else 0.0,
            'attendance': numeric_df["Attendance Penalty"].sum() if "Attendance Penalty" in numeric_df.columns else 0.0
        }

        # Row 1: Dispatcher | Delivery Parcels | Pickup Parcels | Return Parcels | QR Orders
        row1_col1, row1_col2, row1_col3, row1_col4, row1_col5 = st.columns(5)
        with row1_col1:
            st.metric("Dispatchers", f"{len(display_df):,}")
        with row1_col2:
            st.metric("Delivery Parcels", f"{int(numeric_df['Parcels Delivered'].sum()):,}")
        with row1_col3:
            st.metric("Pickup Parcels", f"{total_pickup_parcels:,}")
        with row1_col4:
            st.metric("Return Parcels", f"{total_return_parcels:,}")
        with row1_col5:
            st.metric("QR Orders", f"{total_qr_orders:,}")

        # Row 2: Total Payout | Delivery Parcels Payout | Pickup Parcels Payout | Return Parcels Payout | QR Orders Payout
        row2_col1, row2_col2, row2_col3, row2_col4, row2_col5 = st.columns(5)
        with row2_col1:
            st.metric("Total Payout", f"{currency} {total_payout:,.2f}")
        with row2_col2:
            st.metric("Delivery Parcels Payout", f"{currency} {total_dispatch_payout:,.2f}")
        with row2_col3:
            st.metric("Reward", f"{currency} {total_reward:,.2f}")
        with row2_col4:
            st.metric("Pickup Parcels Payout", f"{currency} {total_pickup_payout:,.2f}")
        with row2_col5:
            st.metric("Return Parcels Payout", f"{currency} {total_return_payout:,.2f}")

        row3_cols = st.columns(5)
        with row3_cols[0]:
            st.metric("QR Orders Payout", f"{currency} {total_qr_order_payout:,.2f}")

        # Penalty breakdown by type
        st.markdown("#### ⚠️ Penalty Breakdown by Type")
        penalty_col1, penalty_col2, penalty_col3, penalty_col4, penalty_col5 = st.columns(5)
        with penalty_col1:
            st.metric("DuitNow Penalty", f"-{currency} {penalty_by_type['duitnow']:,.2f}")
        with penalty_col2:
            st.metric("LDR Penalty", f"-{currency} {penalty_by_type['ldr']:,.2f}")
        with penalty_col3:
            st.metric("Fake Attempt Penalty", f"-{currency} {penalty_by_type['fake_attempt']:,.2f}")
        with penalty_col4:
            st.metric("COD Penalty", f"-{currency} {penalty_by_type['cod']:,.2f}")
        with penalty_col5:
            st.metric("Attendance Penalty", f"-{currency} {penalty_by_type['attendance']:,.2f}")

    # Charts
    with tab_analytics:
        st.markdown("---")
        st.subheader("📊 Performance Analytics")
        charts = DataVisualizer.create_all_charts(daily_parcel_df, numeric_df)

        col1, col2 = st.columns([1, 1])
        with col1:
            if 'daily_trend' in charts:
                st.altair_chart(charts['daily_trend'], use_container_width=True)
            if 'performers' in charts:
                st.altair_chart(charts['performers'], use_container_width=True)

        with col2:
            if 'payout_dist' in charts:
                st.altair_chart(charts['payout_dist'], use_container_width=True)

    # Top Performers and Top Penalties side by side
    with tab_overview:
        top_col1, top_col2 = st.columns(2)

        with top_col1:
            st.markdown("##### 🏆 Top Performers")
            # Sort by Total Payout descending and take top 5
            top_performers = numeric_df.sort_values('Total Payout', ascending=False).head(5)
            for _, row in top_performers.iterrows():
                parcels_delivered = int(row.get('Parcels Delivered', 0))
                pickup_parcels = int(row.get('Pickup Parcels', 0))
                return_parcels = int(row.get('Return Parcels', 0))
                qr_orders = int(row.get('QR Orders', 0))
                delivery_payout = float(row.get('Delivery Parcels Payout', 0.0))
                pickup_payout = float(row.get('Pickup Parcels Payout', 0.0))
                return_payout = float(row.get('Return Parcels Payout', 0.0))
                qr_orders_payout = float(row.get('QR Orders Payout', 0.0))
                reward_amount = float(row.get('Reward', 0.0))

                st.markdown(f"""
                <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin: 8px 0; min-height: 90px; height: 90px; display: flex; flex-direction: column; justify-content: space-between;">
                    <div style="font-weight: 600; color: {ColorScheme.PRIMARY}; margin-bottom: 8px; font-size: 0.95rem;">{row['Dispatcher Name']}</div>
                    <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 4px; line-height: 1.4;">
                        <strong>Parcels:</strong> {parcels_delivered} | <strong>Pickup Parcels:</strong> {pickup_parcels} | <strong>Return Parcels:</strong> {return_parcels} | <strong>QR Orders:</strong> {qr_orders}
                    </div>
                    <div style="color: #64748b; font-size: 0.85rem; line-height: 1.4;">
                        <strong>Parcel Payout:</strong> {currency}{delivery_payout:,.2f} | <strong>Pickup Payout:</strong> {currency}{pickup_payout:,.2f} | <strong>Return Payout:</strong> {currency}{return_payout:,.2f} | <strong>QR Orders Payout:</strong> {currency}{qr_orders_payout:,.2f} | <strong>Reward:</strong> {currency}{reward_amount:,.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with top_col2:
            st.markdown("##### ⚠️ Top Penalties")
            # Filter dispatchers with penalties and sort by penalty amount descending, take top 5
            penalty_df = numeric_df[numeric_df['Penalty'] > 0].copy() if 'Penalty' in numeric_df.columns else pd.DataFrame()
            if not penalty_df.empty:
                top_penalties = penalty_df.sort_values('Penalty', ascending=False).head(5)
                for _, row in top_penalties.iterrows():
                    penalty_amount = row.get('Penalty', 0.0)
                    penalty_count = row.get('Penalty Parcels', 0) if 'Penalty Parcels' in row else 0
                    # Get penalty breakdown if available
                    duitnow_penalty = row.get('DuitNow Penalty', 0.0) if 'DuitNow Penalty' in row else 0.0
                    ldr_penalty = row.get('LDR Penalty', 0.0) if 'LDR Penalty' in row else 0.0
                    fake_attempt_penalty = row.get('Fake Attempt Penalty', 0.0) if 'Fake Attempt Penalty' in row else 0.0
                    cod_penalty = row.get('COD Penalty', 0.0) if 'COD Penalty' in row else 0.0
                    attendance_penalty = row.get('Attendance Penalty', 0.0) if 'Attendance Penalty' in row else 0.0

                    penalty_breakdown = []
                    if duitnow_penalty > 0:
                        penalty_breakdown.append(f"DuitNow: {currency}{duitnow_penalty:,.2f}")
                    if ldr_penalty > 0:
                        penalty_breakdown.append(f"LDR: {currency}{ldr_penalty:,.2f}")
                    if fake_attempt_penalty > 0:
                        penalty_breakdown.append(f"Fake: {currency}{fake_attempt_penalty:,.2f}")
                    if cod_penalty > 0:
                        penalty_breakdown.append(f"COD: {currency}{cod_penalty:,.2f}")
                    if attendance_penalty > 0:
                        penalty_breakdown.append(f"Attendance: {currency}{attendance_penalty:,.2f}")

                    breakdown_text = " • ".join(penalty_breakdown) if penalty_breakdown else ""

                    st.markdown(f"""
                    <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin: 8px 0; min-height: 90px; height: 90px; display: flex; flex-direction: column; justify-content: space-between;">
                        <div style="font-weight: 600; color: {ColorScheme.ERROR}; margin-bottom: 8px; font-size: 0.95rem;">{row['Dispatcher Name']}</div>
                        <div style="color: #64748b; font-size: 0.85rem; margin-bottom: 4px; line-height: 1.4;">
                            {penalty_count} penalty parcels • -{currency}{penalty_amount:,.2f} total
                        </div>
                        {f'<div style="color: #ef4444; font-size: 0.85rem; line-height: 1.4;">{breakdown_text}</div>' if breakdown_text else '<div style="min-height: 20px;"></div>'}
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No penalties recorded")

    # Data table
    with tab_details:
        st.markdown("---")

        with st.expander("👥 Dispatcher Performance Details", expanded=False):
            st.subheader("👥 Dispatcher Performance Details")

            # Reorder columns for better readability
            preferred_order = [
                "Dispatcher ID", "Dispatcher Name", "Parcels Delivered",
                "Delivery Parcels Payout", "Pickup Parcels", "Pickup Parcels Payout",
                "QR Orders", "QR Orders Payout",
                "Return Parcels", "Return Parcels Payout",
                "Reward",
                "Penalty", "DuitNow Penalty", "LDR Penalty", "Fake Attempt Penalty", "COD Penalty", "Attendance Penalty",
                "Total Payout", "Total Weight (kg)", "Avg Weight (kg)",
                "Avg Rate per Parcel", "Parcels 0-5kg", "Parcels 5.01-10kg",
                "Parcels 10.01-30kg", "Parcels 30+kg"
            ]

            # Filter to only include columns that exist in display_df
            existing_columns = [col for col in preferred_order if col in display_df.columns]
            remaining_columns = [col for col in display_df.columns if col not in existing_columns]

            # Create final column order
            final_columns = existing_columns + remaining_columns

            # Reorder the display dataframe
            display_df_reordered = display_df[final_columns]

            st.dataframe(display_df_reordered, use_container_width=True, hide_index=True)

        # Pickup Parcels Details
        if pickup_df is not None and not pickup_df.empty:
            with st.expander("📦 Pickup Parcels Details", expanded=False):
                st.subheader("📦 Pickup Parcels Details")
                # Find pickup dispatcher ID column
                pickup_dispatcher_col = None
                for col in pickup_df.columns:
                    if 'pickup_dispatcher_id' in col.lower() or 'Pickup Dispatcher ID' in col:
                        pickup_dispatcher_col = col
                        break

                # Find waybill column
                waybill_col = None
                for col in pickup_df.columns:
                    if 'waybill' in col.lower() or 'Waybill Number' in col:
                        waybill_col = col
                        break

                if pickup_dispatcher_col and waybill_col:
                    # Show summary by dispatcher
                    pickup_summary = pickup_df.groupby(pickup_dispatcher_col).agg(
                        parcel_count=(waybill_col, 'nunique')
                    ).reset_index()
                    pickup_summary = pickup_summary.rename(columns={pickup_dispatcher_col: 'Dispatcher ID', 'parcel_count': 'Pickup Parcels'})
                    pickup_summary = pickup_summary.sort_values('Pickup Parcels', ascending=False)
                    st.dataframe(pickup_summary, use_container_width=True, hide_index=True)

                    # Show full details
                    st.markdown("#### Full Pickup Data")
                    st.dataframe(pickup_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(pickup_df, use_container_width=True, hide_index=True)

        # Return Parcels Details
        if return_df is not None and not return_df.empty:
            with st.expander("↩️ Return Parcels Details", expanded=False):
                st.subheader("↩️ Return Parcels Details")
                # Find dispatcher ID column
                return_dispatcher_col = None
                for col in return_df.columns:
                    if 'dispatcher_id' in col.lower() or 'Dispatcher ID' in col:
                        return_dispatcher_col = col
                        break

                # Find waybill column
                waybill_col = None
                for col in return_df.columns:
                    if 'waybill' in col.lower() or 'Waybill Number' in col:
                        waybill_col = col
                        break

                if return_dispatcher_col and waybill_col:
                    # Show summary by dispatcher
                    return_summary = return_df.groupby(return_dispatcher_col).agg(
                        parcel_count=(waybill_col, 'nunique')
                    ).reset_index()
                    return_summary = return_summary.rename(columns={return_dispatcher_col: 'Dispatcher ID', 'parcel_count': 'Return Parcels'})
                    return_summary = return_summary.sort_values('Return Parcels', ascending=False)
                    st.dataframe(return_summary, use_container_width=True, hide_index=True)

                    # Show full details
                    st.markdown("#### Full Return Data")
                    st.dataframe(return_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(return_df, use_container_width=True, hide_index=True)

        # QR Orders Details
        if qr_order_df is not None and not qr_order_df.empty:
            with st.expander("📱 QR Orders Details", expanded=False):
                st.subheader("📱 QR Orders Details")
                # Find dispatcher column
                qr_dispatcher_col = None
                for col in qr_order_df.columns:
                    if col.lower() in ('pickup_dispatcher_id', 'dispatcher_id', 'dispatcher'):
                        qr_dispatcher_col = col
                        break

                # Find waybill_number column
                order_no_col = None
                for col in qr_order_df.columns:
                    if col.lower() in ('waybill_number', 'no_awb'):
                        order_no_col = col
                        break

                if qr_dispatcher_col and order_no_col:
                    # Show summary by dispatcher
                    qr_summary = qr_order_df.groupby(qr_dispatcher_col).agg(
                        order_count=(order_no_col, 'nunique')
                    ).reset_index()
                    qr_summary = qr_summary.rename(columns={qr_dispatcher_col: 'Dispatcher ID', 'order_count': 'QR Orders'})
                    qr_summary = qr_summary.sort_values('QR Orders', ascending=False)
                    st.dataframe(qr_summary, use_container_width=True, hide_index=True)

                    # Show full details
                    st.markdown("#### Full QR Orders Data")
                    st.dataframe(qr_order_df, use_container_width=True, hide_index=True)
                else:
                    st.dataframe(qr_order_df, use_container_width=True, hide_index=True)

        # Penalty Details Section
        if 'Penalty Parcels' in numeric_df.columns and numeric_df['Penalty Parcels'].sum() > 0:
            with st.expander("⚠️ Penalty Details", expanded=False):
                st.subheader("⚠️ Penalty Details")

                penalty_rows = []
                penalty_dispatchers = numeric_df[numeric_df['Penalty Parcels'] > 0] if 'Penalty Parcels' in numeric_df.columns else pd.DataFrame()
                for _, row in penalty_dispatchers.iterrows():
                    dispatcher_id = row['Dispatcher ID']
                    dispatcher_name = row['Dispatcher Name']
                    penalty_amount = row.get('Penalty', 0.0)
                    # Check if we have waybills stored (they might be in the grouped dataframe before renaming)
                    waybills_str = ''
                    if 'Penalty Waybills' in numeric_df.columns:
                        waybills_str = str(row.get('Penalty Waybills', ''))
                    waybills_list = [w.strip() for w in waybills_str.split(',') if w.strip() and w.strip().lower() != 'nan'] if waybills_str else []

                    if waybills_list:
                        for waybill_number in waybills_list:
                            penalty_rows.append({
                                'Dispatcher ID': dispatcher_id,
                                'Dispatcher Name': dispatcher_name,
                                'Waybill Number': waybill_number,
                                'Penalty Amount': f"{currency}{penalty_amount:,.2f}"
                            })
                    else:
                        # If no waybills, show summary
                        penalty_rows.append({
                            'Dispatcher ID': dispatcher_id,
                            'Dispatcher Name': dispatcher_name,
                            'Waybill Number': f"{int(row.get('Penalty Parcels', 0))} parcels",
                            'Penalty Amount': f"{currency}{penalty_amount:,.2f}"
                        })

                if penalty_rows:
                    penalty_table = pd.DataFrame(penalty_rows)
                    st.dataframe(penalty_table, use_container_width=True, hide_index=True)
                else:
                    st.info("No penalty waybill numbers available")


    # Forecasting Section - PARCEL FORECAST
    with tab_analytics:
        st.markdown("---")
        st.subheader("📈 Delivery & Payout Forecast")

        if not daily_parcel_df.empty and len(daily_parcel_df) >= 7:
            if PROPHET_AVAILABLE:
                # Parcel Forecast
                st.markdown("##### 📦 Parcel Delivery Forecast")
                with st.spinner("Generating parcel forecast..."):
                    forecast_result = ForecastGenerator.generate_forecast(daily_parcel_df, forecast_days, 'total_parcels')

                    if forecast_result:
                        model, forecast, forecast_summary = forecast_result

                        # Display forecast metrics
                        metrics = ForecastGenerator.calculate_forecast_metrics(daily_parcel_df, forecast_summary, 'total_parcels')

                        if metrics:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Avg Historical Parcels",
                                    f"{metrics['avg_historical']:.0f}",
                                    f"{metrics['change_pct']:+.1f}%"
                                )
                            with col2:
                                st.metric(
                                    "Avg Forecast Parcels",
                                    f"{metrics['avg_forecast']:.0f}",
                                    f"±{metrics['forecast_std']:.0f}"
                                )
                            with col3:
                                st.metric(
                                    "Max Forecast",
                                    f"{metrics['max_forecast']:.0f}"
                                )
                            with col4:
                                st.metric(
                                    "Total Forecast Parcels",
                                    f"{metrics['total_forecast']:.0f}"
                                )

                        # Display forecast chart
                        forecast_chart = ForecastGenerator.create_forecast_chart(
                            daily_parcel_df, forecast_summary,
                            title='30-Day Parcel Delivery Forecast',
                            value_column='total_parcels',
                            y_title='Parcels'
                        )
                        st.altair_chart(forecast_chart, use_container_width=True)

                    else:
                        st.warning("Could not generate parcel forecast.")

                # Payout Forecast (if we have payout data)
                if not daily_payout_df.empty and len(daily_payout_df) >= 7:
                    st.markdown("##### 💰 Payout Forecast")
                    with st.spinner("Generating payout forecast..."):
                        payout_forecast_result = ForecastGenerator.generate_forecast(
                            daily_payout_df, forecast_days, 'total_payout'
                        )

                        if payout_forecast_result:
                            payout_model, payout_forecast, payout_forecast_summary = payout_forecast_result

                            # Display payout forecast metrics
                            payout_metrics = ForecastGenerator.calculate_forecast_metrics(
                                daily_payout_df, payout_forecast_summary, 'total_payout'
                            )

                            if payout_metrics:
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric(
                                        "Avg Historical Payout",
                                        f"{currency} {payout_metrics['avg_historical']:,.2f}",
                                        f"{payout_metrics['change_pct']:+.1f}%"
                                    )
                                with col2:
                                    st.metric(
                                        "Avg Forecast Payout",
                                        f"{currency} {payout_metrics['avg_forecast']:,.2f}",
                                        f"±{currency} {payout_metrics['forecast_std']:,.2f}"
                                    )
                                with col3:
                                    st.metric(
                                        "Max Forecast Payout",
                                        f"{currency} {payout_metrics['max_forecast']:,.2f}"
                                    )
                                with col4:
                                    st.metric(
                                        "Total Forecast Payout",
                                        f"{currency} {payout_metrics['total_forecast']:,.2f}"
                                    )

                            # Display payout forecast chart
                            payout_forecast_chart = ForecastGenerator.create_forecast_chart(
                                daily_payout_df, payout_forecast_summary,
                                title='30-Day Payout Forecast',
                                value_column='total_payout',
                                y_title=f'Payout ({currency})'
                            )
                            st.altair_chart(payout_forecast_chart, use_container_width=True)

                            # Combined forecast table and insights in expander
                            with st.expander("📅 Forecast Details", expanded=False):
                                st.markdown("##### 📅 Forecast Details")

                                # Create combined forecast table
                                if forecast_result and payout_forecast_result:
                                    combined_forecast = forecast_summary.copy()
                                    combined_forecast['Date'] = combined_forecast['ds'].dt.strftime('%Y-%m-%d')
                                    combined_forecast['Parcels Forecast'] = combined_forecast['yhat'].round(0).astype(int)
                                    combined_forecast['Parcels Lower'] = combined_forecast['yhat_lower'].round(0).astype(int)
                                    combined_forecast['Parcels Upper'] = combined_forecast['yhat_upper'].round(0).astype(int)

                                    # Merge with payout forecast
                                    payout_display = payout_forecast_summary.copy()
                                    payout_display['Date'] = payout_display['ds'].dt.strftime('%Y-%m-%d')
                                    payout_display['Payout Forecast'] = payout_display['yhat'].round(2)
                                    payout_display['Payout Lower'] = payout_display['yhat_lower'].round(2)
                                    payout_display['Payout Upper'] = payout_display['yhat_upper'].round(2)

                                    combined_forecast = pd.merge(
                                        combined_forecast[['Date', 'Parcels Forecast', 'Parcels Lower', 'Parcels Upper']],
                                        payout_display[['Date', 'Payout Forecast', 'Payout Lower', 'Payout Upper']],
                                        on='Date'
                                    )

                                    # Format currency
                                    combined_forecast['Payout Forecast'] = combined_forecast['Payout Forecast'].apply(
                                        lambda x: f"{currency} {x:,.2f}"
                                    )
                                    combined_forecast['Payout Lower'] = combined_forecast['Payout Lower'].apply(
                                        lambda x: f"{currency} {x:,.2f}"
                                    )
                                    combined_forecast['Payout Upper'] = combined_forecast['Payout Upper'].apply(
                                        lambda x: f"{currency} {x:,.2f}"
                                    )

                                    st.dataframe(combined_forecast, use_container_width=True, hide_index=True)

                                # Forecast insights
                                st.markdown("##### 🔮 Forecast Insights")

                                if metrics['change_pct'] > 10:
                                    st.info(f"📈 **Growing Parcel Trend:** Forecast shows a {metrics['change_pct']:.1f}% increase in daily parcels.")
                                elif metrics['change_pct'] < -10:
                                    st.warning(f"📉 **Declining Parcel Trend:** Forecast shows a {abs(metrics['change_pct']):.1f}% decrease in daily parcels.")
                                else:
                                    st.success("📊 **Stable Parcel Trend:** Forecast shows relatively stable parcel delivery volumes.")

                                if payout_metrics['change_pct'] > 10:
                                    st.info(f"💰 **Growing Payout Trend:** Forecast shows a {payout_metrics['change_pct']:.1f}% increase in daily payouts.")
                                elif payout_metrics['change_pct'] < -10:
                                    st.warning(f"💸 **Declining Payout Trend:** Forecast shows a {abs(payout_metrics['change_pct']):.1f}% decrease in daily payouts.")
                                else:
                                    st.success("💵 **Stable Payout Trend:** Forecast shows relatively stable payout volumes.")

                        else:
                            st.warning("Could not generate payout forecast.")
                else:
                    st.info("Insufficient payout data for forecasting. Need at least 7 days of payout data.")

            else:
                st.warning("""
                ⚠️ **Prophet forecasting library is not installed.**

                To enable forecasting features, install Prophet:
                ```
                pip install prophet
                ```

                Alternatively, you can use the basic trend analysis shown above.
                """)
        else:
            if daily_parcel_df.empty:
                st.info("⚠️ No daily trend data available for forecasting. Please check your date column selection and ensure data has valid dates.")
            else:
                st.info(f"⚠️ Insufficient historical data for forecasting. Found {len(daily_parcel_df)} unique dates, but need at least 7 days of data.")

    with tab_invoice:
        st.markdown("---")
        st.subheader("📄 Invoice Generation")
        invoice_html = InvoiceGenerator.build_invoice_html(display_df, numeric_df, total_payout, currency, pickup_payout_per_parcel)
        st.components.v1.html(invoice_html, height=1200, scrolling=True)

        st.download_button(
            label="📥 Download Invoice (HTML)",
            data=invoice_html.encode("utf-8"),
            file_name=f"invoice_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            use_container_width=True
        )

    with tab_details:
        st.markdown("---")
        st.subheader("📥 Export Data")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📊 Download Summary CSV",
                data=numeric_df.to_csv(index=False),
                file_name=f"summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                label="📋 Download Raw Data CSV",
                data=df.to_csv(index=False),
                file_name=f"raw_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        with col3:
            if pickup_df is not None and not pickup_df.empty:
                st.download_button(
                    label="📦 Download Pickup Data CSV",
                    data=pickup_df.to_csv(index=False),
                    file_name=f"pickup_data_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

    add_footer()


if __name__ == "__main__":
    main()
