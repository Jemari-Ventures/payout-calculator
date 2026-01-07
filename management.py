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
    'date': ['Delivery Signature', 'Delivery Date', 'Date', 'signature_date', 'delivery_date', 'delivery_signature'],
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
            "pickup_table": "pickup"
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
        "forecast_days": 30
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
                    if "pickup_payout_per_parcel" not in config:
                        config["pickup_payout_per_parcel"] = 1.50
                    if "forecast_days" not in config:
                        config["forecast_days"] = 30
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
    def load_data(config: dict) -> Optional[pd.DataFrame]:
        """Load data based on configuration."""
        data_source = config["data_source"]

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
    def load_penalty_data(config: dict) -> Optional[Dict[str, pd.DataFrame]]:
        """Load penalty data from all penalty tables.

        Returns:
            Dictionary with keys: 'duitnow', 'ldr', 'fake_attempt'
        """
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

        return penalty_data if penalty_data else None

    @staticmethod
    def load_pickup_data(config: dict) -> Optional[pd.DataFrame]:
        """Load pickup data from pickup table."""
        data_source = config["data_source"]
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

# =============================================================================
# DATA PROCESSING
# =============================================================================

class DataProcessor:
    """Handle data cleaning and preparation."""

    @staticmethod
    def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize dataframe columns."""
        df_clean = df.copy()

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
        """Remove duplicate waybills and log count."""
        waybill_col = find_column(df, 'waybill')
        date_col = find_column(df, 'date')

        if waybill_col:
            initial_count = len(df)
            df_to_dedup = df.copy()
            if date_col and date_col in df_to_dedup.columns:
                df_to_dedup[date_col] = pd.to_datetime(df_to_dedup[date_col], errors='coerce')
                df_to_dedup = df_to_dedup.sort_values(by=[date_col, waybill_col])
                df_dedup = df_to_dedup.drop_duplicates(subset=[waybill_col], keep='last')
            else:
                df_dedup = df.drop_duplicates(subset=[waybill_col], keep='first')
            removed = initial_count - len(df_dedup)
            if removed > 0:
                st.info(f"✅ Removed {removed} duplicate waybills")
            return df_dedup

        st.warning("⚠️ No waybill column found; duplicates cannot be removed")
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
                    df_processed['_rider_normalized'] = df_processed[rider_col].astype(str).str.strip().str.lower()

            # Pre-process LDR penalty data
            elif penalty_type == 'ldr':
                employee_id_col = next((col for col in df.columns if col.lower() == 'employee_id'), None)
                if employee_id_col:
                    df_processed['_employee_id_normalized'] = df_processed[employee_id_col].astype(str).str.strip().str.lower()

            # Pre-process Fake Attempt penalty data
            elif penalty_type == 'fake_attempt':
                dispatcher_id_col = next((col for col in df.columns if col.lower() == 'dispatcher_id'), None)
                if dispatcher_id_col:
                    df_processed['_dispatcher_id_normalized'] = df_processed[dispatcher_id_col].astype(str).str.strip().str.lower()

            processed[penalty_type] = df_processed

        return processed

    @staticmethod
    def calculate_penalty(dispatcher_id: str, penalty_data: Optional[Dict[str, pd.DataFrame]]) -> Tuple[float, int, List[str]]:
        """
        Calculate total penalty for a dispatcher from all penalty types.

        Args:
            dispatcher_id: Dispatcher ID to calculate penalty for
            penalty_data: Dictionary containing penalty dataframes from all penalty tables

        Returns:
            (total_penalty_amount, total_penalty_count, waybill_numbers)
        """
        if penalty_data is None or not penalty_data:
            return 0.0, 0, []

        total_penalty = 0.0
        total_count = 0
        waybill_numbers = []
        dispatcher_id_clean = str(dispatcher_id).strip().lower()

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
                        rider_series = duitnow_df[rider_col].astype(str).str.strip().str.lower()
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
                    duitnow_penalty_rounded = sum(rounded_penalties)
                    total_penalty += float(duitnow_penalty_rounded)  # Convert to float only at the end
                    total_count += len(duitnow_records)

        # 2. LDR Penalty: employee_id column = dispatcher_id, penalty = waybill count * RM 100
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
                        employee_series = ldr_df[employee_id_col].astype(str).str.strip().str.lower()
                        ldr_records = ldr_df[employee_series == dispatcher_id_clean]
                    else:
                        ldr_records = pd.DataFrame()

                if not ldr_records.empty:
                    # Count unique waybills (using ticket_no or no_awb if available)
                    waybill_col = next((col for col in ldr_df.columns if col.lower() in ['ticket_no', 'no_awb', 'waybill_number']), None)
                    if waybill_col:
                        waybill_count = ldr_records[waybill_col].nunique()
                        waybill_list = ldr_records[waybill_col].dropna().astype(str).unique().tolist()
                        waybill_numbers.extend([wb for wb in waybill_list if wb and wb.lower() != 'nan'])
                    else:
                        waybill_count = len(ldr_records)

                    ldr_penalty = waybill_count * 100.0  # RM 100 per waybill
                    total_penalty += ldr_penalty
                    total_count += waybill_count

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
                        dispatcher_series = fake_attempt_df[dispatcher_id_col].astype(str).str.strip().str.lower()
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

                    fake_attempt_penalty = waybill_count * 1.0  # RM 1.00 per waybill
                    total_penalty += fake_attempt_penalty
                    total_count += waybill_count

        return float(total_penalty), total_count, waybill_numbers

    @staticmethod
    def calculate_penalty_breakdown(dispatcher_id: str, penalty_data: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, float]:
        """
        Calculate penalty breakdown by type for a dispatcher.

        Args:
            dispatcher_id: Dispatcher ID to calculate penalty for
            penalty_data: Dictionary containing penalty dataframes from all penalty tables

        Returns:
            Dictionary with keys: 'duitnow', 'ldr', 'fake_attempt' and their penalty amounts
        """
        breakdown = {
            'duitnow': 0.0,
            'ldr': 0.0,
            'fake_attempt': 0.0
        }

        if penalty_data is None or not penalty_data:
            return breakdown

        dispatcher_id_clean = str(dispatcher_id).strip().lower()

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
                        rider_series = duitnow_df[rider_col].astype(str).str.strip().str.lower()
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
                    breakdown['duitnow'] = float(sum(rounded_penalties))

        # 2. LDR Penalty: employee_id column = dispatcher_id, penalty = waybill count * RM 100
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
                        employee_series = ldr_df[employee_id_col].astype(str).str.strip().str.lower()
                        ldr_records = ldr_df[employee_series == dispatcher_id_clean]
                    else:
                        ldr_records = pd.DataFrame()

                if not ldr_records.empty:
                    waybill_col = next((col for col in ldr_df.columns if col.lower() in ['ticket_no', 'no_awb', 'waybill_number']), None)
                    if waybill_col:
                        waybill_count = ldr_records[waybill_col].nunique()
                    else:
                        waybill_count = len(ldr_records)

                    breakdown['ldr'] = waybill_count * 100.0  # RM 100 per waybill

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
                        dispatcher_series = fake_attempt_df[dispatcher_id_col].astype(str).str.strip().str.lower()
                        fake_attempt_records = fake_attempt_df[dispatcher_series == dispatcher_id_clean]
                    else:
                        fake_attempt_records = pd.DataFrame()

                if not fake_attempt_records.empty:
                    waybill_col = next((col for col in fake_attempt_df.columns if col.lower() in ['waybill_number', 'waybill']), None)
                    if waybill_col:
                        waybill_count = fake_attempt_records[waybill_col].nunique()
                    else:
                        waybill_count = len(fake_attempt_records)

                    breakdown['fake_attempt'] = waybill_count * 1.0  # RM 1.00 per waybill

        return breakdown

    @staticmethod
    def calculate_penalty_by_type(penalty_data: Optional[Dict[str, pd.DataFrame]]) -> Dict[str, float]:
        """
        Calculate total penalty amounts by type for all dispatchers.

        Args:
            penalty_data: Dictionary containing penalty dataframes from all penalty tables

        Returns:
            Dictionary with keys: 'duitnow', 'ldr', 'fake_attempt' and their total amounts
        """
        penalty_totals = {
            'duitnow': 0.0,
            'ldr': 0.0,
            'fake_attempt': 0.0
        }

        if penalty_data is None or not penalty_data:
            return penalty_totals

        # 1. DuitNow Penalty: sum all penalty amounts (only positive amounts)
        if 'duitnow' in penalty_data:
            duitnow_df = penalty_data['duitnow']
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
                rounded_penalties = [
                    penalty.quantize(Decimal('0.01'), rounding='ROUND_HALF_UP')
                    for penalty in duitnow_filtered['penalty_numeric'].tolist()
                ]
                penalty_totals['duitnow'] = float(sum(rounded_penalties))

        # 2. LDR Penalty: count waybills * RM 100
        if 'ldr' in penalty_data:
            ldr_df = penalty_data['ldr']
            waybill_col = None
            for col in ldr_df.columns:
                if col.lower() in ['ticket_no', 'no_awb', 'waybill_number']:
                    waybill_col = col
                    break

            if waybill_col:
                waybill_count = ldr_df[waybill_col].nunique()
            else:
                waybill_count = len(ldr_df)

            penalty_totals['ldr'] = waybill_count * 100.0  # RM 100 per waybill

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

            penalty_totals['fake_attempt'] = waybill_count * 1.0  # RM 1.00 per waybill

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
    def calculate_payout(df: pd.DataFrame, currency_symbol: str, penalty_data: Optional[Dict[str, pd.DataFrame]] = None,
                        pickup_df: Optional[pd.DataFrame] = None,
                        pickup_payout_per_parcel: float = 1.50) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
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

        # Deduplicate by dispatcher & waybill
        waybill_col = find_column(df_clean, 'waybill') or 'waybill'
        date_col = find_column(df_clean, 'date')
        df_for_unique = df_clean.copy()
        if date_col and date_col in df_for_unique.columns:
            df_for_unique[date_col] = pd.to_datetime(df_for_unique[date_col], errors='coerce')
            df_for_unique = df_for_unique.sort_values(by=[date_col, 'dispatcher_id', waybill_col])
            df_unique = df_for_unique.drop_duplicates(subset=['dispatcher_id', waybill_col], keep='last')
        else:
            df_unique = df_clean.drop_duplicates(subset=['dispatcher_id', waybill_col], keep='first')

        # Diagnostics
        raw_weight = df_clean['weight'].sum()
        dedup_weight = df_unique['weight'].sum()
        duplicates = len(df_clean) - len(df_unique)
        st.info(f"Weight totals – Raw: {raw_weight:,.2f} kg | Deduplicated: {dedup_weight:,.2f} kg | Duplicate rows: {duplicates}")

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
            parcel_count=(waybill_col, 'nunique'),
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

        if penalty_data is not None:
            # Pre-process penalty dataframes once for better performance
            penalty_data_processed = PayoutCalculator._preprocess_penalty_data(penalty_data)

            # Use apply instead of iterrows for better performance
            def calculate_penalties(row):
                dispatcher_id = str(row['dispatcher_id'])
                penalty_amount, penalty_count, penalty_waybills = PayoutCalculator.calculate_penalty(
                    dispatcher_id, penalty_data_processed
                )
                penalty_breakdown = PayoutCalculator.calculate_penalty_breakdown(
                    dispatcher_id, penalty_data_processed
                )
                return pd.Series({
                    'penalty_amount': penalty_amount,
                    'penalty_count': penalty_count,
                    'penalty_waybills': ', '.join(penalty_waybills) if penalty_waybills else '',
                    'duitnow_penalty': penalty_breakdown['duitnow'],
                    'ldr_penalty': penalty_breakdown['ldr'],
                    'fake_attempt_penalty': penalty_breakdown['fake_attempt']
                })

            penalty_results = grouped.apply(calculate_penalties, axis=1)
            grouped[['penalty_amount', 'penalty_count', 'penalty_waybills',
                     'duitnow_penalty', 'ldr_penalty', 'fake_attempt_penalty']] = penalty_results

        # Calculate pickup payout
        grouped = PayoutCalculator.calculate_pickup_payout(pickup_df, grouped, pickup_payout_per_parcel)

        # Calculate total payout: dispatch payout - penalty + pickup payout
        grouped['total_payout'] = grouped['dispatch_payout'] - grouped['penalty_amount'] + grouped['pickup_payout']

        # Create display and numeric dataframes
        numeric_df = grouped.rename(columns={
            "dispatcher_id": "Dispatcher ID",
            "dispatcher_name": "Dispatcher Name",
            "parcel_count": "Parcels Delivered",
            "total_weight": "Total Weight (kg)",
            "avg_weight": "Avg Weight (kg)",
            "avg_rate": "Avg Rate per Parcel",
            "dispatch_payout": "Dispatch Payout",
            "total_payout": "Total Payout",
            "penalty_amount": "Penalty",
            "penalty_count": "Penalty Parcels",
            "penalty_waybills": "Penalty Waybills",
            "duitnow_penalty": "DuitNow Penalty",
            "ldr_penalty": "LDR Penalty",
            "fake_attempt_penalty": "Fake Attempt Penalty",
            "pickup_parcels": "Pickup Parcels",
            "pickup_payout": "Pickup Payout",
            "tier1_parcels": "Parcels 0-5kg",
            "tier2_parcels": "Parcels 5.01-10kg",
            "tier3_parcels": "Parcels 10.01-30kg",
            "tier4_parcels": "Parcels 30+kg"
        }).sort_values(by="Dispatcher Name", ascending=True)

        display_df = numeric_df.copy()
        display_df["Total Weight (kg)"] = display_df["Total Weight (kg)"].apply(lambda x: f"{x:.2f}")
        display_df["Avg Weight (kg)"] = display_df["Avg Weight (kg)"].apply(lambda x: f"{x:.2f}")
        display_df["Avg Rate per Parcel"] = display_df["Avg Rate per Parcel"].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_df["Dispatch Payout"] = display_df["Dispatch Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        display_df["Total Payout"] = display_df["Total Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        display_df["Penalty"] = display_df["Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        if "DuitNow Penalty" in display_df.columns:
            display_df["DuitNow Penalty"] = display_df["DuitNow Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        if "LDR Penalty" in display_df.columns:
            display_df["LDR Penalty"] = display_df["LDR Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        if "Fake Attempt Penalty" in display_df.columns:
            display_df["Fake Attempt Penalty"] = display_df["Fake Attempt Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        display_df["Pickup Payout"] = display_df["Pickup Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")

        # Keep Penalty Waybills and Penalty Parcels in numeric_df but remove from display_df
        if "Penalty Waybills" in display_df.columns:
            display_df = display_df.drop(columns=["Penalty Waybills"])
        if "Penalty Parcels" in display_df.columns:
            display_df = display_df.drop(columns=["Penalty Parcels"])

        total_payout = numeric_df["Total Payout"].sum()
        st.success(f"✅ Processed {len(df_unique)} unique parcels from {len(grouped)} dispatchers")

        # Calculate breakdown for info message
        total_dispatch_payout = numeric_df["Dispatch Payout"].sum()
        total_pickup_payout = numeric_df["Pickup Payout"].sum()
        total_penalty = numeric_df["Penalty"].sum()

        st.info(f"""
        💰 **Payout Breakdown:**
        - Dispatch Payout: {currency_symbol} {total_dispatch_payout:,.2f}
        + Pickup Payout: {currency_symbol} {total_pickup_payout:,.2f}
        - Penalties: {currency_symbol} {total_penalty:,.2f}
        **Total Payout: {currency_symbol} {total_payout:,.2f}**
        """)

        return display_df, numeric_df, total_payout

    @staticmethod
    def get_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
        """Get daily parcel delivery trend."""
        date_col = find_column(df, 'date')
        if date_col:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")
            daily_df = df_copy.groupby(df_copy[date_col].dt.date).size().reset_index(name='total_parcels')
            daily_df = daily_df.rename(columns={date_col: 'signature_date'})
            return daily_df.sort_values('signature_date')
        return pd.DataFrame()

    @staticmethod
    def get_daily_payout_trend(df: pd.DataFrame) -> pd.DataFrame:
        """Get daily payout trend."""
        date_col = find_column(df, 'date')
        if date_col:
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors="coerce")

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
        top_10 = numeric_df.head(10)
        return alt.Chart(top_10).mark_bar(color=ColorScheme.PRIMARY).encode(
            y=alt.Y('Dispatcher Name:N', title='Dispatcher', sort='-x'),
            x=alt.X('Parcels Delivered:Q', title='Parcels Delivered'),
            color=alt.Color('Parcels Delivered:Q', scale=alt.Scale(scheme='blues'), legend=None),
            tooltip=[
                'Dispatcher Name:N',
                'Parcels Delivered:Q',
                alt.Tooltip('Total Weight (kg):Q', format=',.2f'),
                alt.Tooltip('Total Payout:Q', format=',.2f')
            ]
        )

    @staticmethod
    def create_payout_distribution(numeric_df: pd.DataFrame) -> alt.Chart:
        """Create payout distribution donut chart."""
        return alt.Chart(numeric_df).mark_arc(innerRadius=50).encode(
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
            total_dispatch_payout = numeric_df["Dispatch Payout"].sum() if "Dispatch Payout" in numeric_df.columns else 0.0
            total_pickup_payout = numeric_df["Pickup Payout"].sum() if "Pickup Payout" in numeric_df.columns else 0.0
            total_pickup_parcels = int(numeric_df["Pickup Parcels"].sum()) if "Pickup Parcels" in numeric_df.columns else 0
            total_penalty = numeric_df["Penalty"].sum() if "Penalty" in numeric_df.columns else 0.0
            top_3 = display_df.head(3)

            table_columns = ["Dispatcher ID", "Dispatcher Name", "Parcels Delivered",
                           "Dispatch Payout", "Pickup Parcels", "Pickup Payout",
                           "Penalty", "Total Payout"]

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
                  margin-top: 24px; display: flex; gap: 12px; flex-wrap: wrap; justify-content: center;
                }}
                .chip {{
                  border: 1px solid var(--border); border-radius: 12px;
                  padding: 16px; background: var(--surface); min-width: 180px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                  transition: transform 0.2s, box-shadow 0.2s;
                }}
                .chip:hover {{
                  transform: translateY(-2px);
                  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                  border-color: var(--primary-light);
                }}
                .chip .label {{ color: var(--text-secondary); font-size: 12px; text-transform: uppercase;
                               letter-spacing: 0.5px; font-weight: 600; }}
                .chip .value {{ font-size: 18px; font-weight: 700; margin-top: 6px; color: var(--primary); }}
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
                        <div class="chip">
                            <div class="label">Total Dispatchers</div>
                            <div class="value">{total_dispatchers}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Total Parcels</div>
                            <div class="value">{total_parcels:,}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Dispatch Payout</div>
                            <div class="value">{currency_symbol} {total_dispatch_payout:,.2f}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Pickup Parcels</div>
                            <div class="value">{total_pickup_parcels:,}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Total Penalty</div>
                            <div class="value">-{currency_symbol} {total_penalty:,.2f}</div>
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
                            <tr><td>Total Dispatch Payout</td><td style="text-align:right;">{currency_symbol} {total_dispatch_payout:,.2f}</td></tr>
                            <tr><td>Pickup Payout</td><td style="text-align:right;">{currency_symbol} {total_pickup_payout:,.2f}</td></tr>
                            <tr><td>Total Penalty</td><td style="text-align:right;">-{currency_symbol} {total_penalty:,.2f}</td></tr>
                            <tr><td><strong>Total Payout</strong></td><td style="text-align:right;"><strong>{currency_symbol} {total_payout:,.2f}</strong></td></tr>
                        </table>
                    </div>
                    <div class="note">
                        Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} • JMR Management Dashboard v2.1<br>
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
        © 2025 Jemari Ventures. All rights reserved. | JMR Management Dashboard v2.1
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application."""
    st.set_page_config(page_title="JMR Management Dashboard", page_icon="📊", layout="wide")
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

    **📦 Pickup Payout:**
    - RM{pickup_payout_per_parcel:.2f} per parcel

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
                    st.warning(f"⚠️ {len(lost_dispatchers)} dispatcher(s) have no valid dates in '{selected_date_col}' and will show 0 parcels: {sorted(lost_dispatchers)}")
                    df = df_filtered
            else:
                df = df_filtered

            if df.empty:
                st.warning("No records found for the selected date range.")
                add_footer()
                return
        else:
            st.sidebar.warning("Selected date column has no valid date values; showing all data.")

    # Load penalty and pickup data
    penalty_data = DataSource.load_penalty_data(config)
    pickup_df = DataSource.load_pickup_data(config)

    # Filter all data by selected date range if a date column is selected
    if selected_date_col != "-- None --" and start_date is not None and end_date is not None:
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
            # Filter DuitNow penalty
            if 'duitnow' in penalty_data and penalty_data['duitnow'] is not None and not penalty_data['duitnow'].empty:
                duitnow_df = penalty_data['duitnow']
                duitnow_date_col = None
                # Try to find date columns in DuitNow data
                for col in duitnow_df.columns:
                    col_lower = str(col).lower()
                    if any(k in col_lower for k in ["date", "time", "delivery", "signature", "created_at", "updated_at"]):
                        duitnow_date_col = col
                        break

                if duitnow_date_col is not None:
                    duitnow_df[duitnow_date_col] = pd.to_datetime(duitnow_df[duitnow_date_col], errors="coerce")
                    initial_duitnow_count = len(duitnow_df)
                    duitnow_df = duitnow_df[
                        (duitnow_df[duitnow_date_col].dt.date >= start_date) &
                        (duitnow_df[duitnow_date_col].dt.date <= end_date)
                    ]
                    filtered_duitnow_count = len(duitnow_df)
                    penalty_data['duitnow'] = duitnow_df
                    if initial_duitnow_count != filtered_duitnow_count:
                        st.info(f"⚠️ Filtered DuitNow penalty: {initial_duitnow_count:,} → {filtered_duitnow_count:,} records")
                else:
                    st.warning("⚠️ DuitNow penalty table has no detectable date column; penalties are not filtered by date range.")

            # Filter LDR penalty
            if 'ldr' in penalty_data and penalty_data['ldr'] is not None and not penalty_data['ldr'].empty:
                ldr_df = penalty_data['ldr']
                ldr_date_col = None
                # Try to find date columns in LDR data
                for col in ldr_df.columns:
                    col_lower = str(col).lower()
                    if any(k in col_lower for k in ["date", "time", "delivery", "signature", "created_at", "updated_at"]):
                        ldr_date_col = col
                        break

                if ldr_date_col is not None:
                    ldr_df[ldr_date_col] = pd.to_datetime(ldr_df[ldr_date_col], errors="coerce")
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

        # Show summary of date filtering
        if selected_date_col != "-- None --" and start_date is not None and end_date is not None:
            st.success(f"✅ All data filtered by date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Calculate payouts
    currency = config.get("currency_symbol", "RM")

    display_df, numeric_df, total_payout = PayoutCalculator.calculate_payout(
        df, currency, penalty_data, pickup_df, pickup_payout_per_parcel
    )

    if numeric_df.empty:
        st.warning("No data after filtering.")
        add_footer()
        return

    # Get daily trends for forecasting
    daily_parcel_df = PayoutCalculator.get_daily_trend(df)

    # Calculate daily payout trend (requires additional processing)
    df_clean = DataProcessor.prepare_dataframe(df)
    df_clean = DataProcessor.remove_duplicates(df_clean)

    if 'weight' in df_clean.columns:
        tiers = Config.load().get("weight_tiers")
        df_clean['payout_rate'] = df_clean['weight'].apply(
            lambda w: PayoutCalculator.get_rate_by_weight(w, tiers)
        )
        df_clean['payout'] = df_clean['payout_rate']

        daily_payout_df = PayoutCalculator.get_daily_payout_trend(df_clean)

    # Metrics - Updated to include Pickup Payout per Parcel
    st.subheader("📈 Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)

    total_pickup_parcels = int(numeric_df["Pickup Parcels"].sum()) if "Pickup Parcels" in numeric_df.columns else 0
    total_pickup_payout = numeric_df["Pickup Payout"].sum() if "Pickup Payout" in numeric_df.columns else 0.0
    total_dispatch_payout = numeric_df["Dispatch Payout"].sum() if "Dispatch Payout" in numeric_df.columns else 0.0

    total_penalty = numeric_df["Penalty"].sum() if "Penalty" in numeric_df.columns else 0.0

    # Calculate penalty breakdown by type
    penalty_by_type = PayoutCalculator.calculate_penalty_by_type(penalty_data) if penalty_data else {'duitnow': 0.0, 'ldr': 0.0, 'fake_attempt': 0.0}

    col1.metric("Dispatchers", f"{len(display_df):,}")
    col2.metric("Delivery Parcels", f"{int(numeric_df['Parcels Delivered'].sum()):,}")
    col3.metric("Pickup Parcels", f"{total_pickup_parcels:,}")
    col4.metric("Total Payout", f"{currency} {total_payout:,.2f}")
    col5.metric("Dispatch Payout", f"{currency} {total_dispatch_payout:,.2f}")
    col6.metric("Pickup Payout", f"{currency} {total_pickup_payout:,.2f}")
    col7.metric("Pickup Rate", f"{currency} {pickup_payout_per_parcel:.2f}")
    col8.metric("Total Penalty", f"-{currency} {total_penalty:,.2f}")

    # Penalty breakdown by type
    st.markdown("#### ⚠️ Penalty Breakdown by Type")
    penalty_col1, penalty_col2, penalty_col3 = st.columns(3)
    with penalty_col1:
        st.metric("DuitNow Penalty", f"-{currency} {penalty_by_type['duitnow']:,.2f}")
    with penalty_col2:
        st.metric("LDR Penalty", f"-{currency} {penalty_by_type['ldr']:,.2f}")
    with penalty_col3:
        st.metric("Fake Attempt Penalty", f"-{currency} {penalty_by_type['fake_attempt']:,.2f}")

    # Charts
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

        st.markdown("##### 🏆 Top Performers")
        for _, row in numeric_df.head(5).iterrows():
            pickup_parcels = row.get('Pickup Parcels', 0)
            pickup_payout = row.get('Pickup Payout', 0.0)
            dispatch_payout = row.get('Dispatch Payout', 0.0)
            st.markdown(f"""
            <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin: 8px 0;">
                <div style="font-weight: 600; color: {ColorScheme.PRIMARY};">{row['Dispatcher Name']}</div>
                <div style="color: #64748b; font-size: 0.9rem;">
                    {row['Parcels Delivered']} parcels • {pickup_parcels} pickups • {currency}{dispatch_payout:,.2f} dispatch • {currency}{row['Total Payout']:,.2f} total
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Data table
    st.markdown("---")
    st.subheader("👥 Dispatcher Performance Details")

    # Reorder columns for better readability
    preferred_order = [
        "Dispatcher ID", "Dispatcher Name", "Parcels Delivered",
        "Dispatch Payout", "Pickup Parcels", "Pickup Payout",
        "Penalty", "DuitNow Penalty", "LDR Penalty", "Fake Attempt Penalty",
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

    # Forecasting Section - PARCEL FORECAST
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

                        # Combined forecast table
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
        st.info("Insufficient historical data for forecasting. Need at least 7 days of data.")

    # Penalty Details Section
    if 'Penalty Parcels' in numeric_df.columns and numeric_df['Penalty Parcels'].sum() > 0:
        st.markdown("---")
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

    st.markdown("---")
    st.subheader("📄 Invoice Generation")
    invoice_html = InvoiceGenerator.build_invoice_html(display_df, numeric_df, total_payout, currency, pickup_payout_per_parcel)
    st.download_button(
        label="📥 Download Invoice (HTML)",
        data=invoice_html.encode("utf-8"),
        file_name=f"invoice_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html"
    )

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
