import warnings
import urllib3
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)

import io
import re
import json
import os
from typing import Optional, Tuple, Dict, List
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
import altair as alt
import requests
from sqlalchemy import create_engine

# =============================================================================
# CONSTANTS
# =============================================================================

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
            "postgres_table": "dispatcher",
            "penalty_table": "penalty",
            "pickup_table": "pickup"
        },
        "database": {"table_name": "dispatcher"},
        "weight_tiers": [
            {"min": 0, "max": 5, "rate": 1.50},
            {"min": 5, "max": 10, "rate": 1.60},
            {"min": 10, "max": 30, "rate": 2.70},
            {"min": 30, "max": float('inf'), "rate": 4.00}
        ],
        "pickup_fee": 0.00,
        "pickup_payout_per_parcel": 1.50,
        "currency_symbol": "RM",
        "penalty_rate": 100.0
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
                        config["data_source"]["postgres_table"] = "dispatcher"
                    if "penalty_table" not in config.get("data_source", {}):
                        config["data_source"]["penalty_table"] = "penalty"
                    if "pickup_table" not in config.get("data_source", {}):
                        config["data_source"]["pickup_table"] = "pickup"
                    if "penalty_rate" not in config:
                        config["penalty_rate"] = 100.0
                    if "pickup_payout_per_parcel" not in config:
                        config["pickup_payout_per_parcel"] = 1.50
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
            if table_name == 'dispatcher' or table_name.endswith('dispatcher'):
                column_mapping = {
                    'waybill': 'Waybill Number',
                    'delivery_signature': 'Delivery Signature',
                    'dispatcher_id': 'Dispatcher ID',
                    'dispatcher_name': 'Dispatcher Name',
                    'billing_weight': 'Billing Weight',
                    'date_|_pick_up': 'Pick Up Date',
                    'pick_up_dp': 'Pick Up DP',
                    'pick_up_dispatcher_id': 'Pick Up Dispatcher ID',
                    'pick_up_dispatcher_name': 'Pick Up Dispatcher Name'
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
            elif table_name == 'penalty' or table_name.endswith('penalty'):
                column_mapping = {
                    'waybill': 'Waybill Number',
                    'responsible': 'RESPONSIBLE',
                    'penalty_amount_actual': 'Penalty Amount'
                }
            else:
                column_mapping = {}

            # Rename columns that exist in the dataframe
            rename_dict = {old: new for old, new in column_mapping.items() if old in df.columns}
            df = df.rename(columns=rename_dict)

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
            table_name = data_source.get("postgres_table", "dispatcher")
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
    def load_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        """Load penalty data from penalty table."""
        data_source = config["data_source"]
        engine = DataSource.get_postgres_engine()

        if not engine:
            return None

        try:
            penalty_table = data_source.get("penalty_table", "penalty")
            df = DataSource.read_postgres_table(engine, penalty_table)

            # Map penalty table columns
            penalty_mapping = {
                'waybill': 'Waybill Number',
                'responsible': 'RESPONSIBLE',
                'penalty_amount_actual': 'Penalty Amount'
            }

            rename_dict = {old: new for old, new in penalty_mapping.items() if old in df.columns}
            df = df.rename(columns=rename_dict)

            return df
        except Exception as exc:
            st.warning(f"Could not load penalty data from PostgreSQL table '{penalty_table}': {exc}")
            return None

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
        """Get per-parcel rate based on weight tiers."""
        if tiers is None:
            tiers = Config.load().get("weight_tiers", Config.DEFAULT_CONFIG["weight_tiers"])

        w = 0.0 if pd.isna(weight) else float(weight)

        for idx, tier in enumerate(sorted(tiers, key=lambda t: t['min'])):
            tier_max = tier.get('max', float('inf'))
            is_first_tier = idx == 0
            is_last_tier = idx == len(tiers) - 1

            if is_first_tier:
                if w <= tier_max:
                    return tier['rate']
                continue

            if is_last_tier:
                if w > tier['min']:
                    return tier['rate']
                continue

            if (w > tier['min']) and (w <= tier_max):
                return tier['rate']

        return tiers[-1]['rate']

    @staticmethod
    def calculate_penalty(dispatcher_id: str, penalty_df: Optional[pd.DataFrame], penalty_rate: float = 100.0) -> Tuple[float, int, List[str]]:
        """
        Calculate penalty for a dispatcher based on penalty data.

        Returns: (penalty_amount, penalty_count, waybill_numbers)
        """
        if penalty_df is None or penalty_df.empty or not dispatcher_id:
            return 0.0, 0, []

        # Find 'RESPONSIBLE' column (mapped from 'responsible')
        responsible_col = None
        for col in penalty_df.columns:
            if col.strip().replace(' ', '').upper() == "RESPONSIBLE":
                responsible_col = col
                break
        if responsible_col is None:
            return 0.0, 0, []

        # Find waybill column
        waybill_col = None
        for col in penalty_df.columns:
            if 'waybill' in col.lower() or 'Waybill Number' in col:
                waybill_col = col
                break

        dispatcher_id_clean = str(dispatcher_id).strip().lower()
        responsible_series = penalty_df[responsible_col].astype(str).str.strip().str.lower()

        penalty_records = penalty_df[responsible_series == dispatcher_id_clean]
        penalty_count = len(penalty_records)

        # Use actual penalty amount if available
        penalty_amount = 0.0
        if 'Penalty Amount' in penalty_df.columns:
            penalty_amount = penalty_records['Penalty Amount'].sum()
        else:
            penalty_amount = penalty_count * penalty_rate

        waybill_numbers = []
        if waybill_col and penalty_count > 0:
            waybill_series = penalty_records[waybill_col].astype(str).str.strip()
            waybill_numbers = [wb for wb in waybill_series if wb and wb.lower() != 'nan']

        return float(penalty_amount), penalty_count, waybill_numbers

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
    def calculate_payout(df: pd.DataFrame, currency_symbol: str, penalty_df: Optional[pd.DataFrame] = None,
                        penalty_rate: float = 100.0, pickup_fee: float = 0.0,
                        pickup_df: Optional[pd.DataFrame] = None, pickup_payout_per_parcel: float = 1.50) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
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

        grouped['avg_rate'] = grouped['dispatch_payout'] / grouped['parcel_count']

        # Calculate penalties
        grouped['penalty_amount'] = 0.0
        grouped['penalty_count'] = 0
        grouped['penalty_waybills'] = ''

        if penalty_df is not None and not penalty_df.empty:
            for i, row in grouped.iterrows():
                dispatcher_id = row['dispatcher_id']
                penalty_amount, penalty_count, penalty_waybills = PayoutCalculator.calculate_penalty(
                    str(dispatcher_id), penalty_df, penalty_rate
                )
                grouped.at[i, 'penalty_amount'] = penalty_amount
                grouped.at[i, 'penalty_count'] = penalty_count
                grouped.at[i, 'penalty_waybills'] = ', '.join(penalty_waybills) if penalty_waybills else ''

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
            "pickup_parcels": "Pickup Parcels",
            "pickup_payout": "Pickup Payout",
            "tier1_parcels": "Parcels 0-5kg",
            "tier2_parcels": "Parcels 5.01-10kg",
            "tier3_parcels": "Parcels 10.01-30kg",
            "tier4_parcels": "Parcels 30+kg"
        }).sort_values(by="Total Payout", ascending=False)

        display_df = numeric_df.copy()
        display_df["Total Weight (kg)"] = display_df["Total Weight (kg)"].apply(lambda x: f"{x:.2f}")
        display_df["Avg Weight (kg)"] = display_df["Avg Weight (kg)"].apply(lambda x: f"{x:.2f}")
        display_df["Avg Rate per Parcel"] = display_df["Avg Rate per Parcel"].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_df["Dispatch Payout"] = display_df["Dispatch Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        display_df["Total Payout"] = display_df["Total Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")
        display_df["Penalty"] = display_df["Penalty"].apply(lambda x: f"-{currency_symbol}{x:,.2f}" if x > 0 else f"{currency_symbol}0.00")
        display_df["Pickup Payout"] = display_df["Pickup Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")

        if "Penalty Waybills" in display_df.columns:
            display_df = display_df.drop(columns=["Penalty Waybills"])
        if "Penalty Parcels" in display_df.columns:
            display_df = display_df.drop(columns=["Penalty Parcels"])

        total_payout = numeric_df["Total Payout"].sum() + pickup_fee
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
        + Pickup Fee: {currency_symbol} {pickup_fee:,.2f}
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
                          penalty_rate: float = 100.0, pickup_fee: float = 0.0) -> str:
        """Build management summary invoice HTML with original layout."""
        try:
            total_parcels = int(numeric_df["Parcels Delivered"].sum())
            total_dispatchers = len(numeric_df)
            total_weight = numeric_df["Total Weight (kg)"].sum()
            total_penalty = numeric_df["Penalty"].sum()
            total_dispatch_payout = numeric_df["Dispatch Payout"].sum() if "Dispatch Payout" in numeric_df.columns else 0.0
            total_pickup_payout = numeric_df["Pickup Payout"].sum() if "Pickup Payout" in numeric_df.columns else 0.0
            total_pickup_parcels = int(numeric_df["Pickup Parcels"].sum()) if "Pickup Parcels" in numeric_df.columns else 0
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
                            <div class="label">Pickup Payout</div>
                            <div class="value">{currency_symbol} {total_pickup_payout:,.2f}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Total Penalty</div>
                            <div class="value">-{currency_symbol} {total_penalty:,.2f}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Pickup Fee</div>
                            <div class="value">{currency_symbol} {pickup_fee:,.2f}</div>
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
                            <tr><td>Pickup Fee</td><td style="text-align:right;">{currency_symbol} {pickup_fee:,.2f}</td></tr>
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
    st.sidebar.info(f"📊 Table: `{config['data_source'].get('postgres_table', 'dispatcher')}`")
    st.sidebar.info(f"⚠️ Penalty Table: `{config['data_source'].get('penalty_table', 'penalty')}`")
    st.sidebar.info(f"📦 Pickup Table: `{config['data_source'].get('pickup_table', 'pickup')}`")

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

    # Add configuration for penalty rate
    penalty_rate = st.sidebar.number_input(
        "Penalty Rate per Incident",
        min_value=0.0,
        max_value=500.0,
        value=config.get("penalty_rate", 100.0),
        step=10.0,
        help="Penalty amount per penalty incident"
    )

    # Update config with new value
    if penalty_rate != config.get("penalty_rate", 100.0):
        config["penalty_rate"] = penalty_rate
        Config.save(config)

    # Add configuration for pickup fee
    pickup_fee = st.sidebar.number_input(
        "Pickup Fee",
        min_value=0.0,
        max_value=1000.0,
        value=config.get("pickup_fee", 0.0),
        step=10.0,
        help="Fixed pickup fee"
    )

    # Update config with new value
    if pickup_fee != config.get("pickup_fee", 0.0):
        config["pickup_fee"] = pickup_fee
        Config.save(config)

    st.sidebar.info(f"""
    **💰 Weight-Based Payout:**
    - 0-5kg: RM1.50
    - 5-10kg: RM1.60
    - 10-30kg: RM2.70
    - 30kg+: RM4.00

    **📦 Pickup Payout:**
    - RM{pickup_payout_per_parcel:.2f} per parcel

    **⚠️ Penalty Rate:**
    - RM{penalty_rate:.2f} per incident

    **🚚 Pickup Fee:**
    - RM{pickup_fee:.2f}
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

    if selected_date_col != "-- None --":
        df[selected_date_col] = pd.to_datetime(df[selected_date_col], errors="coerce")
        valid_dates = df[selected_date_col].dropna()
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

            df = df[
                (df[selected_date_col].dt.date >= start_date) &
                (df[selected_date_col].dt.date <= end_date)
            ]

            if df.empty:
                st.warning("No records found for the selected date range.")
                add_footer()
                return
        else:
            st.sidebar.warning("Selected date column has no valid date values; showing all data.")

    # Load penalty and pickup data
    penalty_df = DataSource.load_penalty_data(config)
    pickup_df = DataSource.load_pickup_data(config)

    # Filter penalty_df by selected month/date range (same as main df)
    if penalty_df is not None and not penalty_df.empty and selected_date_col != "-- None --":
        penalty_date_col = find_column(penalty_df, "date")
        if penalty_date_col is None:
            for col in penalty_df.columns:
                if any(k in str(col).lower() for k in ["date", "created", "signature", "scan"]):
                    penalty_date_col = col
                    break

        if penalty_date_col is not None:
            penalty_df[penalty_date_col] = pd.to_datetime(penalty_df[penalty_date_col], errors="coerce")
            penalty_df = penalty_df[
                (penalty_df[penalty_date_col].dt.date >= start_date) &
                (penalty_df[penalty_date_col].dt.date <= end_date)
            ]
        else:
            st.warning("Penalty table has no detectable date column; penalties are not filtered by month.")

    # Filter pickup_df by selected month/date range
    if pickup_df is not None and not pickup_df.empty and selected_date_col != "-- None --":
        pickup_date_col = None
        for col in pickup_df.columns:
            if any(k in str(col).lower() for k in ["date", "pick_up", "pickup", "signature"]):
                pickup_date_col = col
                break

        if pickup_date_col is not None:
            pickup_df[pickup_date_col] = pd.to_datetime(pickup_df[pickup_date_col], errors="coerce")
            pickup_df = pickup_df[
                (pickup_df[pickup_date_col].dt.date >= start_date) &
                (pickup_df[pickup_date_col].dt.date <= end_date)
            ]
        else:
            st.warning("Pickup table has no detectable date column; pickup parcels are not filtered by month.")

    # Calculate payouts
    currency = config.get("currency_symbol", "RM")

    display_df, numeric_df, total_payout = PayoutCalculator.calculate_payout(
        df, currency, penalty_df, penalty_rate, pickup_fee, pickup_df, pickup_payout_per_parcel
    )

    if numeric_df.empty:
        st.warning("No data after filtering.")
        add_footer()
        return

    # Metrics
    st.subheader("📈 Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)

    total_pickup_parcels = int(numeric_df["Pickup Parcels"].sum()) if "Pickup Parcels" in numeric_df.columns else 0
    total_pickup_payout = numeric_df["Pickup Payout"].sum() if "Pickup Payout" in numeric_df.columns else 0.0
    total_dispatch_payout = numeric_df["Dispatch Payout"].sum() if "Dispatch Payout" in numeric_df.columns else 0.0

    col1.metric("Dispatchers", f"{len(display_df):,}")
    col2.metric("Delivery Parcels", f"{int(numeric_df['Parcels Delivered'].sum()):,}")
    col3.metric("Pickup Parcels", f"{total_pickup_parcels:,}")
    col4.metric("Total Payout", f"{currency} {total_payout:,.2f}")
    col5.metric("Dispatch Payout", f"{currency} {total_dispatch_payout:,.2f}")
    col6.metric("Pickup Payout", f"{currency} {total_pickup_payout:,.2f}")
    col7.metric("Total Penalty", f"-{currency} {numeric_df['Penalty'].sum():,.2f}")
    col8.metric("Pickup Fee", f"{currency} {pickup_fee:,.2f}")

    # Charts
    st.markdown("---")
    st.subheader("📊 Performance Analytics")
    daily_df = PayoutCalculator.get_daily_trend(df)
    charts = DataVisualizer.create_all_charts(daily_df, numeric_df)

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

    # Data table - Reorder columns to show Dispatch Payout first
    st.markdown("---")
    st.subheader("👥 Dispatcher Performance Details")

    # Reorder columns for better readability
    preferred_order = [
        "Dispatcher ID", "Dispatcher Name", "Parcels Delivered",
        "Dispatch Payout", "Pickup Parcels", "Pickup Payout",
        "Penalty", "Total Payout", "Total Weight (kg)", "Avg Weight (kg)",
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

    if 'Penalty Parcels' in numeric_df.columns and numeric_df['Penalty Parcels'].sum() > 0:
        st.markdown("---")
        st.subheader("⚠️ Penalty Details")

        penalty_rows = []
        penalty_dispatchers = numeric_df[numeric_df['Penalty Parcels'] > 0]
        for _, row in penalty_dispatchers.iterrows():
            dispatcher_id = row['Dispatcher ID']
            dispatcher_name = row['Dispatcher Name']
            waybills_str = row.get('Penalty Waybills', '')
            waybills_list = [w.strip() for w in waybills_str.split(',') if w.strip()] if waybills_str else []
            for waybill_number in waybills_list:
                penalty_rows.append({
                    'Dispatcher ID': dispatcher_id,
                    'Dispatcher Name': dispatcher_name,
                    'Waybill Number': waybill_number,
                    'Penalty Amount': f"{currency}{penalty_rate:.2f}"
                })

        if penalty_rows:
            penalty_table = pd.DataFrame(penalty_rows)
            st.dataframe(penalty_table, use_container_width=True, hide_index=True)
        else:
            st.info("No penalty waybill numbers available")

    # Pickup details
    if 'Pickup Parcels' in numeric_df.columns and numeric_df['Pickup Parcels'].sum() > 0:
        st.markdown("---")
        st.subheader("📦 Pickup Details")

        pickup_details = numeric_df[numeric_df['Pickup Parcels'] > 0][['Dispatcher ID', 'Dispatcher Name', 'Pickup Parcels', 'Pickup Payout']]
        if not pickup_details.empty:
            st.dataframe(pickup_details, use_container_width=True, hide_index=True)

            # Show summary
            st.markdown(f"""
            <div style="background: {ColorScheme.BACKGROUND}; padding: 1rem; border-radius: 8px; border: 1px solid {ColorScheme.BORDER}; margin-top: 1rem;">
                <h4 style="color: {ColorScheme.PRIMARY}; margin: 0 0 0.5rem 0;">📊 Pickup Summary</h4>
                <p style="margin: 0.25rem 0;">• Total Pickup Parcels: <strong>{total_pickup_parcels:,}</strong></p>
                <p style="margin: 0.25rem 0;">• Total Pickup Payout: <strong>{currency} {total_pickup_payout:,.2f}</strong></p>
                <p style="margin: 0.25rem 0;">• Payout per Parcel: <strong>{currency} {pickup_payout_per_parcel:.2f}</strong></p>
                <p style="margin: 0.25rem 0;">• Dispatchers with Pickups: <strong>{len(pickup_details)}</strong></p>
            </div>
            """, unsafe_allow_html=True)

    # Dispatch payout summary
    if 'Dispatch Payout' in numeric_df.columns:
        st.markdown("---")
        st.subheader("🚚 Dispatch Payout Summary")

        dispatch_summary = numeric_df[['Dispatcher ID', 'Dispatcher Name', 'Parcels Delivered', 'Dispatch Payout']]
        if not dispatch_summary.empty:
            # Add a summary row
            total_row = pd.DataFrame({
                'Dispatcher ID': ['TOTAL'],
                'Dispatcher Name': [''],
                'Parcels Delivered': [int(numeric_df['Parcels Delivered'].sum())],
                'Dispatch Payout': [total_dispatch_payout]
            })
            dispatch_summary_with_total = pd.concat([dispatch_summary, total_row], ignore_index=True)

            # Format the display
            dispatch_display = dispatch_summary_with_total.copy()
            dispatch_display['Dispatch Payout'] = dispatch_display['Dispatch Payout'].apply(
                lambda x: f"{currency}{x:,.2f}" if isinstance(x, (int, float)) else x
            )

            st.dataframe(dispatch_display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("📄 Invoice Generation")
    invoice_html = InvoiceGenerator.build_invoice_html(display_df, numeric_df, total_payout, currency, penalty_rate, pickup_fee)
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
