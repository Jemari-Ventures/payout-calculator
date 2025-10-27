# management_bi.py
import warnings
import urllib3
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)

import io
import re
import json
import os
from datetime import datetime
from typing import Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

import pandas as pd
import streamlit as st
import altair as alt
import requests

# =========================
# ColorScheme (unchanged)
# =========================
class ColorScheme:
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

# =========================
# Config (unchanged except default data_source)
# =========================
class Config:
    CONFIG_FILE = "config.json"
    DEFAULT_CONFIG = {
        "data_source": {
            "type": "gsheet",  # or "postgres"
            "gsheet_url": "",  # public CSV-exportable URL preferred
            "sheet_name": None,
            "db_url": ""
        },
        "tiers": [
            {"Tier": "Tier 3", "Min Parcels": 0, "Max Parcels": 60, "Rate (RM)": 0.95},
            {"Tier": "Tier 2", "Min Parcels": 61, "Max Parcels": 120, "Rate (RM)": 1.00},
            {"Tier": "Tier 1", "Min Parcels": 121, "Max Parcels": None, "Rate (RM)": 1.10},
        ],
        "currency_symbol": "RM"
    }

    @classmethod
    def load(cls):
        if os.path.exists(cls.CONFIG_FILE):
            try:
                with open(cls.CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading config: {e}")
                return cls.DEFAULT_CONFIG
        else:
            cls.save(cls.DEFAULT_CONFIG)
            return cls.DEFAULT_CONFIG

    @classmethod
    def save(cls, config):
        try:
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving config: {e}")
            return False

# =========================
# DataSource (supports gsheet + postgres)
# =========================
class DataSource:
    def __init__(self, use_gsheet: bool = False):
        self.use_gsheet = use_gsheet

    @st.cache_data
    def load_data(_self):
        """Load data from Google Sheets or PostgreSQL."""
        if _self.use_gsheet:
            st.info("📊 Loading data from Google Sheets...")
            df = _self._load_from_gsheet()
        else:
            st.info("🗄️ Loading data from PostgreSQL...")
            df = _self._load_from_postgres()
        return df

    # ========== GOOGLE SHEETS ==========
    def _load_from_gsheet(self):
        # Example: simple CSV export link or gspread usage
        sheet_url = st.secrets.get("GSHEET_URL", None)
        if not sheet_url:
            st.error("❌ Missing GSHEET_URL in secrets.toml")
            return pd.DataFrame()
        df = pd.read_csv(sheet_url)
        return df

    # ========== POSTGRES ==========
    def _load_from_postgres(self):
        """Load data from PostgreSQL using Streamlit secrets."""
        try:
            host = st.secrets["DB_HOST"]
            port = st.secrets["DB_PORT"]
            name = st.secrets["DB_NAME"]
            user = st.secrets["DB_USER"]
            password = st.secrets["DB_PASSWORD"]

            # Build connection string
            engine = create_engine(
                f"postgresql://{user}:{password}@{host}:{port}/{name}",
                pool_pre_ping=True
            )

            query = text("SELECT * FROM dispatcher_raw_data")  # Adjust your table name
            df = pd.read_sql(query, engine)
            return df

        except Exception as e:
            st.error(f"❌ Database load failed: {e}")
            return pd.DataFrame()

# =========================
# Helpers: normalization and cleaning
# =========================
def standardize_column_map(cols):
    """Return mapping from original column -> canonical name (case-insensitive)."""
    canonical = {
        "waybill": ["waybill", "waybill number", "awb", "waybill_no"],
        "signature_date": ["signature_date"],
        "dispatcher_id": ["dispatcher id", "dispatcher_id", "dispatcherid", "driver_id"],
        "dispatcher_name": ["dispatcher name", "dispatcher_name", "name", "driver_name"],
        "arriving_dp": ["arriving dp", "arriving_dp", "dp", "arrival_dp"],
        "pickup_dp": ["pick up", "pickup", "pickup_dp", "pick_up_dp"],
        "cod_amount": ["cod", "cod_amount", "cod_value"]
    }
    # normalize incoming columns
    col_map = {}
    lowered = {c: c for c in cols}
    for orig in cols:
        label = orig.strip().lower()
        found = False
        for canon, variants in canonical.items():
            if label in variants or any(v in label for v in variants):
                col_map[orig] = canon
                found = True
                break
        if not found:
            # no mapping -> keep cleaned label as-is
            col_map[orig] = orig
    return col_map

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and normalize dispatcher payout data loaded from Postgres or GSheet."""

    # 1️⃣ Normalize column names early
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # 2️⃣ Detect and drop duplicate columns *after normalization*
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        st.warning(f"⚠️ Dropping duplicate columns: {dupes}")
        df = df.loc[:, ~df.columns.duplicated()]

    # 3️⃣ Parse known date/time columns safely
    date_cols = [
        "pick_up_date",
        "signature_date",
        "arrival_time",
        "last_arrival_time",
    ]
    for col in date_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception as e:
                st.warning(f"⚠️ Could not parse column {col}: {e}")

    # 4️⃣ Parse numeric fields
    for col in ["billing_weight", "cod"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5️⃣ Drop invalid or duplicate waybills
    if "waybill" in df.columns:
        df = df.dropna(subset=["waybill"]).drop_duplicates(subset=["waybill"])

    return df


def clean_dispatcher_name(name: str) -> str:
    prefixes = ['JMR', 'ECP', 'AF', 'PEN', 'KUL', 'JHR']
    cleaned_name = str(name).strip()
    for prefix in prefixes:
        if cleaned_name.upper().startswith(prefix):
            cleaned_name = cleaned_name[len(prefix):].strip()
            cleaned_name = cleaned_name.lstrip(' -')
            break
    return cleaned_name

# =========================
# PayoutCalculator (small tweak to use canonical cols)
# =========================
class PayoutCalculator:
    @staticmethod
    def calculate_flat_rate(df: pd.DataFrame, payout_rate: float, currency_symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        df_clean = df.copy()

        # If canonical names (waybill, dispatcher_id, dispatcher_name) exist, use them.
        # Otherwise try to rename from legacy column names (handled by normalize_df)
        if 'dispatcher_name' in df_clean.columns:
            df_clean['dispatcher_name'] = df_clean['dispatcher_name'].apply(clean_dispatcher_name)
        else:
            df_clean['dispatcher_name'] = 'Unknown'

        if 'dispatcher_id' not in df_clean.columns:
            df_clean['dispatcher_id'] = df_clean.get('dispatcher_id', 'Unknown')

        # If waybill doesn't exist, try other columns
        if 'waybill' not in df_clean.columns:
            # fall back to any first column that looks like an id
            df_clean['waybill'] = df_clean.iloc[:, 0].astype(str)

        grouped = (
            df_clean.groupby(["dispatcher_id", "dispatcher_name"])
            .agg(parcel_count=("waybill", "nunique"))
            .reset_index()
        )
        grouped["payout_rate"] = payout_rate
        grouped["total_payout"] = grouped["parcel_count"] * payout_rate

        numeric_df = grouped.rename(columns={
            "dispatcher_id": "Dispatcher ID",
            "dispatcher_name": "Dispatcher Name",
            "parcel_count": "Parcels Delivered",
            "payout_rate": "Rate per Parcel",
            "total_payout": "Total Payout"
        })

        display_df = numeric_df.copy()
        display_df["Rate per Parcel"] = display_df["Rate per Parcel"].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_df["Total Payout"] = display_df["Total Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")

        total_payout = grouped["total_payout"].sum()
        numeric_df = numeric_df.sort_values(by="Parcels Delivered", ascending=False)
        display_df = display_df.sort_values(by="Parcels Delivered", ascending=False)

        return display_df, numeric_df, total_payout

    @staticmethod
    def get_daily_trend_data(df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        if "signature_date" in df_clean.columns:
            if not pd.api.types.is_datetime64_any_dtype(df_clean["signature_date"]):
                df_clean["signature_date"] = pd.to_datetime(df_clean["signature_date"], errors="coerce")
            daily_df = (
                df_clean.dropna(subset=["signature_date"])
                .groupby(df_clean["signature_date"].dt.date)
                .size()
                .reset_index(name='total_parcels')
            )
            daily_df = daily_df.rename(columns={"signature_date": "signature_date"})
            daily_df = daily_df.rename(columns={daily_df.columns[0]: "signature_date"})
            # convert signature_date back to datetime for Altair
            daily_df["signature_date"] = pd.to_datetime(daily_df["signature_date"])
            return daily_df.sort_values("signature_date")
        return pd.DataFrame()

# =========================
# DataVisualizer (unchanged logic)
# =========================
class DataVisualizer:
    @staticmethod
    def create_management_charts(daily_df: pd.DataFrame, numeric_df: pd.DataFrame, currency_symbol: str):
        charts = {}
        if not daily_df.empty:
            daily_trend = alt.Chart(daily_df).mark_area(
                line={'color': ColorScheme.PRIMARY, 'width': 2},
                color=ColorScheme.PRIMARY_LIGHT,
                opacity=0.6
            ).encode(
                x=alt.X('signature_date:T', title='Date', axis=alt.Axis(format='%b %d')),
                y=alt.Y('total_parcels:Q', title='Parcels Delivered'),
                tooltip=[alt.Tooltip('signature_date:T', title='Date'), alt.Tooltip('total_parcels:Q', title='Parcels')]
            ).properties(title='Daily Parcel Delivery Trend', height=300)
            charts['daily_trend'] = daily_trend

        if not numeric_df.empty:
            top_10 = numeric_df.head(10)
            performers_chart = alt.Chart(top_10).mark_bar().encode(
                y=alt.Y('Dispatcher Name:N', sort='-x', title='Dispatcher'),
                x=alt.X('Parcels Delivered:Q', title='Parcels Delivered'),
                color=alt.Color('Parcels Delivered:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=['Dispatcher Name:N', 'Parcels Delivered:Q', 'Total Payout:Q']
            ).properties(title='Top 10 Performers (Parcels)', height=400)
            charts['performers'] = performers_chart

            payout_chart = alt.Chart(numeric_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Total Payout", type="quantitative", title="Payout Amount"),
                color=alt.Color(field="Dispatcher Name", type="nominal", scale=alt.Scale(range=ColorScheme.CHART_COLORS)),
                tooltip=['Dispatcher Name:N', 'Total Payout:Q']
            ).properties(title='Payout Distribution', height=300, width=400)
            charts['payout_dist'] = payout_chart

        return charts

# =========================
# InvoiceGenerator (unchanged)
# =========================
class InvoiceGenerator:
    @staticmethod
    def build_invoice_html(display_df: pd.DataFrame, total_payout: float, payout_rate: float, currency_symbol: str) -> str:
        try:
            total_parcels = display_df["Parcels Delivered"].sum() if len(display_df) else 0
            total_dispatchers = len(display_df)
            avg_parcels = display_df["Parcels Delivered"].mean() if len(display_df) else 0
            avg_payout = total_payout / total_dispatchers if total_dispatchers > 0 else 0
            top_3 = display_df.head(3).copy()

            # same HTML generation as before (kept concise here)
            html_content = f"<html><body><h1>Invoice — Total {currency_symbol} {total_payout:,.2f}</h1></body></html>"
            return html_content
        except Exception as e:
            st.error(f"Error generating invoice: {e}")
            return f"<html><body><h1>Error: {e}</h1></body></html>"

# =========================
# UI Helpers
# =========================
def apply_custom_styles():
    st.markdown(f"""
    <style>
    .stApp {{ background-color: {ColorScheme.BACKGROUND}; }}
    .stButton>button {{ background-color: {ColorScheme.PRIMARY}; color:white; border-radius:8px; }}
    </style>
    """, unsafe_allow_html=True)

def add_footer():
    st.markdown(f"""
    <div style="margin-top:2rem; padding:1rem; border-radius:8px; background:linear-gradient(135deg,{ColorScheme.PRIMARY}, {ColorScheme.PRIMARY_LIGHT}); color:white; text-align:center;">
        © {datetime.now().year} Jemari Ventures — JMR Management Dashboard
    </div>
    """, unsafe_allow_html=True)

# =========================
# BI Features
# =========================
def compute_tiers(numeric_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    tiers = config.get("tiers", [])
    def assign_tier(parcels):
        for t in tiers:
            mn = t.get("Min Parcels", 0) or 0
            mx = t.get("Max Parcels", None)
            if (parcels >= mn) and (mx is None or parcels <= mx):
                return t["Tier"]
        return "Unassigned"
    df = numeric_df.copy()
    if "Parcels Delivered" in df.columns:
        df["Tier"] = df["Parcels Delivered"].apply(assign_tier)
    return df

def detect_outliers_iqr(numeric_df: pd.DataFrame) -> pd.DataFrame:
    s = numeric_df["Parcels Delivered"]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return numeric_df[(s < lower) | (s > upper)].copy()

# =========================
# Main app
# =========================
def main():
    st.set_page_config(page_title="JMR Management Dashboard", page_icon="📊", layout="wide")
    apply_custom_styles()
    data_source = DataSource(use_gsheet=False)
    df_raw = data_source.load_data()

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {ColorScheme.PRIMARY} 0%, {ColorScheme.PRIMARY_LIGHT} 100%); padding:1.6rem; border-radius:8px; color:white;">
        <h1 style="margin:0">📊 JMR Management Dashboard — BI</h1>
        <p style="margin:0.2rem 0 0 0; color:rgba(255,255,255,0.9);">Dispatcher performance, payouts & BI insights</p>
    </div>
    """, unsafe_allow_html=True)

    config = Config.load()
    if not config:
        st.error("Failed to load config.json")
        add_footer()
        return

    # Sidebar: choose data source
    st.sidebar.header("⚙️ Data & Configuration")
    ds_choice = st.sidebar.radio("Data source", options=["Auto (Config)", "Postgres (secret)", "Google Sheet (URL)"])
    if ds_choice == "Auto (Config)":
        data_source = DataSource(config)
    elif ds_choice == "Postgres (secret)":
        # allow user to override with secret/env input
        db_url_input = st.sidebar.text_input("Postgres URL", value=st.secrets.get("db_url", "") or os.getenv("DATABASE_URL", ""))
        cfg = config.copy()
        cfg["data_source"] = {"type": "postgres", "db_url": db_url_input}
        data_source = DataSource(cfg)
    else:
        gsheet_input = st.sidebar.text_input("Google Sheet URL (public/exportable)", value=config.get("data_source", {}).get("gsheet_url", ""))
        cfg = config.copy()
        cfg["data_source"] = {"type": "gsheet", "gsheet_url": gsheet_input}
        data_source = DataSource(cfg)

    # Payout rate in sidebar
    payout_rate = st.sidebar.number_input("Payout Rate (RM/parcel)", min_value=0.0, value=1.5, step=0.1)

    # Fetch data
    with st.spinner("🔄 Loading data..."):
        df_raw = data_source.load_data()

    if df_raw is None or df_raw.empty:
        st.error("No data loaded. Check your data source settings.")
        add_footer()
        return

    # normalize columns
    df = normalize_df(df_raw)

    # detect date column and provide date filter
    st.sidebar.header("📅 Date Filter")
    if "signature_date" in df.columns and df["signature_date"].notna().any():
        df["signature_date"] = pd.to_datetime(df["signature_date"], errors="coerce")
        min_date = df["signature_date"].min().date()
        max_date = df["signature_date"].max().date()
        date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df["signature_date"].dt.date >= start_date) & (df["signature_date"].dt.date <= end_date)]
    else:
        # fallback: try to find any datetime-like column
        possible = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if possible:
            st.sidebar.info(f"Found potential date columns: {possible}. Consider renaming to 'signature_date' for best results.")
        else:
            st.sidebar.info("No date column detected; date filtering skipped.")

    # ============
    # SUMMARY METRICS
    # ============
    st.subheader("📈 Overview")

    currency_symbol = config.get("currency_symbol", "RM")
    display_df, numeric_df, total_payout = PayoutCalculator.calculate_flat_rate(df, payout_rate, currency_symbol)

    total_dispatchers = len(display_df)
    total_parcels = numeric_df["Parcels Delivered"].sum() if len(numeric_df) else 0
    avg_parcels = numeric_df["Parcels Delivered"].mean() if len(numeric_df) else 0
    avg_payout = total_payout / total_dispatchers if total_dispatchers > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Dispatchers", f"{total_dispatchers:,}")
    c2.metric("Total Parcels", f"{total_parcels:,}")
    c3.metric("Total Payout", f"{currency_symbol} {total_payout:,.2f}")
    c4.metric("Avg Parcels/Dispatcher", f"{avg_parcels:.1f}")

    # ============
    # BI Tab (charts, tiers, outliers)
    # ============
    st.markdown("---")
    st.subheader("📊 BI Analytics")

    daily_df = PayoutCalculator.get_daily_trend_data(df)

    charts = DataVisualizer.create_management_charts(daily_df, numeric_df, currency_symbol)

    left, right = st.columns([2, 1])

    with left:
        if 'daily_trend' in charts:
            st.altair_chart(charts['daily_trend'], use_container_width=True)
        else:
            st.info("No daily trend available (no signature_date).")

        st.markdown("### 🏆 Top Performers")
        st.dataframe(numeric_df.head(10), use_container_width=True)

    with right:
        # Tier analysis
        tiered = compute_tiers(numeric_df, config)
        if not tiered.empty:
            tier_counts = tiered.groupby("Tier").size().reset_index(name="count")
            st.markdown("### 🎯 Tier Impact")
            if not tier_counts.empty:
                pie = alt.Chart(tier_counts).mark_arc(innerRadius=50).encode(
                    theta=alt.Theta(field="count", type="quantitative"),
                    color=alt.Color(field="Tier", type="nominal", scale=alt.Scale(range=ColorScheme.CHART_COLORS)),
                    tooltip=['Tier:N', 'count:Q']
                )
                st.altair_chart(pie, use_container_width=True)
                st.table(tier_counts.set_index("Tier"))
        else:
            st.info("No tier distribution (not enough data).")

        # Outliers
        outliers = detect_outliers_iqr(numeric_df) if len(numeric_df) else pd.DataFrame()
        st.markdown("### ⚠️ Outliers (IQR)")
        if not outliers.empty:
            with st.expander(f"Show {len(outliers)} outlier(s)"):
                st.dataframe(outliers, use_container_width=True)
        else:
            st.write("No outliers detected.")

    # ============
    # Dispatcher details & invoice
    # ============
    st.markdown("---")
    st.subheader("👥 Dispatcher Performance Details")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("#### 📄 Invoice Generation")
    invoice_html = InvoiceGenerator.build_invoice_html(display_df, total_payout, payout_rate, currency_symbol)
    st.download_button("📥 Download Invoice (HTML)", data=invoice_html.encode("utf-8"),
                       file_name=f"invoice_{datetime.now().strftime('%Y%m%d')}.html", mime="text/html")

    # Export buttons
    st.markdown("---")
    st.subheader("📥 Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download Summary CSV", data=numeric_df.to_csv(index=False).encode("utf-8"),
                           file_name=f"management_summary_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    with col2:
        st.download_button("Download Raw CSV", data=df.to_csv(index=False).encode("utf-8"),
                           file_name=f"raw_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")

    add_footer()

if __name__ == "__main__":
    main()
