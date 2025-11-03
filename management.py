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
    'date': ['Delivery Signature', 'Delivery Date', 'Date', 'signature_date', 'delivery_date'],
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
            "type": "gsheet",
            "gsheet_url": "https://docs.google.com/spreadsheets/d/1Ivm-ORxdAyG3g-z0ZJuea6eXu0Smvzcf/edit?gid=1939981789#gid=1939981789",
            "sheet_name": None
        },
        "database": {"table_name": "dispatcher_raw_data"},
        "weight_tiers": [
            {"min": 0, "max": 5, "rate": 1.50},
            {"min": 5, "max": 10, "rate": 1.60},
            {"min": 10, "max": 30, "rate": 2.70},
            {"min": 30, "max": float('inf'), "rate": 4.00}
        ],
        "currency_symbol": "RM"
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
                    cls._cache = json.load(f)
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
    """Handle data loading from Google Sheets."""

    @staticmethod
    def _extract_gsheet_id_and_gid(url_or_id: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract spreadsheet ID and GID from Google Sheets URL."""
        if not url_or_id:
            return None, None

        # Direct ID format
        if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url_or_id):
            return url_or_id, None

        try:
            parsed = urlparse(url_or_id)
            path_parts = [p for p in parsed.path.split('/') if p]

            # Extract spreadsheet ID
            spreadsheet_id = None
            if 'spreadsheets' in path_parts and 'd' in path_parts:
                idx = path_parts.index('d')
                if idx + 1 < len(path_parts):
                    spreadsheet_id = path_parts[idx + 1]

            # Extract GID
            query_gid = parse_qs(parsed.query).get('gid', [None])[0]
            frag_gid_match = re.search(r"gid=(\d+)", parsed.fragment or "")
            gid = query_gid or (frag_gid_match.group(1) if frag_gid_match else None)

            return spreadsheet_id, gid
        except Exception:
            return None, None

    @staticmethod
    def _build_csv_url(spreadsheet_id: str, sheet_name: Optional[str], gid: Optional[str]) -> str:
        """Construct CSV export URL."""
        base = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
        if sheet_name:
            return f"{base}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        if gid:
            return f"{base}/export?format=csv&gid={gid}"
        return f"{base}/export?format=csv"

    @staticmethod
    @st.cache_data(ttl=300)
    def read_google_sheet(url_or_id: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Fetch Google Sheet as DataFrame."""
        spreadsheet_id, gid = DataSource._extract_gsheet_id_and_gid(url_or_id)
        if not spreadsheet_id:
            raise ValueError("Invalid Google Sheet URL or ID.")

        csv_url = DataSource._build_csv_url(spreadsheet_id, sheet_name, gid)
        try:
            resp = requests.get(csv_url, timeout=30)
            resp.raise_for_status()
            return pd.read_csv(io.BytesIO(resp.content))
        except Exception as exc:
            st.error(f"Failed to fetch Google Sheet: {exc}")
            raise

    @staticmethod
    def load_data(config: dict) -> Optional[pd.DataFrame]:
        """Load data based on configuration."""
        data_source = config["data_source"]
        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                return DataSource.read_google_sheet(
                    data_source["gsheet_url"],
                    data_source["sheet_name"]
                )
            except Exception as exc:
                st.error(f"Error reading Google Sheet: {exc}")
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

        if waybill_col:
            initial_count = len(df)
            df_dedup = df.drop_duplicates(subset=[waybill_col])
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

        for tier in sorted(tiers, key=lambda t: t['min']):
            tier_max = tier.get('max', float('inf'))
            if tier['min'] <= w < tier_max:
                return tier['rate']

        return tiers[-1]['rate']

    @staticmethod
    def calculate_payout(df: pd.DataFrame, currency_symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
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
        df_unique = df_clean.drop_duplicates(subset=['dispatcher_id', waybill_col], keep='first')

        # Diagnostics
        raw_weight = df_clean['weight'].sum()
        dedup_weight = df_unique['weight'].sum()
        duplicates = len(df_clean) - len(df_unique)
        st.info(f"Weight totals — Raw: {raw_weight:,.2f} kg | Deduplicated: {dedup_weight:,.2f} kg | Duplicate rows: {duplicates}")

        # Group by dispatcher
        grouped = df_unique.groupby('dispatcher_id').agg(
            dispatcher_name=('dispatcher_name', 'first'),
            parcel_count=(waybill_col, 'nunique'),
            total_weight=('weight', 'sum'),
            avg_weight=('weight', 'mean'),
            total_payout=('payout', 'sum')
        ).reset_index()

        grouped['avg_rate'] = grouped['total_payout'] / grouped['parcel_count']

        # Create display and numeric dataframes
        numeric_df = grouped.rename(columns={
            "dispatcher_id": "Dispatcher ID",
            "dispatcher_name": "Dispatcher Name",
            "parcel_count": "Parcels Delivered",
            "total_weight": "Total Weight (kg)",
            "avg_weight": "Avg Weight (kg)",
            "avg_rate": "Avg Rate per Parcel",
            "total_payout": "Total Payout"
        }).sort_values(by="Total Payout", ascending=False)

        display_df = numeric_df.copy()
        display_df["Total Weight (kg)"] = display_df["Total Weight (kg)"].apply(lambda x: f"{x:.2f}")
        display_df["Avg Weight (kg)"] = display_df["Avg Weight (kg)"].apply(lambda x: f"{x:.2f}")
        display_df["Avg Rate per Parcel"] = display_df["Avg Rate per Parcel"].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_df["Total Payout"] = display_df["Total Payout"].apply(lambda x: f"{currency_symbol}{x:,.2f}")

        total_payout = numeric_df["Total Payout"].sum()

        st.success(f"✅ Processed {len(df_unique)} unique parcels from {len(grouped)} dispatchers")
        st.info(f"💰 Total Payout: {currency_symbol} {total_payout:,.2f}")

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
                          total_payout: float, currency_symbol: str) -> str:
        """Build management summary invoice HTML with original layout."""
        try:
            total_parcels = int(numeric_df["Parcels Delivered"].sum())
            total_dispatchers = len(numeric_df)
            total_weight = numeric_df["Total Weight (kg)"].sum()
            top_3 = display_df.head(3)

            table_columns = ["Dispatcher ID", "Dispatcher Name", "Parcels Delivered",
                           "Total Payout", "Total Weight (kg)"]

            # Build HTML with original styling
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
                .tier-info {{
                  margin-top: 24px;
                  background: var(--surface);
                  border: 1px solid var(--border);
                  border-radius: 12px;
                  padding: 20px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                .tier-info h3 {{ margin: 0 0 12px 0; color: var(--primary); }}
                .tier-grid {{
                  display: grid;
                  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                  gap: 12px;
                  margin-top: 12px;
                }}
                .tier-item {{
                  background: var(--background);
                  padding: 12px;
                  border-radius: 8px;
                  border: 1px solid var(--border);
                }}
                .tier-item .range {{ font-weight: 600; color: var(--text-primary); }}
                .tier-item .rate {{ color: var(--primary); font-weight: 700; margin-top: 4px; }}
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
                .top-performers {{
                  margin-top: 32px;
                }}
                .top-performers h3 {{
                  color: var(--text-primary);
                  border-bottom: 2px solid var(--primary);
                  padding-bottom: 8px;
                  margin-bottom: 16px;
                }}
                .performers-grid {{
                  display: grid;
                  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                  gap: 16px;
                }}
                .performer-card {{
                  background: var(--surface);
                  padding: 16px;
                  border-radius: 8px;
                  border: 1px solid var(--border);
                }}
                .performer-card .name {{
                  font-weight: 700;
                  color: var(--primary);
                  margin-bottom: 8px;
                }}
                .performer-card .stats {{
                  color: var(--text-secondary);
                  font-size: 14px;
                }}
                .performer-card .stats div {{ margin: 4px 0; }}
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
                            <div class="brand">📊 Invoice</div>
                            <div class="subbrand">From: Jemari Ventures</div>
                            <div class="subbrand">To: Niagamatic Sdn Bhd</div>
                            <div class="idline">Invoice No: JMR{datetime.now().strftime('%Y%m')}</div>
                            <div class="idline">Period: {(datetime.now().replace(day=1) - pd.Timedelta(days=1)).strftime('%B %Y')}</div>
                        </div>
                        <div class="total-badge">
                            <div class="label">Total Payout</div>
                            <div class="value">{currency_symbol} {total_payout:,.2f}</div>
                        </div>
                    </div>

                    <div class="tier-info">
                        <h3>⚖️ Weight-Based Payout Tiers</h3>
                        <div class="tier-grid">
                            <div class="tier-item">
                                <div class="range">0 - 5 kg</div>
                                <div class="rate">{currency_symbol}1.50 per parcel</div>
                            </div>
                            <div class="tier-item">
                                <div class="range">5 - 10 kg</div>
                                <div class="rate">{currency_symbol}1.60 per parcel</div>
                            </div>
                            <div class="tier-item">
                                <div class="range">10 - 30 kg</div>
                                <div class="rate">{currency_symbol}2.70 per parcel</div>
                            </div>
                            <div class="tier-item">
                                <div class="range">30+ kg</div>
                                <div class="rate">{currency_symbol}4.00 per parcel</div>
                            </div>
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
                            <div class="label">Total Weight</div>
                            <div class="value">{total_weight:,.2f} kg</div>
                        </div>
                    </div>

                    <table>
                        <thead><tr>"""

            # Add table headers
            for col in table_columns:
                html += f"<th>{col}</th>"
            html += "</tr></thead><tbody>"

            # Add table rows
            for _, row in display_df.iterrows():
                html += "<tr>"
                for col in table_columns:
                    html += f"<td>{row.get(col, '')}</td>"
                html += "</tr>"

            html += "</tbody></table>"

            # Add top performers section
            if len(top_3) > 0:
                html += """
                    <div class="top-performers">
                        <h3>🏆 Top Performers</h3>
                        <div class="performers-grid">
                """

                for idx, row in top_3.iterrows():
                    html += f"""
                        <div class="performer-card">
                            <div class="name">{row['Dispatcher Name']}</div>
                            <div class="stats">
                                <div>📦 {row['Parcels Delivered']} parcels</div>
                                <div>⚖️ {row['Total Weight (kg)']} kg total</div>
                                <div>💰 {row['Total Payout']}</div>
                            </div>
                        </div>
                    """

                html += "</div></div>"

            html += f"""
                    <div class="note">
                        Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} • JMR Management Dashboard<br>
                        <em>Payout calculated using tier-based weight system</em>
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
        © 2025 Jemari Ventures. All rights reserved. | JMR Management Dashboard v2.0
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
            Overview of dispatcher performance and payouts
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load config and data
    config = Config.load()
    with st.spinner("📄 Loading data from Google Sheets..."):
        df = DataSource.load_data(config)

    if df is None or df.empty:
        st.error("❌ No data loaded. Check your Google Sheet configuration.")
        add_footer()
        return

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    st.sidebar.info("**💰 Weight-Based Payout:**\n- 0-5kg: RM1.50\n- 5-10kg: RM1.60\n- 10-30kg: RM2.70\n- 30kg+: RM4.00")

    # Date filter
    date_col = find_column(df, 'date')
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        min_date, max_date = df[date_col].min().date(), df[date_col].max().date()
        date_range = st.sidebar.date_input("Select Date Range", value=(min_date, max_date),
                                          min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = date_range
            df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]

    # Calculate payouts
    currency = config.get("currency_symbol", "RM")
    display_df, numeric_df, total_payout = PayoutCalculator.calculate_payout(df, currency)

    if numeric_df.empty:
        st.warning("No data after filtering.")
        add_footer()
        return

    # Metrics
    st.subheader("📈 Performance Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Dispatchers", f"{len(display_df):,}")
    col2.metric("Parcels", f"{int(numeric_df['Parcels Delivered'].sum()):,}")
    col3.metric("Total Weight", f"{numeric_df['Total Weight (kg)'].sum():,.2f} kg")
    col4.metric("Total Payout", f"{currency} {total_payout:,.2f}")
    col5.metric("Avg Weight", f"{numeric_df['Avg Weight (kg)'].mean():.2f} kg")
    col6.metric("Avg Payout", f"{currency} {total_payout/len(numeric_df):.2f}")

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
            st.markdown(f"""
            <div style="background: white; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; margin: 8px 0;">
                <div style="font-weight: 600; color: {ColorScheme.PRIMARY};">{row['Dispatcher Name']}</div>
                <div style="color: #64748b; font-size: 0.9rem;">
                    {row['Parcels Delivered']} parcels • {row['Total Weight (kg)']:.2f} kg • {currency}{row['Total Payout']:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Data table
    st.markdown("---")
    st.subheader("👥 Dispatcher Performance Details")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Invoice generation
    st.markdown("---")
    st.subheader("📄 Invoice Generation")
    invoice_html = InvoiceGenerator.build_invoice_html(display_df, numeric_df, total_payout, currency)
    st.download_button(
        label="📥 Download Invoice (HTML)",
        data=invoice_html.encode("utf-8"),
        file_name=f"invoice_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html"
    )

    # Export options
    st.markdown("---")
    st.subheader("📥 Export Data")
    col1, col2 = st.columns(2)
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

    add_footer()


if __name__ == "__main__":
    main()
