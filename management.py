import warnings
import urllib3
# Suppress the NotOpenSSLWarning
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)

import io
from typing import List, Optional, Tuple
import re
from urllib.parse import urlparse, parse_qs
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import streamlit as st
import altair as alt
import requests

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
            "gsheet_url": "https://docs.google.com/spreadsheets/d/1_PFvSf1v8g9p_8BdouTFk4lPFI2xOQC_iModlziUmMU/edit?usp=sharing",
            "sheet_name": None
        },
        "database": {
            "table_name": "dispatcher_raw_data"
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
        """Load configuration from file or create default."""
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
        """Save configuration to file."""
        try:
            with open(cls.CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error saving config: {e}")
            return False


# =============================================================================
# DATA SOURCE MANAGEMENT
# =============================================================================

# =============================================================================
# DATABASE CONNECTION (from Streamlit secrets)
# =============================================================================

def get_database_url():
    """Build database URL from Streamlit secrets."""
    try:
        db_host = st.secrets["DB_HOST"]
        db_port = st.secrets["DB_PORT"]
        db_name = st.secrets["DB_NAME"]
        db_user = st.secrets["DB_USER"]
        db_password = st.secrets["DB_PASSWORD"]

        db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        return db_url
    except Exception as e:
        st.error("❌ Database URL not found. Please set it in .streamlit/secrets.toml or environment variables.")
        st.stop()


@st.cache_resource
def get_database_engine(db_url: str):
    """Cached database engine - reused across reruns."""
    return create_engine(db_url, pool_pre_ping=True)


class DataSource:
    def __init__(self, db_url: str):
        """
        Initialize DataSource with a PostgreSQL connection string.
        Example: postgresql://user:password@host:port/dbname
        """
        self.db_url = db_url
        self.engine = get_database_engine(db_url)

    @st.cache_data(ttl=300)
    def fetch_data(_self, query: str = "SELECT * FROM dispatcher_raw_data;") -> pd.DataFrame:
        try:
            with _self.engine.connect() as conn:
                df = pd.read_sql(text(query), conn)
            return df
        except SQLAlchemyError as e:
            st.error(f"Database error: {str(e)}")
            return pd.DataFrame()

# =============================================================================
# NAME CLEANING
# =============================================================================

def clean_dispatcher_name(name: str) -> str:
    """Remove prefixes like JMR, ECP, AF from dispatcher names."""
    prefixes = ['JMR', 'ECP', 'AF', 'PEN', 'KUL', 'JHR']

    cleaned_name = str(name).strip()

    # Remove common prefixes
    for prefix in prefixes:
        if cleaned_name.startswith(prefix):
            cleaned_name = cleaned_name[len(prefix):].strip()
            # Remove any remaining hyphens or spaces at start
            cleaned_name = cleaned_name.lstrip(' -')
            break

    return cleaned_name


# =============================================================================
# PAYOUT CALCULATIONS - MANAGEMENT VIEW
# =============================================================================

class PayoutCalculator:
    """Handle payout calculations for management view."""
    @staticmethod
    def validate_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data before processing."""
        if df.empty:
            return False, "DataFrame is empty"

        if 'waybill' not in df.columns:
            return False, "Missing 'waybill' column"

        # Check for duplicate waybills
        duplicates = df['waybill'].duplicated().sum()
        if duplicates > 0:
            return False, f"Found {duplicates} duplicate waybills"

        return True, "Data validation passed"

    @staticmethod
    def calculate_payout_rate(df: pd.DataFrame, payout_rate: float, currency_symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
        """Calculate payout using flat rate for management view."""
        # Standardize column names
        df_clean = df.copy()

        # Only rename columns that exist
        existing_columns = {}
        for orig, new in {
            "Waybill Number": "waybill",
            "Delivery Signature": "signature_date",
            "Dispatcher ID": "dispatcher_id",
            "Dispatcher Name": "dispatcher_name",
        }.items():
            if orig in df_clean.columns:
                existing_columns[orig] = new

        df_clean = df_clean.rename(columns=existing_columns)

        # Clean dispatcher names if the column exists
        if 'dispatcher_name' in df_clean.columns:
            df_clean['dispatcher_name'] = df_clean['dispatcher_name'].apply(clean_dispatcher_name)

        # Ensure we have the required columns
        if 'dispatcher_id' not in df_clean.columns:
            df_clean['dispatcher_id'] = 'Unknown'
        if 'dispatcher_name' not in df_clean.columns:
            df_clean['dispatcher_name'] = 'Unknown'

        # Count unique waybills per dispatcher
        grouped = (
            df_clean.groupby(["dispatcher_id", "dispatcher_name"])
            .agg(parcel_count=("waybill", "nunique"))
            .reset_index()
        )

        if grouped.empty:
            return pd.DataFrame(), pd.DataFrame(), 0.0
        grouped["payout_rate"] = payout_rate
        grouped["total_payout"] = grouped["parcel_count"] * payout_rate

        # Format for display
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

        # Sort by Parcels Delivered in descending order (highest first)
        numeric_df = numeric_df.sort_values(by="Parcels Delivered", ascending=False)
        display_df = display_df.sort_values(by="Parcels Delivered", ascending=False)

        return display_df, numeric_df, total_payout

    @staticmethod
    def get_daily_trend_data(df: pd.DataFrame) -> pd.DataFrame:
        """Get daily parcel delivery trend data."""
        df_clean = df.copy()

        # Try to find date column
        date_column = None
        for col in ["Delivery Signature", "Delivery Date", "Date", "signature_date"]:
            if col in df_clean.columns:
                date_column = col
                break

        if date_column:
            df_clean[date_column] = pd.to_datetime(df_clean[date_column], errors="coerce")
            daily_df = (
                df_clean.groupby(df_clean[date_column].dt.date)
                .size()
                .reset_index(name='total_parcels')
            )
            daily_df = daily_df.rename(columns={date_column: 'signature_date'})
            return daily_df.sort_values('signature_date')

        return pd.DataFrame()


# =============================================================================
# DATA VISUALIZATION - MANAGEMENT VIEW
# =============================================================================

class DataVisualizer:
    """Create performance charts and graphs for management view."""

    @staticmethod
    def create_management_charts(daily_df: pd.DataFrame, numeric_df: pd.DataFrame, currency_symbol: str):
        """Create charts for management dashboard using numeric data."""
        charts = {}

        if not daily_df.empty:
            # Daily Trend Chart (unchanged)
            daily_trend = alt.Chart(daily_df).mark_area(
                line={'color': ColorScheme.PRIMARY, 'width': 2},
                color=ColorScheme.PRIMARY_LIGHT,
                opacity=0.6
            ).encode(
                x=alt.X('signature_date:T', title='Date', axis=alt.Axis(format='%b %d')),
                y=alt.Y('total_parcels:Q', title='Parcels Delivered'),
                tooltip=['signature_date:T', 'total_parcels:Q']
            ).properties(
                title='Daily Parcel Delivery Trend',
                height=300
            )
            charts['daily_trend'] = daily_trend

        if not numeric_df.empty:
            # Top Performers Bar Chart - Use numeric_df directly
            top_10 = numeric_df.head(10)  # Already sorted by Parcels Delivered

            performers_chart = alt.Chart(top_10).mark_bar(color=ColorScheme.PRIMARY).encode(
                y=alt.Y('Dispatcher Name:N', title='Dispatcher', sort='-x'),
                x=alt.X('Parcels Delivered:Q', title='Parcels Delivered'),
                color=alt.Color('Parcels Delivered:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=[
                    'Dispatcher Name:N',
                    'Parcels Delivered:Q',
                    alt.Tooltip('Total Payout:Q', format=',.2f', title='Payout (RM)')  # Add formatting
                ]
            )
            charts['performers'] = performers_chart

            # Payout Distribution - Use numeric_df directly
            payout_chart = alt.Chart(numeric_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Total Payout", type="quantitative", title="Payout Amount"),
                color=alt.Color(field="Dispatcher Name", type="nominal",
                              scale=alt.Scale(range=ColorScheme.CHART_COLORS),
                              legend=alt.Legend(title="Dispatchers", orient="right")),
                order=alt.Order(field="Total Payout", sort="descending"),
                tooltip=['Dispatcher Name:N', 'Total Payout:Q']
            ).properties(
                title='Payout Distribution',
                height=300,
                width=400
            )
            charts['payout_dist'] = payout_chart

        return charts

# =============================================================================
# INVOICE GENERATION
# =============================================================================

class InvoiceGenerator:
    """Generate professional invoices."""

    @staticmethod
    def build_invoice_html(display_df: pd.DataFrame, total_payout: float, payout_rate: float, currency_symbol: str) -> str:
        """Build a management summary invoice HTML for the entire company."""

        try:
            # Calculate metrics directly from the display_df (Parcels Delivered is still numeric)
            total_parcels = display_df["Parcels Delivered"].sum()
            total_dispatchers = len(display_df)
            avg_parcels = display_df["Parcels Delivered"].mean()
            avg_payout = total_payout / total_dispatchers if total_dispatchers > 0 else 0

            # For top performers, we need to extract numeric values from formatted Total Payout
            # But we can use Parcels Delivered directly since it's not formatted
            top_3 = display_df.head(3).copy()

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
                .container {{ max-width: 1200px; margin: 24px auto; padding: 0 16px; }}
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
                  margin-top: 24px; display: flex; gap: 12px; flex-wrap: wrap;
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
                .chip .label {{ color: var(--text-secondary); font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 600; }}
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
                .footer {{
                  margin-top: 24px;
                  text-align: right;
                  font-weight: 700;
                  font-size: 20px;
                  color: var(--primary);
                  padding: 16px;
                  border-top: 2px solid var(--border);
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
                @media (max-width: 768px) {{
                  .header {{ grid-template-columns: 1fr; text-align: center; }}
                  .summary {{ flex-direction: column; }}
                  .chip {{ min-width: auto; }}
                }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <div>
                            <div class="brand">📊 Management Summary Invoice</div>
                            <div class="idline">From: Jemari Ventures &nbsp;&nbsp;|&nbsp;&nbsp; Period: {datetime.now().strftime('%B %Y')}</div>
                        </div>
                        <div class="total-badge">
                            <div class="label">Total Company Payout</div>
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
                            <div class="label">Payout Rate</div>
                            <div class="value">{currency_symbol}{payout_rate:.2f}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Avg Parcels/Dispatcher</div>
                            <div class="value">{avg_parcels:.1f}</div>
                        </div>
                        <div class="chip">
                            <div class="label">Avg Payout/Dispatcher</div>
                            <div class="value">{currency_symbol} {avg_payout:.2f}</div>
                        </div>
                    </div>
            """

            # Add the table
            if len(display_df) > 0:
                html_content += "<table>"
                html_content += "<thead><tr>"
                for col in display_df.columns:
                    html_content += f"<th>{col}</th>"
                html_content += "</tr></thead>"

                html_content += "<tbody>"
                for _, row in display_df.iterrows():
                    html_content += "<tr>"
                    for col in display_df.columns:
                        html_content += f"<td>{row[col]}</td>"
                    html_content += "</tr>"
                html_content += "</tbody></table>"

            # Add top performers section - use formatted values directly
            if len(top_3) > 0:
                html_content += """
                    <div style="margin-top: 32px;">
                        <h3 style="color: var(--text-primary); border-bottom: 2px solid var(--primary); padding-bottom: 8px;">🏆 Top Performers</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 16px; margin-top: 16px;">
                """

                for idx, row in top_3.iterrows():
                    html_content += f"""
                        <div style="background: var(--surface); padding: 16px; border-radius: 8px; border: 1px solid var(--border);">
                            <div style="font-weight: 700; color: var(--primary); margin-bottom: 8px;">{row['Dispatcher Name']}</div>
                            <div style="color: var(--text-secondary); font-size: 14px;">
                                <div>📦 {row['Parcels Delivered']:,} parcels</div>
                                <div>💰 {row['Total Payout']}</div>
                            </div>
                        </div>
                    """

                html_content += "</div></div>"

            # Footer
            html_content += f"""
                    <div class="note">
                        Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')} • JMR Management Dashboard
                    </div>
                </div>
            </body>
            </html>
            """

            return html_content

        except Exception as e:
            st.error(f"Error generating invoice: {e}")
            # Return a simple error HTML as fallback
            return f"""
            <html>
            <body>
                <h1>Error Generating Invoice</h1>
                <p>There was an error generating the invoice: {str(e)}</p>
                <p>Please try again or contact support.</p>
            </body>
            </html>
            """

# =============================================================================
# STREAMLIT UI ENHANCEMENTS
# =============================================================================

def apply_custom_styles():
    """Apply consistent color scheme to Streamlit components."""
    st.markdown(f"""
    <style>
        /* Main background */
        .stApp {{
            background-color: {ColorScheme.BACKGROUND};
        }}

        /* Override Streamlit default text colors */
        .stMarkdown, .stText, .stWrite {{
            color: {ColorScheme.TEXT_PRIMARY} !important;
        }}

        /* Override Streamlit header colors */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
            color: {ColorScheme.TEXT_PRIMARY} !important;
        }}

        /* Specific override for our custom header */
        .stMarkdown div h1 {{
            color: white !important;
        }}

        /* Sidebar */
        .css-1d391kg, .css-1lcbmhc {{
            background-color: {ColorScheme.SURFACE};
        }}

        /* Buttons */
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

        /* Expanders */
        .streamlit-expanderHeader {{
            background-color: {ColorScheme.SURFACE};
            border: 1px solid {ColorScheme.BORDER};
            border-radius: 8px;
            color: {ColorScheme.TEXT_PRIMARY} !important;
        }}

        /* Expander content */
        .streamlit-expanderContent {{
            color: {ColorScheme.TEXT_PRIMARY} !important;
        }}

        /* Dataframes */
        .dataframe {{
            border: 1px solid {ColorScheme.BORDER};
            border-radius: 8px;
        }}

        /* Metrics */
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

        /* Select boxes */
        .stSelectbox>div>div {{
            border: 1px solid {ColorScheme.BORDER};
            border-radius: 8px;
        }}

        /* Success, warning, error messages */
        .stAlert {{
            border-radius: 8px;
        }}

        /* Caption text */
        .stCaption {{
            color: {ColorScheme.TEXT_SECONDARY} !important;
        }}

        /* Footer styling - FIXED TEXT COLOR */
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

        /* Ensure all text in footer is white */
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
    """Add a professional footer to the main page."""
    st.markdown(f"""
    <div class="footer">
        <div class="footer-content">
            <div class="footer-copyright" style="color: white !important;">
                © 2025 Jemari Ventures. All rights reserved. | JMR Management Dashboard v1.0
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# MAIN MANAGEMENT APPLICATION
# =============================================================================

def main():
    """Main management dashboard application."""

    # Page configuration
    st.set_page_config(
        page_title="JMR Management Dashboard",
        page_icon="📊",
        layout="wide"
    )

    # Apply custom styles
    apply_custom_styles()

    # Custom header with consistent branding
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {ColorScheme.PRIMARY} 0%, {ColorScheme.PRIMARY_LIGHT} 100%);
                padding: 2rem;
                border-radius: 12px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
        <h1 style="color: white !important; margin: 0; padding: 0;">📊 JMR Management Dashboard</h1>
        <p style="color: rgba(255,255,255,0.9) !important; margin: 0.5rem 0 0 0; padding: 0;">
            Overview of dispatcher performance and payouts
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Load configuration
    config = Config.load()
    if not config:
        st.error("❌ Failed to load configuration. Please check config.json file.")
        add_footer()
        return

    # # Load data from Google Sheets
    # with st.spinner("🔄 Loading data from Google Sheets..."):
    #     df = DataSource.load_data(config)
    # Load data from PostgreSQL
    with st.spinner("🔄 Loading data from PostgreSQL database..."):
        db_url = st.secrets.get("db_url") or os.getenv("DATABASE_URL")

        if not db_url:
            st.error("❌ Database URL not found. Please set it in .streamlit/secrets.toml or environment variables.")
            add_footer()
            return

        db_url = get_database_url()
        data_source = DataSource(db_url)

        df = data_source.fetch_data("SELECT * FROM dispatcher_raw_data;")

    if df is None or df.empty:
        st.error("❌ No data loaded. Please check your Google Sheets configuration.")
        st.info("💡 Tip: Check if data exists in the 'dispatcher_raw_data' table")
        add_footer()
        return

    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")

    payout_rate = st.sidebar.number_input(
        "💰 Payout Rate (RM/parcel)",
        min_value=0.0,
        value=1.5,
        step=0.1,
        help="Flat rate per parcel delivered"
    )

    # Date range filter
    st.sidebar.header("📅 Date Range Filter")

    # Try to find date column for filtering
    date_column = None

    for col in ["Delivery Signature", "Delivery Date", "signature_date"]:
        if col in df.columns:
            date_column = col
            break

    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        min_date = df[date_column].min().date()
        max_date = df[date_column].max().date()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df[date_column].dt.date >= start_date) &
                            (df[date_column].dt.date <= end_date)]
            if len(df_filtered) == 0:
                st.warning(f"⚠️ No data found between {start_date} and {end_date}")
            df = df_filtered
        elif len(date_range) == 1:
            st.warning("Please select both start and end dates")
    # ==============================================================
    # 🧮 SUMMARY METRICS
    # ==============================================================

    st.subheader("📈 Performance Overview")

    currency_symbol = config.get("currency_symbol", "RM")
    display_df, numeric_df, total_payout = PayoutCalculator.calculate_payout_rate(df, payout_rate, currency_symbol)

    if numeric_df.empty:
        st.warning("No data available after filtering.")
        add_footer()
        return


    total_dispatchers = len(display_df)
    total_parcels = numeric_df["Parcels Delivered"].sum()
    avg_parcels = numeric_df["Parcels Delivered"].mean()
    avg_payout = total_payout / total_dispatchers if total_dispatchers > 0 else 0

    # Top Metrics Layout
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Dispatchers", f"{total_dispatchers:,}")
    with col2:
        st.metric("Total Parcels", f"{total_parcels:,}")
    with col3:
        st.metric("Total Payout", f"{currency_symbol} {total_payout:,.2f}")
    with col4:
        st.metric("Avg Parcels/Dispatcher", f"{avg_parcels:.1f}")
    with col5:
        st.metric("Avg Payout/Dispatcher", f"{currency_symbol} {avg_payout:.2f}")

    # ==============================================================
    # 📊 PERFORMANCE CHARTS
    # ==============================================================

    st.markdown("---")
    st.subheader("📊 Performance Analytics")

    # Prepare daily data for charts
    daily_df = PayoutCalculator.get_daily_trend_data(df)

    # Create and display charts
    charts = DataVisualizer.create_management_charts(daily_df, numeric_df, currency_symbol)

    if charts:
        # First row: Charts side by side
        col1, col2 = st.columns([1, 1])

        with col1:
            if 'daily_trend' in charts:
                st.altair_chart(charts['daily_trend'], use_container_width=True)

            # Metrics below daily trend chart
            if not daily_df.empty:
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    highest_day = daily_df.loc[daily_df['total_parcels'].idxmax()]
                    st.metric("Peak Delivery Day",
                            f"{highest_day['total_parcels']} parcels",
                            f"on {highest_day['signature_date'].strftime('%b %d')}")
                with metric_col2:
                    avg_daily = daily_df['total_parcels'].mean()
                    st.metric("Average Daily Volume", f"{avg_daily:.1f} parcels")

            # Bar chart below metrics
            if 'performers' in charts:
                st.altair_chart(charts['performers'], use_container_width=True)

        with col2:
            if 'payout_dist' in charts:
                st.altair_chart(charts['payout_dist'], use_container_width=True)

            # Top 5 performers below donut chart
            st.markdown("##### 🏆 Top Performers")
            top_5 = numeric_df.head(5)
            for idx, row in top_5.iterrows():
                st.markdown(f"""
                <div style="background: {ColorScheme.SURFACE}; padding: 0.8rem 1rem; border-radius: 8px; border: 1px solid {ColorScheme.BORDER}; margin: 0.4rem 0;">
                    <div style="font-weight: 600; color: {ColorScheme.PRIMARY}; margin-bottom: 0.2rem;">{row['Dispatcher Name']}</div>
                    <div style="color: {ColorScheme.TEXT_SECONDARY}; font-size: 0.9rem;">
                        {row['Parcels Delivered']} parcels • {row['Total Payout']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ==============================================================
        # 🔮 FORECAST & INSIGHTS (Including Total Payout)
        # ==============================================================

        st.markdown("---")
        st.subheader("🔮 Forecast & Insights")

        try:
            from prophet import Prophet
            import altair as alt

            # --- CONFIG ---
            # Example flat payout rate per parcel (you can replace this with your actual config)
            #payout_rate = payout_rate  # RM5 per parcel, for example
            #currency_symbol = "RM"


            # --- Data Preparation ---
            forecast_df = daily_df.rename(columns={"signature_date": "ds", "total_parcels": "y"})[["ds", "y"]]

            # Ensure valid datetime
            forecast_df["ds"] = pd.to_datetime(forecast_df["ds"], errors="coerce")
            forecast_df = forecast_df.dropna(subset=["ds", "y"])

            if len(forecast_df) < 5:
                st.warning("📉 Not enough historical data for reliable forecasting. Please keep at least 14 days of data.")
            else:
                # --- Prophet Model ---
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    interval_width=0.8
                )
                model.fit(forecast_df)

                # Forecast next 14 days
                future = model.make_future_dataframe(periods=14)
                forecast = model.predict(future)

                # Ensure datetime alignment
                forecast["ds"] = pd.to_datetime(forecast["ds"])
                forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

                # Merge actual + forecast data
                forecast_chart_data = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].merge(
                    forecast_df, on="ds", how="left"
                )

                # --- Compute Payout ---
                forecast_chart_data["yhat_payout"] = forecast_chart_data["yhat"] * payout_rate
                forecast_chart_data["y_payout"] = forecast_chart_data["y"] * payout_rate

                # Split into historical and future predictions
                historical = forecast_chart_data[forecast_chart_data["y"].notna()]
                future = forecast_chart_data[forecast_chart_data["y"].isna()]

                # --- Visualization: Parcel Forecast ---
                st.markdown("### 📦 Parcel Volume Forecast")

                actual_line = alt.Chart(historical).mark_line(color="#1f77b4").encode(
                    x=alt.X("ds:T", title="Date"),
                    y=alt.Y("y:Q", title="Parcels Delivered"),
                    tooltip=["ds:T", "y"]
                )

                forecast_line = alt.Chart(future).mark_line(
                    strokeDash=[4, 4], color="#ffa500"
                ).encode(
                    x="ds:T",
                    y="yhat:Q",
                    tooltip=["ds:T", "yhat"]
                )

                confidence_band = alt.Chart(forecast_chart_data).mark_area(
                    opacity=0.2, color="#ffa500"
                ).encode(
                    x="ds:T",
                    y="yhat_lower:Q",
                    y2="yhat_upper:Q"
                )

                st.altair_chart(confidence_band + actual_line + forecast_line, use_container_width=True)

                # --- Visualization: Payout Forecast ---
                st.markdown(f"### 💸 Total Payout Forecast ({currency_symbol})")

                payout_actual = alt.Chart(forecast_chart_data).mark_line(color="#2ca02c").encode(
                    x=alt.X("ds:T", title="Date"),
                    y=alt.Y("y_payout:Q", title=f"Total Payout ({currency_symbol})"),
                    tooltip=["ds:T", "y_payout"]
                )

                payout_forecast = alt.Chart(forecast_chart_data).mark_line(
                    strokeDash=[4, 4], color="#ff7f0e"
                ).encode(
                    x="ds:T",
                    y="yhat_payout:Q",
                    tooltip=["ds:T", "yhat_payout"]
                )

                st.altair_chart(payout_actual + payout_forecast, use_container_width=True)

                # --- Forecast Metrics ---
                latest_actual = forecast_df["y"].iloc[-1]
                tomorrow_forecast = forecast.iloc[-1]["yhat"]
                next_week_forecast = forecast.tail(7)["yhat"].sum()

                tomorrow_payout = tomorrow_forecast * payout_rate
                next_week_payout = next_week_forecast * payout_rate

                avg_daily = forecast_df["y"].mean()
                growth_rate = ((tomorrow_forecast - avg_daily) / avg_daily * 100) if avg_daily > 0 else 0

                st.markdown("### 📅 Short-Term Outlook")
                metric1, metric2, metric3 = st.columns(3)
                with metric1:
                    st.metric("Tomorrow’s Volume (Est.)", f"{tomorrow_forecast:.0f} parcels",
                            f"{tomorrow_forecast - latest_actual:+.1f} vs last day")
                with metric2:
                    st.metric(f"Next 7 Days Payout (Est.)", f"{currency_symbol}{next_week_payout:,.0f}")
                with metric3:
                    st.metric("Expected Change", f"{growth_rate:+.1f}% vs Avg Daily")

                # --- Auto Insights ---
                st.markdown("### 🧠 Auto Insights")
                if growth_rate > 5:
                    st.success(f"📈 Volume expected to increase — projected payout for next 7 days: {currency_symbol}{next_week_payout:,.0f}. Consider adding dispatch capacity.")
                elif growth_rate < -5:
                    st.warning(f"📉 Lower forecast — estimated payout next 7 days: {currency_symbol}{next_week_payout:,.0f}. May indicate slow demand period.")
                else:
                    st.info(f"📊 Stable forecast — total payout expected around {currency_symbol}{next_week_payout:,.0f} next week.")

        except ImportError:
            st.warning("⚠️ Prophet is not installed. Run `pip install prophet` to enable forecasting.")
        except Exception as e:
            st.error(f"Forecasting failed: {e}")


        # ==============================================================
        # 📋 DISPATCHER DETAILS TABLE
        # ==============================================================

        st.markdown("---")
        st.subheader("👥 Dispatcher Performance Details")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ==============================================================
        # 📥 INVOICE GENERATION
        # ==============================================================
        st.markdown("---")
        st.subheader("📄 Invoice Generation")

        # Generate and download invoice
        invoice_html = InvoiceGenerator.build_invoice_html(
            display_df, total_payout, payout_rate, currency_symbol
        )

        st.download_button(
            label="📥 Download Invoice (HTML)",
            data=invoice_html.encode("utf-8"),
            file_name=f"invoice_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
        )

        st.caption("Invoice generated based on current configuration and data.")

        # ==============================================================
        # 📥 EXPORT OPTIONS
        # ==============================================================

        st.markdown("---")
        st.subheader("📥 Export Data")

        col1, col2 = st.columns(2)

        with col1:
            # Export summary data
            csv_data = numeric_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Summary CSV",
                data=csv_data,
                file_name=f"management_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            # Export raw data
            raw_csv = df.to_csv(index=False)
            st.download_button(
                label="📋 Download Raw Data CSV",
                data=raw_csv,
                file_name=f"raw_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        # Add footer
        add_footer()


if __name__ == "__main__":
    main()
