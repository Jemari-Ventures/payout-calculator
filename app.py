import warnings
import urllib3
# Suppress the NotOpenSSLWarning
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)

import io
from typing import List, Optional, Tuple, Dict, Any
import re
from urllib.parse import urlparse, parse_qs
import json
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import altair as alt
import requests
import html

# Add input validation and sanitization
import html

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS attacks."""
    return html.escape(str(text)).strip()

# Use in invoice generation and other user-facing outputs

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
        "payout_mode": "Tiered daily",
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
                    config = json.load(f)
                # Validate the loaded config
                is_valid, errors = cls.validate_tiers(config.get("tiers", []))
                if not is_valid:
                    st.warning(f"Configuration validation issues: {', '.join(errors)}")
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

    @classmethod
    def validate_tiers(cls, tiers_config: List[dict]) -> Tuple[bool, List[str]]:
        """Enhanced tier validation."""
        errors = []

        if not tiers_config:
            errors.append("At least one tier must be configured")
            return False, errors

        tiers = sorted(tiers_config, key=lambda x: x["Min Parcels"] or 0)

        # Validate each tier
        for i, tier in enumerate(tiers):
            min_parcels = tier.get("Min Parcels", 0)
            max_parcels = tier.get("Max Parcels")
            rate = tier.get("Rate (RM)")

            if min_parcels is None or min_parcels < 0:
                errors.append(f"Tier {i+1}: Min Parcels must be non-negative")

            if max_parcels is not None and max_parcels < min_parcels:
                errors.append(f"Tier {i+1}: Max Parcels cannot be less than Min Parcels")

            if rate is None or rate <= 0:
                errors.append(f"Tier {i+1}: Rate must be positive")

        # Rest of existing validation logic...
        return len(errors) == 0, errors


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
        if sheet_name:
            return f"{base}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        if gid:
            return f"{base}/export?format=csv&gid={gid}"
        return f"{base}/export?format=csv"

    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def read_google_sheet(url_or_id: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Fetch a Google Sheet as CSV and return a DataFrame."""
        spreadsheet_id, gid = DataSource._extract_gsheet_id_and_gid(url_or_id)
        if not spreadsheet_id:
            raise ValueError("Invalid Google Sheet URL or ID.")
        csv_url = DataSource._build_gsheet_csv_url(spreadsheet_id, sheet_name, gid)
        try:
            resp = requests.get(csv_url, timeout=30)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content))
            return DataSource.clean_dataframe(df)
        except Exception as exc:
            st.error(f"Failed to fetch Google Sheet: {exc}")
            raise

    @staticmethod
    def load_data(config: dict) -> Optional[pd.DataFrame]:
        """Load data based on configuration."""
        data_source = config["data_source"]

        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                df = DataSource.read_google_sheet(data_source["gsheet_url"], data_source["sheet_name"])
                # Validate the loaded data
                is_valid, errors = DataSource.validate_dataframe(df)
                if not is_valid:
                    for error in errors:
                        st.warning(f"Data validation: {error}")
                return df
            except Exception as exc:
                st.error(f"Error reading Google Sheet: {exc}")
                return None
        return None

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Enhanced data validation."""
        errors = []
        required_columns = ["Dispatcher ID", "Waybill Number", "Delivery Signature"]

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing columns: {', '.join(missing_columns)}")
            return False, errors

        # Enhanced validation
        validation_checks = [
            ("Dispatcher ID", lambda x: x.notna().all(), "missing Dispatcher ID"),
            ("Waybill Number", lambda x: x.duplicated().sum() == 0, "duplicate waybill numbers"),
            ("Delivery Signature", lambda x: pd.to_datetime(x, errors='coerce').notna().all(), "invalid dates")
        ]

        for col, check_func, error_msg in validation_checks:
            if col in df.columns:
                if not check_func(df[col]):
                    count = len(df) if "missing" in error_msg else df[col].duplicated().sum()
                    errors.append(f"{count} {error_msg}")

        return len(errors) == 0, errors

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Data cleaning and normalization."""
        df_clean = df.copy()

        # Normalize column names - but keep original format for compatibility
        df_clean.columns = [str(col).strip() for col in df_clean.columns]

        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')

        # Standardize date formats
        if "Delivery Signature" in df_clean.columns:
            df_clean["Delivery Signature"] = pd.to_datetime(
                df_clean["Delivery Signature"], errors='coerce'
            )

        return DataSource.optimize_dataframe_memory(df_clean)

    @staticmethod
    def optimize_dataframe_memory(df):
        """Reduce memory usage of DataFrame."""
        df_optimized = df.copy()

        for col in df_optimized.columns:
            if df_optimized[col].dtype == 'object':
                # Convert object columns to category if they have low cardinality
                if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                    df_optimized[col] = df_optimized[col].astype('category')

        return df_optimized


# =============================================================================
# PAYOUT CALCULATIONS
# =============================================================================

class PayoutCalculator:
    """Handle payout calculations."""

    @staticmethod
    def calculate_tiered_daily(filtered_df: pd.DataFrame, tiers_config: List, currency_symbol: str) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
        """Calculate payout for tiered daily mode."""
        # Prepare tiers
        tiers = []
        for tier in tiers_config:
            tmin = tier.get("Min Parcels")
            tmax = tier.get("Max Parcels")
            trate = tier.get("Rate (RM)")
            tname = tier.get("Tier")
            if pd.notna(trate):
                tiers.append((tmin, tmax, trate, tname))

        #tiers.sort(key=lambda t: (t[0] or 0), reverse=True)
        tiers.sort(key=lambda t: (t[0] or 0))

        def map_rate(daily_parcels: float) -> Tuple[str, float]:
            for tmin, tmax, trate, tname in tiers:
                lower_ok = True if pd.isna(tmin) else daily_parcels >= tmin
                upper_ok = True if pd.isna(tmax) else daily_parcels <= tmax
                if lower_ok and upper_ok:
                    return str(tname), float(trate)
            return "Unmatched", 0.0

        # Process data - use fixed column names
        work = filtered_df.copy()
        work["__date"] = pd.to_datetime(work["Delivery Signature"], errors="coerce").dt.date
        work["__waybill"] = work["Waybill Number"].astype(str).str.strip()

        per_day = (
            work.groupby(["__date"], dropna=False)["__waybill"]
            .nunique(dropna=True)
            .reset_index()
            .rename(columns={"__waybill": "daily_parcels"})
        )

        per_day[["tier", "rate_per_parcel"]] = per_day["daily_parcels"].apply(
            lambda x: pd.Series(map_rate(float(x)))
        )
        per_day["payout_per_day"] = per_day["daily_parcels"] * per_day["rate_per_parcel"]
        total_payout = float(per_day["payout_per_day"].sum())

        # Create display dataframe
        display_df = per_day.rename(columns={
            "__date": "Date",
            "daily_parcels": "Total Parcel",
            "tier": "Tier",
            "rate_per_parcel": "Payout Rate",
            "payout_per_day": "Payout",
        })

        # Format currency
        display_df["Payout Rate"] = display_df["Payout Rate"].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_df["Payout"] = display_df["Payout"].apply(lambda x: f"{currency_symbol}{x:.2f}")

        return display_df, total_payout, per_day

    @staticmethod
    def calculate_advanced_metrics(per_day_df: pd.DataFrame) -> dict:
        """Calculate comprehensive performance metrics."""
        if per_day_df.empty:
            return {}

        metrics = {
            "total_payout": per_day_df["payout_per_day"].sum(),
            "total_parcels": per_day_df["daily_parcels"].sum(),
            "total_days": len(per_day_df),
            "avg_daily_parcels": per_day_df["daily_parcels"].mean(),
            "avg_daily_payout": per_day_df["payout_per_day"].mean(),
            "peak_parcels": per_day_df["daily_parcels"].max(),
            "peak_payout": per_day_df["payout_per_day"].max(),
            "consistency_score": (per_day_df["daily_parcels"].std() / per_day_df["daily_parcels"].mean())
                               if per_day_df["daily_parcels"].mean() > 0 else 0,
        }

        # Tier performance analysis
        tier_stats = per_day_df.groupby("tier").agg({
            "daily_parcels": ["count", "sum", "mean"],
            "payout_per_day": ["sum", "mean"]
        }).round(2)

        metrics["tier_performance"] = tier_stats.to_dict()
        return metrics

    @staticmethod
    def calculate_forecast(per_day_df: pd.DataFrame, days: int = 30) -> dict:
        """Simple forecasting based on historical performance."""
        if len(per_day_df) < 7:
            return {}

        avg_parcels = per_day_df["daily_parcels"].mean()
        avg_payout = per_day_df["payout_per_day"].mean()

        return {
            "projected_parcels": avg_parcels * days,
            "projected_payout": avg_payout * days,
            "confidence": min(len(per_day_df) / 30, 1.0)  # Simple confidence score
        }


# =============================================================================
# DATA VISUALIZATION
# =============================================================================

class DataVisualizer:
    """Create performance charts and graphs."""

    @staticmethod
    def create_performance_charts(per_day_df: pd.DataFrame, currency_symbol: str):
        """Create performance charts for tiered daily payout mode."""
        charts = {}

        if per_day_df.empty:
            return charts

        # Prepare data for charts
        chart_data = per_day_df.copy()
        chart_data['date'] = pd.to_datetime(chart_data['__date'])

        # 1. Daily Parcels and Payout Bar Chart
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

        # 2. Performance Metrics Over Time
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

        # 3. Daily Payout Trend
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
    """Remove prefixes like JMR, ECP, AF from dispatcher names."""
    prefixes = ['JMR', 'ECP', 'AF', 'PEN', 'KUL', 'JHR']

    cleaned_name = sanitize_input(name)
    # Remove common prefixes
    for prefix in prefixes:
        if cleaned_name.startswith(prefix):
            cleaned_name = cleaned_name[len(prefix):].strip()
            # Remove any remaining hyphens or spaces at start
            cleaned_name = cleaned_name.lstrip(' -')
            break

    return cleaned_name


# =============================================================================
# INVOICE GENERATION
# =============================================================================

class InvoiceGenerator:
    """Generate professional invoices."""

    @staticmethod
    def build_invoice_html(df_disp: pd.DataFrame, total: float, name: str, dpid: str,
                          currency_symbol: str) -> str:
        """Build a modern, professional invoice HTML with consistent color scheme."""

        # Calculate metrics
        total_parcels = df_disp['Total Parcel'].sum() if 'Total Parcel' in df_disp.columns else 0
        total_days = len(df_disp) if 'Date' in df_disp.columns else 0

        # Clean the dispatcher name for invoice
        cleaned_name = clean_dispatcher_name(name)

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
              margin-top: 24px; display: flex; gap: 12px; flex-wrap: wrap;
            }}
            .chip {{
              border: 1px solid var(--border); border-radius: 12px;
              padding: 16px; background: var(--surface); min-width: 160px;
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
                        <div class="brand">🚚 Invoice</div>
                        <div class="idline">From: Jemari Ventures &nbsp;&nbsp;|&nbsp;&nbsp; To: {cleaned_name}</div>
                    </div>
                    <div class="total-badge">
                        <div class="label">Total Payout</div>
                        <div class="value">{currency_symbol} {total:,.2f}</div>
                    </div>
                </div>

                <div class="summary">
                    <div class="chip">
                        <div class="label">Dispatcher</div>
                        <div class="value">{cleaned_name}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Dispatcher ID</div>
                        <div class="value">{dpid}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Total Parcels</div>
                        <div class="value">{total_parcels:,}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Total Days</div>
                        <div class="value">{total_days:,}</div>
                    </div>
                </div>
        """

        # Add the table
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

        # Footer
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
# BUSINESS INTELLIGENCE & REPORTING
# =============================================================================

class BusinessIntelligence:
    """Advanced analytics and business intelligence."""

    @staticmethod
    def generate_performance_report(per_day_df: pd.DataFrame, dispatcher_name: str) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        metrics = PayoutCalculator.calculate_advanced_metrics(per_day_df)
        forecast = PayoutCalculator.calculate_forecast(per_day_df)

        report = {
            "summary": metrics,
            "forecast": forecast,
            "recommendations": []
        }

        # Business intelligence recommendations
        if metrics:
            avg_parcels = metrics.get("avg_daily_parcels", 0)
            consistency = metrics.get("consistency_score", 0)

            if consistency > 0.5:
                report["recommendations"].append(
                    "📊 Consider strategies to improve delivery consistency"
                )

            if avg_parcels < 50:
                report["recommendations"].append(
                    "🎯 Focus on increasing daily parcel volume to reach higher tiers"
                )

            if avg_parcels > 100:
                report["recommendations"].append(
                    "⭐ Excellent performance! Maintain consistency for maximum earnings"
                )

        return report


# =============================================================================
# UI COMPONENTS
# =============================================================================

def create_sidebar_navigation():
    """Create sidebar navigation for better organization."""
    with st.sidebar:
        st.header("🚚 Navigation")

        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### System Info")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

        # Display configuration status
        config = Config.load()
        is_valid, errors = Config.validate_tiers(config.get("tiers", []))
        if is_valid:
            st.success("✅ Config Valid")
        else:
            st.warning("⚠️ Config Issues")

def show_progress_indicator(step: int, total_steps: int, current_section: str):
    """Show progress indicator for multi-step processes."""
    progress = step / total_steps
    st.progress(progress)
    st.caption(f"Step {step} of {total_steps}: {current_section}")

def create_summary_cards(per_day_df: pd.DataFrame, total_payout: float, total_parcels: int, dispatcher_name: str):
    """Create enhanced summary cards at the top."""
    if per_day_df.empty:
        return

    total_days = len(per_day_df)

    # Create three columns for the metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="👷‍♂️ Dispatcher", value=dispatcher_name)

    with col2:
        st.metric(
            "💰 Total Payout",
            f"RM{total_payout:,.2f}",
            delta=f"{total_days} days" if total_days > 0 else None
        )

    with col3:
        st.metric(
            "📦 Total Parcels",
            f"{total_parcels:,}",
            delta=f"{total_days} days" if total_days > 0 else None
        )

def add_date_filter(filtered_df: pd.DataFrame):
    """Add date range filtering for data analysis."""
    if "Delivery Signature" not in filtered_df.columns:
        return None, None

    dates = pd.to_datetime(filtered_df["Delivery Signature"], errors='coerce').dt.date
    valid_dates = dates.dropna()

    if valid_dates.empty:
        return None, None

    min_date, max_date = valid_dates.min(), valid_dates.max()

    st.subheader("📅 Date Range Filter")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    return start_date, end_date

def get_dispatcher_name(filtered_df: pd.DataFrame, selected_dispatcher: str) -> str:
    """Extract dispatcher name from filtered data with proper column name handling."""
    # Try multiple possible column names for dispatcher name
    possible_name_columns = [
        "Dispatcher Name", "DispatcherName", "Name", "Rider Name", "RiderName",
        "Driver Name", "DriverName", "Courier Name", "CourierName"
    ]

    for col in possible_name_columns:
        if col in filtered_df.columns:
            names = filtered_df[col].dropna().astype(str).unique()
            if len(names) > 0:
                # Return the first non-empty name found
                name = names[0].strip()
                if name:
                    return clean_dispatcher_name(name)

    # If no name found, return the dispatcher ID as fallback
    return selected_dispatcher

def enhanced_dispatcher_selection(df: pd.DataFrame):
    """Enhanced dispatcher selection with search."""
    st.subheader("👤 Dispatcher Selection")

    unique_dispatchers = sorted(df["Dispatcher ID"].dropna().astype(str).unique().tolist())

    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.selectbox("Select Dispatcher", unique_dispatchers, key="dispatcher_select")
    with col2:
        search_term = st.text_input("🔍 Search", placeholder="Filter dispatchers...", key="dispatcher_search")

    if search_term:
        filtered_dispatchers = [d for d in unique_dispatchers if search_term.lower() in d.lower()]
        if filtered_dispatchers:
            selected = st.selectbox("Filtered Dispatchers", filtered_dispatchers, key="filtered_dispatcher_select")
        else:
            st.warning("No dispatchers match your search criteria.")

    return selected

def paginate_dataframe(df: pd.DataFrame, page_size: int = 50, page_number: int = 1):
    """Implement pagination for large datasets."""
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    return df.iloc[start_idx:end_idx]

def create_pagination_controls(total_rows: int, page_size: int, current_page: int):
    """Create pagination UI controls."""
    total_pages = max(1, (total_rows + page_size - 1) // page_size)

    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

    with col2:
        if current_page > 1:
            if st.button("◀ Previous", key="prev_page"):
                current_page -= 1
                st.rerun()

    with col3:
        if current_page < total_pages:
            if st.button("Next ▶", key="next_page"):
                current_page += 1
                st.rerun()

    st.caption(f"Page {current_page} of {total_pages} ({total_rows} total records)")
    return current_page


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
        .stMarkdown div p {{
            color: rgba(255,255,255,0.9) !important;
        }}

        /* Regular text elements */
        .stMarkdown p, .stMarkdown div {{
            color: {ColorScheme.TEXT_PRIMARY} !important;
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
        .footer, .footer * {{
            color: white !important;
        }}

        /* Progress bar styling */
        .stProgress > div > div > div > div {{
            background-color: {ColorScheme.PRIMARY};
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
                © 2025 Jemari Ventures. All rights reserved. | JMR Dispatcher Payout System v2.0
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# CACHING OPTIMIZATIONS
# =============================================================================

@st.cache_data(ttl=600, show_spinner="Loading configuration...")
def load_config_cached():
    return Config.load()

@st.cache_data(ttl=300, show_spinner="Processing payout data...")
def calculate_payout_cached(filtered_df, tiers_config, currency_symbol):
    return PayoutCalculator.calculate_tiered_daily(filtered_df, tiers_config, currency_symbol)

@st.cache_data(ttl=1800, show_spinner="Generating performance charts...")
def create_charts_cached(per_day_df, currency_symbol):
    return DataVisualizer.create_performance_charts(per_day_df, currency_symbol)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application with enhanced features."""
    # Page configuration
    st.set_page_config(
        page_title="JMR Dispatcher Payout System",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Apply custom styles
    apply_custom_styles()

    # Add sidebar navigation
    create_sidebar_navigation()

    # Custom header with consistent branding
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {ColorScheme.PRIMARY} 0%, {ColorScheme.PRIMARY_LIGHT} 100%);
                padding: 2rem;
                border-radius: 12px;
                color: white;
                margin-bottom: 2rem;
                text-align: center;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);">
        <h1 style="color: white !important; margin: 0; padding: 0;">🚚 JMR Dispatcher Payout System</h1>
        <p style="color: rgba(255,255,255,0.9) !important; margin: 0.5rem 0 0 0; padding: 0;">Enhanced Tiered Daily Payout Calculator</p>
    </div>
    """, unsafe_allow_html=True)

    # Load configuration with caching
    config = load_config_cached()

    # Load data from Google Sheets
    with st.spinner("📊 Loading data from Google Sheets..."):
        df = DataSource.load_data(config)

    if df is None:
        st.error("❌ Failed to load data from Google Sheets.")
        st.info("Please check the configuration file or ensure the Google Sheet is accessible.")
        st.stop()
        add_footer()
        return

    if df.empty:
        st.warning("No data found in the Google Sheet.")
        add_footer()
        return

    # Normalize column names - use original format for compatibility
    #df = df.rename(columns={c: str(c).strip() for c in df.columns})
    df.columns = [str(col).strip() for col in df.columns]
    # Check required columns
    required_columns = ["Dispatcher ID", "Waybill Number", "Delivery Signature"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Available columns: {', '.join(df.columns.tolist())}")
        add_footer()
        return


    unique_dispatchers = sorted(df["Dispatcher ID"].dropna().astype(str).unique().tolist())
    if not unique_dispatchers:
        st.error("No dispatcher IDs found in the data.")
        add_footer()
        return

    selected_dispatcher = enhanced_dispatcher_selection(df)

    # Apply date filtering if available
    filtered = df[df["Dispatcher ID"].astype(str) == str(selected_dispatcher)].copy()
    start_date, end_date = add_date_filter(filtered)

    if start_date and end_date:
        filtered = filtered[
            (pd.to_datetime(filtered["Delivery Signature"]).dt.date >= start_date) &
            (pd.to_datetime(filtered["Delivery Signature"]).dt.date <= end_date)
        ]

    # Get dispatcher name using the improved function
    dispatcher_name = get_dispatcher_name(filtered, selected_dispatcher)

    # Pagination for filtered data view
    with st.expander("📋 View Filtered Data", expanded=False):
        page_size = 50
        current_page = 1

        if len(filtered) > page_size:
            current_page = create_pagination_controls(len(filtered), page_size, current_page)
            paginated_data = paginate_dataframe(filtered, page_size, current_page)
            st.dataframe(paginated_data, use_container_width=True)
        else:
            st.dataframe(filtered, use_container_width=True)

    # Step 2: Payout Calculation with Caching
    st.subheader("💰 Payout Calculation")

    currency_symbol = config["currency_symbol"]

    # Calculate tiered daily payout with caching
    display_df, total_payout, per_day = calculate_payout_cached(
        filtered, config["tiers"], currency_symbol
    )

    # Enhanced Summary Cards
    total_parcels = per_day["daily_parcels"].sum()
    create_summary_cards(per_day, total_payout, total_parcels, dispatcher_name)

    # Display detailed payout table
    st.dataframe(display_df, use_container_width=True)

    # Step 3: Performance Visualization with Caching
    st.subheader("📊 Performance Visualization & Analytics")

    # Create and display charts with caching
    charts = create_charts_cached(per_day, currency_symbol)

    if charts:
        # Show charts in a responsive layout
        col1, col2, col3 = st.columns(3)

        with col1:
            if 'parcels_payout' in charts:
                st.altair_chart(charts['parcels_payout'], use_container_width=True)
        with col2:
            if 'payout_trend' in charts:
                st.altair_chart(charts['payout_trend'], use_container_width=True)
        with col3:
            if 'performance_scatter' in charts:
                st.altair_chart(charts['performance_scatter'], use_container_width=True)

        # Advanced Performance Metrics
        st.markdown("---")
        st.subheader("📈 Advanced Performance Analytics")

        # Business Intelligence Report
        report = BusinessIntelligence.generate_performance_report(per_day, dispatcher_name)

        if report["summary"]:
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                avg_parcels = report["summary"]["avg_daily_parcels"]
                st.metric("📦 Average Daily Parcels", f"{avg_parcels:.1f}")

            with col2:
                max_parcels = report["summary"]["peak_parcels"]
                st.metric("🗳️ Maximum Daily Parcels", f"{max_parcels:.0f}")

            with col3:
                avg_payout = report["summary"]["avg_daily_payout"]
                st.metric("💵 Average Daily Payout", f"{currency_symbol}{avg_payout:.2f}")

            with col4:
                if not per_day.empty and "tier" in per_day.columns:
                    tier_distribution = per_day["tier"].value_counts()
                    most_used_tier = tier_distribution.index[0] if not tier_distribution.empty else "N/A"
                    st.metric("🏆 Most Used Tier", most_used_tier)

            with col5:
                consistency = report["summary"]["consistency_score"]
                st.metric("📊 Consistency Score", f"{consistency:.2f}")

        # Forecasting Section
        if report["forecast"]:
            st.subheader("🔮 30-Day Forecast")
            forecast = report["forecast"]
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "📦📦 Projected Parcels",
                    f"{forecast['projected_parcels']:,.0f}",
                    delta=f"{forecast['confidence']:.0%} confidence"
                )

            with col2:
                st.metric(
                    "🏦 Projected Payout",
                    f"{currency_symbol}{forecast['projected_payout']:,.2f}",
                    delta=f"{forecast['confidence']:.0%} confidence"
                )

            with col3:
                st.metric("📅 Forecast Period", "30 days")

        # Recommendations
        if report["recommendations"]:
            st.subheader("💡 Performance Recommendations")
            for recommendation in report["recommendations"]:
                st.info(recommendation)
    else:
        st.info("No performance data available for visualization.")

    # Step 4: Invoice Generation
    st.subheader("📄 Invoice Generation")

    # Use the already extracted dispatcher name
    inv_name = dispatcher_name
    inv_id = selected_dispatcher

    # Generate and download invoice
    invoice_html = InvoiceGenerator.build_invoice_html(
        display_df, total_payout, inv_name, inv_id, currency_symbol
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        st.download_button(
            label="📥 Download Invoice (HTML)",
            data=invoice_html.encode("utf-8"),
            file_name=f"invoice_{selected_dispatcher}_{datetime.now().strftime('%Y%m%d')}.html",
            mime="text/html",
            use_container_width=True
        )

    with col2:
        st.caption("Professional invoice with detailed breakdown and company branding.")

    # Add footer at the end of the main content
    add_footer()


if __name__ == "__main__":
    main()
