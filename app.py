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
            "description": "Advance Payout"
        }
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
    @st.cache_data(ttl=300)
    def read_google_sheet(url_or_id: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Fetch a Google Sheet as CSV and return a DataFrame."""
        spreadsheet_id, gid = DataSource._extract_gsheet_id_and_gid(url_or_id)
        if not spreadsheet_id:
            raise ValueError("Invalid Google Sheet URL or ID.")
        csv_url = DataSource._build_gsheet_csv_url(spreadsheet_id, sheet_name, gid)
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
                return DataSource.read_google_sheet(data_source["gsheet_url"], data_source["sheet_name"])
            except Exception as exc:
                st.error(f"Error reading Google Sheet: {exc}")
                return None
        return None


# =============================================================================
# PAYOUT CALCULATIONS
# =============================================================================

class PayoutCalculator:
    """Handle payout calculations."""

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
                return float(bonus), description

        return 0.0, "No KPI achieved"

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
            return float(bonus_amount), description, qualified_days
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
    def calculate_tiered_daily(filtered_df: pd.DataFrame, tiers_config: List,
                              kpi_config: List, special_rates_config: List,
                              attendance_config: dict, currency_symbol: str) -> Tuple[pd.DataFrame, float, float, str, float, str, int, pd.DataFrame]:
        """Calculate payout for tiered daily mode with KPI bonus, attendance bonus, and special rates."""
        tiers = []
        for tier in tiers_config:
            tmin = tier.get("Min Parcels")
            tmax = tier.get("Max Parcels")
            trate = tier.get("Rate (RM)")
            tname = tier.get("Tier")
            if pd.notna(trate):
                tiers.append((tmin, tmax, trate, tname))

        tiers.sort(key=lambda t: (t[0] or 0), reverse=True)

        def map_rate(daily_parcels: float) -> Tuple[str, float]:
            for tmin, tmax, trate, tname in tiers:
                lower_ok = True if pd.isna(tmin) else daily_parcels >= tmin
                upper_ok = True if pd.isna(tmax) else daily_parcels <= tmax
                if lower_ok and upper_ok:
                    return str(tname), float(trate)
            return "Unmatched", 0.0

        work = filtered_df.copy()
        work["__date"] = pd.to_datetime(work["Delivery Signature"], errors="coerce").dt.date
        work["__waybill"] = work["Waybill Number"].astype(str).str.strip()

        per_day = (
            work.groupby(["__date"], dropna=False)["__waybill"]
            .nunique(dropna=True)
            .reset_index()
            .rename(columns={"__waybill": "daily_parcels"})
        )

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
        base_payout = float(per_day["payout_per_day"].sum())

        total_parcels = int(per_day["daily_parcels"].sum())
        kpi_bonus, kpi_description = PayoutCalculator.calculate_kpi_bonus(total_parcels, kpi_config)

        attendance_bonus, attendance_desc, qualified_days = PayoutCalculator.calculate_attendance_bonus(
            per_day, attendance_config
        )

        total_payout = base_payout + kpi_bonus + attendance_bonus

        display_df = per_day[["__date", "daily_parcels", "tier", "rate_per_parcel", "payout_per_day"]].copy()
        display_df = display_df.rename(columns={
            "__date": "Date",
            "daily_parcels": "Total Parcel",
            "tier": "Tier",
            "rate_per_parcel": "Payout Rate",
            "payout_per_day": "Payout",
        })

        display_df["Payout Rate"] = display_df["Payout Rate"].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_df["Payout"] = display_df["Payout"].apply(lambda x: f"{currency_symbol}{x:.2f}")

        return display_df, total_payout, kpi_bonus, kpi_description, attendance_bonus, attendance_desc, qualified_days, per_day


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
        df_disp: pd.DataFrame, base_payout: float, kpi_bonus: float,
        attendance_bonus: float, total_payout: float, kpi_description: str,
        attendance_description: str, name: str, dpid: str, currency_symbol: str,
        advance_payout: float, advance_payout_desc: str
    ) -> str:
        total_parcels = df_disp['Total Parcel'].sum() if 'Total Parcel' in df_disp.columns else 0
        total_days = len(df_disp) if 'Date' in df_disp.columns else 0
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
                        <div class="label">Total Payout</div>
                        <div class="value">{currency_symbol} {total_payout:,.2f}</div>
                    </div>
                </div>

                <div class="summary">
                    <div class="chip">
                        <div class="label">Dispatcher ID</div>
                        <div class="value">{dpid}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Total Parcels</div>
                        <div class="value">{total_parcels:,}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Advance Payout</div>
                        <div class="value">{currency_symbol} {(advance_payout):,.2f}</div>
                    </div>
                    <div class="chip">
                        <div class="label">Balance Payout</div>
                        <div class="value">{currency_symbol} {(total_payout - advance_payout):,.2f}</div>
                    </div>
                </div>
        """

        if kpi_bonus > 0 or attendance_bonus > 0:
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
            html_content += '</div>'

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

        html_content += f"""
            <div class="payout-summary">
                <div class="payout-row">
                    <span>Base Payout (Daily Rates):</span>
                    <span>{currency_symbol} {base_payout:,.2f}</span>
                </div>
                <div class="payout-row">
                    <span>KPI Incentive Bonus:</span>
                    <span>{currency_symbol} {kpi_bonus:,.2f}</span>
                </div>
                <div class="payout-row">
                    <span>Attendance Incentive Bonus:</span>
                    <span>{currency_symbol} {attendance_bonus:,.2f}</span>
                </div>
                <div class="payout-row">
                    <span>{advance_payout_desc}:</span>
                    <span>- {currency_symbol} {advance_payout:,.2f}</span>
                </div>
                <div class="payout-row total">
                    <span>Balance Payout:</span>
                    <span>{currency_symbol} {(total_payout - advance_payout):,.2f}</span>
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
        layout="wide"
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
    required_columns = ["Dispatcher ID", "Waybill Number", "Delivery Signature"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Available columns: {', '.join(df.columns.tolist())}")
        add_footer()
        return

    st.subheader("👤 Dispatcher Selection")
    dispatcher_mapping = {}
    for candidate_col in ["Dispatcher Name", "Name", "Rider Name"]:
        if candidate_col in df.columns:
            temp_mapping = df[["Dispatcher ID", candidate_col]].dropna()
            temp_mapping["Dispatcher ID"] = temp_mapping["Dispatcher ID"].astype(str)
            temp_mapping[candidate_col] = temp_mapping[candidate_col].astype(str)
            for _, row in temp_mapping.iterrows():
                dispatcher_id = row["Dispatcher ID"]
                dispatcher_name = clean_dispatcher_name(row[candidate_col])
                if dispatcher_id not in dispatcher_mapping:
                    dispatcher_mapping[dispatcher_id] = dispatcher_name
            if dispatcher_mapping:
                break
    unique_dispatchers = sorted(df["Dispatcher ID"].dropna().astype(str).unique().tolist())
    if not unique_dispatchers:
        st.error("No dispatcher IDs found in the data.")
        add_footer()
        return
    dispatcher_options = []
    for dispatcher_id in unique_dispatchers:
        dispatcher_name = dispatcher_mapping.get(dispatcher_id, "Unknown")
        display_text = f"{dispatcher_name} ({dispatcher_id})"
        dispatcher_options.append(display_text)
    dispatcher_options.sort(key=lambda x: x.lower())
    display_to_id = {
        f"{dispatcher_mapping.get(did, 'Unknown')} ({did})": did
        for did in unique_dispatchers
    }
    selected_display = st.selectbox(
        "Select Dispatcher",
        options=dispatcher_options,
        help="Choose a dispatcher to view their payout details"
    )
    selected_dispatcher = display_to_id[selected_display]
    filtered = df[df["Dispatcher ID"].astype(str) == str(selected_dispatcher)].copy()
    with st.expander("View Filtered Data", expanded=False):
        st.dataframe(filtered.head(100), use_container_width=True)
    st.subheader("💰 Payout Calculation")
    dispatcher_name = ""
    for candidate_col in ["Dispatcher Name", "Name", "Rider Name"]:
        if candidate_col in filtered.columns:
            values = filtered[candidate_col].dropna().astype(str).unique().tolist()
            if values:
                dispatcher_name = clean_dispatcher_name(values[0])
                break
    currency_symbol = config["currency_symbol"]
    display_df, total_payout, kpi_bonus, kpi_description, attendance_bonus, attendance_desc, qualified_days, per_day = PayoutCalculator.calculate_tiered_daily(
        filtered, config["tiers"], config.get("kpi_incentives", []),
        config.get("special_rates", []), config.get("attendance_incentive", {}), currency_symbol
    )
    base_payout = total_payout - kpi_bonus - attendance_bonus

    advance_payout_config = config.get("advance_payout", {})
    advance_payout_enabled = advance_payout_config.get("enabled", False)
    advance_payout_percentage = advance_payout_config.get("percentage", 0.0)
    advance_payout_desc = advance_payout_config.get("description", "Advance Payout")
    advance_payout = 0.0
    if advance_payout_enabled and advance_payout_percentage > 0:
        advance_payout = round(total_payout * (advance_payout_percentage / 100.0), 2)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Dispatcher", value=dispatcher_name)
    with col2:
        total_parcels = per_day["daily_parcels"].sum()
        st.metric(label="Total Parcels", value=f"{total_parcels:,}")
    with col3:
        st.metric(label="Base Payout", value=f"{currency_symbol}{base_payout:,.2f}")
    with col4:
        st.metric(label="Total Payout", value=f"{currency_symbol}{total_payout:,.2f}")
    with col5:
        st.metric(label=advance_payout_desc, value=f"{currency_symbol}{advance_payout:,.2f}", help=f"{advance_payout_percentage}% of Total Payout")

    if kpi_bonus > 0:
        st.success(f"🎯 **KPI Incentive Achieved!** +{currency_symbol}{kpi_bonus:,.2f} ({kpi_description})")
    if attendance_bonus > 0:
        st.success(f"📅 **Attendance Incentive Achieved!** +{currency_symbol}{attendance_bonus:,.2f} ({attendance_desc})")
    else:
        st.info(f"📅 Attendance Progress: {attendance_desc}")

    st.dataframe(display_df, use_container_width=True)

    with st.expander("📊 Payout Breakdown", expanded=False):
        breakdown_col1, breakdown_col2, breakdown_col3, breakdown_col4, breakdown_col5 = st.columns(5)
        with breakdown_col1:
            st.metric("Base Payout", f"{currency_symbol}{base_payout:,.2f}", help="Daily rate payouts")
        with breakdown_col2:
            st.metric("KPI Bonus", f"{currency_symbol}{kpi_bonus:,.2f}", help=kpi_description)
        with breakdown_col3:
            st.metric("Attendance Bonus", f"{currency_symbol}{attendance_bonus:,.2f}", help=attendance_desc)
        with breakdown_col4:
            st.metric("Total", f"{currency_symbol}{total_payout:,.2f}", help="Base + KPI + Attendance")
        with breakdown_col5:
            st.metric(advance_payout_desc, f"{currency_symbol}{advance_payout:,.2f}", help=f"{advance_payout_percentage}% of total payout")

    st.subheader("📊 Performance Visualization")
    charts = DataVisualizer.create_performance_charts(per_day, currency_symbol)
    if charts:
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'parcels_payout' in charts:
                st.altair_chart(charts['parcels_payout'], use_container_width=True)
            else:
                st.info("No parcels and payout data available")
        with col2:
            if 'performance_scatter' in charts:
                st.altair_chart(charts['performance_scatter'], use_container_width=True)
            else:
                st.info("No performance scatter data available")
        with col3:
            if 'payout_trend' in charts:
                st.altair_chart(charts['payout_trend'], use_container_width=True)
            else:
                st.info("No payout trend data available")

        st.markdown("---")
        st.subheader("📈 Performance Summary")
        if per_day is not None and not per_day.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_parcels = per_day["daily_parcels"].mean()
                st.metric("Average Daily Parcels", f"{avg_parcels:.1f}")
            with col2:
                max_parcels = per_day["daily_parcels"].max()
                st.metric("Maximum Daily Parcels", f"{max_parcels:.0f}")
            with col3:
                avg_payout = per_day["payout_per_day"].mean()
                st.metric("Average Daily Payout", f"{currency_symbol}{avg_payout:.2f}")
            with col4:
                total_days = len(per_day)
                st.metric("Total Working Days", f"{total_days}")
            st.markdown("---")
            st.subheader("📅 Attendance Details")
            att_col1, att_col2, att_col3 = st.columns(3)
            with att_col1:
                st.metric("Qualified Days", f"{qualified_days}",
                         help=f"Days with ≥{config.get('attendance_incentive', {}).get('min_parcels_per_day', 30)} parcels")
            with att_col2:
                required_days = config.get('attendance_incentive', {}).get('required_days', 26)
                st.metric("Required Days", f"{required_days}",
                         help="Days needed for attendance bonus")
            with att_col3:
                if qualified_days >= required_days:
                    st.metric("Status", "✅ Achieved", delta="Bonus Earned")
                else:
                    remaining = required_days - qualified_days
                    st.metric("Status", f"❌ Not Yet", delta=f"{remaining} more days needed")
    else:
        st.info("No performance data available for visualization.")

    st.subheader("📄 Invoice Generation")
    inv_name, inv_id = "", ""
    for candidate_col in ["Dispatcher Name", "Name", "Rider Name"]:
        if candidate_col in filtered.columns:
            values = filtered[candidate_col].dropna().astype(str).unique().tolist()
            if values:
                inv_name = values[0]
                break
    if not inv_name:
        inv_name = str(selected_dispatcher)
    inv_id = str(selected_dispatcher)

    invoice_html = InvoiceGenerator.build_invoice_html(
        display_df, base_payout, kpi_bonus, attendance_bonus, total_payout,
        kpi_description, attendance_desc, inv_name, inv_id, currency_symbol,
        advance_payout, advance_payout_desc
    )

    st.download_button(
        label="📥 Download Invoice (HTML)",
        data=invoice_html.encode("utf-8"),
        file_name=f"invoice_{selected_dispatcher}_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
    )
    st.caption("Invoice generated based on current configuration and data.")
    add_footer()


if __name__ == "__main__":
    main()
