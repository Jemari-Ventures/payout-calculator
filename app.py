import warnings
import urllib3
# Suppress the NotOpenSSLWarning
warnings.filterwarnings('ignore', category=urllib3.exceptions.NotOpenSSLWarning)

import io
from typing import List, Optional, Tuple, Dict
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
            "gsheet_url": "https://docs.google.com/spreadsheets/d/1an9G3ryAuy8tEEyPW4j-7Hfhs7kTtYzvOyvEzwLgXzs/edit?gid=0#gid=0",
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
        """Load dispatcher delivery data from Sheet1."""
        data_source = config["data_source"]

        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                return DataSource.read_google_sheet(data_source["gsheet_url"], "Sheet1")
            except Exception as exc:
                st.error(f"Error reading dispatcher data from Sheet1: {exc}")
                return None
        return None

    @staticmethod
    def load_pickup_data(config: dict) -> Optional[pd.DataFrame]:
        """Load pickup data from Sheet2."""
        data_source = config["data_source"]
        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                return DataSource.read_google_sheet(data_source["gsheet_url"], sheet_name="Pickup")
            except Exception as exc:
                st.warning(f"Could not load pickup data from Sheet2: {exc}")
                return None
        return None

    @staticmethod
    def load_duitnow_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        """Load DuitNow penalty data from Sheet3."""
        data_source = config["data_source"]
        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                return DataSource.read_google_sheet(data_source["gsheet_url"], sheet_name="DuitNow")
            except Exception as exc:
                st.warning(f"Could not load DuitNow penalty data from Sheet3: {exc}")
                return None
        return None

    @staticmethod
    def load_ldr_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        """Load LD&R penalty data from Sheet4."""
        data_source = config["data_source"]
        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                return DataSource.read_google_sheet(data_source["gsheet_url"], sheet_name="LDR")
            except Exception as exc:
                st.warning(f"Could not load LD&R penalty data from Sheet4: {exc}")
                return None
        return None

    @staticmethod
    def load_fake_attempt_penalty_data(config: dict) -> Optional[pd.DataFrame]:
        """Load Fake Attempt penalty data from Sheet5."""
        data_source = config["data_source"]
        if data_source["type"] == "gsheet" and data_source["gsheet_url"]:
            try:
                return DataSource.read_google_sheet(data_source["gsheet_url"], sheet_name="Fake Attempt")
            except Exception as exc:
                st.warning(f"Could not load Fake Attempt penalty data from Sheet5: {exc}")
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
    def calculate_penalty(dispatcher_id: str,
                         duitnow_df: Optional[pd.DataFrame] = None,
                         ldr_df: Optional[pd.DataFrame] = None,
                         fake_df: Optional[pd.DataFrame] = None) -> Dict:
        """Calculate total penalty for a dispatcher from three penalty sheets.

        Args:
            dispatcher_id: The dispatcher ID to check
            duitnow_df: DataFrame from Sheet3 (DuitNow penalty)
            ldr_df: DataFrame from Sheet4 (LD&R penalty)
            fake_df: DataFrame from Sheet5 (Fake attempt penalty)

        Returns:
            Dictionary containing penalty breakdown by type
        """
        penalty_breakdown = {
            'duitnow': {'amount': 0.0, 'count': 0, 'waybills': []},
            'ldr': {'amount': 0.0, 'count': 0, 'waybills': []},
            'fake_attempt': {'amount': 0.0, 'count': 0, 'waybills': []},
            'total_amount': 0.0,
            'total_count': 0
        }

        # 1. Process DuitNow penalty (Sheet3) - No waybills
        if duitnow_df is not None and not duitnow_df.empty:
            rider_col = None
            for col in duitnow_df.columns:
                if col.upper().strip() == "RIDER":
                    rider_col = col
                    break

            if rider_col is not None:
                duitnow_rows = duitnow_df[
                    duitnow_df[rider_col].astype(str).str.strip() == str(dispatcher_id).strip()
                ]

                if not duitnow_rows.empty:
                    penalty_col = None
                    for col in duitnow_df.columns:
                        if "PENALTY" in col.upper():
                            penalty_col = col
                            break

                    if penalty_col is not None:
                        penalty_amount = duitnow_rows[penalty_col].fillna(0).astype(float).sum()
                        penalty_breakdown['duitnow']['amount'] = float(penalty_amount)
                        penalty_breakdown['duitnow']['count'] = len(duitnow_rows)

        # 2. Process LD&R penalty (Sheet4)
        if ldr_df is not None and not ldr_df.empty:
            emp_id_col = None
            for col in ldr_df.columns:
                if "EMPLOYEE ID" in col.upper() or "EMPLOYEEID" in col.upper().replace(" ", ""):
                    emp_id_col = col
                    break

            if emp_id_col is not None:
                ldr_rows = ldr_df[
                    ldr_df[emp_id_col].astype(str).str.strip() == str(dispatcher_id).strip()
                ]

                if not ldr_rows.empty:
                    amount_col = None
                    for col in ldr_df.columns:
                        if col.upper().strip() == "AMOUNT":
                            amount_col = col
                            break

                    if amount_col is not None:
                        penalty_amount = ldr_rows[amount_col].fillna(0).astype(float).sum()
                        penalty_breakdown['ldr']['amount'] = float(penalty_amount)
                        penalty_breakdown['ldr']['count'] = len(ldr_rows)

                        # Get waybill numbers
                        awb_col = None
                        for col in ldr_df.columns:
                            if "NO. AWB" in col.upper() or "AWB" in col.upper():
                                awb_col = col
                                break

                        if awb_col is not None:
                            waybills = ldr_rows[awb_col].dropna().astype(str).str.strip().tolist()
                            penalty_breakdown['ldr']['waybills'] = [wb for wb in waybills if wb and wb.lower() != 'nan']

        # 3. Process Fake attempt penalty (Sheet5)
        if fake_df is not None and not fake_df.empty:
            disp_id_col = None
            for col in fake_df.columns:
                if "DISPATCHER ID" in col.upper() or "DISPATCHERID" in col.upper().replace(" ", ""):
                    disp_id_col = col
                    break

            if disp_id_col is not None:
                fake_rows = fake_df[
                    fake_df[disp_id_col].astype(str).str.strip() == str(dispatcher_id).strip()
                ]

                if not fake_rows.empty:
                    # Fixed RM1.00 per fake attempt
                    penalty_amount = len(fake_rows) * 1.00
                    penalty_breakdown['fake_attempt']['amount'] = float(penalty_amount)
                    penalty_breakdown['fake_attempt']['count'] = len(fake_rows)

                    # Get waybill numbers
                    waybill_col = None
                    for col in fake_df.columns:
                        if "WAYBILL NUMBER" in col.upper() or "WAYBILL" in col.upper():
                            waybill_col = col
                            break

                    if waybill_col is not None:
                        waybills = fake_rows[waybill_col].dropna().astype(str).str.strip().tolist()
                        penalty_breakdown['fake_attempt']['waybills'] = [wb for wb in waybills if wb and wb.lower() != 'nan']

        # Calculate totals
        penalty_breakdown['total_amount'] = (
            penalty_breakdown['duitnow']['amount'] +
            penalty_breakdown['ldr']['amount'] +
            penalty_breakdown['fake_attempt']['amount']
        )
        penalty_breakdown['total_count'] = (
            penalty_breakdown['duitnow']['count'] +
            penalty_breakdown['ldr']['count'] +
            penalty_breakdown['fake_attempt']['count']
        )

        return penalty_breakdown

    @staticmethod
    def calculate_pickup(pickup_df: pd.DataFrame, dispatcher_id: str, rate: float = 1.00) -> Tuple[int, float, pd.DataFrame]:
        """
        Counts unique Waybill Numbers for selected dispatcher in Sheet2
        based on Pick Up Dispatcher ID matching the dispatcher_id.

        Returns:
            Tuple of (parcel_count, payout, filtered_pickup_df)
        """
        if pickup_df is None or pickup_df.empty or not dispatcher_id:
            return 0, 0.0, pd.DataFrame()

        # Make sure columns exist
        if "Pick Up Dispatcher ID" not in pickup_df.columns or "Waybill Number" not in pickup_df.columns:
            return 0, 0.0, pd.DataFrame()

        # Convert dispatcher_id to string for comparison
        dispatcher_id_str = str(dispatcher_id).strip()

        # Clean and prepare the Pick Up Dispatcher ID column
        pickup_df = pickup_df.copy()
        pickup_df["Pick Up Dispatcher ID"] = pickup_df["Pick Up Dispatcher ID"].astype(str).str.strip()

        # Filter records for this dispatcher
        matched_records = pickup_df[pickup_df["Pick Up Dispatcher ID"] == dispatcher_id_str]

        if matched_records.empty:
            return 0, 0.0, pd.DataFrame()

        # Count unique waybills
        unique_waybills = matched_records["Waybill Number"].dropna().astype(str).str.strip()
        parcel_count = len(unique_waybills)
        payout = parcel_count * rate

        return parcel_count, payout, matched_records

    @staticmethod
    def calculate_tiered_daily(filtered_df: pd.DataFrame, tiers_config: List,
                              kpi_config: List, special_rates_config: List,
                              attendance_config: dict, currency_symbol: str,
                              duitnow_df: Optional[pd.DataFrame] = None,
                              ldr_df: Optional[pd.DataFrame] = None,
                              fake_df: Optional[pd.DataFrame] = None) -> Tuple:
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

        work = work[work["__waybill"].notna() & (work["__waybill"] != "") & (work["__waybill"].str.lower() != "nan")]

        work = work.sort_values(by=["__date", "__waybill", "Delivery Signature"])
        work = work.drop_duplicates(subset=["__date", "__waybill"], keep="last")

        per_day = (
            work.groupby(["__date"], dropna=False)["__waybill"]
            .nunique()
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

        # Calculate penalty breakdown
        penalty_breakdown = {'duitnow': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'ldr': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'fake_attempt': {'amount': 0.0, 'count': 0, 'waybills': []},
                           'total_amount': 0.0, 'total_count': 0}

        dispatcher_id = None
        for col in ["Dispatcher ID", "Dispatcher Id", "dispatcher_id"]:
            if col in filtered_df.columns:
                dispatcher_id = filtered_df[col].iloc[0] if len(filtered_df) > 0 else None
                break

        if dispatcher_id:
            penalty_breakdown = PayoutCalculator.calculate_penalty(
                str(dispatcher_id), duitnow_df, ldr_df, fake_df
            )

        # Calculate gross payout (base + bonuses - penalties)
        gross_payout = base_payout + kpi_bonus + attendance_bonus - penalty_breakdown['total_amount']

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

        return (display_df, base_payout, gross_payout, kpi_bonus, kpi_description, attendance_bonus,
                attendance_desc, qualified_days, per_day, penalty_breakdown)


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
        pickup_payout: float = 0.0,
        pickup_parcels: int = 0,
        kpi_description: str = "",
        attendance_description: str = ""
    ) -> str:
        total_parcels = df_disp['Total Parcel'].sum() if 'Total Parcel' in df_disp.columns else 0
        total_days = len(df_disp) if 'Date' in df_disp.columns else 0
        cleaned_name = clean_dispatcher_name(name)

        # Calculate final payout after advance
        final_payout = gross_payout + pickup_payout - advance_payout

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
                        <div class="label">Final Payout</div>
                        <div class="value">{currency_symbol} {final_payout:,.2f}</div>
                    </div>
                </div>

                <div class="summary">
                    <div class="chip">
                        <div class="label">Dispatcher ID</div>
                        <div class="value">{dpid}</div>
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
                        <div class="bonus-description">{penalty_breakdown['total_count']} parcel(s) affected</div>
                        <div class="penalty-detail">
                """

                # DuitNow penalties
                if penalty_breakdown['duitnow']['count'] > 0:
                    html_content += f"""
                        <div class="penalty-item">
                            <span><strong>DuitNow:</strong> {penalty_breakdown['duitnow']['count']} parcel(s)</span>
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
                    waybills_fake = ", ".join(penalty_breakdown['fake_attempt']['waybills'][:3])
                    if len(penalty_breakdown['fake_attempt']['waybills']) > 3:
                        waybills_fake += f" (+{len(penalty_breakdown['fake_attempt']['waybills']) - 3} more)"

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

        # Payout summary
        html_content += f"""
            <div class="payout-summary">
                <div class="payout-row">
                    <span>Base Delivery Payout:</span>
                    <span>{currency_symbol} {base_payout:,.2f}</span>
                </div>
                <div class="payout-row">
                    <span>Pickup Payout ({pickup_parcels} parcel(s) × {currency_symbol}1.00):</span>
                    <span>+ {currency_symbol} {pickup_payout:,.2f}</span>
                </div>
                <div class="payout-row">
                    <span>KPI Incentive Bonus:</span>
                    <span>+ {currency_symbol} {kpi_bonus:,.2f}</span>
                </div>
                <div class="payout-row">
                    <span>Attendance Incentive Bonus:</span>
                    <span>+ {currency_symbol} {attendance_bonus:,.2f}</span>
                </div>"""

        # Penalty details
        if penalty_breakdown['total_amount'] > 0:
            html_content += f"""
                <div class="payout-row" style="color: var(--error);">
                    <span>Total Penalty:</span>
                    <span>- {currency_symbol} {penalty_breakdown['total_amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['duitnow']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ DuitNow ({penalty_breakdown['duitnow']['count']} parcel(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['duitnow']['amount']:,.2f}</span>
                </div>"""

            if penalty_breakdown['ldr']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ LD&R ({penalty_breakdown['ldr']['count']} parcel(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['ldr']['amount']:,.2f}</span>
                </div>"""
                if penalty_breakdown['ldr']['waybills']:
                    waybills_display = ", ".join(penalty_breakdown['ldr']['waybills'][:5])
                    if len(penalty_breakdown['ldr']['waybills']) > 5:
                        waybills_display += f" (+{len(penalty_breakdown['ldr']['waybills']) - 5} more)"
                    html_content += f"""
                    <div class="payout-row penalty-detail-row" style="font-size: 12px; padding-left: 40px;">
                        <span style="color: var(--text-secondary);">Waybills: {waybills_display}</span>
                    </div>"""

            if penalty_breakdown['fake_attempt']['count'] > 0:
                html_content += f"""
                <div class="payout-row penalty-detail-row">
                    <span>↳ Fake Attempt ({penalty_breakdown['fake_attempt']['count']} parcel(s)):</span>
                    <span>- {currency_symbol} {penalty_breakdown['fake_attempt']['amount']:,.2f}</span>
                </div>"""
                if penalty_breakdown['fake_attempt']['waybills']:
                    waybills_display = ", ".join(penalty_breakdown['fake_attempt']['waybills'][:5])
                    if len(penalty_breakdown['fake_attempt']['waybills']) > 5:
                        waybills_display += f" (+{len(penalty_breakdown['fake_attempt']['waybills']) - 5} more)"
                    html_content += f"""
                    <div class="payout-row penalty-detail-row" style="font-size: 12px; padding-left: 40px;">
                        <span style="color: var(--text-secondary);">Waybills: {waybills_display}</span>
                    </div>"""

        # Add gross payout calculation
        html_content += f"""
                <div class="payout-row" style="border-top: 1px dashed var(--border); margin-top: 8px; padding-top: 8px;">
                    <span><strong>Gross Payout (Delivery + Pickup + Bonuses - Penalties):</strong></span>
                    <span><strong>{currency_symbol} {gross_payout + pickup_payout:,.2f}</strong></span>
                </div>
                <div class="payout-row">
                    <span>{advance_payout_desc}:</span>
                    <span>- {currency_symbol} {advance_payout:,.2f}</span>
                </div>
                <div class="payout-row total">
                    <span>Final Payout (Gross - Advance):</span>
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
    required_columns = ["Dispatcher ID", "Waybill Number", "Delivery Signature"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        st.info(f"Available columns: {', '.join(df.columns.tolist())}")
        add_footer()
        return

    df["Delivery Signature"] = pd.to_datetime(df["Delivery Signature"], errors="coerce")
    today = datetime.now()
    last_month_period = (today - pd.DateOffset(months=1)).to_period("M")
    monthly_mask = df["Delivery Signature"].dt.to_period("M") == last_month_period
    df = df[monthly_mask].copy()

    if df.empty:
        st.warning("No records found for last month. Please update the data source.")
        add_footer()
        return

    st.caption(f"Showing deliveries for {last_month_period.start_time:%B %Y}.")

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
    # Create a list of options for the dropdown with ID and cleaned name
    dispatcher_options = []
    for dispatcher_id, dispatcher_name in dispatcher_mapping.items():
        display_name = f"{dispatcher_id} - {dispatcher_name}" if dispatcher_name else dispatcher_id
        dispatcher_options.append((dispatcher_id, display_name))

    # Sort by dispatcher ID
    dispatcher_options.sort(key=lambda x: x[0])

    selected_display = st.selectbox(
        "Select Dispatcher",
        options=[opt[1] for opt in dispatcher_options],
        index=0 if dispatcher_options else None,
        help="Choose a dispatcher to view their payout details"
    )

    # Extract selected dispatcher ID
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

    # Filter data for selected dispatcher
    dispatcher_df = df[df["Dispatcher ID"].astype(str) == selected_dispatcher_id].copy()

    if dispatcher_df.empty:
        st.warning(f"No delivery records found for {selected_dispatcher_name or selected_dispatcher_id}.")
        add_footer()
        return

    # Load additional data sheets
    pickup_df = DataSource.load_pickup_data(config)
    duitnow_df = DataSource.load_duitnow_penalty_data(config)
    ldr_df = DataSource.load_ldr_penalty_data(config)
    fake_attempt_df = DataSource.load_fake_attempt_penalty_data(config)

    # Calculate payout
    st.subheader(f"💰 Payout Calculation for {selected_dispatcher_name or selected_dispatcher_id}")

    # Get advance payout configuration
    advance_config = config.get("advance_payout", {"enabled": False, "percentage": 0.0, "description": "Advance Payout"})
    advance_enabled = advance_config.get("enabled", False)
    advance_percentage = advance_config.get("percentage", 0.0)
    advance_description = advance_config.get("description", "Advance Payout (40% of Base Delivery)")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Payout Details", "📈 Performance Charts", "🧾 Invoice"])

    with tab1:
        # Calculate tiered daily payout
        try:
            (display_df, base_delivery_payout, gross_delivery_payout, kpi_bonus, kpi_description,
             attendance_bonus, attendance_desc, qualified_days,
             per_day_df, penalty_breakdown) = PayoutCalculator.calculate_tiered_daily(
                dispatcher_df,
                config["tiers"],
                config.get("kpi_incentives", []),
                config.get("special_rates", []),
                config.get("attendance_incentive", {}),
                config["currency_symbol"],
                duitnow_df,
                ldr_df,
                fake_attempt_df
            )

            # Calculate pickup payout (assuming RM1.00 per pickup parcel)
            pickup_parcels, pickup_payout, pickup_filtered_df = PayoutCalculator.calculate_pickup(
                pickup_df, selected_dispatcher_id, rate=1.00
            )

            # Calculate gross total payout (delivery + pickup + bonuses - penalties)
            gross_total_payout = gross_delivery_payout + pickup_payout

            # Calculate advance payout (40% of base delivery payout only, not including pickup, bonuses, penalties)
            advance_payout = 0.0
            if advance_enabled and advance_percentage > 0:
                advance_payout = (base_delivery_payout * advance_percentage) / 100.0

            # Calculate final payout
            final_payout = gross_total_payout - advance_payout

            # Display metrics in columns
            col1, col2, col3, col4, col5, col6 = st.columns(6)

            with col1:
                st.metric(
                    "Total Delivery Parcels",
                    f"{display_df['Total Parcel'].sum():,}",
                    help="Total parcels delivered in the month"
                )

            with col2:
                st.metric(
                    "Working Days",
                    f"{len(display_df)}",
                    help="Number of days worked"
                )

            with col3:
                st.metric(
                    "Base Delivery Payout",
                    f"{config['currency_symbol']}{base_delivery_payout:,.2f}",
                    help="Total base payout from daily deliveries"
                )

            with col4:
                st.metric(
                    "Gross Payout",
                    f"{config['currency_symbol']}{gross_total_payout:,.2f}",
                    help="Total gross payout including bonuses and penalties"
                )

            with col5:
                st.metric(
                    "Advance Payout",
                    f"{config['currency_symbol']}{advance_payout:,.2f}",
                    help="40% of base delivery payout"
                )

            with col6:
                st.metric(
                    "Final Payout",
                    f"{config['currency_symbol']}{final_payout:,.2f}",
                    help="Gross payout minus advance payout"
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
                if pickup_parcels > 0:
                    st.metric(
                        "Pickup Payout",
                        f"{config['currency_symbol']}{pickup_payout:,.2f}",
                        delta=f"{pickup_parcels} parcels",
                        delta_color="normal"
                    )
                else:
                    st.metric(
                        "Pickup Payout",
                        f"{config['currency_symbol']}0.00",
                        delta="No pickups",
                        delta_color="off"
                    )

            with col4:
                if penalty_breakdown['total_amount'] > 0:
                    st.metric(
                        "Total Penalty",
                        f"-{config['currency_symbol']}{penalty_breakdown['total_amount']:,.2f}",
                        delta=f"{penalty_breakdown['total_count']} parcels",
                        delta_color="inverse"
                    )
                else:
                    st.metric(
                        "Total Penalty",
                        f"{config['currency_symbol']}0.00",
                        delta="No penalties",
                        delta_color="off"
                    )

            # Display pickup details if available
            if pickup_parcels > 0 and not pickup_filtered_df.empty:
                with st.expander("📦 Pickup Details", expanded=False):
                    st.info(f"**Total Pickup Parcels: {pickup_parcels}** | **Payout: {config['currency_symbol']}{pickup_payout:,.2f}** (RM1.00 per parcel)")

                    # Clean up column names for display
                    display_pickup_df = pickup_filtered_df.copy()
                    display_pickup_df.columns = [str(col).title() for col in display_pickup_df.columns]

                    # Display the filtered pickup dataframe
                    st.dataframe(
                        display_pickup_df,
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                with st.expander("📦 Pickup Details", expanded=False):
                    st.info("No pickup records found for this dispatcher.")

            # Display penalty details if any
            if penalty_breakdown['total_amount'] > 0:
                with st.expander("⚠️ Penalty Details"):
                    penalty_col1, penalty_col2, penalty_col3 = st.columns(3)

                    with penalty_col1:
                        if penalty_breakdown['duitnow']['count'] > 0:
                            st.error(f"**DuitNow:** {penalty_breakdown['duitnow']['count']} parcel(s) - {config['currency_symbol']}{penalty_breakdown['duitnow']['amount']:,.2f}")

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
                - Gross Total: {config['currency_symbol']}{gross_total_payout:,.2f}
                """)

            with breakdown_col3:
                if advance_enabled:
                    st.warning(f"""
                    **Advance & Final:**
                    - Advance (40% of Base Delivery): -{config['currency_symbol']}{advance_payout:,.2f}
                    - **Final Payout:** {config['currency_symbol']}{final_payout:,.2f}
                    """)
                else:
                    st.success(f"""
                    **Final Payout:**
                    - **Gross Total:** {config['currency_symbol']}{gross_total_payout:,.2f}
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
                name=selected_dispatcher_name or selected_dispatcher_id,
                dpid=selected_dispatcher_id,
                currency_symbol=config["currency_symbol"],
                pickup_payout=pickup_payout,
                pickup_parcels=pickup_parcels,
                kpi_description=kpi_description,
                attendance_description=attendance_desc
            )

            # Display invoice
            st.components.v1.html(invoice_html, height=1200, scrolling=True)

            # Download buttons
            col1, col2, col3 = st.columns(3)

            with col1:
                # Download as HTML
                st.download_button(
                    label="📥 Download Invoice (HTML)",
                    data=invoice_html,
                    file_name=f"invoice_{selected_dispatcher_id}_{last_month_period.start_time:%Y%m}.html",
                    mime="text/html",
                    use_container_width=True
                )

            with col2:
                # Download as CSV (daily breakdown)
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Daily Breakdown (CSV)",
                    data=csv_data,
                    file_name=f"breakdown_{selected_dispatcher_id}_{last_month_period.start_time:%Y%m}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col3:
                # Download summary as text
                summary_text = f"""
INVOICE SUMMARY
===============
Dispatcher: {selected_dispatcher_name or selected_dispatcher_id}
Dispatcher ID: {selected_dispatcher_id}
Period: {last_month_period.start_time:%B %Y}

SUMMARY
-------
Total Delivery Parcels: {display_df['Total Parcel'].sum():,}
Working Days: {len(display_df)}
Pickup Parcels: {pickup_parcels:,}

PAYOUT BREAKDOWN
----------------
Base Delivery Payout: {config['currency_symbol']}{base_delivery_payout:,.2f}
Pickup Payout: +{config['currency_symbol']}{pickup_payout:,.2f}
KPI Bonus: +{config['currency_symbol']}{kpi_bonus:,.2f}
Attendance Bonus: +{config['currency_symbol']}{attendance_bonus:,.2f}
Total Penalties: -{config['currency_symbol']}{penalty_breakdown['total_amount']:,.2f}

GROSS TOTAL PAYOUT
------------------
Gross Payout: {config['currency_symbol']}{gross_total_payout:,.2f}

ADVANCE PAYOUT
--------------
Advance (40% of Base Delivery): -{config['currency_symbol']}{advance_payout:,.2f}

FINAL PAYOUT
------------
Final Payout (Gross - Advance): {config['currency_symbol']}{final_payout:,.2f}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """
                st.download_button(
                    label="📝 Download Summary (TXT)",
                    data=summary_text,
                    file_name=f"summary_{selected_dispatcher_id}_{last_month_period.start_time:%Y%m}.txt",
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
