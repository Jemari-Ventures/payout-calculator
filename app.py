import io
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def read_excel(file_bytes: bytes, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Read an Excel sheet safely with openpyxl."""
    buffer = io.BytesIO(file_bytes)
    try:
        if sheet_name:
            return pd.read_excel(buffer, engine="openpyxl", sheet_name=sheet_name)
        return pd.read_excel(buffer, engine="openpyxl")
    except Exception as exc:
        st.error(f"Failed to read Excel file: {exc}")
        raise


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalize column names."""
    renamed = {c: str(c).strip() for c in df.columns}
    return df.rename(columns=renamed)


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_string_like_columns(df: pd.DataFrame) -> List[str]:
    return df.columns.astype(str).tolist()


# -----------------------------
# Main App
# -----------------------------
def main() -> None:
    st.set_page_config(page_title="Dispatcher Payout Calculator", page_icon="🧮", layout="wide")
    st.title("🧮 Dispatcher Payout Calculator")
    st.caption("Upload an Excel sheet, filter by Dispatcher ID, and compute payout.")

    with st.sidebar:
        st.header("1️⃣ Upload Data")
        uploaded = st.file_uploader("Excel file (.xlsx)", type=["xlsx"])
        sheet_name: Optional[str] = None

        if uploaded is not None:
            try:
                xl = pd.ExcelFile(uploaded, engine="openpyxl")
                if len(xl.sheet_names) > 1:
                    sheet_name = st.selectbox("Sheet", options=xl.sheet_names)
                else:
                    sheet_name = xl.sheet_names[0]
            except Exception as exc:
                st.error(f"Could not open workbook: {exc}")

        st.header("2️⃣ Payout Settings")
        payout_mode = st.radio(
            "Payout mode",
            options=["Per parcel", "Tiered daily"],
            index=0,
        )

        currency_symbol = "RM"
        rate_per_parcel = 0.0
        tier_df = None

        if payout_mode == "Per parcel":
            rate_per_parcel = st.number_input(
                f"Rate per parcel ({currency_symbol})", min_value=0.0, value=1.0, step=0.05
            )
        else:
            st.caption("Tiered daily: sum parcels per day, map to tiers, pay per-parcel rate.")
            default_tiers = pd.DataFrame(
                {
                    "Tier": ["Tier 3", "Tier 2", "Tier 1"],
                    "Min Parcels": [0, 61, 121],
                    "Max Parcels": [60, 120, None],
                    "Rate (RM)": [0.95, 1.00, 1.10],
                }
            )
            tier_df = st.data_editor(default_tiers, use_container_width=True, num_rows="dynamic", key="tier_table")

        # Invoice info sidebar removed; invoice will auto-fill and download is enabled by default

    # -------------------------------------
    # Step 1: Upload check
    # -------------------------------------
    if uploaded is None:
        st.info("📤 Upload an Excel file to begin.")
        return

    df = read_excel(uploaded.getvalue(), sheet_name)
    if df is None or df.empty:
        st.warning("No data found in the selected sheet.")
        return

    df = normalize_column_names(df)
    all_columns = get_string_like_columns(df)

    # -------------------------------------
    # Step 2: Dispatcher selection
    # -------------------------------------
    st.subheader("Dispatcher Filter")

    default_dispatcher_col = "Dispatcher ID" if "Dispatcher ID" in all_columns else all_columns[0]
    dispatcher_col = st.selectbox("Dispatcher ID column", options=all_columns, index=all_columns.index(default_dispatcher_col))
    unique_dispatchers = sorted(df[dispatcher_col].dropna().astype(str).unique().tolist())
    selected_dispatcher = st.selectbox("Select Dispatcher ID", options=unique_dispatchers)

    filtered = df[df[dispatcher_col].astype(str) == str(selected_dispatcher)].copy()
    st.write(f"Matched rows: {len(filtered)}")

    st.dataframe(filtered.head(100), use_container_width=True)

    # -------------------------------------
    # Step 3: Payout Calculation
    # -------------------------------------
    st.subheader("Payout Calculation")

    if payout_mode == "Per parcel":
        parcel_col = st.selectbox("Parcel/Waybill column", options=all_columns)
        wb = filtered[parcel_col].astype(str).str.strip()
        is_valid_wb = wb.ne("") & wb.ne("nan")
        filtered["_parcels"] = is_valid_wb.astype(int)
        filtered["_payout"] = filtered["_parcels"] * float(rate_per_parcel)
        total_payout = float(filtered["_payout"].sum())

        st.metric(label="💰 Total payout", value=f"{currency_symbol} {total_payout:,.2f}")
        st.caption(f"{currency_symbol}{rate_per_parcel:.2f} per parcel")

        display_df = pd.DataFrame([
            {"Total Parcel": int(is_valid_wb.sum()), "Payout Rate": f"{currency_symbol}{rate_per_parcel:.2f}", "Payout": f"{currency_symbol}{total_payout:,.2f}"}
        ])

    else:
        default_date_col = "Delivery Signature" if "Delivery Signature" in all_columns else all_columns[0]
        date_col = st.selectbox(
            "Delivery date column",
            options=all_columns,
            index=all_columns.index(default_date_col),
        )
        parcel_col = st.selectbox("Parcel/Waybill column", options=all_columns)

        df_tiers = tier_df if tier_df is not None else pd.DataFrame()
        tiers = []
        for _, r in df_tiers.iterrows():
            tmin = r.get("Min Parcels")
            tmax = r.get("Max Parcels")
            trate = r.get("Rate (RM)")
            tname = r.get("Tier")
            if pd.notna(trate):
                tiers.append((tmin, tmax, trate, tname))

        if not tiers:
            st.error("Please define at least one tier.")
            st.stop()

        # Sort tiers
        tiers.sort(key=lambda t: (t[0] or 0), reverse=True)

        def map_rate(daily_parcels: float) -> Tuple[str, float]:
            for tmin, tmax, trate, tname in tiers:
                lower_ok = True if pd.isna(tmin) else daily_parcels >= tmin
                upper_ok = True if pd.isna(tmax) else daily_parcels <= tmax
                if lower_ok and upper_ok:
                    return str(tname), float(trate)
            return "Unmatched", 0.0

        work = filtered.copy()
        work["__date"] = pd.to_datetime(work[date_col], errors="coerce").dt.date
        work["__waybill"] = work[parcel_col].astype(str).str.strip()
        per_day = (
            work.groupby(["__date"], dropna=False)["__waybill"]
            .nunique(dropna=True)
            .reset_index()
            .rename(columns={"__waybill": "daily_parcels"})
        )

        per_day[["tier", "rate_per_parcel"]] = per_day["daily_parcels"].apply(lambda x: pd.Series(map_rate(float(x))))
        per_day["payout_per_day"] = per_day["daily_parcels"] * per_day["rate_per_parcel"]
        total_payout = float(per_day["payout_per_day"].sum())

        display_df = per_day.rename(
            columns={
                "__date": "Date",
                "daily_parcels": "Total Parcel",
                "tier": "Tier",
                "rate_per_parcel": "Payout Rate",
                "payout_per_day": "Payout",
            }
        )
        display_df["Payout Rate"] = display_df["Payout Rate"].apply(lambda x: f"{currency_symbol}{x:.2f}")
        display_df["Payout"] = display_df["Payout"].apply(lambda x: f"{currency_symbol}{x:.2f}")

        st.dataframe(display_df, use_container_width=True)
        st.metric(label="💰 Total payout", value=f"{currency_symbol} {total_payout:,.2f}")

        # Charts
        chart_df = per_day.rename(columns={"__date": "Date"})
        chart_df["Date"] = pd.to_datetime(chart_df["Date"])
        st.markdown("### 📈 Payout performance")
        st.altair_chart(
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    "Date:T",
                    timeUnit="yearmonthdate",
                    axis=alt.Axis(format="%Y-%m-%d", title="Date"),
                ),
                y=alt.Y("payout_per_day:Q", axis=alt.Axis(title=None)),
            )
            .properties(height=200, width="container"),
            use_container_width=True,
        )

    # -------------------------------------
    # Step 4: Invoice download
    # -------------------------------------
    # Invoice download is enabled by default; auto-fill values
    if True:
        inv_name = ""
        inv_id = ""

        # Auto-fetch dispatcher name from data if input left blank
        if not inv_name:
            for candidate_col in ["Dispatcher Name", "Name", "Rider Name"]:
                if candidate_col in filtered.columns:
                    values = (
                        filtered[candidate_col].dropna().astype(str).unique().tolist()
                    )
                    if values:
                        inv_name = values[0]
                        break
        if not inv_name:
            inv_name = str(selected_dispatcher)

        # Auto-fetch dispatcher id from data if input left blank
        if not inv_id:
            for candidate_col in ["Dispatcher ID", "ID", "Rider ID"]:
                if candidate_col in filtered.columns:
                    values = (
                        filtered[candidate_col].dropna().astype(str).unique().tolist()
                    )
                    if values:
                        inv_id = values[0]
                        break
        if not inv_id:
            inv_id = str(selected_dispatcher)

        def build_invoice_html(df_disp: pd.DataFrame, total: float, name: str, dpid: str) -> str:
            styles = """
                <style>
                body { font-family: Arial, sans-serif; color: #111; }
                h1 { margin-bottom: 0; }
                .small { color: #555; margin-top: 4px; }
                table { border-collapse: collapse; width: 100%; margin-top: 16px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background: #f5f5f5; }
                .total { font-weight: bold; margin-top: 16px; }
                .header-box { border: 1px solid #ddd; padding: 12px; border-radius: 6px; }
                </style>
            """
            head = f"""
                <div class="header-box">
                    <h1>Dispatcher Invoice</h1>
                    <div class="small">Name: {name} &nbsp;&nbsp;|&nbsp;&nbsp; Dispatcher ID: {dpid}</div>
                </div>
            """
            header_html = "<tr>" + "".join([f"<th>{c}</th>" for c in df_disp.columns]) + "</tr>"
            rows_html = "".join([
                "<tr>" + "".join([f"<td>{row[c]}</td>" for c in df_disp.columns]) + "</tr>"
                for _, row in df_disp.iterrows()
            ])
            table = f"<table>{header_html}{rows_html}</table>"
            footer = f"<p class='total'>Total payout: {currency_symbol} {total:,.2f}</p>"
            return f"<html><head>{styles}</head><body>{head}{table}{footer}</body></html>"

        invoice_html = build_invoice_html(display_df, total_payout, inv_name, inv_id)
        st.download_button(
            label="💾 Download Invoice (HTML)",
            data=invoice_html.encode("utf-8"),
            file_name=f"invoice_{selected_dispatcher}.html",
            mime="text/html",
        )

    st.caption("Tip: Adjust settings in sidebar for custom payout structures.")


if __name__ == "__main__":
    main()
