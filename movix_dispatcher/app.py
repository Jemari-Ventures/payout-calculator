"""Streamlit UI for Movix IC courier payout (standalone)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from movix_dispatcher.calculator import MovixPayoutCalculator, results_to_summary_frame
from movix_dispatcher.config import MovixConfig
from movix_dispatcher.ingest import date_bounds, driver_directory, load_sheet1

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config_movix.json"


def _money(amount: float, symbol: str) -> str:
    return f"{symbol}{amount:,.2f}"


def main() -> None:
    st.set_page_config(
        page_title="Movix IC Payout",
        page_icon=None,
        layout="wide",
    )
    st.title("Movix IC Courier Payout")
    st.caption(
        "Standalone calculator for Kulim Independent Contractor couriers. "
        "Uses Sheet1 of the IC Payment export and the Movix weight / basic / attendance rate card."
    )

    config = MovixConfig.load(CONFIG_PATH)
    calc = MovixPayoutCalculator(config)

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload IC Payment Excel (.xlsx)", type=["xlsx"])
        st.markdown("**Rate card**")
        st.dataframe(pd.DataFrame(config.rate_table_rows()), hide_index=True, use_container_width=True)
        st.markdown(
            f"- Basic salary: **{config.currency_symbol}{config.basic_amount:,.0f}** "
            f"(≥{config.basic_required_working_days} days & avg ≥{config.basic_min_avg_parcels_per_day:.0f}/day)\n"
            f"- Attendance: **{config.currency_symbol}{config.attendance_amount:,.0f}** "
            f"(same gate; MC deducts {config.currency_symbol}{config.mc_deduction_per_day:.0f}/day; "
            f"{config.mc_days_disqualify_at}+ MC days → no allowance)\n"
            f"- Absent penalty: **{config.currency_symbol}{config.absent_penalty_per_day:.0f}/day**\n"
            f"- Billable services: `{', '.join(config.billable_service_prefixes)}*`"
        )

    if uploaded is None:
        st.info("Upload an IC Payment `.xlsx` file (Sheet1) to begin.")
        return

    try:
        df = load_sheet1(uploaded, config)
    except Exception as exc:  # noqa: BLE001 — surface ingest errors in UI
        st.error(f"Failed to load Sheet1: {exc}")
        return

    if df.empty:
        st.warning("No billable forward-delivery rows found after filtering.")
        return

    start, end = date_bounds(df)
    drivers = driver_directory(df)

    st.success(
        f"Loaded **{len(df):,}** billable parcels · "
        f"**{len(drivers)}** drivers · "
        f"period **{start.date() if start is not None else '—'}** → "
        f"**{end.date() if end is not None else '—'}**"
    )

    tab_batch, tab_driver = st.tabs(["Batch payout", "Single driver"])

    with tab_batch:
        st.subheader("Batch payout")
        st.caption("MC / absent days default to 0. Adjust per driver in the editor, then recalculate.")

        adj_seed = drivers[["driver_id", "driver_name"]].copy()
        adj_seed["mc_days"] = 0
        adj_seed["absent_days"] = 0
        edited = st.data_editor(
            adj_seed,
            hide_index=True,
            use_container_width=True,
            disabled=["driver_id", "driver_name"],
            column_config={
                "mc_days": st.column_config.NumberColumn("MC days", min_value=0, step=1),
                "absent_days": st.column_config.NumberColumn("Absent days", min_value=0, step=1),
            },
            key="movix_adj_editor",
        )

        if st.button("Calculate all drivers", type="primary"):
            adjustments: Dict[str, Dict[str, int]] = {}
            for row in edited.itertuples(index=False):
                adjustments[str(row.driver_id)] = {
                    "mc_days": int(row.mc_days or 0),
                    "absent_days": int(row.absent_days or 0),
                }
            results = calc.calculate_all(df, adjustments=adjustments)
            summary = results_to_summary_frame(results)
            st.session_state["movix_batch_summary"] = summary
            st.session_state["movix_batch_results"] = results

        summary = st.session_state.get("movix_batch_summary")
        if summary is not None and not summary.empty:
            display_cols = [
                "driver_id",
                "driver_name",
                "total_parcels",
                "working_days",
                "avg_parcels_per_day",
                "commission_total",
                "basic_salary",
                "attendance_allowance",
                "absent_penalty",
                "net_payout",
                "qualifies_for_basic",
                "eligibility_reason",
            ]
            present = [c for c in display_cols if c in summary.columns]
            st.dataframe(summary[present], hide_index=True, use_container_width=True)

            totals = {
                "Parcels": int(summary["total_parcels"].sum()),
                "Commission": _money(float(summary["commission_total"].sum()), config.currency_symbol),
                "Basic": _money(float(summary["basic_salary"].sum()), config.currency_symbol),
                "Attendance": _money(float(summary["attendance_allowance"].sum()), config.currency_symbol),
                "Absent penalties": _money(float(summary["absent_penalty"].sum()), config.currency_symbol),
                "Net payout": _money(float(summary["net_payout"].sum()), config.currency_symbol),
            }
            cols = st.columns(len(totals))
            for col, (label, value) in zip(cols, totals.items()):
                col.metric(label, value)

            csv_buf = io.StringIO()
            summary.to_csv(csv_buf, index=False)
            st.download_button(
                "Download batch summary CSV",
                data=csv_buf.getvalue(),
                file_name="movix_ic_payout_summary.csv",
                mime="text/csv",
            )

    with tab_driver:
        st.subheader("Single driver")
        options = [
            f"{row.driver_id} — {row.driver_name}" if row.driver_name else str(row.driver_id)
            for row in drivers.itertuples(index=False)
        ]
        choice = st.selectbox("Driver", options)
        driver_id = choice.split(" — ", 1)[0].strip() if choice else ""
        c1, c2 = st.columns(2)
        mc_days = c1.number_input("MC / compassionate days", min_value=0, value=0, step=1)
        absent_days = c2.number_input("Unauthorized absent days", min_value=0, value=0, step=1)

        if driver_id:
            result = calc.calculate_driver(df, driver_id, mc_days=mc_days, absent_days=absent_days)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Parcels", f"{result.total_parcels:,}")
            m2.metric("Working days", result.working_days)
            m3.metric("Avg / day", f"{result.avg_parcels_per_day:.2f}")
            m4.metric("Net payout", _money(result.net_payout, config.currency_symbol))

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Commission", _money(result.commission_total, config.currency_symbol))
            b2.metric("Basic salary", _money(result.basic_salary, config.currency_symbol))
            b3.metric("Attendance", _money(result.attendance_allowance, config.currency_symbol))
            b4.metric("Absent penalty", _money(result.absent_penalty, config.currency_symbol))

            st.info(result.eligibility_reason)

            if result.parcels_by_band:
                band_df = pd.DataFrame(
                    [
                        {
                            "Weight band": band,
                            "Parcels": result.parcels_by_band.get(band, 0),
                            "Commission": result.commission_by_band.get(band, 0.0),
                        }
                        for band in result.parcels_by_band
                    ]
                )
                st.markdown("**Commission by weight band**")
                st.dataframe(band_df, hide_index=True, use_container_width=True)

            if not result.daily_summary.empty:
                st.markdown("**Daily summary**")
                st.dataframe(result.daily_summary, hide_index=True, use_container_width=True)

            if not result.parcel_detail.empty:
                export_cols = [
                    c
                    for c in [
                        "service_date",
                        "service_type",
                        "lm_type",
                        "shipment_id",
                        "chargeable_weight",
                        "weight_band",
                        "commission_rate",
                        "commission_amount",
                        "driver_id",
                        "driver_name",
                        "tier",
                    ]
                    if c in result.parcel_detail.columns
                ]
                detail = result.parcel_detail[export_cols].copy()
                detail["service_date"] = pd.to_datetime(detail["service_date"]).dt.strftime("%Y-%m-%d")
                csv_detail = io.StringIO()
                detail.to_csv(csv_detail, index=False)
                st.download_button(
                    "Download parcel detail CSV",
                    data=csv_detail.getvalue(),
                    file_name=f"movix_ic_parcels_{driver_id}.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
