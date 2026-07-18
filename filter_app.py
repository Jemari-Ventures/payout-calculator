"""
JMR Filter Prep — upload raw JMS Excel exports and download Template JMR–shaped workbooks.

Uses the same hub_filter pipeline as the CLI (`python -m hub_filter --hub …`).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from hub_filter.contract import OUTPUT_SHEETS
from hub_filter.pipeline import (
    hub_task_preset,
    list_excel_sheet_names,
    list_hub_presets,
    load_hub_config,
    run_upload_filter,
)
from return_sender_filter import DEFAULT_RETURN_SENDER_NAMES

ROOT = Path(__file__).resolve().parent
HUBS_DIR = ROOT / "hubs"


def _parse_dispatcher_ids(raw: str) -> List[str]:
    tokens: List[str] = []
    for line in (raw or "").replace(",", "\n").splitlines():
        part = line.strip()
        if part:
            tokens.append(part)
    return tokens or ["*"]


def _hub_options() -> Dict[str, Path]:
    presets = list_hub_presets(HUBS_DIR)
    return {p.stem: p for p in presets}


def main() -> None:
    st.set_page_config(
        page_title="JMR Filter Prep",
        page_icon="🧹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("JMR Filter Prep")
    st.caption(
        "Upload raw JMS Excel exports → trim columns → Template JMR format for Google Sheet / Streamlit."
    )

    hubs = _hub_options()
    if not hubs:
        st.error(f"No hub presets found in `{HUBS_DIR}`.")
        return

    with st.sidebar:
        st.header("Hub preset")
        hub_key = st.selectbox("Hub config", options=list(hubs.keys()), index=0)
        hub_cfg = load_hub_config(hubs[hub_key])
        st.caption(f"Hub ID: **{hub_cfg.get('hub_id', hub_key)}**")

        default_ids = hub_cfg.get("dispatcher_ids") or ["*"]
        if isinstance(default_ids, str):
            default_ids = [default_ids]
        dispatcher_raw = st.text_area(
            "Dispatcher IDs (exact or wildcards)",
            value="\n".join(str(x) for x in default_ids),
            height=120,
            help="One per line. Use `*` for all, or `PEN364*` for a prefix.",
        )
        dispatcher_ids = _parse_dispatcher_ids(dispatcher_raw)

        apply_return_sender_filter = st.checkbox(
            "Filter Return by allowed sender_name",
            value=True,
            help="Uses the default merchant allowlist when enabled.",
        )
        if apply_return_sender_filter:
            with st.expander("Allowed Return sender names", expanded=False):
                st.code("\n".join(DEFAULT_RETURN_SENDER_NAMES))

        st.markdown("---")
        st.caption("CLI equivalent still works: `python -m hub_filter --hub pen353`")

    task_presets: Dict[str, Dict[str, Any]] = {
        name: hub_task_preset(cfg) for name, cfg in (hub_cfg.get("tasks") or {}).items()
    }
    # Always offer contract tabs even if hub JSON only lists a subset
    available_tabs = list(dict.fromkeys([*task_presets.keys(), *OUTPUT_SHEETS.keys()]))

    st.subheader("1. Choose sheets & upload files")
    selected_tabs = st.multiselect(
        "Output sheets to build",
        options=available_tabs,
        default=list(task_presets.keys()) or ["Dispatch", "Pickup", "Return"],
        help="Only sheets with an uploaded file will be processed.",
    )

    built_tasks: Dict[str, Dict[str, Any]] = {}
    upload_summary: List[str] = []

    for tab in selected_tabs:
        preset = task_presets.get(tab, {})
        with st.expander(f"{tab}", expanded=tab in ("Dispatch", "Pickup", "Return")):
            uploaded = st.file_uploader(
                f"Raw Excel for {tab}",
                type=["xlsx", "xls"],
                key=f"upload_{tab}",
            )
            if uploaded is None:
                st.info("Upload a file to include this sheet.")
                continue

            file_bytes = uploaded.getvalue()
            try:
                sheet_names = list_excel_sheet_names(file_bytes)
            except Exception as exc:
                st.error(f"Could not read workbook: {exc}")
                continue

            default_sheet = str(preset.get("sheet", sheet_names[0] if sheet_names else "0"))
            if default_sheet not in sheet_names and sheet_names:
                default_sheet = sheet_names[0]
            sheet_index = sheet_names.index(default_sheet) if default_sheet in sheet_names else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                sheet_name = st.selectbox(
                    "Worksheet",
                    options=sheet_names,
                    index=sheet_index,
                    key=f"sheet_{tab}",
                )
            with col2:
                header_row = st.number_input(
                    "Header row (0-based)",
                    min_value=0,
                    max_value=20,
                    value=int(preset.get("header", 0)),
                    key=f"header_{tab}",
                )
            with col3:
                filter_col = st.text_input(
                    "Dispatcher filter column",
                    value=str(preset.get("column") or preset.get("filter_column") or ""),
                    key=f"col_{tab}",
                    help="Leave blank only if you enable skip filter.",
                )

            skip_filter = st.checkbox(
                "Skip dispatcher ID filter (keep all rows)",
                value=False,
                key=f"skip_{tab}",
            )

            rename_preview = preset.get("rename_columns") or {}
            if rename_preview:
                st.caption(f"Preset renames: `{json.dumps(rename_preview)}`")

            task_cfg: Dict[str, Any] = {
                **preset,
                "source": file_bytes,
                "sheet": sheet_name,
                "header": int(header_row),
                "skip_row_filter": skip_filter,
            }
            if filter_col.strip():
                task_cfg["column"] = filter_col.strip()
            if tab.lower() == "return":
                task_cfg["sender_names"] = True if apply_return_sender_filter else False

            built_tasks[tab] = task_cfg
            upload_summary.append(f"{tab}: {uploaded.name} → `{sheet_name}`")

    st.subheader("2. Run filter")
    if not built_tasks:
        st.warning("Upload at least one Excel file for a selected sheet.")
        return

    st.write("Ready:", ", ".join(upload_summary))
    st.write("Dispatcher filter:", ", ".join(dispatcher_ids))

    if st.button("Run filter", type="primary"):
        hub_runtime = dict(hub_cfg)
        if not apply_return_sender_filter:
            hub_runtime["return_sender_names"] = False

        with st.spinner("Filtering…"):
            try:
                frames, messages, payload = run_upload_filter(
                    tasks=built_tasks,
                    dispatcher_ids=dispatcher_ids,
                    hub_cfg=hub_runtime,
                )
            except Exception as exc:
                st.error(f"Filter failed: {exc}")
                st.exception(exc)
                return

        st.session_state["filter_frames"] = frames
        st.session_state["filter_messages"] = messages
        st.session_state["filter_payload"] = payload
        st.session_state["filter_hub"] = hub_cfg.get("hub_id", hub_key)
        st.success("Filter complete.")

    frames: Optional[Dict[str, pd.DataFrame]] = st.session_state.get("filter_frames")
    messages = st.session_state.get("filter_messages") or []
    payload: Optional[bytes] = st.session_state.get("filter_payload")

    if messages:
        with st.expander("Log", expanded=True):
            for line in messages:
                st.text(line)

    if frames:
        st.subheader("3. Preview & download")
        counts = {name: len(df) for name, df in frames.items()}
        st.write(counts)

        preview_tab = st.selectbox("Preview sheet", options=list(frames.keys()))
        preview_df = frames[preview_tab]
        st.dataframe(preview_df.head(100), hide_index=True, use_container_width=True)
        if len(preview_df) > 100:
            st.caption(f"Showing first 100 of {len(preview_df):,} rows.")

        if payload:
            stamp = datetime.now().strftime("%Y%m%d-%H%M")
            hub_label = st.session_state.get("filter_hub", hub_key)
            file_name = f"JMR-{hub_label}-filtered-{stamp}.xlsx"
            st.download_button(
                "Download Template JMR workbook",
                data=payload,
                file_name=file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            st.caption(
                "Next: paste/upload tabs into that hub’s Google Sheet, then open the dispatcher or management app."
            )


if __name__ == "__main__":
    main()
