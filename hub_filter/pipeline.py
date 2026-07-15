"""Hub filter pipeline: JMS Excel → Template JMR–shaped workbook."""
from __future__ import annotations

import fnmatch
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from hub_filter.contract import COLUMN_ALIASES, OUTPUT_SHEETS, contract_columns
from return_sender_filter import (
    build_sender_name_allowlist,
    find_sender_name_column,
    normalize_sender_name,
    resolve_return_sender_names,
)


def _is_wildcard_pass_through(filter_values: Sequence[str]) -> bool:
    """True when config means 'keep all rows' (no dispatcher ID filter)."""
    if not filter_values:
        return True
    tokens = [str(v).strip() for v in filter_values if str(v).strip()]
    if not tokens:
        return True
    return any(token in ("*", "**") for token in tokens)


def row_matches_dispatcher_filters(value: Any, filter_values: Sequence[str]) -> bool:
    """Match a dispatcher ID against exact IDs and/or fnmatch patterns (e.g. PEN364*)."""
    text = str(value).strip() if pd.notna(value) else ""
    if not text:
        return False
    for pattern in filter_values:
        pat = str(pattern).strip()
        if not pat:
            continue
        if any(ch in pat for ch in "*?[]"):
            if fnmatch.fnmatch(text, pat):
                return True
        elif text == pat:
            return True
    return False


def _normalize_col_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("|", "").replace(".", "")


def find_column(df: pd.DataFrame, preferred_names: Sequence[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    lower = {str(c).strip().lower(): c for c in cols}
    for name in preferred_names:
        if name in cols:
            return name
        key = str(name).strip().lower()
        if key in lower:
            return lower[key]
    norms = {_normalize_col_name(c): c for c in cols}
    for name in preferred_names:
        hit = norms.get(_normalize_col_name(name))
        if hit is not None:
            return hit
    return None


def build_contract_frame(df: pd.DataFrame, sheet_tab: str, messages: List[str]) -> pd.DataFrame:
    """Project/rename to exact Template JMR columns for this tab."""
    wanted = contract_columns(sheet_tab)
    out = pd.DataFrame(index=df.index)
    for canon in wanted:
        aliases = COLUMN_ALIASES.get(canon, [canon])
        source = find_column(df, aliases)
        if source is None:
            messages.append(f"⚠ {sheet_tab}: column '{canon}' not found; writing blank.")
            out[canon] = ""
        else:
            out[canon] = df[source]
    return out


def process_task(
    sheet_tab: str,
    cfg: Dict[str, Any],
    filter_values: Sequence[str],
    hub_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[str, pd.DataFrame, List[str]]:
    messages: List[str] = []
    path = cfg["file"]
    if not Path(path).exists():
        messages.append(f"❌ {sheet_tab}: input file not found: {path}")
        cols = contract_columns(sheet_tab) if sheet_tab in OUTPUT_SHEETS else []
        return sheet_tab, pd.DataFrame(columns=cols), messages

    df = pd.read_excel(path, sheet_name=cfg["sheet"], header=cfg.get("header", 0))
    df.columns = df.columns.astype(str).str.strip()

    rename_columns = cfg.get("rename_columns") or {}
    if rename_columns:
        existing = {k: v for k, v in rename_columns.items() if k in df.columns}
        if existing:
            df = df.rename(columns=existing)

    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

    if cfg.get("skip_row_filter"):
        filtered = df.copy()
    else:
        filter_column_config = cfg.get("column") or cfg.get("filter_column")
        filter_col = None
        if isinstance(filter_column_config, list):
            for candidate in filter_column_config:
                filter_col = find_column(df, [candidate])
                if filter_col is not None:
                    break
        elif filter_column_config:
            filter_col = find_column(df, [filter_column_config])

        task_values = cfg.get("values", list(filter_values))
        if filter_col is None:
            messages.append(
                f"⚠ {sheet_tab}: filter column '{filter_column_config}' not found. "
                f"Available: {list(df.columns)}"
            )
            filtered = df.iloc[0:0].copy()
        elif _is_wildcard_pass_through(task_values):
            filtered = df.copy()
            messages.append(f"ℹ {sheet_tab}: dispatcher filter is wildcard/empty — keeping all rows.")
        else:
            mask = df[filter_col].apply(lambda v: row_matches_dispatcher_filters(v, task_values))
            filtered = df[mask].copy()

    sender_names_cfg = cfg.get("sender_names")
    if sender_names_cfg is None and hub_cfg is not None:
        sender_names_cfg = hub_cfg.get("return_sender_names")
    if sender_names_cfg is None and str(sheet_tab).strip().lower() == "return":
        sender_names_cfg = True  # use DEFAULT_RETURN_SENDER_NAMES

    allowlist_names = resolve_return_sender_names(sender_names_cfg)
    if allowlist_names:
        sender_col = find_sender_name_column(filtered)
        if sender_col is None:
            messages.append(f"⚠ {sheet_tab}: sender_name filter configured but column not found.")
            filtered = filtered.iloc[0:0].copy()
        else:
            allowed = build_sender_name_allowlist(allowlist_names)
            before = len(filtered)
            filtered = filtered[
                filtered[sender_col].apply(lambda v: normalize_sender_name(v) in allowed)
            ].copy()
            messages.append(
                f"ℹ {sheet_tab}: sender_name filter {before:,} → {len(filtered):,} rows"
            )

    if sheet_tab in OUTPUT_SHEETS:
        shaped = build_contract_frame(filtered, sheet_tab, messages)
    else:
        messages.append(f"⚠ {sheet_tab}: not in OUTPUT_SHEETS contract; writing raw filtered rows.")
        shaped = filtered

    messages.append(f"✔ {sheet_tab}: {len(shaped)} rows")
    return sheet_tab, shaped, messages


def load_hub_config(config_path: Path) -> Dict[str, Any]:
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        return json.loads(text)
    raise ValueError(f"Unsupported hub config format: {config_path.suffix} (use .json)")


def run_hub_filter(config_path: Path, *, max_workers: Optional[int] = None) -> Path:
    cfg = load_hub_config(config_path)
    # Omit, [], ["*"], or patterns like "PEN364*" are supported.
    dispatcher_ids = cfg.get("dispatcher_ids", ["*"])
    output_file = Path(cfg["output_file"])
    tasks: Dict[str, Dict[str, Any]] = cfg["tasks"]
    if not tasks:
        raise ValueError("Hub config has no tasks")

    workers = max_workers or min(len(tasks), os.cpu_count() or 4)
    results: Dict[str, pd.DataFrame] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_task, name, task_cfg, dispatcher_ids, cfg): name
            for name, task_cfg in tasks.items()
        }
        for future in as_completed(futures):
            sheet_name, filtered_df, messages = future.result()
            results[sheet_name] = filtered_df
            for message in messages:
                print(message)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name in tasks:
            results[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"\n✅ Hub filter wrote: {output_file}")
    return output_file
