#!/usr/bin/env python3
"""
Export dispatch rows to Excel after removing any AWB/waybill that appears on the Return sheet.

Matches payout-calculator logic: same _norm_waybill() as app.py / management.py so dispatch
and return identifiers align (numeric floats, trailing ".0", scientific notation).

Usage (single workbook, two sheets):
  python export_dispatch_without_returns.py \\
    --workbook monthly.xlsx \\
    --dispatch-sheet Dispatch \\
    --return-sheet Return \\
    -o dispatch_delivery_only.xlsx

Usage (two files):
  python export_dispatch_without_returns.py \\
    --dispatch dispatch.xlsx \\
    --return return.xlsx \\
    -o dispatch_delivery_only.xlsx

Google Sheet (CSV export; sheet names must match tab names):
  python export_dispatch_without_returns.py \\
    --gsheet-url 'https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit' \\
    --dispatch-sheet Dispatch \\
    --return-sheet Return \\
    -o dispatch_delivery_only.xlsx

Optional: load default sheet names from config.json (same keys as the app):
  python export_dispatch_without_returns.py --config config.json -o out.xlsx
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from typing import Optional, Tuple
from urllib.parse import parse_qs, urlparse

import pandas as pd


def _extract_gsheet_id_and_gid(url_or_id: str) -> Tuple[Optional[str], Optional[str]]:
    if not url_or_id:
        return None, None
    if re.fullmatch(r"[A-Za-z0-9_-]{20,}", url_or_id):
        return url_or_id, None
    try:
        parsed = urlparse(url_or_id)
        path_parts = [p for p in parsed.path.split("/") if p]
        spreadsheet_id = None
        if "spreadsheets" in path_parts and "d" in path_parts:
            try:
                idx = path_parts.index("d")
                spreadsheet_id = path_parts[idx + 1]
            except Exception:
                spreadsheet_id = None
        query_gid = parse_qs(parsed.query).get("gid", [None])[0]
        frag_gid_match = re.search(r"gid=(\d+)", parsed.fragment or "")
        frag_gid = frag_gid_match.group(1) if frag_gid_match else None
        gid = query_gid or frag_gid
        return spreadsheet_id, gid
    except Exception:
        return None, None


def _build_gsheet_csv_url(spreadsheet_id: str, sheet_name: Optional[str], gid: Optional[str]) -> str:
    base = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
    if sheet_name:
        from urllib.parse import quote

        return f"{base}/gviz/tq?tqx=out:csv&sheet={quote(str(sheet_name), safe='')}"
    if gid:
        return f"{base}/export?format=csv&gid={gid}"
    return f"{base}/export?format=csv"


def read_google_sheet_csv(url_or_id: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    try:
        import requests
    except ImportError as e:
        raise ImportError("Google Sheet export requires the 'requests' package: pip install requests") from e

    spreadsheet_id, gid = _extract_gsheet_id_and_gid(url_or_id)
    if not spreadsheet_id:
        raise ValueError("Invalid Google Sheet URL or ID.")
    csv_url = _build_gsheet_csv_url(spreadsheet_id, sheet_name, gid)
    resp = requests.get(csv_url, timeout=60)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        if sheet_name and str(sheet_name).strip() and resp.status_code in (400, 404):
            sample = (resp.text or "")[:1000].lower()
            if any(
                x in sample
                for x in ("worksheet", "sheet", "unable to parse range", "not found", "invalid")
            ):
                return pd.DataFrame()
        raise
    content = resp.content
    if sheet_name and str(sheet_name).strip():
        head = content[:500].decode("utf-8", errors="ignore").lower().strip()
        if (
            "<html" in head
            or "<!doctype html" in head
            or ("worksheet" in head and "not found" in head)
            or ("sheet" in head and "not found" in head)
        ):
            return pd.DataFrame()
    return pd.read_csv(
        io.BytesIO(content),
        keep_default_na=False,
        na_values=[],
        encoding="utf-8",
        low_memory=False,
    )


def _norm_waybill(v) -> str:
    """Same normalization as payout-calculator app.py / management.py."""
    if pd.isna(v):
        return ""
    if isinstance(v, (int, float)):
        try:
            if isinstance(v, float) and v == int(v):
                return str(int(v))
            if isinstance(v, float):
                if v == int(v):
                    return str(int(v))
                return str(v).strip()
            return str(int(v))
        except (ValueError, OverflowError):
            return str(v).strip()
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none", "null", ""):
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if "e" in s.lower():
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return str(f).strip()
        except (ValueError, OverflowError):
            pass
    return s


def _is_fallback_dispatch_sheet(candidate: pd.DataFrame, dispatch: pd.DataFrame) -> bool:
    """If Return tab is missing, Google may return Dispatch CSV — treat Return as empty."""
    if candidate.empty or dispatch.empty:
        return False
    ccols = [str(c).strip().lower() for c in candidate.columns]
    dcols = [str(c).strip().lower() for c in dispatch.columns]
    if ccols != dcols:
        return False
    n = min(10, len(candidate), len(dispatch))
    if n <= 0:
        return False
    return candidate.head(n).reset_index(drop=True).equals(dispatch.head(n).reset_index(drop=True))


def find_waybill_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve AWB / waybill column (dispatch or return)."""
    if df is None or df.empty:
        return None
    exact = [
        "Waybill Number",
        "waybill_number",
        "Waybill",
        "waybill",
        "AWB No.",
        "AWB No",
        "No. AWB",
        "awb_no",
        "AWB",
    ]
    for name in exact:
        if name in df.columns:
            return name
    for col in df.columns:
        sl = str(col).strip().lower()
        if "waybill" in sl:
            return col
    for col in df.columns:
        sl = str(col).strip().lower()
        if "awb" in sl:
            return col
    return None


def filter_dispatch_excluding_returns(
    dispatch_df: pd.DataFrame, return_df: pd.DataFrame
) -> Tuple[pd.DataFrame, dict]:
    """
    Return dispatch_df with rows removed whose normalized waybill appears in return_df.
    """
    dispatch_wb = find_waybill_column(dispatch_df)
    if dispatch_wb is None:
        raise ValueError(
            "Could not find a waybill/AWB column on the dispatch sheet. "
            f"Columns: {list(dispatch_df.columns)}"
        )

    stats = {
        "dispatch_rows_in": len(dispatch_df),
        "return_rows": len(return_df) if return_df is not None else 0,
        "return_waybill_column": None,
        "return_awbs_unique": 0,
        "rows_removed": 0,
        "dispatch_rows_out": 0,
    }

    if return_df is None or return_df.empty:
        stats["dispatch_rows_out"] = len(dispatch_df)
        return dispatch_df.copy(), stats

    return_wb = find_waybill_column(return_df)
    stats["return_waybill_column"] = return_wb
    if return_wb is None:
        stats["dispatch_rows_out"] = len(dispatch_df)
        return dispatch_df.copy(), stats

    return_set = set()
    for x in return_df[return_wb].dropna().unique():
        n = _norm_waybill(x)
        if n:
            return_set.add(n)
    stats["return_awbs_unique"] = len(return_set)

    if not return_set:
        stats["dispatch_rows_out"] = len(dispatch_df)
        return dispatch_df.copy(), stats

    norm_dispatch = dispatch_df[dispatch_wb].apply(_norm_waybill)
    keep = ~norm_dispatch.isin(return_set)
    removed = int((~keep).sum())
    stats["rows_removed"] = removed
    out = dispatch_df.loc[keep].copy()
    stats["dispatch_rows_out"] = len(out)
    return out, stats


def _sheet_names_from_config(config_path: str) -> Tuple[str, str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    sheets = cfg.get("data_source", {}).get("excel_sheets", {})
    dispatch_name = sheets.get("dispatch", "Dispatch")
    return_name = sheets.get("return", "Return")
    return str(dispatch_name), str(return_name)


def _gsheet_url_from_config(config_path: str) -> Optional[str]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg.get("data_source", {}).get("gsheet_url")


def main() -> int:
    p = argparse.ArgumentParser(description="Export dispatch list minus return AWBs to Excel.")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--workbook", help="Excel file containing both dispatch and return sheets")
    src.add_argument("--gsheet-url", help="Google Sheets URL or spreadsheet ID")
    p.add_argument("--dispatch", metavar="PATH", help="Excel file containing dispatch sheet only")
    p.add_argument("--return", dest="return_path", metavar="PATH", help="Excel file containing return sheet only")
    p.add_argument("--dispatch-sheet", default="Dispatch", help="Dispatch sheet name (default: Dispatch)")
    p.add_argument("--return-sheet", default="Return", help="Return sheet name (default: Return)")
    p.add_argument("--config", help="config.json: use gsheet_url and excel_sheets dispatch/return names")
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output .xlsx path (e.g. dispatch_delivery_only.xlsx)",
    )
    args = p.parse_args()

    dispatch_sheet = args.dispatch_sheet
    return_sheet = args.return_sheet

    if args.config:
        dispatch_sheet, return_sheet = _sheet_names_from_config(args.config)
        gsheet = _gsheet_url_from_config(args.config)
        if not args.workbook and not args.gsheet_url and not args.dispatch:
            if not gsheet:
                print("config.json has no data_source.gsheet_url; use --workbook or --gsheet-url.", file=sys.stderr)
                return 1
            args.gsheet_url = gsheet

    if args.workbook:
        dispatch_df = pd.read_excel(args.workbook, sheet_name=dispatch_sheet)
        return_df = pd.read_excel(args.workbook, sheet_name=return_sheet)
    elif args.dispatch and args.return_path:
        dispatch_df = pd.read_excel(args.dispatch, sheet_name=dispatch_sheet)
        return_df = pd.read_excel(args.return_path, sheet_name=return_sheet)
    elif args.gsheet_url:
        dispatch_df = read_google_sheet_csv(args.gsheet_url, dispatch_sheet)
        return_df = read_google_sheet_csv(args.gsheet_url, return_sheet)
        if not return_df.empty and not dispatch_df.empty:
            if _is_fallback_dispatch_sheet(return_df, dispatch_df):
                return_df = pd.DataFrame()
    else:
        p.print_help()
        print(
            "\nProvide one of: --workbook | (--dispatch and --return) | --gsheet-url "
            "(or --config with gsheet_url).",
            file=sys.stderr,
        )
        return 1

    filtered, stats = filter_dispatch_excluding_returns(dispatch_df, return_df)

    summary = pd.DataFrame(
        [
            {"metric": "dispatch_rows_in", "value": stats["dispatch_rows_in"]},
            {"metric": "return_rows", "value": stats["return_rows"]},
            {"metric": "return_waybill_column", "value": stats["return_waybill_column"] or ""},
            {"metric": "unique_return_awbs_norm", "value": stats["return_awbs_unique"]},
            {"metric": "dispatch_rows_removed", "value": stats["rows_removed"]},
            {"metric": "dispatch_rows_out", "value": stats["dispatch_rows_out"]},
        ]
    )

    try:
        with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
            filtered.to_excel(writer, sheet_name="DispatchFiltered", index=False)
            summary.to_excel(writer, sheet_name="Summary", index=False)
    except ImportError:
        print("Writing .xlsx requires openpyxl: pip install openpyxl", file=sys.stderr)
        return 1

    print(f"Wrote: {args.output}")
    print(
        f"  dispatch in: {stats['dispatch_rows_in']}, "
        f"removed (in return set): {stats['rows_removed']}, "
        f"out: {stats['dispatch_rows_out']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
