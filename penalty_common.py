"""Shared penalty helpers used by app.py and management.py."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PENALTY_DATE_COLUMNS: Dict[str, List[str]] = {
    "duitnow": ["created_at", "date", "Date"],
    "ldr": ["created_at", "date", "Date"],
    "fake_attempt": ["Date", "date", "created_at"],
    "cod": ["created_at", "date", "Date"],
    "binding": ["created_at", "date", "Date"],
    "pending_parcel": ["date", "created_at", "Date"],
    "parcel_lost": ["date", "created_at", "Date"],
    "no_outbound_scan": [
        "Scanning Time | Last",
        "Scanning Time|Last",
        "scanning_time_last",
        "Scanning Time",
        "date",
        "created_at",
    ],
}


def find_column(
    df: pd.DataFrame,
    possible_names: Sequence[str],
    case_sensitive: bool = False,
) -> Optional[str]:
    if df is None or df.empty:
        return None
    for name in possible_names:
        if name in df.columns:
            return name
    if case_sensitive:
        return None
    cols_lower = {str(col).lower().strip(): col for col in df.columns}
    for name in possible_names:
        key = str(name).lower().strip()
        if key in cols_lower:
            return cols_lower[key]
    cols_norm = {
        str(col).lower().strip().replace(" ", "_").replace(".", "").replace("|", "").replace("/", "_"): col
        for col in df.columns
    }
    for name in possible_names:
        key = str(name).lower().strip().replace(" ", "_").replace(".", "").replace("|", "").replace("/", "_")
        if key in cols_norm:
            return cols_norm[key]
    return None


def clean_penalty_dispatcher_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return ""
    if re.fullmatch(r".+\.0", s):
        s = s[:-2]
    return s.upper()


def normalize_waybill(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, float)):
        try:
            if isinstance(value, float) and value == int(value):
                return str(int(value))
            return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value).strip()
        except (ValueError, OverflowError):
            return str(value).strip()
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "null", ""):
        return ""
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    # Only treat scientific notation when the whole token is numeric sci-notation.
    # Do not use `"e" in s` — that breaks alphanumeric AWBs containing the letter E.
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?[eE][+-]?\d+", s):
        try:
            f = float(s)
            return str(int(f)) if f == int(f) else str(f).strip()
        except (ValueError, OverflowError):
            pass
    return s


def is_waybill_column_name(name) -> bool:
    """Return True if a column header is a waybill/AWB field."""
    s = str(name).replace("\ufeff", "").strip().lower()
    if not s:
        return False
    if s in {
        "waybill_number",
        "waybill number",
        "waybill",
        "waybill no",
        "waybill no.",
        "awb",
        "awb no.",
        "awb no",
        "awb_no",
        "no. awb",
        "awbno",
        "awbno.",
    }:
        return True
    return "waybill" in s or "awb" in s


def waybill_column_read_dtypes(columns) -> Dict[str, str]:
    """Build a pandas read_csv dtype map so waybill columns load as text."""
    dtype: Dict[str, str] = {}
    for raw in columns:
        col = str(raw).replace("\ufeff", "").strip()
        if is_waybill_column_name(col):
            # Use the raw header token so pandas matches the CSV column name.
            dtype[str(raw)] = str
    return dtype


def find_penalty_waybill_column(df: pd.DataFrame) -> Optional[str]:
    col = find_column(
        df,
        [
            "AWB No.",
            "AWB No",
            "AWB NO.",
            "AWB NO",
            "Waybill Number",
            "waybill_number",
            "Waybill",
            "waybill",
            "AWB",
            "awb_no",
        ],
    )
    if col is not None:
        return col
    for c in df.columns:
        s = str(c).strip().lower().replace(" ", "")
        if s in ("awbno", "awbno.", "waybillnumber", "waybillno") or s == "awb_no":
            return c
    for c in df.columns:
        s = str(c).strip().lower()
        if ("awb" in s and "no" in s) or "waybill" in s:
            return c
    return None


def find_pickup_commission_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve pickup commission column (handles common typo 'commisson')."""
    col = find_column(
        df,
        ["commission", "Commission", "COMMISSION", "commisson", "Commisson", "COMMISSON"],
    )
    if col is not None:
        return col
    for c in df.columns:
        if "commis" in str(c).strip().lower():
            return c
    return None


PICKUP_JTD_QR_COMMISSION = 1.80
PICKUP_EASYPARCEL_LOW_WEIGHT_COMMISSION = 1.00
PICKUP_EASYPARCEL_HIGH_WEIGHT_COMMISSION = 3.00
PICKUP_EASYPARCEL_WEIGHT_THRESHOLD_KG = 10.0
PICKUP_STANDARD_SOURCE_COMMISSION = 1.00
PICKUP_DEFAULT_SOURCE_COMMISSION = 0.05

PICKUP_ONE_RM_ORDER_SOURCES = frozenset({
    "TT-RETURN",
    "LAZADA-RETURN",
    "SHEIN-RETURN",
    "CAINIAO-RETURN",
    "APP-ANDROID",
    "APP-IOS",
    "APP-HORMONY",
    "WEBSITE",
    "WHATSAPP",
    "WECHAT",
})


def normalize_pickup_order_source(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return ""
    return text.upper()


def find_pickup_order_source_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, ["Order Source", "order_source", "ORDER SOURCE", "Order source"])


def find_pickup_billing_weight_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(
        df,
        ["Billing Weight", "billing_weight", "Billing weight", "Weight", "weight", "weight_kg"],
    )


def pickup_commission_for_source_weight(order_source, billing_weight=0.0) -> float:
    """Commission (RM) from Order Source and Billing Weight."""
    src = normalize_pickup_order_source(order_source)
    if src == "JTD QR":
        return PICKUP_JTD_QR_COMMISSION
    if src == "EASYPARCEL":
        weight = penalty_cell_to_float(billing_weight)
        if weight > PICKUP_EASYPARCEL_WEIGHT_THRESHOLD_KG:
            return PICKUP_EASYPARCEL_HIGH_WEIGHT_COMMISSION
        return PICKUP_EASYPARCEL_LOW_WEIGHT_COMMISSION
    if src in PICKUP_ONE_RM_ORDER_SOURCES:
        return PICKUP_STANDARD_SOURCE_COMMISSION
    if src:
        return PICKUP_DEFAULT_SOURCE_COMMISSION
    return PICKUP_DEFAULT_SOURCE_COMMISSION


def compute_pickup_commission_series(
    df: pd.DataFrame,
    *,
    fallback_rate: float = 1.0,
) -> pd.Series:
    """Per-row pickup commission: Order Source rules, else sheet commission, else flat rate."""
    if df is None or df.empty:
        return pd.Series(dtype=float)

    order_source_col = find_pickup_order_source_column(df)
    if order_source_col is not None:
        weight_col = find_pickup_billing_weight_column(df)
        if weight_col is not None:
            weights = df[weight_col]
        else:
            weights = pd.Series(0.0, index=df.index)
        return pd.Series(
            [
                pickup_commission_for_source_weight(src, weight)
                for src, weight in zip(df[order_source_col], weights)
            ],
            index=df.index,
            dtype=float,
        )

    commission_col = find_pickup_commission_column(df)
    if commission_col is not None:
        return df[commission_col].map(penalty_cell_to_float)

    return pd.Series(float(fallback_rate), index=df.index, dtype=float)


def sum_pickup_commission(df: pd.DataFrame, *, fallback_rate: float = 1.0) -> float:
    if df is None or df.empty:
        return 0.0
    return round(float(compute_pickup_commission_series(df, fallback_rate=fallback_rate).sum()), 2)


def find_penalty_dispatcher_column(df: pd.DataFrame) -> Optional[str]:
    col = find_column(
        df,
        [
            "Dispatcher ID",
            "dispatcher_id",
            "Dispatcher Id",
            "DISPATCHER ID",
            "Operator | Last",
            "Operator|Last",
        ],
    )
    if col is not None:
        return col
    for c in df.columns:
        cl = str(c).strip().lower()
        if ("operator" in cl and "last" in cl) or ("dispatcher" in cl and "id" in cl):
            return c
    return None


def find_employee_id_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, ["Employee ID", "employee_id", "Employee Id", "EMPLOYEE ID", "EmployeeID"])


def find_rider_column(df: pd.DataFrame) -> Optional[str]:
    col = find_column(df, ["Rider", "rider", "RIDER"])
    if col is not None:
        return col
    for c in df.columns:
        if "rider" in str(c).lower():
            return c
    return None


def find_achieve_column(df: pd.DataFrame) -> Optional[str]:
    col = find_column(df, ["Achieve", "achieve", "ACHIEVE"])
    if col is not None:
        return col
    for c in df.columns:
        if "achieve" in str(c).lower():
            return c
    return None


def find_amount_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve amount column on hub/socso/overpaid/reward style sheets."""
    return find_column(df, ["Amount", "amount", "AMOUNT"])


def find_reward_dispatcher_name_column(df: pd.DataFrame) -> Optional[str]:
    """Resolve dispatcher_name on Reward sheet."""
    return find_column(df, ["dispatcher_name", "Dispatcher Name", "Dispatcher name"])


def preprocess_dispatcher_amount_penalty_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize hub/socso/overpaid sheets: Dispatcher ID + Amount."""
    df_processed = df.copy()
    dispatcher_id_col = find_penalty_dispatcher_column(df)
    amount_col = find_amount_column(df)

    if dispatcher_id_col:
        df_processed["_dispatcher_id_normalized"] = df_processed[dispatcher_id_col].apply(
            clean_penalty_dispatcher_id
        )

    if amount_col:
        if "penalty_numeric" not in df_processed.columns:
            df_processed["penalty_numeric"] = df_processed[amount_col].apply(penalty_cell_to_decimal)
        df_processed = df_processed[df_processed["penalty_numeric"] > 0].copy()

    return df_processed


def filter_dispatcher_amount_records(df: pd.DataFrame, dispatcher_id_clean: str) -> pd.DataFrame:
    """Return dispatcher rows from a hub/socso/overpaid style sheet."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "_dispatcher_id_normalized" in df.columns:
        return df[df["_dispatcher_id_normalized"] == dispatcher_id_clean]
    dispatcher_id_col = find_penalty_dispatcher_column(df)
    if dispatcher_id_col is None:
        return pd.DataFrame()
    dispatcher_series = df[dispatcher_id_col].apply(clean_penalty_dispatcher_id)
    return df[dispatcher_series == dispatcher_id_clean]


def sum_rounded_penalty_numeric_records(
    records: pd.DataFrame,
    source_df: Optional[pd.DataFrame] = None,
) -> Tuple[float, int]:
    """Sum positive amount values for matched dispatcher records."""
    if records is None or records.empty:
        return 0.0, 0
    if "penalty_numeric" in records.columns:
        rounded_penalties = [
            penalty.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            for penalty in records["penalty_numeric"].tolist()
        ]
        return float(sum(rounded_penalties)), len(records)

    amount_df = source_df if source_df is not None else records
    amount_col = find_amount_column(amount_df)
    if amount_col is None:
        return 0.0, 0
    amount_values = records[amount_col].apply(penalty_cell_to_decimal)
    rounded_penalties = [
        penalty.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        for penalty in amount_values.tolist()
        if penalty > 0
    ]
    if not rounded_penalties:
        return 0.0, 0
    return float(sum(rounded_penalties)), len(rounded_penalties)


BENEFIT_DEDUCTION_SHEET_KEYS = ("socso", "overpaid")


def build_dispatcher_amount_deduction_map(
    df: Optional[pd.DataFrame],
    normalize_id: Callable[[object], str],
) -> Dict[str, float]:
    """Build per-dispatcher deduction totals from dispatcher_id + amount sheets."""
    if df is None or df.empty:
        return {}

    disp_col = find_penalty_dispatcher_column(df)
    amount_col = find_amount_column(df)
    if disp_col is None or amount_col is None:
        return {}

    work = df.copy()
    work["_dispatcher_id_normalized"] = work[disp_col].apply(normalize_id)
    work["penalty_numeric"] = work[amount_col].apply(penalty_cell_to_decimal)
    work = work[work["penalty_numeric"] > 0].copy()
    if work.empty:
        return {}

    grouped = (
        work.groupby("_dispatcher_id_normalized")["penalty_numeric"]
        .sum()
        .round(2)
    )
    return {str(key): float(value) for key, value in grouped.items() if key}


def sum_dispatcher_amount_penalty_float(
    df: Optional[pd.DataFrame],
    dispatcher_id: str,
) -> Tuple[float, int]:
    """Sum Amount column for one dispatcher (app.py style)."""
    if df is None or df.empty:
        return 0.0, 0
    disp_col = find_penalty_dispatcher_column(df)
    if disp_col is None:
        return 0.0, 0
    rows = filter_rows_for_penalty_dispatcher(df, disp_col, dispatcher_id)
    if rows.empty:
        return 0.0, 0
    amount_col = find_amount_column(df)
    if amount_col is None:
        return 0.0, 0
    amount_values = rows[amount_col].apply(penalty_cell_to_float)
    amount_values = amount_values[amount_values > 0]
    return round(float(amount_values.sum()), 2), int(len(amount_values))


def sum_benefit_deduction_float(
    df: Optional[pd.DataFrame],
    dispatcher_id: str,
) -> Tuple[float, int]:
    """Sum benefit deduction amount for one dispatcher (SOCSO, Overpaid, etc.)."""
    return sum_dispatcher_amount_penalty_float(df, dispatcher_id)


def sum_all_dispatcher_amount_penalty(df: Optional[pd.DataFrame]) -> float:
    """Sum all positive Amount values on a hub/socso/overpaid sheet."""
    if df is None or df.empty:
        return 0.0
    if "penalty_numeric" in df.columns:
        filtered = df[df["penalty_numeric"] > 0]
        if filtered.empty:
            return 0.0
        rounded_penalties = [
            penalty.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
            for penalty in filtered["penalty_numeric"].tolist()
        ]
        return float(sum(rounded_penalties))

    amount_col = find_amount_column(df)
    if amount_col is None:
        return 0.0
    work = df.copy()
    work["penalty_numeric"] = work[amount_col].apply(penalty_cell_to_decimal)
    filtered = work[work["penalty_numeric"] > 0]
    if filtered.empty:
        return 0.0
    rounded_penalties = [
        penalty.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        for penalty in filtered["penalty_numeric"].tolist()
    ]
    return float(sum(rounded_penalties))


def find_penalty_amount_column(df: pd.DataFrame) -> Optional[str]:
    col = find_column(df, ["Penalty", "penalty", "PENALTY", "PENALTY BINDING RM5"])
    if col is not None:
        return col
    for c in df.columns:
        cl = str(c).strip().lower()
        if "penalty" in cl and "sum of" not in cl:
            return c
    return None


def extract_waybill_list(series: pd.Series) -> List[str]:
    if series is None or series.empty:
        return []
    seen = set()
    result: List[str] = []
    for value in series.dropna():
        wb = normalize_waybill(value)
        if wb and wb not in seen:
            seen.add(wb)
            result.append(wb)
    return result


def format_waybills_display(waybills: List[str], limit: int = 5) -> str:
    if not waybills:
        return ""
    shown = waybills[:limit]
    text = ", ".join(shown)
    if len(waybills) > limit:
        text += f" (+{len(waybills) - limit} more)"
    return text


def penalty_cell_to_decimal(value) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, bool):
        return Decimal(int(value))
    if isinstance(value, Decimal):
        if value.is_nan() or value.is_infinite():
            return Decimal("0")
        return value
    try:
        if pd.isna(value):
            return Decimal("0")
    except TypeError:
        pass
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return Decimal("0")
        return Decimal(str(v))
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "-", "n/a", "na", "--", ""):
        return Decimal("0")
    for sym in ("RM", "MYR", "S$", "SGD", "USD", "$", "€", "£"):
        s = s.replace(sym, "")
    s = s.replace(",", "").strip()
    if not s:
        return Decimal("0")
    try:
        return Decimal(s)
    except InvalidOperation:
        return Decimal("0")


def penalty_cell_to_float(value) -> float:
    return float(penalty_cell_to_decimal(value))


def sanitize_no_outbound_scan_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    disp_col = find_penalty_dispatcher_column(work)
    awb_col = find_penalty_waybill_column(work)
    if disp_col is not None:
        disp_keys = work[disp_col].apply(clean_penalty_dispatcher_id)
        work = work[disp_keys.astype(str).str.len() > 0]
    if awb_col is not None and not work.empty:
        awb_norm = work[awb_col].apply(normalize_waybill)
        work = work[awb_norm.astype(str).str.len() > 0]
    return work


def dedupe_no_outbound_scan_by_awb(records: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty:
        return pd.DataFrame()
    awb_col = find_penalty_waybill_column(source_df) or find_penalty_waybill_column(records)
    if awb_col is None:
        return pd.DataFrame()
    work = records.copy()
    work["_awb_norm"] = work[awb_col].apply(normalize_waybill)
    work = work[work["_awb_norm"].astype(str).str.len() > 0]
    if work.empty:
        return work
    return work.drop_duplicates(subset=["_awb_norm"], keep="first")


def filter_penalty_sheet_by_date(
    sheet_df: Optional[pd.DataFrame],
    start_date,
    end_date,
    date_column_names: Sequence[str],
) -> Optional[pd.DataFrame]:
    if sheet_df is None or sheet_df.empty:
        return sheet_df
    date_col = find_column(sheet_df, list(date_column_names))
    if date_col is None:
        return sheet_df
    work = sheet_df.copy()
    parsed = pd.to_datetime(work[date_col], errors="coerce")
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return work[(parsed >= start_ts) & (parsed <= end_ts)].copy()


def filter_penalty_data_by_date(
    penalty_data: Dict[str, pd.DataFrame],
    start_date,
    end_date,
) -> Dict[str, pd.DataFrame]:
    filtered: Dict[str, pd.DataFrame] = {}
    for key, df in penalty_data.items():
        if df is None or df.empty:
            continue
        cols = PENALTY_DATE_COLUMNS.get(key, ["date", "created_at"])
        out = filter_penalty_sheet_by_date(df, start_date, end_date, cols)
        if key == "no_outbound_scan" and out is not None and not out.empty:
            out = sanitize_no_outbound_scan_df(out)
        if out is not None and not out.empty:
            filtered[key] = out
    return filtered


def collect_dispatcher_ids_from_penalty_sheets(
    penalty_sheets: Dict[str, Optional[pd.DataFrame]],
) -> List[str]:
    ids = set()
    for key, df in penalty_sheets.items():
        if df is None or df.empty:
            continue
        if key == "ldr":
            col = find_employee_id_column(df)
        elif key == "duitnow":
            col = find_rider_column(df)
        else:
            col = find_penalty_dispatcher_column(df)
        if col is None:
            continue
        for value in df[col].dropna():
            cleaned = clean_penalty_dispatcher_id(value)
            if cleaned:
                ids.add(cleaned)
    return sorted(ids)


def filter_rows_for_penalty_dispatcher(
    df: pd.DataFrame,
    dispatcher_col: str,
    dispatcher_id: str,
) -> pd.DataFrame:
    target = clean_penalty_dispatcher_id(dispatcher_id)
    if not target or dispatcher_col not in df.columns:
        return pd.DataFrame()
    keys = df[dispatcher_col].apply(clean_penalty_dispatcher_id)
    return df[keys == target]


def find_bulky_date_column(df: pd.DataFrame) -> Optional[str]:
    """Find the delivery date column on the Bulky sheet."""
    if df is None or df.empty:
        return None
    date_col = find_column(
        df,
        ["Delivery Signature", "delivery_signature", "delivery_sig", "Created At", "created_at"],
    )
    if date_col is not None:
        return date_col
    for col_name in df.columns:
        col_lower = str(col_name).lower()
        if "delivery" in col_lower and "signature" in col_lower:
            return col_name
        if col_lower in ("created_at", "created at", "date"):
            return col_name
    return None


def filter_bulky_for_dispatcher(bulky_df: pd.DataFrame, dispatcher_id: str) -> pd.DataFrame:
    """Return bulky sheet rows for one dispatcher."""
    if bulky_df is None or bulky_df.empty or not dispatcher_id:
        return pd.DataFrame()
    dispatcher_col = find_penalty_dispatcher_column(bulky_df)
    if dispatcher_col is None:
        return pd.DataFrame()
    return filter_rows_for_penalty_dispatcher(bulky_df, dispatcher_col, dispatcher_id)


def count_bulky_parcels_for_dispatcher(bulky_df: pd.DataFrame, dispatcher_id: str) -> int:
    """Count bulky sheet rows for one dispatcher (matches management.py)."""
    matched = filter_bulky_for_dispatcher(bulky_df, dispatcher_id)
    return int(len(matched))


def bulky_only_records_for_dispatcher(
    bulky_df: pd.DataFrame,
    dispatcher_id: str,
    dispatch_waybills: set,
) -> pd.DataFrame:
    """Return bulky sheet rows not already present on the Dispatch sheet."""
    matched = filter_bulky_for_dispatcher(bulky_df, dispatcher_id)
    if matched.empty:
        return pd.DataFrame()

    waybill_col = _find_bulky_waybill_column(matched)
    if not waybill_col:
        return pd.DataFrame()

    skip = dispatch_waybills or set()
    work = matched.copy()
    work["_wb_key"] = work[waybill_col].apply(normalize_waybill)
    work = work[(work["_wb_key"] != "") & (~work["_wb_key"].isin(skip))]
    return work.drop(columns=["_wb_key"], errors="ignore")


def _find_bulky_waybill_column(df: pd.DataFrame) -> Optional[str]:
    waybill_col = find_penalty_waybill_column(df)
    if waybill_col:
        return waybill_col
    for col in df.columns:
        cl = str(col).lower()
        if "waybill" in cl or "awb" in cl:
            return col
    return None


def bulky_waybills_for_dispatcher(bulky_df: pd.DataFrame, dispatcher_id: str) -> set:
    """Normalized waybill set from the Bulky sheet for one dispatcher."""
    matched = filter_bulky_for_dispatcher(bulky_df, dispatcher_id)
    if matched.empty:
        return set()
    waybill_col = find_penalty_waybill_column(matched)
    if not waybill_col:
        return set()
    return {
        wb for wb in matched[waybill_col].apply(normalize_waybill)
        if wb
    }


def route_penalty_dispatcher_key(dispatcher_id) -> str:
    if dispatcher_id is None:
        return ""
    try:
        if pd.isna(dispatcher_id):
            return ""
    except TypeError:
        pass
    s = str(dispatcher_id).strip()
    if not s or s.lower() == "nan":
        return ""
    try:
        v = float(s.replace(",", ""))
        if v == int(v):
            return str(int(v))
    except (ValueError, OverflowError):
        pass
    return s
