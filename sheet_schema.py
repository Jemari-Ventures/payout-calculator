"""Canonical Google Sheet headers — rename once at load, then use direct column access."""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

# Internal column names used everywhere after standardize_sheet().
SHEET_COLUMNS: Dict[str, List[str]] = {
    "dispatch": [
        "waybill_number", "delivery_signature", "dispatcher_id", "dispatcher_name", "billing_weight",
    ],
    "bulky": [
        "waybill_number", "delivery_signature", "dispatcher_id", "dispatcher_name", "billing_weight",
    ],
    "pickup": [
        "waybill_number", "date_pick_up", "pickup_dispatcher_id", "pickup_dispatcher_name",
        "billing_weight", "order_source",
    ],
    "return": [
        "waybill_number", "sender_name", "delivery_signature", "dispatcher_id",
    ],
    "duitnow": [
        "dispatcher_id", "dispatcher_name", "cod_quantity", "duitnow_quantity", "target", "penalty",
    ],
    "ldr": [
        "waybill_number", "penalty", "penalty_amount", "dispatcher_id",
        "declaration_time", "generation_time",
    ],
    "cod": [
        "date", "dispatcher_id", "dispatcher_name", "total_cod",
        "receivable_amount", "uncollected_amount", "penalty_amount", "remark",
    ],
    "binding": ["dispatcher_id", "dispatcher_name", "penalty"],
    "hub": ["dispatcher_id", "dispatcher_name", "penalty"],
    "pending_parcel": ["waybill_number", "penalty", "dispatcher_name", "dispatcher_id", "reason"],
    "parcel_lost": ["waybill_number", "amount", "dispatcher_id"],
    "reward": ["dispatcher_id", "amount"],
    "attendance": ["dispatcher_id", "penalty"],
    "socso": ["dispatcher_id", "dispatcher_name", "amount"],
    "rental": ["dispatcher_id", "amount"],
    "fake_attempt": ["waybill_number", "dispatcher_id", "date"],
    "no_outbound_scan": ["waybill_number", "dispatcher_id", "scanning_time_last", "date"],
    "overpaid": ["dispatcher_id", "amount"],
    "qr_order": ["waybill_number", "dispatcher_id", "date"],
}

# Map normalized header text → canonical internal name.
_HEADER_MAP: Dict[str, str] = {
    "waybill_number": "waybill_number",
    "waybill number": "waybill_number",
    "delivery_signature": "delivery_signature",
    "delivery signature": "delivery_signature",
    "dispatcher_id": "dispatcher_id",
    "dispatcher name": "dispatcher_name",
    "dispatcher_name": "dispatcher_name",
    "billing_weight": "billing_weight",
    "billing weight": "billing_weight",
    "date_pick_up": "date_pick_up",
    "date pick up": "date_pick_up",
    "pickup_dispatcher_id": "pickup_dispatcher_id",
    "pickup dispatcher id": "pickup_dispatcher_id",
    "pickup_dispatcher_name": "pickup_dispatcher_name",
    "pickup dispatcher name": "pickup_dispatcher_name",
    "order_source": "order_source",
    "order source": "order_source",
    "sender name": "sender_name",
    "sender_name": "sender_name",
    "cod_quantity": "cod_quantity",
    "duitnow_quantity": "duitnow_quantity",
    "target": "target",
    "penalty": "penalty",
    "penalty_amount": "penalty_amount",
    "penalty amount": "penalty_amount",
    "declaration_time": "declaration_time",
    "generation_time": "generation_time",
    "date": "date",
    "total_cod": "total_cod",
    "receivable amount": "receivable_amount",
    "receivable_amount": "receivable_amount",
    "uncollected_amount": "uncollected_amount",
    "remark": "remark",
    "amount": "amount",
    "reason": "reason",
    "scanning time | last": "scanning_time_last",
    "scanning_time_last": "scanning_time_last",
    # Legacy optional columns (ignored if absent)
    "achieve": "achieve",
}


def _header_key(raw: str) -> str:
    return str(raw).replace("\ufeff", "").strip().lower()


def canonical_header(raw: str) -> str:
    """Map a sheet header cell to the internal column name."""
    key = _header_key(raw)
    if key in _HEADER_MAP:
        return _HEADER_MAP[key]
    return key.replace(" ", "_").replace("|", "").replace("/", "_").strip("_")


def standardize_sheet(df: Optional[pd.DataFrame], sheet_key: str) -> pd.DataFrame:
    """Rename headers to canonical names and keep only columns used by this sheet."""
    if df is None or df.empty:
        return pd.DataFrame()

    rename = {col: canonical_header(col) for col in df.columns}
    out = df.rename(columns=rename)

    wanted = SHEET_COLUMNS.get(sheet_key)
    if wanted:
        keep = [col for col in wanted if col in out.columns]
        if keep:
            return out[keep].copy()
    return out.copy()


def sheet_col(df: Optional[pd.DataFrame], name: str) -> Optional[str]:
    """Return *name* if present after standardization (no legacy alias scan)."""
    if df is None or df.empty:
        return None
    return name if name in df.columns else None
