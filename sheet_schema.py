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
        "waybill_number", "penalty", "dispatcher_id",
        "declaration_time", "generation_time",
    ],
    "cod": [
        "date", "dispatcher_id", "dispatcher_name", "total_cod",
        "receivable_amount", "uncollected_amount", "penalty", "remark",
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
    "penalty_amount": "penalty",
    "penalty amount": "penalty",
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


def _penalty_source_priority(raw_col: str) -> int:
    """Prefer the literal penalty header when legacy penalty_amount also exists."""
    key = _header_key(raw_col)
    if key == "penalty":
        return 0
    if key in ("penalty_amount", "penalty amount"):
        return 1
    return 2


def _coalesce_sheet_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merge source columns that canonicalize to the same name (e.g. penalty + penalty_amount)."""
    canon_groups: Dict[str, List[int]] = {}
    for i, col in enumerate(df.columns):
        canon = canonical_header(col)
        canon_groups.setdefault(canon, []).append(i)

    if all(len(indices) == 1 for indices in canon_groups.values()):
        rename = {df.columns[i]: canon for canon, indices in canon_groups.items() for i in indices}
        return df.rename(columns=rename)

    out = pd.DataFrame(index=df.index)
    for canon, indices in canon_groups.items():
        if len(indices) == 1:
            out[canon] = df.iloc[:, indices[0]]
            continue

        ordered = sorted(
            indices,
            key=lambda i: (
                _penalty_source_priority(df.columns[i]) if canon == "penalty" else 0,
                i,
            ),
        )
        combined = df.iloc[:, ordered[0]]
        for idx in ordered[1:]:
            fallback = df.iloc[:, idx]
            empty = combined.isna()
            if combined.dtype == object:
                stripped = combined.astype(str).str.strip()
                empty = empty | stripped.eq("") | stripped.str.lower().eq("nan")
            combined = combined.where(~empty, fallback)
        out[canon] = combined
    return out


def standardize_sheet(df: Optional[pd.DataFrame], sheet_key: str) -> pd.DataFrame:
    """Rename headers to canonical names and keep only columns used by this sheet."""
    if df is None or df.empty:
        return pd.DataFrame()

    out = _coalesce_sheet_columns(df)

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
