"""Shared hub payout data contract — filter output must match Template JMR + app schema."""
from __future__ import annotations

from typing import Dict, List, Tuple

# Excel / Google Sheet tab name → (sheet_schema key, output columns in order)
# Columns are the snake_case headers Streamlit expects after upload.
OUTPUT_SHEETS: Dict[str, Tuple[str, List[str]]] = {
    "Dispatch": (
        "dispatch",
        [
            "waybill_number",
            "delivery_signature",
            "dispatcher_id",
            "dispatcher_name",
            "billing_weight",
        ],
    ),
    "Bulky": (
        "bulky",
        [
            "waybill_number",
            "delivery_signature",
            "dispatcher_id",
            "dispatcher_name",
            "billing_weight",
        ],
    ),
    "Pickup": (
        "pickup",
        [
            "waybill_number",
            "date_pick_up",
            "pickup_dispatcher_id",
            "pickup_dispatcher_name",
            "billing_weight",
            "order_source",
        ],
    ),
    "Return": (
        "return",
        [
            "waybill_number",
            "sender_name",
            "delivery_signature",
            "dispatcher_id",
        ],
    ),
    "DuitNow": (
        "duitnow",
        [
            "dispatcher_id",
            "dispatcher_name",
            "cod_quantity",
            "duitnow_quantity",
            "target",
            "penalty",
        ],
    ),
    "LDR": (
        "ldr",
        [
            "waybill_number",
            "penalty",
            "dispatcher_id",
            "pushed_time",
        ],
    ),
    "COD": (
        "cod",
        [
            "date",
            "dispatcher_id",
            "dispatcher_name",
            "total_cod",
            "receivable_amount",
            "uncollected_amount",
            "penalty",
            "remark",
        ],
    ),
    "Binding": (
        "binding",
        ["dispatcher_id", "dispatcher_name", "penalty"],
    ),
    "Hub": (
        "hub",
        ["dispatcher_id", "dispatcher_name", "penalty"],
    ),
    "Pending Parcel": (
        "pending_parcel",
        [
            "waybill_number",
            "penalty",
            "dispatcher_name",
            "dispatcher_id",
            "reason",
        ],
    ),
    "Parcel Lost": (
        "parcel_lost",
        ["waybill_number", "amount", "dispatcher_id"],
    ),
    "Reward": (
        "reward",
        ["dispatcher_id", "amount"],
    ),
    "Attendance": (
        "attendance",
        ["dispatcher_id", "penalty"],
    ),
    "Socso": (
        "socso",
        ["dispatcher_id", "dispatcher_name", "amount"],
    ),
    "Rental": (
        "rental",
        ["dispatcher_id", "amount"],
    ),
    "Fake Attempt": (
        "fake_attempt",
        ["date", "waybill_number", "dispatcher_id", "dispatcher_name"],
    ),
    "No Outbound Scan": (
        "no_outbound_scan",
        ["waybill_number", "dispatcher_id"],
    ),
    "Overpaid": (
        "overpaid",
        ["dispatcher_id", "dispatcher_name", "amount"],
    ),
}

# JMS / Title Case aliases → canonical snake_case (first match wins when building output).
COLUMN_ALIASES: Dict[str, List[str]] = {
    "waybill_number": [
        "waybill_number",
        "AWB No.",
        "AWB No",
        "waybill number",
        "Waybill Number",
        "WAYBILL NO",
        "Ticket No",
        "no_awb",
    ],
    "delivery_signature": [
        "delivery_signature",
        "Delivery Signature",
        "delivery signature",
    ],
    "dispatcher_id": [
        "dispatcher_id",
        "Dispatcher ID",
        "Dispatcher Id",
        "Employee ID",
        "ID STAFF",
        "ID",
    ],
    "dispatcher_name": [
        "dispatcher_name",
        "Dispatcher Name",
        "dispatcher",
        "Dispatcher",
        "Rider",
    ],
    "billing_weight": [
        "billing_weight",
        "Billing Weight",
        "billing weight",
    ],
    "date_pick_up": [
        "date_pick_up",
        "Date | Pick Up",
        "Date Pick Up",
        "date pick up",
    ],
    "pickup_dispatcher_id": [
        "pickup_dispatcher_id",
        "Pick Up Dispatcher ID",
        "Pickup Dispatcher ID",
    ],
    "pickup_dispatcher_name": [
        "pickup_dispatcher_name",
        "Pick Up Dispatcher Name",
        "Pickup Dispatcher Name",
    ],
    "order_source": [
        "order_source",
        "Order Source",
        "order source",
    ],
    "sender_name": [
        "sender_name",
        "Sender Name",
        "sender name",
    ],
    "cod_quantity": [
        "cod_quantity",
        "Volume | Delivery Signature",
        "cod quantity",
    ],
    "duitnow_quantity": [
        "duitnow_quantity",
        "DuitNow Usage",
        "duitnow_usage",
    ],
    "target": [
        "target",
        "Target",
        "Target 23%",
        "target_23",
    ],
    "penalty": [
        "penalty",
        "Penalty",
        "PENALTY",
        "PENALTY (RM)",
        "PENALTY BINDING RM5",
    ],
    "pushed_time": [
        "pushed_time",
        "pushed time",
        "Pushed Time",
    ],
    "date": ["date", "Date", "DATE"],
    "total_cod": ["total_cod", "Total COD", "total COD"],
    "receivable_amount": [
        "receivable_amount",
        "receivable amount",
        "Receivable Amount",
    ],
    "uncollected_amount": [
        "uncollected_amount",
        "uncollected amount",
        "Uncollected Amount",
    ],
    "remark": ["remark", "Remark"],
    "amount": ["amount", "Amount", "AMOUNT"],
    "reason": ["reason", "Reason", "Problematic Reason"],
}


def contract_columns(sheet_tab: str) -> List[str]:
    entry = OUTPUT_SHEETS.get(sheet_tab)
    if entry is None:
        raise KeyError(f"Unknown output sheet tab: {sheet_tab}")
    return list(entry[1])


def schema_key(sheet_tab: str) -> str:
    entry = OUTPUT_SHEETS.get(sheet_tab)
    if entry is None:
        raise KeyError(f"Unknown output sheet tab: {sheet_tab}")
    return entry[0]
