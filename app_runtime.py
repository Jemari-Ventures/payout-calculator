"""Runtime helpers for app.py — memory trimming, caching keys, and sheet prep."""
from __future__ import annotations

import json
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

from sheet_schema import standardize_sheet

PAYOUT_CONFIG_KEYS = (
    "currency_symbol",
    "tiers",
    "kpi_incentives",
    "special_rates",
    "attendance_incentive",
    "advance_payout",
    "designated_driver",
    "fake_attempt_penalty_per_parcel",
    "pending_parcel_penalty_per_parcel",
    "no_outbound_scan_penalty_per_parcel",
    "route_penalty_amount",
    "route_penalty_app_enabled",
    "pickup_payout_per_parcel",
    "return_payout_per_parcel",
    "bulky_rates",
)


def trim_sheet_columns(df: Optional[pd.DataFrame], sheet_key: str) -> pd.DataFrame:
    """Rename to canonical headers and keep only columns needed for payout."""
    return standardize_sheet(df, sheet_key)


def payout_config_fingerprint(config: dict) -> str:
    """Stable hash input for payout result caching."""
    payload = {key: config.get(key) for key in PAYOUT_CONFIG_KEYS}
    return json.dumps(payload, sort_keys=True, default=str)


def iso_date(value: date) -> str:
    return value.isoformat()


def build_dispatcher_mapping(
    dispatch_df: pd.DataFrame,
    dispatcher_id_col: Optional[str],
    dispatcher_name_col: Optional[str],
    penalty_sheets: Dict[str, Optional[pd.DataFrame]],
    reward_df: Optional[pd.DataFrame],
    bulky_df: Optional[pd.DataFrame],
    clean_id,
    clean_name,
    collect_penalty_ids,
    find_reward_employee_column,
    find_reward_dispatcher_name_column,
    find_dispatch_id_column,
    find_column_fn,
    pickup_df: Optional[pd.DataFrame] = None,
    find_pickup_dispatcher_column_fn=None,
    find_pickup_dispatcher_name_column_fn=None,
) -> Dict[str, str]:
    """Build dispatcher_id -> name map without row-wise Python loops on large frames."""
    mapping: Dict[str, str] = {}

    if (
        dispatch_df is not None
        and not dispatch_df.empty
        and dispatcher_id_col
        and dispatcher_id_col in dispatch_df.columns
    ):
        work = dispatch_df[[dispatcher_id_col] + ([dispatcher_name_col] if dispatcher_name_col else [])].copy()
        work["_id"] = work[dispatcher_id_col].map(clean_id)
        work = work[work["_id"].astype(bool)]
        if dispatcher_name_col and dispatcher_name_col in work.columns:
            work["_name"] = work[dispatcher_name_col].astype(str).map(clean_name)
            for disp_id, name in work.drop_duplicates("_id")[["_id", "_name"]].itertuples(index=False):
                if disp_id and disp_id not in mapping:
                    mapping[str(disp_id)] = str(name or "")
        else:
            for disp_id in work["_id"].drop_duplicates():
                if disp_id and disp_id not in mapping:
                    mapping[str(disp_id)] = ""

    if (
        pickup_df is not None
        and not pickup_df.empty
        and find_pickup_dispatcher_column_fn is not None
    ):
        pickup_id_col = find_pickup_dispatcher_column_fn(pickup_df)
        pickup_name_col = (
            find_pickup_dispatcher_name_column_fn(pickup_df)
            if find_pickup_dispatcher_name_column_fn is not None
            else None
        )
        if pickup_id_col:
            cols = [pickup_id_col] + ([pickup_name_col] if pickup_name_col else [])
            pickup_work = pickup_df[cols].dropna(subset=[pickup_id_col]).copy()
            pickup_work["_id"] = pickup_work[pickup_id_col].map(clean_id)
            pickup_work = pickup_work[pickup_work["_id"].astype(bool)]
            if pickup_name_col and pickup_name_col in pickup_work.columns:
                pickup_work["_name"] = pickup_work[pickup_name_col].astype(str).map(clean_name)
                for disp_id, name in pickup_work.drop_duplicates("_id")[["_id", "_name"]].itertuples(index=False):
                    key = str(disp_id)
                    if not key:
                        continue
                    nm = str(name or "").strip()
                    if nm and not str(mapping.get(key, "")).strip():
                        mapping[key] = nm
                    elif key not in mapping:
                        mapping[key] = nm
            else:
                for disp_id in pickup_work["_id"].drop_duplicates():
                    mapping.setdefault(str(disp_id), mapping.get(str(disp_id), ""))

    for penalty_id in collect_penalty_ids(penalty_sheets):
        mapping.setdefault(penalty_id, "")

    if reward_df is not None and not reward_df.empty:
        reward_id_col = find_reward_employee_column(reward_df)
        reward_name_col = find_reward_dispatcher_name_column(reward_df)
        if reward_id_col:
            cols = [reward_id_col] + ([reward_name_col] if reward_name_col else [])
            reward_work = reward_df[cols].dropna(subset=[reward_id_col]).copy()
            reward_work["_id"] = reward_work[reward_id_col].map(clean_id)
            reward_work = reward_work[reward_work["_id"].astype(bool)]
            if reward_name_col and reward_name_col in reward_work.columns:
                reward_work["_name"] = reward_work[reward_name_col].astype(str).map(clean_name)
                for disp_id, name in reward_work.drop_duplicates("_id")[["_id", "_name"]].itertuples(index=False):
                    mapping.setdefault(str(disp_id), str(name or ""))
            else:
                for disp_id in reward_work["_id"].drop_duplicates():
                    mapping.setdefault(str(disp_id), "")

    if bulky_df is not None and not bulky_df.empty:
        bulky_id_col = find_dispatch_id_column(bulky_df)
        bulky_name_col = find_column_fn(
            bulky_df, ["dispatcher_name", "Dispatcher Name", "Rider Name", "rider_name", "Name", "name"]
        )
        if bulky_id_col:
            cols = [bulky_id_col] + ([bulky_name_col] if bulky_name_col else [])
            bulky_work = bulky_df[cols].dropna(subset=[bulky_id_col]).copy()
            bulky_work["_id"] = bulky_work[bulky_id_col].map(clean_id)
            bulky_work = bulky_work[bulky_work["_id"].astype(bool)]
            if bulky_name_col and bulky_name_col in bulky_work.columns:
                bulky_work["_name"] = bulky_work[bulky_name_col].astype(str).map(clean_name)
                for disp_id, name in bulky_work.drop_duplicates("_id")[["_id", "_name"]].itertuples(index=False):
                    key = str(disp_id)
                    if not key:
                        continue
                    nm = str(name or "").strip()
                    if nm and not str(mapping.get(key, "")).strip():
                        mapping[key] = nm
                    elif key not in mapping:
                        mapping[key] = nm
            else:
                for disp_id in bulky_work["_id"].drop_duplicates():
                    mapping.setdefault(str(disp_id), "")

    return mapping
