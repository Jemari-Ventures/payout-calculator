"""Return sheet sender_name allowlist used by hub_filter and Streamlit apps."""
from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Set

import pandas as pd

DEFAULT_RETURN_SENDER_NAMES: tuple[str, ...] = (
    "ACE STORY AQUATIC",
    "BG POLO CS",
    "DANNY",
    "Envie",
    "ENVIE FASHION",
    "Guang Hong",
    "HIALIA",
    "Jersiku",
    "MYKUTSU Signature",
    "MYKUTSU TIK TOK",
    "MYSAPATOS",
    "POLO",
    "SUKMA",
    "SUPER YY",
    "Syauqi",
    "YY",
    "YY HOME",
)

SENDER_NAME_COLUMN_ALIASES: tuple[str, ...] = (
    "sender_name",
    "Sender Name",
    "sender name",
    "Sender",
)


def normalize_sender_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def build_sender_name_allowlist(sender_names: Sequence[str]) -> Set[str]:
    return {
        normalize_sender_name(name)
        for name in sender_names
        if normalize_sender_name(name)
    }


def find_sender_name_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    columns_by_lower = {str(c).strip().lower(): c for c in df.columns}
    for alias in SENDER_NAME_COLUMN_ALIASES:
        key = alias.strip().lower()
        if key in columns_by_lower:
            return columns_by_lower[key]
    for col in df.columns:
        if "sender" in str(col).strip().lower():
            return col
    return None


def resolve_return_sender_names(
    configured: Optional[Iterable[str] | bool] = None,
) -> Optional[tuple[str, ...]]:
    """Return allowlist to apply, or None when filtering is disabled.

    - False → disable filter
    - None / True → default merchant allowlist
    - sequence of names → custom allowlist
    """
    if configured is False:
        return None
    if configured is None or configured is True:
        return DEFAULT_RETURN_SENDER_NAMES
    names = tuple(str(n).strip() for n in configured if str(n).strip())
    return names or None


def filter_return_by_sender_name(
    df: Optional[pd.DataFrame],
    sender_names: Optional[Iterable[str] | bool] = None,
) -> pd.DataFrame:
    """Keep only return rows whose sender_name is in the allowlist."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    allowlist_names = resolve_return_sender_names(sender_names)
    if not allowlist_names:
        return df.copy()

    sender_col = find_sender_name_column(df)
    if not sender_col:
        return df.iloc[0:0].copy()

    allowed = build_sender_name_allowlist(allowlist_names)
    mask = df[sender_col].apply(lambda v: normalize_sender_name(v) in allowed)
    return df.loc[mask].copy()
