"""
Utility functions for data processing and formatting.
"""
from typing import Optional
import pandas as pd

# Centralized column mappings
COLUMN_MAPPINGS = {
    'waybill': ['Waybill Number', 'Waybill', 'waybill_number', 'waybill'],
    'date': ['Delivery Signature', 'Delivery Date', 'Date', 'signature_date', 'delivery_date'],
    'dispatcher_id': ['Dispatcher ID', 'Dispatcher Id', 'dispatcher_id'],
    'dispatcher_name': ['Dispatcher Name', 'Dispatcher', 'dispatcher_name', 'Name', 'Rider Name'],
    'weight': ['Billing Weight', 'Weight', 'weight', 'billing_weight']
}

DISPATCHER_PREFIXES = ['JMR', 'ECP', 'AF', 'PEN', 'KUL', 'JHR']


def find_column(df: pd.DataFrame, column_type: str) -> Optional[str]:
    """
    Find first matching column name from mappings.

    Args:
        df: DataFrame to search
        column_type: Type of column to find ('waybill', 'date', 'dispatcher_id', etc.)

    Returns:
        Column name if found, None otherwise
    """
    for col in COLUMN_MAPPINGS.get(column_type, []):
        if col in df.columns:
            return col
    return None


def clean_dispatcher_name(name: str) -> str:
    """
    Remove prefixes from dispatcher names.

    Args:
        name: Dispatcher name to clean

    Returns:
        Cleaned dispatcher name
    """
    cleaned = str(name).strip()
    for prefix in DISPATCHER_PREFIXES:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].lstrip(' -')
    return cleaned


def normalize_weight(series: pd.Series) -> pd.Series:
    """
    Normalize weight column to numeric.

    Handles:
    - Comma as decimal separator
    - Non-numeric characters
    - Missing values

    Args:
        series: Series containing weight data

    Returns:
        Series with normalized numeric weights
    """
    return pd.to_numeric(
        series.astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace(r'[^0-9\.-]', '', regex=True),
        errors='coerce'
    )
