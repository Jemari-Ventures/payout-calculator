"""Weight → commission rate mapping for Movix IC parcels."""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .config import MovixConfig, WeightBand


def resolve_weight_band(weight_kg: float, config: MovixConfig) -> Optional[WeightBand]:
    """Pick the first matching band. Weights are inclusive on both ends per poster."""
    if pd.isna(weight_kg):
        return None
    weight = float(weight_kg)
    # Guard negative / nonsense weights
    if weight < 0:
        return None
    for band in config.weight_commission:
        if band.contains(weight):
            return band
    # Fallback: if weight sits in the 5.00–5.01 gap, treat as upper of first band
    # Poster uses 0–5.00 then 5.01–20.00; float noise at exactly 5.005 is rare.
    for band in config.weight_commission:
        if band.max_kg is None or weight <= band.max_kg + 0.01:
            return band
    return None


def commission_rate(weight_kg: float, config: MovixConfig) -> Tuple[float, str]:
    band = resolve_weight_band(weight_kg, config)
    if band is None:
        return 0.0, "Unmapped"
    return float(band.rate), band.label or "Weight band"


def apply_commission_rates(df: pd.DataFrame, config: MovixConfig) -> pd.DataFrame:
    """Add commission_rate / weight_band / commission_amount columns."""
    out = df.copy()
    if out.empty:
        out["commission_rate"] = pd.Series(dtype=float)
        out["weight_band"] = pd.Series(dtype=object)
        out["commission_amount"] = pd.Series(dtype=float)
        return out

    rates = []
    labels = []
    for weight in out["chargeable_weight"]:
        rate, label = commission_rate(weight, config)
        rates.append(rate)
        labels.append(label)

    out["commission_rate"] = rates
    out["weight_band"] = labels
    out["commission_amount"] = pd.Series(rates, index=out.index).round(2)
    return out
