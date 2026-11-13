"""Calculator and ingest tests, including optional live IC Payment.xlsx check."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from movix_dispatcher.calculator import MovixPayoutCalculator, results_to_summary_frame
from movix_dispatcher.config import MovixConfig
from movix_dispatcher.ingest import filter_billable, normalize_frame

SAMPLE_XLSX = Path("/Users/UMARI.ZULKIFLI/Documents/IC Payment.xlsx")


def _sample_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Service Date": "2026-10-21",
                "Service Type": "Del - FWD D2D",
                "LM Type": "D2D",
                "Shipment ID": "A1",
                "Chargeable Weight": 0.5,
                "Driver ID": 1001,
                "Driver Name": "[1001] Test Driver",
                "TIER": "T1",
            },
            {
                "Service Date": "2026-10-21",
                "Service Type": "Del - FWD BSC",
                "LM Type": "BSC",
                "Shipment ID": "A2",
                "Chargeable Weight": 7.0,
                "Driver ID": 1001,
                "Driver Name": "[1001] Test Driver",
                "TIER": "T2",
            },
            {
                "Service Date": "2026-10-21",
                "Service Type": "Failed Deliveries",
                "LM Type": "D2D",
                "Shipment ID": "A3",
                "Chargeable Weight": 1.0,
                "Driver ID": 1001,
                "Driver Name": "[1001] Test Driver",
                "TIER": "T1",
            },
            {
                "Service Date": "2026-10-22",
                "Service Type": "Del - FWD D2D",
                "LM Type": "D2D",
                "Shipment ID": "A4",
                "Chargeable Weight": 25.0,
                "Driver ID": 1001,
                "Driver Name": "[1001] Test Driver",
                "TIER": "T3",
            },
            {
                "Service Date": "2026-10-21",
                "Service Type": "Del - FWD D2D",
                "LM Type": "D2D",
                "Shipment ID": "B1",
                "Chargeable Weight": 1.0,
                "Driver ID": 2002,
                "Driver Name": "[2002] Other",
                "TIER": "T1",
            },
        ]
    )


def test_normalize_filters_failed_deliveries():
    cfg = MovixConfig.load()
    df = normalize_frame(_sample_rows(), cfg)
    assert set(df["shipment_id"]) == {"A1", "A2", "A4", "B1"}
    assert "A3" not in set(df["shipment_id"])


def test_driver_commission_math():
    cfg = MovixConfig.load()
    df = normalize_frame(_sample_rows(), cfg)
    calc = MovixPayoutCalculator(cfg)
    result = calc.calculate_driver(df, "1001")
    # A1: 1.00 + A2: 1.80 + A4: 3.00 = 5.80
    assert result.total_parcels == 3
    assert result.working_days == 2
    assert result.commission_total == 5.80
    assert result.basic_salary == 0.0  # only 2 days
    assert result.net_payout == 5.80
    assert result.parcels_by_band["0 – 5.00 kg"] == 1
    assert result.parcels_by_band["5.01 – 20.00 kg"] == 1
    assert result.parcels_by_band["> 20.00 kg"] == 1


def test_batch_summary():
    cfg = MovixConfig.load()
    df = normalize_frame(_sample_rows(), cfg)
    calc = MovixPayoutCalculator(cfg)
    results = calc.calculate_all(df)
    summary = results_to_summary_frame(results)
    assert len(summary) == 2
    assert set(summary["driver_id"]) == {"1001", "2002"}


def test_filter_billable_prefixes():
    cfg = MovixConfig.load()
    raw = normalize_frame(_sample_rows(), cfg)
    # already filtered; ensure helper is idempotent
    again = filter_billable(raw, cfg)
    assert len(again) == len(raw)


@pytest.mark.skipif(not SAMPLE_XLSX.exists(), reason="IC Payment.xlsx not available")
def test_live_ic_payment_sheet1():
    from movix_dispatcher.ingest import load_sheet1

    cfg = MovixConfig.load()
    df = load_sheet1(SAMPLE_XLSX, cfg)
    assert not df.empty
    assert {"service_date", "shipment_id", "chargeable_weight", "driver_id"} <= set(df.columns)
    # Failed / RTS / PU rows excluded
    assert df["service_type"].str.startswith("Del - FWD").all()

    calc = MovixPayoutCalculator(cfg)
    results = calc.calculate_all(df)
    assert len(results) >= 1
    summary = results_to_summary_frame(results)
    # Partial-week sample cannot meet 26-day basic gate
    assert (summary["basic_salary"] == 0).all()
    assert float(summary["commission_total"].sum()) > 0
