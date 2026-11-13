"""Unit tests for Movix basic salary and attendance eligibility."""

from __future__ import annotations

from movix_dispatcher.config import MovixConfig
from movix_dispatcher.eligibility import evaluate_eligibility


def test_commission_only_when_below_volume_gate():
    cfg = MovixConfig.load()
    # 5 working days, 250 parcels => avg 50 but days < 26
    result = evaluate_eligibility(working_days=5, total_parcels=250, config=cfg)
    assert result.qualifies_for_basic is False
    assert result.basic_amount == 0.0
    assert result.attendance_amount == 0.0
    assert "Commission only" in result.reason


def test_commission_only_when_avg_below_50():
    cfg = MovixConfig.load()
    # 26 days, 1000 parcels => avg ~38.5
    result = evaluate_eligibility(working_days=26, total_parcels=1000, config=cfg)
    assert result.qualifies_for_basic is False
    assert result.basic_amount == 0.0
    assert result.attendance_amount == 0.0


def test_full_basic_and_attendance():
    cfg = MovixConfig.load()
    # 26 days × 50 = 1300 parcels
    result = evaluate_eligibility(working_days=26, total_parcels=1300, config=cfg)
    assert result.qualifies_for_basic is True
    assert result.basic_amount == 1700.0
    assert result.qualifies_for_attendance is True
    assert result.attendance_amount == 200.0
    assert result.absent_penalty == 0.0


def test_mc_one_day_deducts_100():
    cfg = MovixConfig.load()
    result = evaluate_eligibility(
        working_days=26, total_parcels=1300, config=cfg, mc_days=1
    )
    assert result.attendance_amount == 100.0
    assert result.attendance_deduction == 100.0
    assert result.qualifies_for_attendance is True


def test_mc_two_days_disqualifies_attendance():
    cfg = MovixConfig.load()
    result = evaluate_eligibility(
        working_days=26, total_parcels=1300, config=cfg, mc_days=2
    )
    assert result.attendance_amount == 0.0
    assert result.qualifies_for_attendance is False
    assert result.basic_amount == 1700.0  # basic still paid


def test_absent_penalty():
    cfg = MovixConfig.load()
    result = evaluate_eligibility(
        working_days=26, total_parcels=1300, config=cfg, absent_days=2
    )
    assert result.absent_penalty == 60.0
