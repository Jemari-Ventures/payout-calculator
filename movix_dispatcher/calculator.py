"""Pure Movix IC payout engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .config import MovixConfig
from .eligibility import EligibilityResult, evaluate_eligibility
from .rates import apply_commission_rates


@dataclass
class PayoutResult:
    driver_id: str
    driver_name: str
    period_start: Optional[str]
    period_end: Optional[str]
    total_parcels: int
    working_days: int
    avg_parcels_per_day: float
    commission_total: float
    basic_salary: float
    attendance_allowance: float
    attendance_deduction: float
    absent_penalty: float
    gross_payout: float
    net_payout: float
    qualifies_for_basic: bool
    qualifies_for_attendance: bool
    eligibility_reason: str
    parcels_by_band: Dict[str, int] = field(default_factory=dict)
    commission_by_band: Dict[str, float] = field(default_factory=dict)
    daily_summary: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    parcel_detail: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def to_summary_row(self) -> Dict:
        row = asdict(self)
        row.pop("daily_summary", None)
        row.pop("parcel_detail", None)
        row.pop("parcels_by_band", None)
        row.pop("commission_by_band", None)
        # Flatten band counts for CSV export
        for band, count in self.parcels_by_band.items():
            row[f"parcels::{band}"] = count
        for band, amount in self.commission_by_band.items():
            row[f"commission::{band}"] = amount
        return row


class MovixPayoutCalculator:
    def __init__(self, config: Optional[MovixConfig] = None):
        self.config = config or MovixConfig.load()

    def calculate_driver(
        self,
        df: pd.DataFrame,
        driver_id: str,
        mc_days: int = 0,
        absent_days: int = 0,
    ) -> PayoutResult:
        driver_id = str(driver_id).strip()
        scoped = df[df["driver_id"].astype(str) == driver_id].copy()
        return self._calculate_frame(scoped, driver_id=driver_id, mc_days=mc_days, absent_days=absent_days)

    def calculate_all(
        self,
        df: pd.DataFrame,
        adjustments: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> List[PayoutResult]:
        """Batch payout for every driver in the normalized frame.

        adjustments: optional {driver_id: {"mc_days": n, "absent_days": n}}
        """
        adjustments = adjustments or {}
        results: List[PayoutResult] = []
        if df.empty:
            return results
        for driver_id in sorted(df["driver_id"].astype(str).unique()):
            adj = adjustments.get(driver_id, {}) or adjustments.get(str(driver_id), {})
            results.append(
                self.calculate_driver(
                    df,
                    driver_id,
                    mc_days=int(adj.get("mc_days", 0) or 0),
                    absent_days=int(adj.get("absent_days", 0) or 0),
                )
            )
        return results

    def _calculate_frame(
        self,
        scoped: pd.DataFrame,
        driver_id: str,
        mc_days: int = 0,
        absent_days: int = 0,
    ) -> PayoutResult:
        cfg = self.config
        if scoped.empty:
            return PayoutResult(
                driver_id=driver_id,
                driver_name="",
                period_start=None,
                period_end=None,
                total_parcels=0,
                working_days=0,
                avg_parcels_per_day=0.0,
                commission_total=0.0,
                basic_salary=0.0,
                attendance_allowance=0.0,
                attendance_deduction=0.0,
                absent_penalty=round(max(0, absent_days) * cfg.absent_penalty_per_day, 2),
                gross_payout=0.0,
                net_payout=round(-max(0, absent_days) * cfg.absent_penalty_per_day, 2),
                qualifies_for_basic=False,
                qualifies_for_attendance=False,
                eligibility_reason="No billable parcels for this driver in the selected period.",
            )

        priced = apply_commission_rates(scoped, cfg)
        driver_name = ""
        for value in priced["driver_name"]:
            text = str(value).strip()
            if text and text.lower() != "nan":
                driver_name = text
                break

        total_parcels = int(len(priced))
        working_days = int(priced["service_date"].nunique())
        eligibility: EligibilityResult = evaluate_eligibility(
            working_days=working_days,
            total_parcels=total_parcels,
            config=cfg,
            mc_days=mc_days,
            absent_days=absent_days,
        )

        commission_total = round(float(priced["commission_amount"].sum()), 2)
        band_counts = priced.groupby("weight_band").size().to_dict()
        band_amounts = (
            priced.groupby("weight_band")["commission_amount"].sum().round(2).to_dict()
        )

        daily = (
            priced.groupby("service_date", as_index=False)
            .agg(
                parcels=("shipment_id", "count"),
                commission=("commission_amount", "sum"),
            )
            .sort_values("service_date")
        )
        daily["commission"] = daily["commission"].round(2)
        daily["service_date"] = daily["service_date"].dt.strftime("%Y-%m-%d")

        gross = round(
            commission_total + eligibility.basic_amount + eligibility.attendance_amount,
            2,
        )
        net = round(gross - eligibility.absent_penalty, 2)

        period_start = priced["service_date"].min()
        period_end = priced["service_date"].max()

        return PayoutResult(
            driver_id=driver_id,
            driver_name=driver_name,
            period_start=period_start.strftime("%Y-%m-%d") if pd.notna(period_start) else None,
            period_end=period_end.strftime("%Y-%m-%d") if pd.notna(period_end) else None,
            total_parcels=total_parcels,
            working_days=eligibility.working_days,
            avg_parcels_per_day=eligibility.avg_parcels_per_day,
            commission_total=commission_total,
            basic_salary=eligibility.basic_amount,
            attendance_allowance=eligibility.attendance_amount,
            attendance_deduction=eligibility.attendance_deduction,
            absent_penalty=eligibility.absent_penalty,
            gross_payout=gross,
            net_payout=net,
            qualifies_for_basic=eligibility.qualifies_for_basic,
            qualifies_for_attendance=eligibility.qualifies_for_attendance,
            eligibility_reason=eligibility.reason,
            parcels_by_band={str(k): int(v) for k, v in band_counts.items()},
            commission_by_band={str(k): float(v) for k, v in band_amounts.items()},
            daily_summary=daily,
            parcel_detail=priced,
        )


def results_to_summary_frame(results: List[PayoutResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([r.to_summary_row() for r in results])
