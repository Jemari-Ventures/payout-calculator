"""Basic salary and attendance eligibility for Movix IC couriers."""

from __future__ import annotations

from dataclasses import dataclass

from .config import MovixConfig


@dataclass(frozen=True)
class EligibilityResult:
    qualifies_for_basic: bool
    qualifies_for_attendance: bool
    working_days: int
    avg_parcels_per_day: float
    basic_amount: float
    attendance_amount: float
    attendance_deduction: float
    absent_penalty: float
    reason: str


def _meets_volume_gate(
    working_days: int,
    avg_parcels_per_day: float,
    required_days: int,
    min_avg: float,
) -> bool:
    return working_days >= required_days and avg_parcels_per_day >= min_avg


def evaluate_eligibility(
    working_days: int,
    total_parcels: int,
    config: MovixConfig,
    mc_days: int = 0,
    absent_days: int = 0,
) -> EligibilityResult:
    """
    Poster rules:
    - Basic RM1,700 requires min 26 working days AND avg >= 50 parcels/day.
    - Couriers failing the 50/day average receive commission only (no basic, no allowance).
    - Attendance RM200 uses the same volume/days gate.
    - Valid MC/death: deduct RM100/day from allowance; 2+ days => no allowance.
    - Unauthorized absent: RM30/day penalty.
    """
    working_days = max(0, int(working_days))
    total_parcels = max(0, int(total_parcels))
    mc_days = max(0, int(mc_days))
    absent_days = max(0, int(absent_days))

    avg = (total_parcels / working_days) if working_days > 0 else 0.0

    qualifies_volume = _meets_volume_gate(
        working_days,
        avg,
        config.basic_required_working_days,
        config.basic_min_avg_parcels_per_day,
    )
    qualifies_for_basic = qualifies_volume

    attendance_raw = 0.0
    qualifies_for_attendance = False
    attendance_deduction = 0.0
    reasons = []

    if not qualifies_volume:
        reasons.append(
            f"Commission only: need ≥{config.basic_required_working_days} working days "
            f"and avg ≥{config.basic_min_avg_parcels_per_day:.0f} parcels/day "
            f"(got {working_days} days, avg {avg:.2f})."
        )
    else:
        reasons.append("Meets basic salary gate (26 days + 50 parcels/day avg).")
        # Attendance gate mirrors basic volume requirements from the poster.
        attendance_volume_ok = _meets_volume_gate(
            working_days,
            avg,
            config.attendance_required_working_days,
            config.attendance_min_avg_parcels_per_day,
        )
        if not attendance_volume_ok:
            reasons.append("Attendance allowance not earned (volume/days gate).")
        elif mc_days >= config.mc_days_disqualify_at:
            reasons.append(
                f"Attendance disqualified: MC/compassionate leave ≥"
                f"{config.mc_days_disqualify_at} days."
            )
        else:
            qualifies_for_attendance = True
            attendance_raw = float(config.attendance_amount)
            if mc_days > 0:
                attendance_deduction = min(
                    attendance_raw,
                    mc_days * float(config.mc_deduction_per_day),
                )
                reasons.append(
                    f"Attendance RM{attendance_raw:.0f} minus "
                    f"RM{attendance_deduction:.0f} MC deduction ({mc_days} day(s))."
                )
            else:
                reasons.append(f"Attendance allowance RM{attendance_raw:.0f} earned.")

    attendance_net = max(0.0, attendance_raw - attendance_deduction) if qualifies_for_attendance else 0.0
    if not qualifies_for_attendance:
        attendance_deduction = 0.0

    absent_penalty = round(absent_days * float(config.absent_penalty_per_day), 2)
    if absent_days > 0:
        reasons.append(
            f"Absent penalty RM{absent_penalty:.2f} ({absent_days} day(s) × "
            f"RM{config.absent_penalty_per_day:.0f})."
        )

    basic_amount = float(config.basic_amount) if qualifies_for_basic else 0.0

    return EligibilityResult(
        qualifies_for_basic=qualifies_for_basic,
        qualifies_for_attendance=qualifies_for_attendance and attendance_net > 0,
        working_days=working_days,
        avg_parcels_per_day=round(avg, 2),
        basic_amount=basic_amount,
        attendance_amount=round(attendance_net, 2),
        attendance_deduction=round(attendance_deduction, 2),
        absent_penalty=absent_penalty,
        reason=" ".join(reasons),
    )
