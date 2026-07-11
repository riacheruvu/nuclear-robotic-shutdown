"""Evaluate pass/fail verdicts against scenario thresholds."""

from __future__ import annotations

from typing import Any, Dict, Optional


def evaluate_thresholds(
    *,
    mission_success_rate: float,
    mission_success_ci95_halfwidth: float,
    p_fail: Optional[float],
    max_dose_observed: Optional[float],
    min_mission_success: float,
    max_p_fail: Optional[float],
    max_dose_msv: float,
) -> Dict[str, Any]:
    """Return structured verdicts with plain-English notes."""
    lower = mission_success_rate - mission_success_ci95_halfwidth
    # Conservative rule: require the lower end of the 95% CI to clear the bar.
    liveness = "pass" if lower >= min_mission_success else "fail"

    pfail_verdict = "n/a"
    if max_p_fail is not None and p_fail is not None:
        pfail_verdict = "pass" if p_fail <= max_p_fail else "fail"

    dose_verdict = "n/a"
    if max_dose_observed is not None:
        # Informational: whether any sampled trajectory exceeded D_max
        # (failure definition already uses D_max; this flags stress).
        dose_verdict = "pass" if max_dose_observed <= max_dose_msv else "watch"

    overall = "pass"
    if liveness == "fail" or pfail_verdict == "fail":
        overall = "fail"

    return {
        "min_mission_success": min_mission_success,
        "max_p_fail": max_p_fail,
        "max_dose_msv": max_dose_msv,
        "mission_success_ci95_lower": lower,
        "verdict": {
            "mission_liveness": liveness,
            "p_fail": pfail_verdict,
            "dose_samples": dose_verdict,
            "overall": overall,
        },
        "notes": {
            "mission_liveness": (
                "PASS if the lower end of the 95% CI on mission success "
                f"is ≥ {min_mission_success:.0%} (conservative CI rule)."
            ),
            "p_fail": (
                "Compared to max_p_fail only if that threshold is set in the scenario."
            ),
        },
    }
