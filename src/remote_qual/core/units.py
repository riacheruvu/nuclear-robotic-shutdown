"""
Unit conventions and conversions for the research dose model.

Plain English
-------------
Real radiation protection uses careful units (gray, sievert) and often heavy
Monte-Carlo transport codes (e.g. MCNP). This toolkit uses a *teaching /
research* dose model so students can see how dose risk couples to motion.

What we compute
---------------
At distance r (metres) from a point source of activity A (Ci):

    Ḋ_source(r) = Γ · A / r² · exp(-μ · r)     [≈ R/h for the source term]

We then form a *per-second* rate used in discrete-time integration:

    Ḋ_step = (Ḋ_source(r) + D_bg) / 3600

and accumulate:

    D ← D + Ḋ_step · Δt     for each time step while the mission is active.

ASSUMPTIONS (read these before trusting numbers)
------------------------------------------------
1. **Point isotropic source.** No walls, no self-shielding, no build-up factor.
2. **Inverse-square + simple exponential air attenuation.** Good for intuition;
   not a substitute for plant ALARA tools (VISIPLAN, MCNP, etc.).
3. **Unit simplification.** Γ is historically quoted in roentgen-based units
   (R·m²/(h·Ci)). The framework treats the combined expression as a *relative
   dose-like rate* and labels mission totals as "mSv" for threshold demos.
   **We do not claim SI-perfect conversion from R to mSv.** For regulatory
   work, replace the hazard plugin with a calibrated dose-rate map.
4. **D_bg is a lumped ambient term** included inside the same /3600 conversion
   path as the source term (legacy research model consistency).
5. **Clamp r ≥ 0.1 m** to avoid 1/r² singularities at the origin.

References (methods & context, not identical software)
------------------------------------------------------
- Wright et al., "Simulating Ionising Radiation in Gazebo for Robotic Nuclear
  Inspection Challenges," *Robotics*, 2021 (radiation-aware robot simulation).
- Introductory health-physics point-source / specific gamma-ray constant usage
  (standard Co-60 Γ ≈ 1.32 R·m²/(h·Ci) order of magnitude).
"""

from __future__ import annotations

import numpy as np


SECONDS_PER_HOUR = 3600.0
MIN_RANGE_M = 0.1


def point_source_dose_rate_per_second(
    distance_m: np.ndarray | float,
    *,
    gamma_const: float,
    activity_ci: float,
    mu_air: float,
    d_bg: float,
    min_range_m: float = MIN_RANGE_M,
) -> np.ndarray:
    """Return simplified dose-like rate in model units per second.

    Parameters
    ----------
    distance_m:
        Distance(s) from the source in metres.
    gamma_const:
        Specific gamma-ray constant Γ [R·m²/(h·Ci)] in the research model.
    activity_ci:
        Source activity A [Ci].
    mu_air:
        Linear attenuation coefficient μ [1/m].
    d_bg:
        Lumped ambient term (same pre-conversion path as source term).
    min_range_m:
        Floor on distance to avoid singularities.

    Returns
    -------
    Dose-like rate per second (numpy array), ready to multiply by Δt.
    """
    r = np.clip(np.asarray(distance_m, dtype=float), min_range_m, None)
    hourly = (gamma_const * activity_ci / r**2) * np.exp(-mu_air * r) + d_bg
    return hourly / SECONDS_PER_HOUR


def accumulate_dose(dose_rate_per_s: float, dt: float) -> float:
    """Integrate a constant rate over one step: D = rate * dt."""
    return float(dose_rate_per_s * dt)
