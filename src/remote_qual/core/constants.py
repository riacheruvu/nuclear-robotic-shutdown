"""
Default physical and task constants.

These are *defaults*; scenarios may override them. Every number below has a
documented role so non-specialists can follow the model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DefaultConstants:
    """Shared defaults for the remote-valve qualification demo."""

    # ── Task geometry ─────────────────────────────────────────────────────
    # Valve is placed at the origin for a simple, inspectable geometry.
    valve_x: float = 0.0
    valve_y: float = 0.0
    # Success: base of the robot within this radius of the valve (metres).
    # ASSUMPTION: navigation-only task — no arm / handwheel model yet.
    task_radius_m: float = 0.15
    # Soft "corridor" half-width (metres) used as a tracking safety bound.
    # ASSUMPTION: empty open plane; no walls or obstacles in v1.
    corridor_radius_m: float = 0.5

    # ── Timing ────────────────────────────────────────────────────────────
    dt: float = 0.1  # integration step (seconds)
    horizon: int = 100  # max steps (10 s at dt=0.1)

    # ── Controller defaults ───────────────────────────────────────────────
    k_p: float = 2.5  # proportional heading gain
    v_nominal: float = 0.5  # nominal forward speed (m/s)

    # ── Radiation model (simplified research model) ───────────────────────
    # Co-60 specific gamma-ray constant Γ ≈ 1.32 R·m²/(h·Ci) is a standard
    # order-of-magnitude value used in introductory health-physics tables
    # for unshielded point-source estimates (not a full transport solution).
    gamma_const: float = 1.32  # R·m²/(h·Ci)
    activity_ci: float = 15.0  # source activity (Ci)
    # Very small linear attenuation in air — almost free-space falloff.
    mu_air: float = 1.0e-4  # 1/m
    # Ambient term in the *same converted units path* as the source term.
    # See units.py and ASSUMPTIONS.md for the research simplification.
    d_bg: float = 0.05
    # Mission dose cap used as a failure threshold (model units ≈ mSv).
    d_max_msv: float = 50.0

    # ── Sensor noise ──────────────────────────────────────────────────────
    # Bernoulli "salt-and-pepper" full-scramble of (x,y) observations.
    p_scramble: float = 0.05
    sigma_slip: float = 0.05  # position slip std-dev per step (m)
    sigma_obs: float = 0.02  # obs noise used in reachability bound (rad-ish)


DEFAULTS = DefaultConstants()
