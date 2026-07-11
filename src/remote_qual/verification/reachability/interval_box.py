"""
Linearized interval-arithmetic reachability (axis-aligned boxes).

Plain English
-------------
Imagine drawing a rectangle around every place the robot *might* be, given
bounded noise. We push that rectangle forward in time with a linear
approximation of the dynamics, then fatten it a bit for noise. If the
rectangle ever strays too far from the planned path, we flag a formal
"unsafe" warning.

Open-loop vs receding-horizon re-certification
----------------------------------------------
**Open-loop:** grow one box for the whole mission. Axis-aligned intervals
often *explode* (wrapping effect) even when real trajectories are fine —
a classic limitation of naive interval arithmetic in feedback loops.

**Receding horizon (default):** every ``receding_window`` steps, re-center a
fresh small box on the *nominal* state and re-propagate for a short window.
This is a standard runtime-assurance style pattern: "if we are still near the
plan, re-certify the next few seconds." It does **not** replace Hamilton–Jacobi
value functions; it mitigates interval blow-up for practical qualification.

What this does *not* prove
--------------------------
- Still linearized + axis-aligned (conservative / approximate).
- Receding mode assumes re-certification is valid when the true state remains
  inside the previous window's tube near the nominal (operational assumption).
- Not a full HJ PDE solution; not zonotopes/ellipsoids.

ASSUMPTIONS
-----------
1. Noise bounds use a 3σ-style heading error proxy for observation noise.
2. Slip and control-noise bounds scale with √Δt (random-walk style).
3. Safety check: max XY deviation of box corners from the nominal path
   vs corridor radius R_safe.
4. Receding resets use a fixed initial width around the nominal (not a
   measured belief covariance).

Literature context
------------------
Set-based reachability (JuliaReach, CORA); interval blow-up / wrapping effect;
receding-horizon / runtime assurance monitors; HJ reachability as the tighter
but harder gold standard for nonlinear safety envelopes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from remote_qual.plugins.base import ControllerPlugin, DynamicsPlugin


@dataclass
class IntervalBox:
    lo: np.ndarray
    hi: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2.0

    @property
    def width(self) -> np.ndarray:
        return self.hi - self.lo

    def xy_area(self) -> float:
        """Axis-aligned area in the (x, y) plane (m²). Diagnostic for blow-up."""
        w = self.width
        return float(max(w[0], 0.0) * max(w[1], 0.0))

    def max_xy_deviation_from(self, nominal_path: np.ndarray) -> float:
        corners_xy = np.array(
            [
                [self.lo[0], self.lo[1]],
                [self.lo[0], self.hi[1]],
                [self.hi[0], self.lo[1]],
                [self.hi[0], self.hi[1]],
            ]
        )
        return max(
            np.min(np.linalg.norm(nominal_path[:, :2] - c, axis=1))
            for c in corners_xy
        )


@dataclass
class ReachabilityResult:
    """Rich formal-layer outcome for reports and diagnostics."""

    is_unsafe: bool
    first_unsafe_step: int  # -1 if never
    boxes: List[IntervalBox]
    mode: str  # "open_loop" | "receding"
    receding_window: Optional[int] = None
    xy_area_series: List[float] = field(default_factory=list)
    max_xy_area: float = 0.0
    initial_xy_area: float = 0.0
    growth_ratio: float = 1.0
    recert_steps: List[int] = field(default_factory=list)
    blowup_suspected: bool = False
    notes: str = ""

    # Optional open-loop companion when mode is receding and compare_open_loop=True
    open_loop_is_unsafe: Optional[bool] = None
    open_loop_first_unsafe_step: Optional[int] = None
    open_loop_max_xy_area: Optional[float] = None
    open_loop_growth_ratio: Optional[float] = None


def _closed_loop_map(
    s: np.ndarray,
    dynamics: DynamicsPlugin,
    controller: ControllerPlugin,
    dt: float,
) -> np.ndarray:
    u = controller(s)
    return dynamics.step(s, u, dt)


def closed_loop_jacobian(
    s: np.ndarray,
    dynamics: DynamicsPlugin,
    controller: ControllerPlugin,
    dt: float,
    eps: float = 1e-5,
) -> np.ndarray:
    f0 = _closed_loop_map(s, dynamics, controller, dt)
    n = len(s)
    A = np.zeros((n, n))
    for i in range(n):
        s_plus = s.copy()
        s_plus[i] += eps
        A[:, i] = (_closed_loop_map(s_plus, dynamics, controller, dt) - f0) / eps
    return A


def _obs_noise_control_bound(sigma_obs: float, k_p: float, v_nominal: float) -> np.ndarray:
    delta_heading = 3.0 * sigma_obs
    delta_omega = k_p * delta_heading
    delta_v = v_nominal * 0.5 * delta_heading / np.pi
    return np.array([delta_v, delta_omega])


def linearized_interval_step(
    box: IntervalBox,
    s_nom: np.ndarray,
    A: np.ndarray,
    f_nom: np.ndarray,
    sigma_slip: float,
    sigma_obs: float,
    dt: float,
    k_p: float,
    v_nominal: float,
) -> IntervalBox:
    delta_lo = box.lo - s_nom
    delta_hi = box.hi - s_nom
    new_lo = np.zeros(7)
    new_hi = np.zeros(7)
    for i in range(7):
        for j in range(7):
            terms = [A[i, j] * delta_lo[j], A[i, j] * delta_hi[j]]
            new_lo[i] += min(terms)
            new_hi[i] += max(terms)
    new_lo += f_nom
    new_hi += f_nom

    scale = np.sqrt(dt)
    slip_bound = np.array(
        [sigma_slip * scale, sigma_slip * scale, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    ctrl = _obs_noise_control_bound(sigma_obs, k_p, v_nominal) * scale
    obs_bound = np.zeros(7)
    obs_bound[3] = ctrl[0]
    obs_bound[4] = ctrl[1]
    noise = slip_bound + obs_bound
    return IntervalBox(lo=new_lo - noise, hi=new_hi + noise)


def _seed_box(s: np.ndarray, init_width: np.ndarray) -> IntervalBox:
    return IntervalBox(lo=s - init_width, hi=s + init_width)


def _propagate(
    *,
    s0: np.ndarray,
    dynamics: DynamicsPlugin,
    controller: ControllerPlugin,
    nominal_path: np.ndarray,
    sigma_slip: float,
    sigma_obs: float,
    dt: float,
    horizon: int,
    corridor_radius: float,
    k_p: float,
    v_nominal: float,
    mode: str,
    receding_window: int,
    init_width: np.ndarray,
    blowup_growth_threshold: float,
) -> ReachabilityResult:
    box = _seed_box(s0, init_width)
    boxes: List[IntervalBox] = [box]
    areas: List[float] = [box.xy_area()]
    recert_steps: List[int] = [0]
    first_unsafe = -1
    is_unsafe = False

    steps_in_window = 0
    for t in range(horizon):
        # Receding: re-center on nominal at window boundaries (except t=0 already seeded)
        if mode == "receding" and steps_in_window >= receding_window:
            s_nom_reset = nominal_path[min(t, len(nominal_path) - 1)]
            box = _seed_box(s_nom_reset, init_width)
            recert_steps.append(t)
            steps_in_window = 0

        s_nom = nominal_path[min(t, len(nominal_path) - 1)]
        A = closed_loop_jacobian(s_nom, dynamics, controller, dt)
        f_nom = _closed_loop_map(s_nom, dynamics, controller, dt)
        box = linearized_interval_step(
            box,
            s_nom,
            A,
            f_nom,
            sigma_slip,
            sigma_obs,
            dt,
            k_p,
            v_nominal,
        )
        boxes.append(box)
        area = box.xy_area()
        areas.append(area)
        steps_in_window += 1

        if (not is_unsafe) and box.max_xy_deviation_from(nominal_path) > corridor_radius:
            is_unsafe = True
            first_unsafe = t + 1

    initial = max(areas[0], 1e-12)
    max_area = float(max(areas)) if areas else 0.0
    # Peak growth within series (not just final — receding final may be small)
    growth = max_area / initial
    blowup = growth >= blowup_growth_threshold

    note_parts = []
    if mode == "open_loop":
        note_parts.append(
            "Open-loop interval tube over full horizon (prone to wrapping / blow-up)."
        )
    else:
        note_parts.append(
            f"Receding-horizon re-certification every {receding_window} steps "
            f"({len(recert_steps)} seeds including t=0)."
        )
    if blowup:
        note_parts.append(
            f"XY box area grew ≥{blowup_growth_threshold:.0f}× peak vs initial — "
            "interval method may be vacuous; prefer receding mode or tighter set reps."
        )
    if is_unsafe:
        note_parts.append(
            f"Tube left corridor (R_safe) at step {first_unsafe} under stated noise bounds."
        )
    else:
        note_parts.append("No corridor exit detected for this mode under stated bounds.")

    return ReachabilityResult(
        is_unsafe=is_unsafe,
        first_unsafe_step=first_unsafe,
        boxes=boxes,
        mode=mode,
        receding_window=receding_window if mode == "receding" else None,
        xy_area_series=areas,
        max_xy_area=max_area,
        initial_xy_area=float(areas[0]) if areas else 0.0,
        growth_ratio=float(growth),
        recert_steps=recert_steps if mode == "receding" else [],
        blowup_suspected=blowup,
        notes=" ".join(note_parts),
    )


def reachability_check(
    s0_3d: np.ndarray,
    *,
    dynamics: DynamicsPlugin,
    controller: ControllerPlugin,
    nominal_path: np.ndarray,
    sigma_slip: float,
    sigma_obs: float,
    dt: float,
    horizon: int,
    corridor_radius: float,
    k_p: float,
    v_nominal: float,
    mode: str = "receding",
    receding_window: int = 20,
    compare_open_loop: bool = True,
    blowup_growth_threshold: float = 50.0,
    init_width: Optional[np.ndarray] = None,
) -> ReachabilityResult:
    """Run interval reachability; default is receding-horizon re-certification.

    Parameters
    ----------
    mode:
        ``"receding"`` (default), ``"open_loop"``, or ``"both"`` (receding primary
        + open-loop companion diagnostics).
    receding_window:
        Steps between re-seeds on the nominal trajectory.
    compare_open_loop:
        If mode is ``receding``, also run open-loop for contrast in the report.
    blowup_growth_threshold:
        Peak XY area / initial area above which we flag suspected interval blow-up.
    """
    mode = (mode or "receding").lower().strip()
    if mode not in ("receding", "open_loop", "both"):
        raise ValueError(f"Unknown reachability mode: {mode!r}")

    s0 = dynamics.augment_initial(np.asarray(s0_3d, dtype=float))
    if init_width is None:
        init_width = np.array([0.02, 0.02, 0.05, 0.0, 0.0, 0.0, 0.0])
    else:
        init_width = np.asarray(init_width, dtype=float)

    primary_mode = "open_loop" if mode == "open_loop" else "receding"
    result = _propagate(
        s0=s0,
        dynamics=dynamics,
        controller=controller,
        nominal_path=nominal_path,
        sigma_slip=sigma_slip,
        sigma_obs=sigma_obs,
        dt=dt,
        horizon=horizon,
        corridor_radius=corridor_radius,
        k_p=k_p,
        v_nominal=v_nominal,
        mode=primary_mode,
        receding_window=max(1, int(receding_window)),
        init_width=init_width,
        blowup_growth_threshold=blowup_growth_threshold,
    )

    want_ol = mode == "both" or (mode == "receding" and compare_open_loop)
    if want_ol and primary_mode != "open_loop":
        ol = _propagate(
            s0=s0,
            dynamics=dynamics,
            controller=controller,
            nominal_path=nominal_path,
            sigma_slip=sigma_slip,
            sigma_obs=sigma_obs,
            dt=dt,
            horizon=horizon,
            corridor_radius=corridor_radius,
            k_p=k_p,
            v_nominal=v_nominal,
            mode="open_loop",
            receding_window=receding_window,
            init_width=init_width,
            blowup_growth_threshold=blowup_growth_threshold,
        )
        result.open_loop_is_unsafe = ol.is_unsafe
        result.open_loop_first_unsafe_step = ol.first_unsafe_step
        result.open_loop_max_xy_area = ol.max_xy_area
        result.open_loop_growth_ratio = ol.growth_ratio
        result.notes += (
            f" Open-loop companion: unsafe={ol.is_unsafe}, "
            f"max_xy_area={ol.max_xy_area:.4g} m², growth={ol.growth_ratio:.1f}×."
        )
        if ol.is_unsafe and not result.is_unsafe:
            result.notes += (
                " Receding holds while open-loop does not — classic interval blow-up "
                "pattern; prefer receding or tighter sets for long horizons."
            )

    return result
