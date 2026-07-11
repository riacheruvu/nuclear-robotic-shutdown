"""
Linearized interval-arithmetic reachability (axis-aligned boxes).

Plain English
-------------
Imagine drawing a rectangle around every place the robot *might* be, given
bounded noise. We push that rectangle forward in time with a linear
approximation of the dynamics, then fatten it a bit for noise. If the
rectangle ever strays too far from the planned path, we flag a formal
"unsafe" warning.

What this does *not* prove
--------------------------
- It is **conservative but approximate**: linearization and axis-aligned boxes
  can over-approximate (false alarms) or, if the linear model is poor, miss
  nonlinear effects. Treat σ* as a *model-based certificate under stated
  bounds*, not absolute truth.
- Observation noise does **not** kick the true pose; it corrupts the control
  that enters the lag buffer (see original project bugfix).

ASSUMPTIONS
-----------
1. Noise bounds use a 3σ-style heading error proxy for observation noise.
2. Slip and control-noise bounds scale with √Δt (random-walk style).
3. Safety check: max XY deviation of box corners from the nominal path
   vs corridor radius R_safe.

Literature context
------------------
Set-based / interval reachability is standard in formal verification of
dynamical systems (see JuliaReach, CORA, and interval-arithmetic surveys).
This module is a lightweight domain-specific implementation for teaching
and rapid scenario qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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
) -> Tuple[bool, int, List[IntervalBox]]:
    """Propagate an IntervalBox; return (is_unsafe, first_step, boxes)."""
    s0 = dynamics.augment_initial(np.asarray(s0_3d, dtype=float))
    init_width = np.array([0.02, 0.02, 0.05, 0.0, 0.0, 0.0, 0.0])
    box = IntervalBox(lo=s0 - init_width, hi=s0 + init_width)
    boxes = [box]

    for t in range(horizon):
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
        if box.max_xy_deviation_from(nominal_path) > corridor_radius:
            return True, t + 1, boxes
    return False, -1, boxes
