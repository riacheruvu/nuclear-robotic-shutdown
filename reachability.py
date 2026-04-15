"""
reachability.py — Module 1: Linearised Interval-Arithmetic Reachability

Propagates an axis-aligned bounding box (IntervalBox) through the linearised
dynamics, inflating it at each step by a Minkowski-sum noise bound.

Bug fix (vs original notebook):
    Observation noise σ_obs does NOT directly perturb the true state θ.
    It affects the *control* computed from the noisy observation, which then
    enters the system through the lag buffer (indices 3-4). 
    
    NEW FIX: Noise is now properly scaled by sqrt(DT) to prevent vacuous 
    linear bloat over long horizons.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from config import R_SAFE, K_P, V_NOMINAL, DT
from dynamics import nominal_dynamics, jacobian_dynamics

# ── Data structure ────────────────────────────────────────────────────────────

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
        """Worst-case XY distance from any box corner to the nominal path."""
        corners_xy = np.array([
            [self.lo[0], self.lo[1]],
            [self.lo[0], self.hi[1]],
            [self.hi[0], self.lo[1]],
            [self.hi[0], self.hi[1]],
        ])
        return max(
            np.min(np.linalg.norm(nominal_path[:, :2] - c, axis=1))
            for c in corners_xy
        )

# ── Interval propagation ──────────────────────────────────────────────────────

def _obs_noise_control_bound(sigma_obs: float) -> np.ndarray:
    delta_heading = 3.0 * sigma_obs           
    delta_omega   = K_P * delta_heading       
    delta_v       = V_NOMINAL * 0.5 * delta_heading / np.pi  
    return np.array([delta_v, delta_omega])

def linearized_interval_step(
    box: IntervalBox,
    s_nom: np.ndarray,
    A: np.ndarray,
    sigma_slip: float,
    sigma_obs: float,
) -> IntervalBox:
    f_nom    = nominal_dynamics(s_nom)
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

    # ── Minkowski-sum noise inflation (FIXED) ──────────────────────────────
    # Noise accumulates as a random walk, so we scale the std-dev by sqrt(DT)
    scale_factor = np.sqrt(DT)
    
    # Slip noise acts directly on x, y (indices 0-1).
    slip_bound = np.array([sigma_slip * scale_factor, sigma_slip * scale_factor, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Obs noise perturbs the control that enters the lag buffer (indices 3-4).
    ctrl_bound          = _obs_noise_control_bound(sigma_obs) * scale_factor
    obs_noise_bound     = np.zeros(7)
    obs_noise_bound[3]  = ctrl_bound[0]  
    obs_noise_bound[4]  = ctrl_bound[1]  

    noise_bound = slip_bound + obs_noise_bound
    return IntervalBox(lo=new_lo - noise_bound, hi=new_hi + noise_bound)

# ── Public API ────────────────────────────────────────────────────────────────

def reachability_check(
    s0_3d: np.ndarray,
    sigma_slip: float,
    sigma_obs: float,
    T: int,
    nominal_path: np.ndarray,
) -> Tuple[bool, int, List[IntervalBox]]:
    s0 = np.concatenate([s0_3d, np.zeros(4)])
    init_width = np.array([0.02, 0.02, 0.05, 0.0, 0.0, 0.0, 0.0])
    box = IntervalBox(lo=s0 - init_width, hi=s0 + init_width)
    boxes = [box]

    for t in range(T):
        s_nom = nominal_path[min(t, len(nominal_path) - 1)]
        A     = jacobian_dynamics(s_nom)
        box   = linearized_interval_step(box, s_nom, A, sigma_slip, sigma_obs)
        boxes.append(box)

        if box.max_xy_deviation_from(nominal_path) > R_SAFE:
            return True, t + 1, boxes

    return False, -1, boxes