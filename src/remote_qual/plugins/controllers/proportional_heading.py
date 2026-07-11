"""
Proportional heading controller toward the valve (origin by default).

Plain English
-------------
The robot always tries to point toward the valve and drive forward.
If it is pointing the wrong way, it turns harder and slows down a bit.

Control law
-----------
    desired_heading = atan2(-y, -x)   # toward origin / valve
    e = wrap(desired_heading - θ)
    ω = K_p · e
    v = V_nom · (1 - 0.5 · |e| / π)

ASSUMPTIONS
-----------
1. Full state pose is available to the controller except when the noise
   plugin corrupts the observation.
2. No path planner — pure reactive homing.
3. No collision avoidance (empty plane in v1).
"""

from __future__ import annotations

import numpy as np

from remote_qual.plugins.base import ControllerPlugin


class ProportionalHeading(ControllerPlugin):
    def __init__(
        self,
        k_p: float = 2.5,
        v_nominal: float = 0.5,
        target_xy: tuple[float, float] = (0.0, 0.0),
    ):
        self.k_p = float(k_p)
        self.v_nominal = float(v_nominal)
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))

    def __call__(self, s_obs: np.ndarray) -> np.ndarray:
        x, y, theta = s_obs[:3]
        tx, ty = self.target_xy
        desired = np.arctan2(ty - y, tx - x)
        err = desired - theta
        err = (err + np.pi) % (2 * np.pi) - np.pi
        omega = self.k_p * err
        v = self.v_nominal * (1.0 - 0.5 * abs(err) / np.pi)
        return np.array([v, omega], dtype=float)
