"""
Simplified Co-60 point-source dose field + valve geometry.

See ``remote_qual.core.units`` for unit conventions and ASSUMPTIONS.
"""

from __future__ import annotations

import numpy as np
import torch

from remote_qual.core.units import point_source_dose_rate_per_second
from remote_qual.plugins.base import HazardPlugin


class PointSourceCo60(HazardPlugin):
    def __init__(
        self,
        activity_ci: float = 15.0,
        gamma_const: float = 1.32,
        mu_air: float = 1.0e-4,
        d_bg: float = 0.05,
        valve_pos=(0.0, 0.0),
        source_pos=None,
    ):
        self.activity_ci = float(activity_ci)
        self.gamma_const = float(gamma_const)
        self.mu_air = float(mu_air)
        self.d_bg = float(d_bg)
        self.valve_pos = np.asarray(valve_pos, dtype=float).reshape(2)
        # Default: hot source colocated with valve (stressful approach).
        self.source_pos = (
            np.asarray(source_pos, dtype=float).reshape(2)
            if source_pos is not None
            else self.valve_pos.copy()
        )

    def distance_to_valve(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.linalg.norm(xy - self.valve_pos, axis=-1)

    def distance_to_source(self, xy):
        xy = np.asarray(xy, dtype=float)
        return np.linalg.norm(xy - self.source_pos, axis=-1)

    def dose_rate(self, xy):
        r = self.distance_to_source(xy)
        return point_source_dose_rate_per_second(
            r,
            gamma_const=self.gamma_const,
            activity_ci=self.activity_ci,
            mu_air=self.mu_air,
            d_bg=self.d_bg,
        )

    def task_reached(self, xy, task_radius: float):
        return self.distance_to_valve(xy) < task_radius

    def dose_rate_torch(self, xy: torch.Tensor) -> torch.Tensor:
        """Batched torch version matching the research model exactly."""
        src = torch.tensor(self.source_pos, dtype=xy.dtype, device=xy.device)
        dists = torch.clamp(torch.norm(xy - src, dim=-1), min=0.1)
        hourly = (
            (self.gamma_const * self.activity_ci / dists**2)
            * torch.exp(-self.mu_air * dists)
            + self.d_bg
        )
        return hourly / 3600.0
