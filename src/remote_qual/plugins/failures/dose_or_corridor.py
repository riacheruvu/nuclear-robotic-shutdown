"""
Failure = cumulative dose too high OR tracking corridor breached.

Plain English
-------------
A rollout "fails" if either:

1. Total accumulated dose exceeds D_max, or
2. At some time, the robot's (x, y) is farther than R_safe from the
   entire nominal path (nearest-point distance).

ASSUMPTIONS
-----------
1. Corridor is defined relative to the *noise-free nominal path*, not
   free-space geometry (empty plane).
2. Collision with walls is not modelled in v1.
3. Task failure (never reaching the valve) is tracked separately as
   mission success, not as this safety failure bit.
"""

from __future__ import annotations

import numpy as np
import torch

from remote_qual.plugins.base import FailurePlugin


class DoseOrCorridorFailure(FailurePlugin):
    def __call__(
        self,
        trajs: torch.Tensor,
        doses: torch.Tensor,
        nominal_path: np.ndarray,
        *,
        corridor_radius: float,
        d_max: float,
        device: str,
    ) -> torch.Tensor:
        nom = torch.tensor(nominal_path[:, :2], dtype=torch.float32, device=device)
        dose_fails = doses > d_max

        # trajs: (T+1, B, 7)
        diffs = trajs[:, :, :2].unsqueeze(2) - nom.unsqueeze(0).unsqueeze(0)
        dists = torch.norm(diffs, dim=3)
        min_dists, _ = torch.min(dists, dim=2)
        max_dev, _ = torch.max(min_dists, dim=0)
        corridor_fails = max_dev > corridor_radius
        return dose_fails | corridor_fails
