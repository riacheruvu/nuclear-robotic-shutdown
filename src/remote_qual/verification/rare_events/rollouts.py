"""GPU/CPU batched stochastic rollouts under lag dynamics + noise + dose."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from remote_qual.plugins.base import ControllerPlugin, DynamicsPlugin, HazardPlugin, NoisePlugin


def batched_rollouts(
    s0_3d: np.ndarray,
    *,
    dynamics: DynamicsPlugin,
    controller: ControllerPlugin,
    noise: NoisePlugin,
    hazard: HazardPlugin,
    dt: float,
    horizon: int,
    task_radius: float,
    n_rollouts: int,
    device: str | None = None,
    bias_factor: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate B trajectories in parallel.

    Returns
    -------
    trajs : (T+1, B, 7)
    cum_doses : (B,)
    slips : (T, B, 2)
    scrambles : (T, B)
    done_mask : (B,) bool — reached task region at some point
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    B = n_rollouts
    s0 = dynamics.augment_initial(np.asarray(s0_3d, dtype=float))
    s = torch.tensor(s0, dtype=torch.float32, device=device).repeat(B, 1)

    trajs = torch.zeros((horizon + 1, B, 7), device=device)
    trajs[0] = s
    cum_doses = torch.zeros(B, device=device)
    slips = noise.sample_slips(horizon, B, device, bias_factor=bias_factor)
    # For scrambles we need the probability consistent with bias_factor
    p = noise.p_scramble if bias_factor == 1.0 else noise.biased_scramble_prob(bias_factor)  # type: ignore[attr-defined]
    scrambles = (torch.rand(horizon, B, device=device) < p).float()
    done_mask = torch.zeros(B, dtype=torch.bool, device=device)

    # Controller params for batched torch reimplementation of proportional law
    # (keeps GPU path fast; must match ProportionalHeading for default plugin)
    k_p = float(getattr(controller, "k_p", 2.5))
    v_nom = float(getattr(controller, "v_nominal", 0.5))
    tx, ty = getattr(controller, "target_xy", (0.0, 0.0))

    scramble_range = float(getattr(noise, "scramble_xy_range", 2.0))

    for t in range(horizon):
        scramble_mask = scrambles[t]
        s_obs = s.clone()
        noise_xy = (torch.rand(B, 2, device=device) * (2 * scramble_range)) - scramble_range
        s_obs[:, :2] += scramble_mask.unsqueeze(1) * noise_xy

        x, y, theta = s_obs[:, 0], s_obs[:, 1], s_obs[:, 2]
        desired = torch.atan2(ty - y, tx - x)
        err = desired - theta
        err = (err + torch.pi) % (2 * torch.pi) - torch.pi
        omega = k_p * err
        v = v_nom * (1.0 - 0.5 * torch.abs(err) / torch.pi)
        u_new = torch.stack([v, omega], dim=1)

        s_next = s.clone()
        v_lag, omega_lag = s[:, 5], s[:, 6]
        s_next[:, 0] += v_lag * torch.cos(s[:, 2]) * dt
        s_next[:, 1] += v_lag * torch.sin(s[:, 2]) * dt
        s_next[:, 2] += omega_lag * dt
        s_next[:, :2] += slips[t]
        s_next[:, 5:7] = s[:, 3:5]
        s_next[:, 3:5] = u_new

        s_next = torch.where(done_mask.unsqueeze(1), s, s_next)

        dose_rate = hazard.dose_rate_torch(s_next[:, :2])  # type: ignore[attr-defined]
        cum_doses += dose_rate * dt * (~done_mask).float()

        # task success relative to valve
        valve = torch.tensor(
            getattr(hazard, "valve_pos", np.zeros(2)),
            dtype=torch.float32,
            device=device,
        )
        dists_valve = torch.norm(s_next[:, :2] - valve, dim=1)
        done_mask = done_mask | (dists_valve < task_radius)

        s = s_next
        trajs[t + 1] = s

    return trajs, cum_doses, slips, scrambles, done_mask
