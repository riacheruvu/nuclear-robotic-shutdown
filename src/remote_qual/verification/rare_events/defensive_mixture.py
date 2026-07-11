"""
Defensive mixture importance sampling for rare safety failures.

Plain English
-------------
Naive Monte Carlo: run many random missions; count failures. If failures are
rare, you need a huge number of runs.

Importance sampling: deliberately make failures *more common* (bigger slip,
more sensor scrambles), then re-weight each run so the estimate of the true
failure probability stays unbiased.

Defensive mixture (Owen & Zhou style idea / mixture proposals):
  sample from a mix of the true noise law and a heavier-tailed biased law:

      q = α · p_nominal + (1-α) · p_biased

so weights stay stable (the "defensive" part).

ASSUMPTIONS
-----------
1. We know the probability laws of slip (Gaussian) and scramble (Bernoulli).
2. Bias multiplies slip σ and scramble p (capped at 0.9).
3. Joint likelihood includes both channels (critical for correct weights).
4. Mission success rate is estimated on the *nominal* subsample only
   (unbiased under the true noise), while P_fail uses the full mixture.

Literature context
------------------
- Importance sampling for rare events (classic MC literature; e.g. L'Ecuyer).
- Defensive mixture proposals for robust IS weights.
- Domain application: risk-informed robotic remote shutdown (this project).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from remote_qual.plugins.base import (
    ControllerPlugin,
    DynamicsPlugin,
    FailurePlugin,
    HazardPlugin,
    NoisePlugin,
)
from remote_qual.verification.rare_events.rollouts import batched_rollouts


def defensive_mixture_is(
    s0_3d: np.ndarray,
    *,
    dynamics: DynamicsPlugin,
    controller: ControllerPlugin,
    noise: NoisePlugin,
    hazard: HazardPlugin,
    failure: FailurePlugin,
    nominal_path: np.ndarray,
    dt: float,
    horizon: int,
    task_radius: float,
    corridor_radius: float,
    d_max: float,
    n_rollouts: int = 2000,
    alpha: float = 0.7,
    bias_factor: float = 2.2,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_nom = int(n_rollouts * alpha)
    n_bias = n_rollouts - n_nom
    if n_nom < 1 or n_bias < 1:
        raise ValueError("Need alpha in (0,1) with n_rollouts large enough for both mixture parts.")

    trajs_nom, doses_nom, slips_nom, scram_nom, done_nom = batched_rollouts(
        s0_3d,
        dynamics=dynamics,
        controller=controller,
        noise=noise,
        hazard=hazard,
        dt=dt,
        horizon=horizon,
        task_radius=task_radius,
        n_rollouts=n_nom,
        device=device,
        bias_factor=1.0,
    )
    trajs_bias, doses_bias, slips_bias, scram_bias, _ = batched_rollouts(
        s0_3d,
        dynamics=dynamics,
        controller=controller,
        noise=noise,
        hazard=hazard,
        dt=dt,
        horizon=horizon,
        task_radius=task_radius,
        n_rollouts=n_bias,
        device=device,
        bias_factor=bias_factor,
    )

    all_trajs = torch.cat([trajs_nom, trajs_bias], dim=1)
    all_doses = torch.cat([doses_nom, doses_bias], dim=0)
    all_slips = torch.cat([slips_nom, slips_bias], dim=1)
    all_scram = torch.cat([scram_nom, scram_bias], dim=1)

    failures = failure(
        all_trajs,
        all_doses,
        nominal_path,
        corridor_radius=corridor_radius,
        d_max=d_max,
        device=device,
    )

    log_p = noise.log_prob_joint(all_slips, all_scram, biased=False, bias_factor=bias_factor)
    log_q_nom = log_p
    log_q_bias = noise.log_prob_joint(
        all_slips, all_scram, biased=True, bias_factor=bias_factor
    )

    log_alpha = np.log(alpha)
    log_1m = np.log(1.0 - alpha)
    log_q_mix = torch.logaddexp(log_q_nom + log_alpha, log_q_bias + log_1m)
    log_w = log_p - log_q_mix
    weights = torch.exp(torch.clamp(log_w, -50, 50))

    fail_w = failures.float() * weights
    p_fail = fail_w.mean().item()
    p_fail_std = (fail_w.std() / np.sqrt(n_rollouts)).item()
    ess = (weights.sum() ** 2 / (weights**2).sum()).item()

    # Failures on nominal subsample for mission success
    fails_nom = failure(
        trajs_nom,
        doses_nom,
        nominal_path,
        corridor_radius=corridor_radius,
        d_max=d_max,
        device=device,
    )
    true_success = done_nom & (~fails_nom)
    mission_success = true_success.float().mean().item()
    se = np.sqrt(mission_success * (1.0 - mission_success) / n_nom) if n_nom > 0 else 0.0
    ci95 = 1.96 * se

    return {
        "p_fail": p_fail,
        "p_fail_std": p_fail_std,
        "ess": ess,
        "mission_success_rate": mission_success,
        "mission_success_ci95_halfwidth": ci95,
        "n_rollouts": n_rollouts,
        "n_nominal": n_nom,
        "mean_dose": all_doses.mean().item(),
        "max_dose": all_doses.max().item(),
        "trajs": all_trajs.detach().cpu().numpy(),
        "doses": all_doses.detach().cpu().numpy(),
        "method": "defensive_mixture",
        "alpha": alpha,
        "bias_factor": bias_factor,
        "device": device,
    }


def naive_monte_carlo(
    s0_3d: np.ndarray,
    *,
    dynamics: DynamicsPlugin,
    controller: ControllerPlugin,
    noise: NoisePlugin,
    hazard: HazardPlugin,
    failure: FailurePlugin,
    nominal_path: np.ndarray,
    dt: float,
    horizon: int,
    task_radius: float,
    corridor_radius: float,
    d_max: float,
    n_rollouts: int = 2000,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    trajs, doses, _, _, done = batched_rollouts(
        s0_3d,
        dynamics=dynamics,
        controller=controller,
        noise=noise,
        hazard=hazard,
        dt=dt,
        horizon=horizon,
        task_radius=task_radius,
        n_rollouts=n_rollouts,
        device=device,
        bias_factor=1.0,
    )
    fails = failure(
        trajs,
        doses,
        nominal_path,
        corridor_radius=corridor_radius,
        d_max=d_max,
        device=device,
    ).float()
    p_fail = fails.mean().item()
    p_fail_std = (fails.std() / np.sqrt(n_rollouts)).item()
    success = (done & (~fails.bool())).float().mean().item()
    se = np.sqrt(success * (1.0 - success) / n_rollouts)
    return {
        "p_fail": p_fail,
        "p_fail_std": p_fail_std,
        "ess": float(n_rollouts),
        "mission_success_rate": success,
        "mission_success_ci95_halfwidth": 1.96 * se,
        "n_rollouts": n_rollouts,
        "mean_dose": doses.mean().item(),
        "max_dose": doses.max().item(),
        "trajs": trajs.detach().cpu().numpy(),
        "method": "monte_carlo",
        "device": device,
    }


def estimate_failure_probability(
    method: str,
    *args,
    **kwargs,
) -> Dict[str, Any]:
    method = method.lower()
    if method in ("defensive_mixture", "defensive", "is"):
        return defensive_mixture_is(*args, **kwargs)
    if method in ("monte_carlo", "mc", "naive"):
        # Drop IS-only kwargs if present
        kwargs.pop("alpha", None)
        kwargs.pop("bias_factor", None)
        return naive_monte_carlo(*args, **kwargs)
    raise ValueError(f"Unknown rare-event method: {method!r}")
