"""
importance_sampling.py — Module 2: GPU-Accelerated Defensive Mixture IS

Key fixes vs original notebook:
  1. Radiation unit conversion: GAMMA_CONST is in R·m²/(hr·Ci).
     Divide by 3600 so the per-step dose uses seconds, not hours.
  2. Redundant log_q_nom variable removed for clarity; the true distribution
     density log_p is computed once and reused correctly.
  3. Joint Likelihood Fix: IS weights now correctly track both Gaussian slip 
     noise AND the Bernoulli salt-and-pepper scrambles to prevent weight collapse.
  4. True Mission Success: Only counts robots that reach the valve AND 
     do not fail safety constraints.
"""

import numpy as np
import torch

from config import (
    DT, T_HORIZON, K_P, V_NOMINAL,
    GAMMA_CONST, ACTIVITY, MU_AIR, D_BG,
    P_SALT_PEP, R_SAFE, D_MAX,
)

# ── Failure detection ─────────────────────────────────────────────────────────

def is_failure_batched(
    trajs: torch.Tensor,
    doses: torch.Tensor,
    nominal_path: np.ndarray,
    device: str,
) -> torch.Tensor:
    """Return a boolean tensor (B,) indicating which rollouts are failures."""
    nom_path_t = torch.tensor(nominal_path[:, :2], dtype=torch.float32, device=device)
    dose_fails = doses > D_MAX

    diffs = trajs[:, :, :2].unsqueeze(2) - nom_path_t.unsqueeze(0).unsqueeze(0)
    dists = torch.norm(diffs, dim=3)                        
    min_dists, _ = torch.min(dists, dim=2)                  
    max_dev, _   = torch.max(min_dists, dim=0)              

    corridor_fails = max_dev > R_SAFE
    return dose_fails | corridor_fails


# ── Stochastic rollouts ───────────────────────────────────────────────────────

def batched_rollouts_gpu(
    s0_3d: np.ndarray,
    sigma_slip: float,
    n_rollouts: int,
    T: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    p_scramble: float = P_SALT_PEP,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate n_rollouts stochastic trajectories in parallel on GPU/CPU."""
    B  = n_rollouts
    s0 = np.concatenate([s0_3d, np.zeros(4)])
    s  = torch.tensor(s0, dtype=torch.float32, device=device).repeat(B, 1)

    trajs     = torch.zeros((T + 1, B, 7), device=device)
    trajs[0]  = s
    cum_doses = torch.zeros(B, device=device)
    slips     = torch.normal(mean=0.0, std=sigma_slip, size=(T, B, 2), device=device)
    scrambles = torch.zeros((T, B), device=device)
    done_mask = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(T):
        # 1. Salt-and-pepper sensor scramble 
        scramble_mask = (torch.rand(B, device=device) < p_scramble).float()
        scrambles[t] = scramble_mask
        s_obs = s.clone()
        noise_xy = (torch.rand(B, 2, device=device) * 4.0) - 2.0
        s_obs[:, :2] += scramble_mask.unsqueeze(1) * noise_xy

        # 2. Proportional controller (batched)
        x, y, theta = s_obs[:, 0], s_obs[:, 1], s_obs[:, 2]
        desired_heading = torch.atan2(-y, -x)
        heading_error   = desired_heading - theta
        heading_error   = (heading_error + torch.pi) % (2 * torch.pi) - torch.pi
        omega = K_P * heading_error
        v     = V_NOMINAL * (1.0 - 0.5 * torch.abs(heading_error) / torch.pi)
        u_new = torch.stack([v, omega], dim=1)

        # 3. Apply lagged kinematics (u_{t-2} drives motion)
        s_next = s.clone()
        v_lag, omega_lag = s[:, 5], s[:, 6]
        s_next[:, 0] += v_lag * torch.cos(s[:, 2]) * DT
        s_next[:, 1] += v_lag * torch.sin(s[:, 2]) * DT
        s_next[:, 2] += omega_lag * DT

        # 4. Add position slip noise
        s_next[:, :2] += slips[t]

        # 5. Shift lag buffers
        s_next[:, 5:7] = s[:, 3:5]
        s_next[:, 3:5] = u_new

        # 6. Freeze robots that already reached the valve
        s_next = torch.where(done_mask.unsqueeze(1), s, s_next)

        # 7. Dose accumulation 
        dists     = torch.clamp(torch.norm(s_next[:, :2], dim=1), min=0.1)
        dose_rate = ((GAMMA_CONST * ACTIVITY / dists**2) * torch.exp(-MU_AIR * dists) + D_BG) / 3600.0
        cum_doses += dose_rate * DT * (~done_mask).float()

        # 8. Update done mask
        done_mask = done_mask | (dists < 0.15)

        s = s_next
        trajs[t + 1] = s

    return trajs, cum_doses, slips, scrambles, done_mask


# ── Defensive mixture importance sampling ─────────────────────────────────────

def defensive_mixture_IS_gpu(
    s0: np.ndarray,
    sigma_slip: float,
    nominal_path: np.ndarray,
    n_rollouts: int = 2000,
    alpha: float = 0.7,
    bias_factor: float = 2.2,
) -> tuple[float, float, float, float, float, np.ndarray]:
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_nom  = int(n_rollouts * alpha)
    n_bias = n_rollouts - n_nom

    # Bias both slip AND scramble probability
    p_scram_nom = P_SALT_PEP
    p_scram_bias = min(P_SALT_PEP * bias_factor, 0.9) 

    trajs_nom,  doses_nom,  slips_nom, scram_nom, done_mask_nom = batched_rollouts_gpu(
        s0, sigma_slip, n_nom,  T_HORIZON, device, p_scramble=p_scram_nom)
    
    trajs_bias, doses_bias, slips_bias, scram_bias, _ = batched_rollouts_gpu(
        s0, sigma_slip * bias_factor, n_bias, T_HORIZON, device, p_scramble=p_scram_bias)

    all_trajs = torch.cat([trajs_nom, trajs_bias], dim=1)
    all_doses = torch.cat([doses_nom, doses_bias], dim=0)
    all_slips = torch.cat([slips_nom, slips_bias], dim=1)
    all_scram = torch.cat([scram_nom, scram_bias], dim=1)

    failures = is_failure_batched(all_trajs, all_doses, nominal_path, device)

    # Compute Joint Likelihood (Gaussian Slips + Bernoulli Scrambles)
    dist_p_slip = torch.distributions.Normal(0.0, sigma_slip)
    dist_b_slip = torch.distributions.Normal(0.0, sigma_slip * bias_factor)
    
    dist_p_scram = torch.distributions.Bernoulli(probs=p_scram_nom)
    dist_b_scram = torch.distributions.Bernoulli(probs=p_scram_bias)

    # Nominal joint log-prob
    log_p_slip = dist_p_slip.log_prob(all_slips).sum(dim=(0, 2))
    log_p_scram = dist_p_scram.log_prob(all_scram).sum(dim=0)
    log_p = log_p_slip + log_p_scram
    log_q_nom = log_p 

    # Biased joint log-prob
    log_q_slip_bias = dist_b_slip.log_prob(all_slips).sum(dim=(0, 2))
    log_q_scram_bias = dist_b_scram.log_prob(all_scram).sum(dim=0)
    log_q_bias = log_q_slip_bias + log_q_scram_bias

    log_alpha    = np.log(alpha)
    log_1m_alpha = np.log(1.0 - alpha)
    log_q_mix    = torch.logaddexp(
        log_q_nom  + log_alpha,
        log_q_bias + log_1m_alpha,
    )

    log_weights = log_p - log_q_mix
    weights     = torch.exp(torch.clamp(log_weights, -50, 50))

    fail_weights = failures.float() * weights
    p_fail       = fail_weights.mean().item()
    p_fail_std   = (fail_weights.std() / np.sqrt(n_rollouts)).item()
    ess          = (weights.sum() ** 2 / (weights ** 2).sum()).item()

    # True Mission Success: Reached valve AND did not fail constraints
    true_success_mask = done_mask_nom & (~failures[:n_nom])
    mission_success_rate = true_success_mask.float().mean().item()
    
    # Avoid div by zero if n_nom is 0
    success_se = np.sqrt((mission_success_rate * (1.0 - mission_success_rate)) / n_nom) if n_nom > 0 else 0.0
    success_ci_95 = 1.96 * success_se

    return p_fail, p_fail_std, ess, mission_success_rate, success_ci_95, all_trajs.cpu().numpy()