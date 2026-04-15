"""
ablation.py — Module 4: Sampling Strategy Ablation Study

Compares three estimators on the same scenario:
  1. Naive Monte Carlo              (baseline, no IS)
  2. Defensive Mixture IS           (70/30 split, 2.2σ bias)
  3. Aggressive Tail IS             (50/50 split, 3.2σ bias)
"""

import numpy as np
import torch

from config import T_HORIZON
from importance_sampling import batched_rollouts_gpu, is_failure_batched, defensive_mixture_IS_gpu


def run_ablation_study(
    s0_3d: np.ndarray,
    sigma_slip: float,
    nominal_path: np.ndarray,
    n_rollouts: int = 2000,
) -> None:
    print("\n" + "=" * 60)
    print("  ABLATION STUDY: SAMPLING STRATEGY COMPARISON")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Naive Monte Carlo ────────────────────────────────────────────────────
    # FIX: Unpack 5 values now since batched_rollouts_gpu returns scrambles
    trajs_mc, doses_mc, _, _, _ = batched_rollouts_gpu(
        s0_3d, sigma_slip, n_rollouts, T_HORIZON, device
    )
    fails_mc  = is_failure_batched(trajs_mc, doses_mc, nominal_path, device).float()
    p_fail_mc = fails_mc.mean().item()
    std_mc    = (fails_mc.std() / np.sqrt(n_rollouts)).item()
    print(f"  1. Naive Monte Carlo       | P(fail): {p_fail_mc:.6f} ± {std_mc:.6f}"
          f" | ESS: {n_rollouts:.1f}")

    # 2. Defensive Mixture IS (70/30, 2.2σ) ───────────────────────────────────
    pf_dm, std_dm, ess_dm, _, _, _ = defensive_mixture_IS_gpu(
        s0_3d, sigma_slip, nominal_path, n_rollouts, alpha=0.7, bias_factor=2.2
    )
    print(f"  2. Defensive Mix (2.2σ)    | P(fail): {pf_dm:.6f} ± {std_dm:.6f}"
          f" | ESS: {ess_dm:.1f}")

    # 3. Aggressive Tail IS (50/50, 3.2σ) ─────────────────────────────────────
    pf_agg, std_agg, ess_agg, _, _, _ = defensive_mixture_IS_gpu(
        s0_3d, sigma_slip, nominal_path, n_rollouts, alpha=0.5, bias_factor=3.2
    )
    print(f"  3. Aggressive Tail (3.2σ)  | P(fail): {pf_agg:.6f} ± {std_agg:.6f}"
          f" | ESS: {ess_agg:.1f}")

    print("=" * 60)