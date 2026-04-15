"""
main.py — Entry Point

A Risk-Informed Qualification Framework for Robotic Remote Shutdown
AA228V / CS238V Final Project

Run:
    python main.py

Outputs:
    nuclear_shutdown_final.png   — static validation snapshot
    shutdown_animation.mp4       — step-by-step animated dashboard
    (metrics printed to stdout)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import T_HORIZON
from dynamics import compute_nominal_trajectory
from reachability import reachability_check
from importance_sampling import defensive_mixture_IS_gpu
from visualization import plot_realistic_environment, create_animated_dashboard
from ablation import run_ablation_study


def run_full_analysis(s0_3d: np.ndarray, sigma_obs: float = 0.02) -> None:
    """Execute the full validation pipeline for a given starting pose.

    Steps:
      1. Compute the noise-free nominal trajectory.
      2. Linearised interval-arithmetic reachability check.
      3. Defensive Mixture Importance Sampling for P(fail).
      4. Save static plot and animated dashboard.
      5. Print final metrics and run the ablation study.

    Args:
        s0_3d:     Initial robot pose [x, y, θ] (3-D).
        sigma_obs: Observation noise std-dev (m); used by reachability module.
    """
    print("Executing GPU-Accelerated Validation Pipeline...")
    nominal_path = compute_nominal_trajectory(s0_3d, T_HORIZON)

    # ── 1. Reachability ────────────────────────────────────────────────────────
    sigma_test = 0.05
    is_unsafe, _, boxes = reachability_check(
        s0_3d, sigma_test, sigma_obs, T_HORIZON, nominal_path
    )

    # ── 2. Importance Sampling ─────────────────────────────────────────────────
    pf, pf_std, ess, mission_success, success_ci_95, mc_trajs = defensive_mixture_IS_gpu(
        s0_3d, sigma_test, nominal_path, n_rollouts=10000, alpha=0.7, bias_factor=2.2
    )

    # ── 3. Animation ───────────────────────────────────────────────────────────
    create_animated_dashboard(
        s0_3d, nominal_path, boxes, mc_trajs, filename="shutdown_animation.mp4"
    )

    # ── 4. Static snapshot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="#94a3b8")

    plot_realistic_environment(ax, s0_3d, nominal_path, mc_trajs)

    for box in boxes[::2]:
        rect = patches.Rectangle(
            (box.lo[0], box.lo[1]), box.width[0], box.width[1],
            lw=0.8, edgecolor="#60a5fa", facecolor="none", alpha=0.8, zorder=5,
        )
        ax.add_patch(rect)

    ax.plot(nominal_path[:, 0], nominal_path[:, 1], "--", color="#e2e8f0", lw=1.5, zorder=5)
    ax.set_aspect("equal")
    ax.set_title(
        f"Validation Snapshot | $\\sigma_{{slip}}={sigma_test}$ | $P(fail)={pf:.4f}$",
        color="white",
    )
    plt.tight_layout()
    plt.savefig("nuclear_shutdown_final.png", dpi=200, facecolor="#0f0f1a")
    print("Analysis complete. Saved to nuclear_shutdown_final.png")

    # ── 5. Metrics ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL VALIDATION METRICS")
    print("=" * 60)
    print(f"  1. Mission Success Rate (Task Liveness): "
          f"{mission_success * 100:.1f}% ± {success_ci_95 * 100:.1f}%")
    if (mission_success - success_ci_95) >= 0.90:
        print("     ✅ SUCCESS: 95% CI strictly exceeds 90% performance threshold.")
    else:
        print("     ❌ FAILED: Did not definitively meet 90% performance threshold.")

    print(f"\n  2. Rare Safety Failure Rate (P_fail):    {pf:.6f} ± {pf_std:.6f}")
    print(f"     Effective Sample Size (ESS):          {ess:.1f} / 10000")
    print("=" * 60)

    # ── 6. Ablation study ──────────────────────────────────────────────────────
    run_ablation_study(s0_3d, sigma_test, nominal_path, n_rollouts=10000)


if __name__ == "__main__":
    s0 = np.array([1.2, 0.6, np.deg2rad(200)])
    run_full_analysis(s0, sigma_obs=0.02)
