"""
visualization.py — Module 3: Plotting & Animation

Fixes vs original notebook:
  1. Indentation bug: `update()` closure is now correctly nested inside
     `create_animated_dashboard()` so it captures mc_lines, box_patches, etc.
  2. blit=True: FuncAnimation now uses blitting for performance; the `update`
     function already returns the full list of modified artists.
  3. Dose-rate heatmap uses the corrected /3600 unit conversion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from config import (
    GAMMA_CONST, ACTIVITY, MU_AIR, D_BG,
    R_SAFE, T_HORIZON,
)
from reachability import IntervalBox


# ── Environment heatmap helper ────────────────────────────────────────────────

def _compute_dose_grid(nominal_path: np.ndarray, s0: np.ndarray):
    """Return (X, Y, Dose_Rate) meshgrids for the environment heatmap."""
    x = np.linspace(nominal_path[:, 0].min() - R_SAFE - 0.5, s0[0] + 0.5, 200)
    y = np.linspace(
        nominal_path[:, 1].min() - R_SAFE - 0.5,
        nominal_path[:, 1].max() + R_SAFE + 0.5,
        200,
    )
    X, Y = np.meshgrid(x, y)
    R = np.clip(np.sqrt(X ** 2 + Y ** 2), 0.1, None)
    # FIX: divide entire sum (including D_BG) by 3600 — all terms are hourly rates
    Dose_Rate = ((GAMMA_CONST * ACTIVITY / R ** 2) * np.exp(-MU_AIR * R) + D_BG) / 3600.0
    return X, Y, Dose_Rate


# ── Static environment panel ──────────────────────────────────────────────────

def plot_realistic_environment(ax, s0: np.ndarray, nominal_path: np.ndarray, mc_trajs=None):
    """Draw the gamma-dose heatmap, success zone, and (optionally) MC rollouts.

    Args:
        ax:           Matplotlib axes to draw onto.
        s0:           Initial robot pose [x, y, θ, ...].
        nominal_path: Nominal trajectory array.
        mc_trajs:     Optional (T+1, B, 7) array; up to 15 trajectories shown.
    """
    X, Y, Dose_Rate = _compute_dose_grid(nominal_path, s0)
    levels = np.logspace(np.log10(D_BG / 3600.0), np.log10(np.max(Dose_Rate)), 30)
    ax.contourf(
        X, Y, Dose_Rate,
        levels=levels,
        cmap="inferno",
        alpha=0.6,
        locator=plt.matplotlib.ticker.LogLocator(),
    )

    ax.add_patch(patches.Circle((0, 0), 0.15, color="#4ade80", alpha=0.3, zorder=3,
                                label=r"$\phi_{task}$ Cone"))
    ax.plot(0, 0, "*", color="#4ade80", ms=12, markeredgecolor="black", zorder=6)

    if mc_trajs is not None:
        for i in range(min(15, mc_trajs.shape[1])):
            ax.plot(mc_trajs[:, i, 0], mc_trajs[:, i, 1],
                    color="#f472b6", lw=1, alpha=0.4, zorder=4)


# ── Animated dashboard ────────────────────────────────────────────────────────

def create_animated_dashboard(
    s0_3d: np.ndarray,
    nominal_path: np.ndarray,
    boxes: list,
    mc_trajs: np.ndarray,
    filename: str = "shutdown_animation.mp4",
):
    """Save a step-by-step animated validation dashboard to an MP4 file.

    Fixes applied:
      • `update` is correctly indented inside this function (closure works).
      • blit=True so matplotlib only redraws the returned artists each frame.

    Args:
        s0_3d:        Initial [x, y, θ].
        nominal_path: Nominal trajectory.
        boxes:        List of IntervalBox objects from reachability_check.
        mc_trajs:     (T+1, B, 7) stochastic trajectories.
        filename:     Output MP4 path.
    """
    print("Generating step-by-step animation...")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.tick_params(colors="#94a3b8")

    # ── Static background ──────────────────────────────────────────────────
    X, Y, Dose_Rate = _compute_dose_grid(nominal_path, s0_3d)
    levels = np.logspace(np.log10(D_BG / 3600.0), np.log10(np.max(Dose_Rate)), 30)
    ax.contourf(
        X, Y, Dose_Rate,
        levels=levels,
        cmap="inferno",
        alpha=0.6,
        locator=plt.matplotlib.ticker.LogLocator(),
    )

    ax.add_patch(patches.Circle((0, 0), 0.15, color="#4ade80", alpha=0.3, zorder=3,
                                label=r"Success Zone ($\phi_{task}$)"))
    ax.plot(0, 0, "*", color="#4ade80", ms=12, markeredgecolor="black", zorder=6,
            label="Manual Valve")
    ax.plot(s0_3d[0], s0_3d[1], "s", color="#60a5fa", ms=8, zorder=6,
            label="Robot Start")
    ax.plot(nominal_path[:, 0], nominal_path[:, 1], "--", color="#e2e8f0", lw=1.5,
            zorder=5, label="Nominal Path")

    # R_safe corridor walls
    path_angle = np.arctan2(0.0 - s0_3d[1], 0.0 - s0_3d[0])
    perp_angle = path_angle + np.pi / 2
    offset_x   = R_SAFE * np.cos(perp_angle)
    offset_y   = R_SAFE * np.sin(perp_angle)
    ax.plot([s0_3d[0] + offset_x, offset_x], [s0_3d[1] + offset_y, offset_y],
            color="#f87171", lw=2, alpha=0.7, ls=":", label=r"$R_{safe}$ Boundary")
    ax.plot([s0_3d[0] - offset_x, -offset_x], [s0_3d[1] - offset_y, -offset_y],
            color="#f87171", lw=2, alpha=0.7, ls=":")

    # ── Animated artists (must be initialised before blit=True) ───────────
    n_show     = min(15, mc_trajs.shape[1])
    mc_lines   = [ax.plot([], [], color="#f472b6", lw=1, alpha=0.4, zorder=4)[0]
                  for _ in range(n_show)]
    box_patches: list = []

    # Legend placeholders
    ax.plot([], [], color="#f472b6", lw=1, alpha=0.4, label="Stochastic MC Rollouts")
    ax.plot([], [], color="#60a5fa", lw=1.5,          label="Formal Reachability Bounds")

    robot_marker, = ax.plot(
        [], [], "o", color="#0ea5e9", ms=10,
        markeredgecolor="white", markeredgewidth=1.5,
        zorder=10, label="Actual Robot Pose",
    )

    ax.legend(fontsize=9, loc="upper right", facecolor="#1e293b",
              edgecolor="none", labelcolor="#e2e8f0")
    ax.set_aspect("equal")

    # ── FIX: update() is now correctly indented inside the outer function ──
    def update(frame):
        ax.set_title(f"Validation Dashboard | Step: {frame}/{T_HORIZON}",
                     color="white", fontsize=12)

        # 1. Stochastic ghost trajectories
        for i, line in enumerate(mc_lines):
            line.set_data(mc_trajs[:frame + 1, i, 0], mc_trajs[:frame + 1, i, 1])

        # 2. Moving robot marker (follows trajectory 0)
        current_step = min(frame, len(mc_trajs) - 1)
        robot_marker.set_data(
            [mc_trajs[current_step, 0, 0]],
            [mc_trajs[current_step, 0, 1]],
        )

        # 3. Fade older reachability boxes into "ghost trail"
        for rect in box_patches:
            rect.set_edgecolor("#94a3b8")
            rect.set_alpha(0.6)
            rect.set_linewidth(0.8)

        # 4. Add current frame's bounding box (bright & bold)
        if frame < len(boxes):
            box  = boxes[frame]
            rect = patches.Rectangle(
                (box.lo[0], box.lo[1]), box.width[0], box.width[1],
                lw=2.0, edgecolor="#38bdf8", facecolor="none", alpha=1.0, zorder=5,
            )
            ax.add_patch(rect)
            box_patches.append(rect)

        # FIX: return artists so blit=True can selectively redraw them
        return mc_lines + box_patches + [robot_marker]

    # FIX: blit=True — only redraws artists returned by update(), much faster
    ani = animation.FuncAnimation(
        fig, update, frames=T_HORIZON, interval=100, blit=True
    )
    ani.save(filename, writer="ffmpeg", fps=10, dpi=150,
             savefig_kwargs={"facecolor": "#0f0f1a"})
    plt.close()
    print(f"Animation successfully saved to {filename}")
