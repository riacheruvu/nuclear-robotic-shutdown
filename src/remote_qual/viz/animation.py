"""Animated dashboard: rollouts + evolving reachability boxes over the dose field."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np

PathLike = Union[str, Path]


def save_animated_dashboard(
    s0: np.ndarray,
    nominal_path: np.ndarray,
    boxes: List,
    trajs: np.ndarray,
    *,
    path: PathLike,
    hazard,
    corridor_radius: float,
    horizon: int,
    task_radius: float = 0.15,
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm

    from remote_qual.viz.static import _dose_grid

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xs_data = np.concatenate([nominal_path[:, 0], [s0[0], 0.0]])
    ys_data = np.concatenate([nominal_path[:, 1], [s0[1], 0.0]])
    pad = corridor_radius + 0.6
    x = np.linspace(xs_data.min() - pad, xs_data.max() + pad, 180)
    y = np.linspace(ys_data.min() - pad, ys_data.max() + pad, 180)
    X, Y, Dose = _dose_grid(hazard, x, y)
    positive = Dose[Dose > 0]
    vmin = max(positive.min(), 1e-8) if positive.size else 1e-8
    vmax = max(Dose.max(), vmin * 10)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0b1020")
    ax.set_facecolor("#0b1020")
    ax.contourf(
        X,
        Y,
        Dose,
        levels=np.logspace(np.log10(vmin), np.log10(vmax), 24),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno",
        alpha=0.8,
    )
    ax.add_patch(
        patches.Circle(
            (hazard.valve_pos[0], hazard.valve_pos[1]),
            task_radius,
            color="#4ade80",
            alpha=0.3,
            zorder=3,
        )
    )
    ax.plot(hazard.valve_pos[0], hazard.valve_pos[1], "*", color="#4ade80", ms=12, zorder=6)
    ax.plot(nominal_path[:, 0], nominal_path[:, 1], "--", color="#e2e8f0", lw=1.5, zorder=5)
    ax.plot(s0[0], s0[1], "s", color="#60a5fa", ms=8, zorder=6)

    n_show = min(15, trajs.shape[1])
    mc_lines = [
        ax.plot([], [], color="#f472b6", lw=1, alpha=0.4, zorder=4)[0] for _ in range(n_show)
    ]
    box_patches: list = []
    robot_marker, = ax.plot(
        [],
        [],
        "o",
        color="#0ea5e9",
        ms=10,
        markeredgecolor="white",
        markeredgewidth=1.5,
        zorder=10,
    )
    ax.set_aspect("equal")
    ax.tick_params(colors="#94a3b8")

    frames = min(horizon, trajs.shape[0] - 1, max(len(boxes) - 1, 1))

    def update(frame):
        ax.set_title(f"Validation dashboard | step {frame}/{frames}", color="white")
        for i, line in enumerate(mc_lines):
            line.set_data(trajs[: frame + 1, i, 0], trajs[: frame + 1, i, 1])
        step = min(frame, trajs.shape[0] - 1)
        robot_marker.set_data([trajs[step, 0, 0]], [trajs[step, 0, 1]])
        for rect in box_patches:
            rect.set_edgecolor("#94a3b8")
            rect.set_alpha(0.5)
            rect.set_linewidth(0.7)
        if frame < len(boxes):
            box = boxes[frame]
            rect = patches.Rectangle(
                (box.lo[0], box.lo[1]),
                box.width[0],
                box.width[1],
                lw=2.0,
                edgecolor="#38bdf8",
                facecolor="none",
                alpha=1.0,
                zorder=5,
            )
            ax.add_patch(rect)
            box_patches.append(rect)
        return mc_lines + box_patches + [robot_marker]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    try:
        ani.save(path, writer="ffmpeg", fps=10, dpi=140, savefig_kwargs={"facecolor": "#0b1020"})
    except Exception:
        # Fallback when ffmpeg is unavailable
        gif_path = path.with_suffix(".gif")
        ani.save(gif_path, writer="pillow", fps=8)
        path = gif_path
    plt.close(fig)
    return path
