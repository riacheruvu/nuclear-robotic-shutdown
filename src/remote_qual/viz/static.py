"""
Static qualification snapshot: dose field + paths + reachability boxes.

Visual encoding (so the plot stays honest)
------------------------------------------
- Background: log-scaled dose-*rate* field (model units / second) from the
  same point-source formula used in the simulator (not decorative noise).
- Green circle: task success region (task_radius).
- Pink curves: sample of stochastic rollouts (not all rollouts — for clarity).
- Blue rectangles: axis-aligned reachability boxes (x–y only).
- White dashed: nominal (noise-free) path.

ASSUMPTION callout on figure: simplified dose model (see units docs).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

PathLike = Union[str, Path]


def _dose_grid(hazard, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    rates = np.asarray(hazard.dose_rate(pts), dtype=float).reshape(X.shape)
    return X, Y, rates


def save_qualification_figure(
    s0: np.ndarray,
    nominal_path: np.ndarray,
    boxes: List,
    trajs: np.ndarray,
    *,
    path: PathLike,
    hazard,
    corridor_radius: float,
    task_radius: float,
    title: str = "Qualification snapshot",
    n_traj_show: int = 20,
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LogNorm

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Bounds from data
    xs_data = np.concatenate([nominal_path[:, 0], [s0[0], 0.0]])
    ys_data = np.concatenate([nominal_path[:, 1], [s0[1], 0.0]])
    pad = corridor_radius + 0.6
    x = np.linspace(xs_data.min() - pad, xs_data.max() + pad, 220)
    y = np.linspace(ys_data.min() - pad, ys_data.max() + pad, 220)
    X, Y, Dose = _dose_grid(hazard, x, y)

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0b1020")
    ax.set_facecolor("#0b1020")

    # Dose field — scientifically tied to hazard plugin
    positive = Dose[Dose > 0]
    vmin = max(positive.min(), 1e-8) if positive.size else 1e-8
    vmax = max(Dose.max(), vmin * 10)
    cf = ax.contourf(
        X,
        Y,
        Dose,
        levels=np.logspace(np.log10(vmin), np.log10(vmax), 28),
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="inferno",
        alpha=0.85,
    )
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dose-like rate (model units / s)\n[simplified point-source model]", color="#e2e8f0")
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")

    # Task region + valve
    ax.add_patch(
        patches.Circle(
            (hazard.valve_pos[0], hazard.valve_pos[1]),
            task_radius,
            color="#4ade80",
            alpha=0.28,
            zorder=3,
            label=f"Task region (r={task_radius} m)",
        )
    )
    ax.plot(
        hazard.valve_pos[0],
        hazard.valve_pos[1],
        marker="*",
        color="#4ade80",
        ms=14,
        markeredgecolor="black",
        zorder=6,
        label="Valve / source (default colocated)",
    )

    # Stochastic rollouts (subsample for visual clarity — stated in legend)
    n_show = min(n_traj_show, trajs.shape[1])
    for i in range(n_show):
        ax.plot(
            trajs[:, i, 0],
            trajs[:, i, 1],
            color="#f472b6",
            lw=0.9,
            alpha=0.35,
            zorder=4,
        )
    ax.plot([], [], color="#f472b6", lw=1.2, alpha=0.8, label=f"Stochastic rollouts (n={n_show} shown)")

    # Reachability boxes (every other for clutter control)
    for box in boxes[::2]:
        ax.add_patch(
            patches.Rectangle(
                (box.lo[0], box.lo[1]),
                box.width[0],
                box.width[1],
                lw=0.9,
                edgecolor="#38bdf8",
                facecolor="none",
                alpha=0.85,
                zorder=5,
            )
        )
    ax.plot([], [], color="#38bdf8", lw=1.5, label="Reachability boxes (x–y)")

    # Nominal path + start
    ax.plot(
        nominal_path[:, 0],
        nominal_path[:, 1],
        "--",
        color="#e2e8f0",
        lw=1.8,
        zorder=5,
        label="Nominal (noise-free) path",
    )
    ax.plot(s0[0], s0[1], "s", color="#60a5fa", ms=9, zorder=7, label="Start pose")

    # Corridor guide lines (perpendicular offset of start→valve)
    path_angle = np.arctan2(hazard.valve_pos[1] - s0[1], hazard.valve_pos[0] - s0[0])
    perp = path_angle + np.pi / 2
    ox, oy = corridor_radius * np.cos(perp), corridor_radius * np.sin(perp)
    ax.plot(
        [s0[0] + ox, hazard.valve_pos[0] + ox],
        [s0[1] + oy, hazard.valve_pos[1] + oy],
        color="#f87171",
        lw=1.5,
        ls=":",
        alpha=0.75,
        label=f"Corridor guide (±{corridor_radius} m)",
    )
    ax.plot(
        [s0[0] - ox, hazard.valve_pos[0] - ox],
        [s0[1] - oy, hazard.valve_pos[1] - oy],
        color="#f87171",
        lw=1.5,
        ls=":",
        alpha=0.75,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color="#cbd5e1")
    ax.set_ylabel("y (m)", color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.set_title(title, color="white", fontsize=12, pad=12)
    ax.legend(
        fontsize=8,
        loc="upper right",
        facecolor="#1e293b",
        edgecolor="none",
        labelcolor="#e2e8f0",
    )
    fig.text(
        0.5,
        0.01,
        "ASSUMPTION: simplified point-source dose model (not MCNP/ALARA plant software). "
        "Research visualization only.",
        ha="center",
        color="#64748b",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path
