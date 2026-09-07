"""Scenario configuration dataclasses with plain-English field docs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioConfig:
    """A complete qualification experiment definition."""

    name: str
    description: str = ""
    seed: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    # Flattened convenience fields (filled by loader)
    initial_pose: List[float] = field(default_factory=lambda: [1.2, 0.6, 3.49066])
    dt: float = 0.1
    horizon: int = 100
    task_radius_m: float = 0.15
    corridor_radius_m: float = 0.5
    d_max_msv: float = 50.0
    sigma_obs: float = 0.02
    n_rollouts: int = 2000
    rare_method: str = "defensive_mixture"
    alpha: float = 0.7
    bias_factor: float = 2.2
    reachability_enabled: bool = True
    reachability_mode: str = "receding"  # receding | open_loop | both
    receding_window: int = 20
    compare_open_loop: bool = True
    blowup_growth_threshold: float = 50.0
    rare_events_enabled: bool = True
    ablation: bool = False
    min_mission_success: float = 0.90
    max_p_fail: Optional[float] = None
    save_plot: bool = True
    save_animation: bool = False
    report_path: Optional[str] = None

    def validate_thresholds(self):
        """Validate parameters are within meaningful ranges for ablation and real runs."""
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1) for defensive mixture IS, got {self.alpha}")
        if self.bias_factor <= 0.0:
            raise ValueError(f"bias_factor must be positive, got {self.bias_factor}")
        if self.horizon <= 0:
            raise ValueError(f"horizon must be positive, got {self.horizon}")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if self.task_radius_m <= 0.0 or self.corridor_radius_m <= 0.0:
            raise ValueError("Task and corridor radii must be positive.")
        if self.n_rollouts < 1:
            raise ValueError(f"n_rollouts must be >= 1, got {self.n_rollouts}")

    def as_dict(self) -> Dict[str, Any]:
        return self.raw
