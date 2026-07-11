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
    rare_events_enabled: bool = True
    ablation: bool = False
    min_mission_success: float = 0.90
    max_p_fail: Optional[float] = None
    save_plot: bool = True
    save_animation: bool = False
    report_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return self.raw
