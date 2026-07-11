"""Load and validate scenario YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from remote_qual.scenario.schema import ScenarioConfig

PathLike = Union[str, Path]


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ValueError(f"Scenario missing required key '{key}' in {ctx}.")
    return d[key]


def load_scenario(
    path: PathLike,
    overrides: Optional[Dict[str, Any]] = None,
) -> ScenarioConfig:
    """Parse a YAML scenario into a ScenarioConfig.

    Parameters
    ----------
    path:
        Path to the YAML file.
    overrides:
        Optional nested dict merged on top (e.g. CLI seed override).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError("Scenario YAML must be a mapping/object at the top level.")

    if overrides:
        raw = _deep_merge(raw, overrides)

    name = str(_require(raw, "name", "root"))
    robot = _require(raw, "robot", "root")
    env = _require(raw, "environment", "root")
    noise = _require(raw, "noise", "root")
    verification = raw.get("verification") or {}
    thresholds = raw.get("thresholds") or {}
    output = raw.get("output") or {}

    pose = list(robot.get("initial_pose") or [1.2, 0.6, 3.49066])
    if len(pose) != 3:
        raise ValueError("robot.initial_pose must be [x, y, theta] (length 3).")

    rare = verification.get("rare_events") or {}
    reach = verification.get("reachability") or {}

    cfg = ScenarioConfig(
        name=name,
        description=str(raw.get("description") or ""),
        seed=int(raw.get("seed", 0)),
        raw=raw,
        initial_pose=pose,
        dt=float(env.get("dt", 0.1)),
        horizon=int(env.get("horizon", 100)),
        task_radius_m=float(env.get("task_radius_m", 0.15)),
        corridor_radius_m=float(env.get("corridor_radius_m", 0.5)),
        d_max_msv=float(thresholds.get("max_dose_msv", 50.0)),
        sigma_obs=float(reach.get("sigma_obs", 0.02)),
        n_rollouts=int(rare.get("n_rollouts", 2000)),
        rare_method=str(rare.get("method", "defensive_mixture")),
        alpha=float(rare.get("alpha", 0.7)),
        bias_factor=float(rare.get("bias_factor", 2.2)),
        reachability_enabled=bool(reach.get("enabled", True)),
        rare_events_enabled=bool(rare.get("enabled", True)),
        ablation=bool(verification.get("ablation", False)),
        min_mission_success=float(thresholds.get("min_mission_success", 0.90)),
        max_p_fail=thresholds.get("max_p_fail", None),
        save_plot=bool(output.get("save_plot", True)),
        save_animation=bool(output.get("save_animation", False)),
        report_path=output.get("report_path"),
    )
    # noise block must exist
    _require(noise, "model", "noise")
    return cfg


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
