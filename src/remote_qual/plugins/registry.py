"""Plugin name → class registry and scenario-driven construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

from remote_qual.plugins.base import (
    ControllerPlugin,
    DynamicsPlugin,
    FailurePlugin,
    HazardPlugin,
    NoisePlugin,
)
from remote_qual.plugins.controllers.proportional_heading import ProportionalHeading
from remote_qual.plugins.dynamics.lag_unicycle import LagUnicycleDynamics
from remote_qual.plugins.failures.dose_or_corridor import DoseOrCorridorFailure
from remote_qual.plugins.hazards.point_source_co60 import PointSourceCo60
from remote_qual.plugins.noise.slip_scramble import SlipScrambleNoise

DYNAMICS: Dict[str, Type[DynamicsPlugin]] = {
    "lag_unicycle": LagUnicycleDynamics,
}
CONTROLLERS: Dict[str, Type[ControllerPlugin]] = {
    "proportional_heading": ProportionalHeading,
}
NOISE: Dict[str, Type[NoisePlugin]] = {
    "slip_scramble": SlipScrambleNoise,
}
HAZARDS: Dict[str, Type[HazardPlugin]] = {
    "point_source_co60": PointSourceCo60,
}
FAILURES: Dict[str, Type[FailurePlugin]] = {
    "dose_or_corridor": DoseOrCorridorFailure,
}


@dataclass
class PluginBundle:
    dynamics: DynamicsPlugin
    controller: ControllerPlugin
    noise: NoisePlugin
    hazard: HazardPlugin
    failure: FailurePlugin


def list_plugins() -> Dict[str, list]:
    return {
        "dynamics": sorted(DYNAMICS),
        "controllers": sorted(CONTROLLERS),
        "noise": sorted(NOISE),
        "hazards": sorted(HAZARDS),
        "failures": sorted(FAILURES),
    }


def _build(registry: Dict[str, Type], name: str, params: Dict[str, Any] | None):
    if name not in registry:
        raise KeyError(
            f"Unknown plugin {name!r}. Available: {sorted(registry)}. "
            "Add yours in remote_qual.plugins.registry."
        )
    return registry[name](**(params or {}))


def build_bundle(cfg: Dict[str, Any]) -> PluginBundle:
    """Build plugins from a scenario config dictionary."""
    robot = cfg["robot"]
    env = cfg["environment"]
    noise_cfg = cfg["noise"]

    dynamics = _build(
        DYNAMICS,
        robot.get("dynamics", "lag_unicycle"),
        {"lag_steps": robot.get("lag_steps", 2)},
    )

    cparams = dict(robot.get("controller_params") or {})
    # Map YAML-friendly keys
    if "k_p" in cparams:
        cparams["k_p"] = cparams.pop("k_p") if "k_p" in cparams else cparams.get("k_p")
    # target from valve
    valve = env.get("hazard_params", {}).get("valve_pos", [0.0, 0.0])
    cparams.setdefault("target_xy", tuple(valve))
    # normalize keys
    mapped = {}
    if "k_p" in cparams:
        mapped["k_p"] = cparams["k_p"]
    if "v_nominal" in cparams:
        mapped["v_nominal"] = cparams["v_nominal"]
    if "target_xy" in cparams:
        mapped["target_xy"] = tuple(cparams["target_xy"])
    controller = _build(
        CONTROLLERS,
        robot.get("controller", "proportional_heading"),
        mapped,
    )

    nparams = dict(noise_cfg.get("params") or {})
    noise = _build(NOISE, noise_cfg.get("model", "slip_scramble"), nparams)

    hparams = dict(env.get("hazard_params") or {})
    # rename yaml keys to constructor names when needed
    if "activity_ci" not in hparams and "activity" in hparams:
        hparams["activity_ci"] = hparams.pop("activity")
    if "d_bg_msv_s" in hparams:
        hparams["d_bg"] = hparams.pop("d_bg_msv_s")
    if "valve_pos" in hparams:
        hparams["valve_pos"] = tuple(hparams["valve_pos"])
    if "source_pos" in hparams and hparams["source_pos"] is not None:
        hparams["source_pos"] = tuple(hparams["source_pos"])
    hazard = _build(HAZARDS, env.get("hazard", "point_source_co60"), hparams)

    failure = _build(
        FAILURES,
        cfg.get("failure", {}).get("model", "dose_or_corridor"),
        {},
    )
    return PluginBundle(
        dynamics=dynamics,
        controller=controller,
        noise=noise,
        hazard=hazard,
        failure=failure,
    )
