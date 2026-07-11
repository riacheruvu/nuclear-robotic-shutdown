"""
End-to-end qualification pipeline: scenario → evidence.

Plain English
-------------
1. Load the scenario (YAML).
2. Build the robot / noise / radiation plugins.
3. Roll out a noise-free "nominal" path (what the controller intends).
4. Optionally push formal reachability boxes along that path.
5. Optionally run rare-event Monte Carlo / importance sampling.
6. Package everything into a QualificationReport with assumptions called out.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from remote_qual._version import __version__
from remote_qual.plugins.registry import build_bundle
from remote_qual.report.export import export_markdown_summary, export_report
from remote_qual.report.model import QualificationReport
from remote_qual.report.thresholds import evaluate_thresholds
from remote_qual.scenario.loader import load_scenario
from remote_qual.scenario.schema import ScenarioConfig
from remote_qual.verification.rare_events.defensive_mixture import (
    defensive_mixture_is,
    estimate_failure_probability,
    naive_monte_carlo,
)
from remote_qual.verification.reachability.interval_box import reachability_check

PathLike = Union[str, Path]

DEFAULT_ASSUMPTIONS: List[str] = [
    "Point isotropic Co-60-like source with inverse-square + simple air attenuation "
    "(not a full radiation-transport solution such as MCNP).",
    "Dose units are a simplified research model (Γ in R-based tables; totals labelled "
    "mSv for threshold demos — see remote_qual.core.units).",
    "Communication lag is a fixed 2-step FIFO buffer (no random jitter/packet loss in v1).",
    "Empty planar workspace: no walls/obstacles; corridor is relative to the nominal path.",
    "Task success = base within task_radius of the valve (no manipulator / handwheel model).",
    "Sensor scrambles corrupt (x, y) only; heading is not scrambled in v1.",
    "Reachability uses linearized interval boxes (conservative approximation, not exact sets).",
    "Default formal mode is receding-horizon re-certification (re-seed small boxes on the "
    "nominal every N steps) to mitigate interval wrapping/blow-up; not full Hamilton–Jacobi.",
    "XY box-area growth is a diagnostic for vacuous certificates — large growth ⇒ distrust open-loop tubes.",
    "Outputs are research evidence, not regulatory certification.",
]


def compute_nominal_trajectory(
    s0_3d: np.ndarray,
    *,
    dynamics,
    controller,
    dt: float,
    horizon: int,
    task_radius: float,
    hazard,
) -> np.ndarray:
    s = dynamics.augment_initial(np.asarray(s0_3d, dtype=float))
    traj = [s.copy()]
    for _ in range(horizon):
        u = controller(s)
        s = dynamics.step(s, u, dt)
        traj.append(s.copy())
        if float(np.linalg.norm(s[:2] - hazard.valve_pos)) < task_radius:
            break
    return np.asarray(traj)


def run_qualification(
    scenario_path: PathLike,
    *,
    overrides: Optional[Dict[str, Any]] = None,
    output_dir: Optional[PathLike] = None,
    device: Optional[str] = None,
) -> QualificationReport:
    """Run the full qualification stack on a scenario file."""
    cfg = load_scenario(scenario_path, overrides=overrides)
    return run_qualification_config(cfg, output_dir=output_dir, device=device)


def run_qualification_config(
    cfg: ScenarioConfig,
    *,
    output_dir: Optional[PathLike] = None,
    device: Optional[str] = None,
) -> QualificationReport:
    np.random.seed(cfg.seed)
    try:
        import torch

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    except Exception:
        pass

    bundle = build_bundle(cfg.raw)
    s0 = np.asarray(cfg.initial_pose, dtype=float)

    nominal = compute_nominal_trajectory(
        s0,
        dynamics=bundle.dynamics,
        controller=bundle.controller,
        dt=cfg.dt,
        horizon=cfg.horizon,
        task_radius=cfg.task_radius_m,
        hazard=bundle.hazard,
    )

    # Reachability (receding-horizon by default + volume diagnostics)
    reach_result = None
    reach_unsafe = False
    reach_step = -1
    boxes = []
    if cfg.reachability_enabled:
        sigma_slip = float((cfg.raw.get("noise") or {}).get("params", {}).get("sigma_slip", 0.05))
        k_p = float(getattr(bundle.controller, "k_p", 2.5))
        v_nom = float(getattr(bundle.controller, "v_nominal", 0.5))
        reach_result = reachability_check(
            s0,
            dynamics=bundle.dynamics,
            controller=bundle.controller,
            nominal_path=nominal,
            sigma_slip=sigma_slip,
            sigma_obs=cfg.sigma_obs,
            dt=cfg.dt,
            horizon=cfg.horizon,
            corridor_radius=cfg.corridor_radius_m,
            k_p=k_p,
            v_nominal=v_nom,
            mode=cfg.reachability_mode,
            receding_window=cfg.receding_window,
            compare_open_loop=cfg.compare_open_loop,
            blowup_growth_threshold=cfg.blowup_growth_threshold,
        )
        reach_unsafe = reach_result.is_unsafe
        reach_step = reach_result.first_unsafe_step
        boxes = reach_result.boxes

    # Rare events
    rare: Dict[str, Any] = {}
    trajs = None
    if cfg.rare_events_enabled:
        rare = estimate_failure_probability(
            cfg.rare_method,
            s0,
            dynamics=bundle.dynamics,
            controller=bundle.controller,
            noise=bundle.noise,
            hazard=bundle.hazard,
            failure=bundle.failure,
            nominal_path=nominal,
            dt=cfg.dt,
            horizon=cfg.horizon,
            task_radius=cfg.task_radius_m,
            corridor_radius=cfg.corridor_radius_m,
            d_max=cfg.d_max_msv,
            n_rollouts=cfg.n_rollouts,
            alpha=cfg.alpha,
            bias_factor=cfg.bias_factor,
            device=device,
        )
        trajs = rare.get("trajs")

    ablation_results = None
    if cfg.ablation and cfg.rare_events_enabled:
        ablation_results = _run_ablation(cfg, bundle, s0, nominal, device)

    # Thresholds
    ms = float(rare.get("mission_success_rate", float("nan")))
    ci = float(rare.get("mission_success_ci95_halfwidth", float("nan")))
    pf = rare.get("p_fail")
    thr = evaluate_thresholds(
        mission_success_rate=ms if ms == ms else 0.0,
        mission_success_ci95_halfwidth=ci if ci == ci else 1.0,
        p_fail=pf,
        max_dose_observed=rare.get("max_dose"),
        min_mission_success=cfg.min_mission_success,
        max_p_fail=cfg.max_p_fail,
        max_dose_msv=cfg.d_max_msv,
    )

    # Artifacts
    out = Path(output_dir) if output_dir else Path("out") / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Any] = {}
    if cfg.save_plot and trajs is not None:
        try:
            from remote_qual.viz.static import save_qualification_figure

            plot_path = out / f"{cfg.name}_snapshot.png"
            save_qualification_figure(
                s0,
                nominal,
                boxes,
                trajs,
                path=plot_path,
                hazard=bundle.hazard,
                corridor_radius=cfg.corridor_radius_m,
                task_radius=cfg.task_radius_m,
                title=f"{cfg.name} | P(fail)={rare.get('p_fail', float('nan')):.4f}",
            )
            artifacts["plot"] = str(plot_path)
        except Exception as exc:  # viz optional
            artifacts["plot_error"] = str(exc)

    if cfg.save_animation and trajs is not None and boxes:
        try:
            from remote_qual.viz.animation import save_animated_dashboard

            anim_path = out / f"{cfg.name}_animation.mp4"
            save_animated_dashboard(
                s0,
                nominal,
                boxes,
                trajs,
                path=anim_path,
                hazard=bundle.hazard,
                corridor_radius=cfg.corridor_radius_m,
                horizon=cfg.horizon,
            )
            artifacts["animation"] = str(anim_path)
        except Exception as exc:
            artifacts["animation_error"] = str(exc)

    summary = _plain_english_summary(cfg, rare, reach_result, thr)

    reach_metrics: Dict[str, Any] = {
        "reachability_unsafe": reach_unsafe,
        "reachability_first_unsafe_step": reach_step if reach_unsafe else None,
        "reachability_mode": None,
        "reachability_receding_window": None,
        "reachability_max_xy_area_m2": None,
        "reachability_initial_xy_area_m2": None,
        "reachability_growth_ratio": None,
        "reachability_blowup_suspected": None,
        "reachability_recert_steps": None,
        "reachability_open_loop_unsafe": None,
        "reachability_open_loop_growth_ratio": None,
        "reachability_notes": None,
    }
    if reach_result is not None:
        reach_metrics.update(
            {
                "reachability_mode": reach_result.mode,
                "reachability_receding_window": reach_result.receding_window,
                "reachability_max_xy_area_m2": reach_result.max_xy_area,
                "reachability_initial_xy_area_m2": reach_result.initial_xy_area,
                "reachability_growth_ratio": reach_result.growth_ratio,
                "reachability_blowup_suspected": reach_result.blowup_suspected,
                "reachability_recert_steps": list(reach_result.recert_steps),
                "reachability_open_loop_unsafe": reach_result.open_loop_is_unsafe,
                "reachability_open_loop_growth_ratio": reach_result.open_loop_growth_ratio,
                "reachability_notes": reach_result.notes,
                # Downsample area series for JSON (full series can be long)
                "reachability_xy_area_series_m2": _downsample_series(
                    reach_result.xy_area_series, max_points=40
                ),
            }
        )

    report = QualificationReport(
        toolkit_version=__version__,
        scenario_name=cfg.name,
        seed=cfg.seed,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        metrics={
            "mission_success_rate": rare.get("mission_success_rate"),
            "mission_success_ci95_halfwidth": rare.get("mission_success_ci95_halfwidth"),
            "mission_success_ci95_lower": thr.get("mission_success_ci95_lower"),
            "p_fail": rare.get("p_fail"),
            "p_fail_std": rare.get("p_fail_std"),
            "ess": rare.get("ess"),
            "n_rollouts": rare.get("n_rollouts"),
            "mean_dose_msv": rare.get("mean_dose"),
            "max_dose_msv_observed": rare.get("max_dose"),
            "nominal_length_steps": int(len(nominal) - 1),
            **reach_metrics,
        },
        thresholds=thr,
        methods={
            "dynamics": cfg.raw.get("robot", {}).get("dynamics"),
            "controller": cfg.raw.get("robot", {}).get("controller"),
            "noise": cfg.raw.get("noise", {}).get("model"),
            "hazard": cfg.raw.get("environment", {}).get("hazard"),
            "reachability": (
                f"interval_box/{reach_result.mode}" if reach_result is not None else None
            ),
            "reachability_params": {
                "mode": cfg.reachability_mode,
                "receding_window": cfg.receding_window,
                "compare_open_loop": cfg.compare_open_loop,
                "blowup_growth_threshold": cfg.blowup_growth_threshold,
            },
            "rare_events": rare.get("method"),
            "rare_events_params": {
                "alpha": cfg.alpha,
                "bias_factor": cfg.bias_factor,
            },
            "device": rare.get("device"),
        },
        artifacts=artifacts,
        assumptions=list(DEFAULT_ASSUMPTIONS),
        ablation=ablation_results,
        plain_english_summary=summary,
    )

    report_path = Path(cfg.report_path) if cfg.report_path else out / f"{cfg.name}_report.json"
    export_report(report, report_path)
    export_markdown_summary(report, out / f"{cfg.name}_report.md")
    artifacts["report_json"] = str(report_path)
    artifacts["report_md"] = str(out / f"{cfg.name}_report.md")
    report.artifacts = artifacts
    # re-export with artifact paths filled
    export_report(report, report_path)
    return report


def _run_ablation(cfg, bundle, s0, nominal, device):
    common = dict(
        dynamics=bundle.dynamics,
        controller=bundle.controller,
        noise=bundle.noise,
        hazard=bundle.hazard,
        failure=bundle.failure,
        nominal_path=nominal,
        dt=cfg.dt,
        horizon=cfg.horizon,
        task_radius=cfg.task_radius_m,
        corridor_radius=cfg.corridor_radius_m,
        d_max=cfg.d_max_msv,
        n_rollouts=cfg.n_rollouts,
        device=device,
    )
    mc = naive_monte_carlo(s0, **common)
    dm = defensive_mixture_is(s0, alpha=0.7, bias_factor=2.2, **common)
    agg = defensive_mixture_is(s0, alpha=0.5, bias_factor=3.2, **common)
    return {
        "naive_monte_carlo": {
            "p_fail": mc["p_fail"],
            "p_fail_std": mc["p_fail_std"],
            "ess": mc["ess"],
        },
        "defensive_mixture_2.2sigma": {
            "p_fail": dm["p_fail"],
            "p_fail_std": dm["p_fail_std"],
            "ess": dm["ess"],
        },
        "aggressive_tail_3.2sigma": {
            "p_fail": agg["p_fail"],
            "p_fail_std": agg["p_fail_std"],
            "ess": agg["ess"],
        },
    }


def _downsample_series(series, max_points: int = 40):
    if not series:
        return []
    if len(series) <= max_points:
        return [float(x) for x in series]
    idx = np.linspace(0, len(series) - 1, max_points).astype(int)
    return [float(series[i]) for i in idx]


def _plain_english_summary(cfg, rare, reach_result, thr) -> str:
    ms = rare.get("mission_success_rate")
    ci = rare.get("mission_success_ci95_halfwidth")
    pf = rare.get("p_fail")
    verdict = thr.get("verdict", {}).get("overall", "n/a")
    parts = [
        f"Scenario '{cfg.name}' finished with overall verdict **{verdict}**.",
    ]
    if ms is not None and ci is not None:
        parts.append(
            f"About {ms:.0%} of missions reached the valve without safety failure "
            f"(95% CI half-width ±{ci:.1%})."
        )
    if pf is not None:
        parts.append(
            f"Estimated safety failure probability P(fail) ≈ {pf:.4f} "
            f"(lower is better; rare-event estimator used)."
        )
    if reach_result is None:
        parts.append("Formal reachability was disabled for this run.")
    else:
        mode = reach_result.mode
        if reach_result.is_unsafe:
            parts.append(
                f"Formal reachability ({mode}) flagged a corridor exit at step "
                f"{reach_result.first_unsafe_step} under stated noise bounds."
            )
        else:
            parts.append(
                f"Formal reachability ({mode}) did not find a corridor exit "
                "(still an approximate interval model — see assumptions)."
            )
        parts.append(
            f"Peak XY box area growth ≈ {reach_result.growth_ratio:.1f}× initial "
            f"(blow-up flag={'yes' if reach_result.blowup_suspected else 'no'})."
        )
        if (
            reach_result.open_loop_is_unsafe is True
            and not reach_result.is_unsafe
        ):
            parts.append(
                "Open-loop companion tube failed while receding held — typical "
                "interval wrapping; receding re-certification is the preferred default."
            )
    return " ".join(parts)
