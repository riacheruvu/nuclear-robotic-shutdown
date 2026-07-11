"""Command-line interface: remote-qual."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="remote-qual",
        description=(
            "Risk-informed qualification for remote robots "
            "(reachability + rare-event statistics)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run qualification on a scenario YAML")
    p_run.add_argument("scenario", type=str, help="Path to scenario YAML")
    p_run.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    p_run.add_argument("--seed", type=int, default=None, help="Override scenario seed")
    p_run.add_argument("--rollouts", type=int, default=None, help="Override n_rollouts")
    p_run.add_argument("--ablation", action="store_true", help="Run sampling ablation")
    p_run.add_argument("--no-plot", action="store_true", help="Skip static plot")
    p_run.add_argument("--animation", action="store_true", help="Save animation if possible")
    p_run.add_argument("--device", type=str, default=None, help="cpu or cuda")

    p_val = sub.add_parser("validate", help="Validate a scenario YAML without running")
    p_val.add_argument("scenario", type=str)

    p_list = sub.add_parser("list-plugins", help="List registered plugin names")

    args = parser.parse_args(argv)

    if args.cmd == "list-plugins":
        from remote_qual.plugins.registry import list_plugins

        print(json.dumps(list_plugins(), indent=2))
        return 0

    if args.cmd == "validate":
        from remote_qual.scenario.loader import load_scenario

        cfg = load_scenario(args.scenario)
        print(f"OK: scenario '{cfg.name}' ({args.scenario})")
        print(f"  pose={cfg.initial_pose}  horizon={cfg.horizon}  rollouts={cfg.n_rollouts}")
        return 0

    if args.cmd == "run":
        from remote_qual.pipeline import run_qualification

        overrides = {}
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.rollouts is not None:
            overrides.setdefault("verification", {}).setdefault("rare_events", {})[
                "n_rollouts"
            ] = args.rollouts
        if args.ablation:
            overrides.setdefault("verification", {})["ablation"] = True
        if args.no_plot:
            overrides.setdefault("output", {})["save_plot"] = False
        if args.animation:
            overrides.setdefault("output", {})["save_animation"] = True

        print("Running qualification pipeline...")
        print("(Research evidence — not regulatory certification.)")
        report = run_qualification(
            args.scenario,
            overrides=overrides or None,
            output_dir=args.output_dir,
            device=args.device,
        )
        m = report.metrics
        v = report.thresholds.get("verdict", {})
        print()
        print("=" * 60)
        print(f"  Scenario: {report.scenario_name}")
        print("=" * 60)
        print(report.plain_english_summary)
        print()
        if m.get("mission_success_rate") is not None:
            print(
                f"  Mission success: {m['mission_success_rate']*100:.1f}% "
                f"± {m['mission_success_ci95_halfwidth']*100:.1f}% (95% CI half-width)"
            )
        if m.get("p_fail") is not None:
            print(f"  P(fail):         {m['p_fail']:.6f} ± {m['p_fail_std']:.6f}")
            print(f"  ESS:             {m['ess']:.1f} / {m['n_rollouts']}")
        print(f"  Reachability:    unsafe={m.get('reachability_unsafe')}")
        print(f"  Verdict:         {v.get('overall', 'n/a').upper()}")
        print()
        print("  Key assumptions:")
        for a in report.assumptions[:4]:
            print(f"   - {a}")
        print("   - … see report JSON for full list")
        print()
        art = report.artifacts
        if art.get("report_json"):
            print(f"  Report: {art['report_json']}")
        if art.get("plot"):
            print(f"  Plot:   {art['plot']}")
        print("=" * 60)
        return 0 if v.get("overall") != "fail" else 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
