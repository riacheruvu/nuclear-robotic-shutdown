"""End-to-end smoke test with a tiny rollout budget."""

from pathlib import Path

from remote_qual.pipeline import run_qualification

ROOT = Path(__file__).resolve().parents[1]


def test_smoke_baseline(tmp_path):
    report = run_qualification(
        ROOT / "scenarios" / "valve_baseline.yaml",
        overrides={
            "seed": 0,
            "verification": {
                "rare_events": {"n_rollouts": 40, "method": "defensive_mixture"},
                "ablation": False,
            },
            "output": {"save_plot": True, "save_animation": False},
        },
        output_dir=tmp_path,
        device="cpu",
    )
    assert report.scenario_name == "valve_baseline"
    assert "p_fail" in report.metrics
    assert report.metrics["p_fail"] is not None
    assert (tmp_path / "valve_baseline_report.json").exists()
    assert report.assumptions, "assumptions must be non-empty"
