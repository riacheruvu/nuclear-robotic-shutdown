"""
Legacy entry point — thin wrapper around the installable toolkit.

Prefer:
    pip install -e ".[viz]"
    remote-qual run scenarios/valve_baseline.yaml --rollouts 10000

Or:
    python main.py
"""

from pathlib import Path

from remote_qual.pipeline import run_qualification


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    report = run_qualification(
        root / "scenarios" / "valve_baseline.yaml",
        overrides={
            "verification": {
                "rare_events": {"n_rollouts": 2000},
                "ablation": True,
            },
            "output": {"save_plot": True, "save_animation": False},
        },
        output_dir=root / "out" / "legacy_main",
    )
    print(report.plain_english_summary)
    print("Report:", report.artifacts.get("report_json"))
