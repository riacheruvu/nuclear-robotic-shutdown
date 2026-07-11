from pathlib import Path

from remote_qual.scenario.loader import load_scenario

ROOT = Path(__file__).resolve().parents[1]


def test_load_baseline():
    cfg = load_scenario(ROOT / "scenarios" / "valve_baseline.yaml")
    assert cfg.name == "valve_baseline"
    assert len(cfg.initial_pose) == 3
    assert cfg.n_rollouts >= 100


def test_override_seed():
    cfg = load_scenario(
        ROOT / "scenarios" / "valve_baseline.yaml",
        overrides={"seed": 123},
    )
    assert cfg.seed == 123
