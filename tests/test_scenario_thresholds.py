import pytest
from remote_qual.scenario.schema import ScenarioConfig

def test_threshold_validation_alpha():
    cfg = ScenarioConfig(name="test", alpha=1.5)
    with pytest.raises(ValueError, match="alpha must be in"):
        cfg.validate_thresholds()

def test_threshold_validation_bias():
    cfg = ScenarioConfig(name="test", bias_factor=0.0)
    with pytest.raises(ValueError, match="bias_factor must be positive"):
        cfg.validate_thresholds()
