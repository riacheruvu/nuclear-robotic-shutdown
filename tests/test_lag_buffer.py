"""Lag buffer semantics: u_{t-2} drives motion."""

import numpy as np

from remote_qual.plugins.controllers.proportional_heading import ProportionalHeading
from remote_qual.plugins.dynamics.lag_unicycle import LagUnicycleDynamics


def test_first_steps_do_not_move_before_lag_fills():
    dyn = LagUnicycleDynamics(lag_steps=2)
    s = dyn.augment_initial(np.array([1.0, 0.0, 0.0]))
    u = np.array([0.5, 0.0])
    s1 = dyn.step(s, u, dt=0.1)
    # First step still uses zero lagged command → no translation
    assert np.allclose(s1[:2], s[:2])
    s2 = dyn.step(s1, u, dt=0.1)
    assert np.allclose(s2[:2], s1[:2])
    s3 = dyn.step(s2, u, dt=0.1)
    # After two buffer shifts, the first u should act
    assert s3[0] > s2[0]


def test_controller_turns_toward_origin():
    ctl = ProportionalHeading(k_p=2.5, v_nominal=0.5, target_xy=(0.0, 0.0))
    # Facing +x while target is behind toward origin from (1,0) → desired heading π
    u = ctl(np.array([1.0, 0.0, 0.0]))
    assert u[1] != 0.0  # nonzero yaw command
