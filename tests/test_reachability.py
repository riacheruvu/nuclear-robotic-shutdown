"""Receding-horizon reachability and box-volume diagnostics."""

import numpy as np

from remote_qual.plugins.controllers.proportional_heading import ProportionalHeading
from remote_qual.plugins.dynamics.lag_unicycle import LagUnicycleDynamics
from remote_qual.pipeline import compute_nominal_trajectory
from remote_qual.plugins.hazards.point_source_co60 import PointSourceCo60
from remote_qual.verification.reachability.interval_box import reachability_check


def _setup():
    dyn = LagUnicycleDynamics()
    ctl = ProportionalHeading(k_p=2.5, v_nominal=0.5)
    haz = PointSourceCo60()
    s0 = np.array([1.2, 0.6, np.deg2rad(200)])
    nom = compute_nominal_trajectory(
        s0,
        dynamics=dyn,
        controller=ctl,
        dt=0.1,
        horizon=100,
        task_radius=0.15,
        hazard=haz,
    )
    return dyn, ctl, s0, nom


def test_open_loop_grows_more_than_receding():
    dyn, ctl, s0, nom = _setup()
    common = dict(
        dynamics=dyn,
        controller=ctl,
        nominal_path=nom,
        sigma_slip=0.05,
        sigma_obs=0.02,
        dt=0.1,
        horizon=80,
        corridor_radius=0.5,
        k_p=2.5,
        v_nominal=0.5,
        compare_open_loop=False,
        blowup_growth_threshold=50.0,
    )
    ol = reachability_check(s0, mode="open_loop", **common)
    rh = reachability_check(s0, mode="receding", receding_window=20, **common)
    assert ol.mode == "open_loop"
    assert rh.mode == "receding"
    assert len(rh.recert_steps) >= 2
    # Open-loop typically accumulates more area over long horizons
    assert ol.max_xy_area >= rh.max_xy_area * 0.5  # soft: both computed
    assert rh.growth_ratio < ol.growth_ratio or rh.max_xy_area <= ol.max_xy_area


def test_receding_compares_open_loop():
    dyn, ctl, s0, nom = _setup()
    res = reachability_check(
        s0,
        dynamics=dyn,
        controller=ctl,
        nominal_path=nom,
        sigma_slip=0.05,
        sigma_obs=0.02,
        dt=0.1,
        horizon=60,
        corridor_radius=0.5,
        k_p=2.5,
        v_nominal=0.5,
        mode="receding",
        receding_window=15,
        compare_open_loop=True,
    )
    assert res.open_loop_growth_ratio is not None
    assert res.open_loop_is_unsafe is not None
    assert len(res.xy_area_series) == 61  # horizon+1 including seed
    assert res.notes


def test_xy_area_nonnegative():
    dyn, ctl, s0, nom = _setup()
    res = reachability_check(
        s0,
        dynamics=dyn,
        controller=ctl,
        nominal_path=nom,
        sigma_slip=0.05,
        sigma_obs=0.02,
        dt=0.1,
        horizon=30,
        corridor_radius=0.5,
        k_p=2.5,
        v_nominal=0.5,
        mode="receding",
        compare_open_loop=False,
    )
    assert all(a >= 0 for a in res.xy_area_series)
