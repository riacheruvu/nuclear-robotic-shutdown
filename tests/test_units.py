"""Dose unit helper sanity checks."""

import numpy as np

from remote_qual.core.units import SECONDS_PER_HOUR, point_source_dose_rate_per_second


def test_dose_rate_decreases_with_distance():
    near = point_source_dose_rate_per_second(
        0.5, gamma_const=1.32, activity_ci=15.0, mu_air=1e-4, d_bg=0.05
    )
    far = point_source_dose_rate_per_second(
        2.0, gamma_const=1.32, activity_ci=15.0, mu_air=1e-4, d_bg=0.05
    )
    assert float(near) > float(far)


def test_hourly_to_per_second_factor():
    # With mu=0 and r=1, source term is Γ A; plus bg; then /3600.
    r = 1.0
    gamma, A, bg = 1.32, 15.0, 0.05
    got = float(
        point_source_dose_rate_per_second(
            r, gamma_const=gamma, activity_ci=A, mu_air=0.0, d_bg=bg
        )
    )
    expect = (gamma * A / r**2 + bg) / SECONDS_PER_HOUR
    assert np.isclose(got, expect)


def test_min_range_clamp():
    a = point_source_dose_rate_per_second(
        0.0, gamma_const=1.32, activity_ci=15.0, mu_air=0.0, d_bg=0.0
    )
    b = point_source_dose_rate_per_second(
        0.1, gamma_const=1.32, activity_ci=15.0, mu_air=0.0, d_bg=0.0
    )
    assert np.isclose(float(a), float(b))
