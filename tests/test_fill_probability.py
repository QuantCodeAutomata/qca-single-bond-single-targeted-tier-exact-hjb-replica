"""
Unit tests for fill probability functions.
"""

import numpy as np
import pytest
from src.primitives.fill_probability import (
    fill_prob,
    fill_prob_d1,
    fill_intensity,
    riskless_spread,
    linearization_coefficient,
)


def test_fill_prob_in_unit_interval():
    """Fill probability must lie in [0, 1] for all inputs."""
    delta = np.linspace(-100.0, 200.0, 1000)
    alpha, beta = 2.0, 1.5
    f = fill_prob(delta, alpha, beta)
    assert np.all(f >= 0.0) and np.all(f <= 1.0)


def test_fill_prob_monotone_decreasing():
    """Fill probability is strictly decreasing in delta."""
    delta = np.linspace(-10.0, 50.0, 500)
    alpha, beta = 2.0, 1.5
    f = fill_prob(delta, alpha, beta)
    diffs = np.diff(f)
    assert np.all(diffs < 0), "fill_prob should be strictly decreasing"


def test_fill_prob_at_zero():
    """Fill probability at delta=0 is 1/(1+exp(alpha))."""
    alpha, beta = 2.0, 1.5
    expected = 1.0 / (1.0 + np.exp(alpha))
    got = fill_prob(0.0, alpha, beta)
    assert abs(float(got) - expected) < 1e-12


def test_fill_prob_d1_negative():
    """First derivative must be non-positive."""
    delta = np.linspace(-20.0, 100.0, 500)
    alpha, beta = 1.5, 1.0
    fd = fill_prob_d1(delta, alpha, beta)
    assert np.all(fd <= 0.0)


def test_fill_prob_d1_finite_diff():
    """Analytical derivative matches forward finite difference."""
    delta0 = 5.0
    alpha, beta = 2.0, 1.5
    h = 1e-5
    analytic = float(fill_prob_d1(delta0, alpha, beta))
    numeric  = float((fill_prob(delta0 + h, alpha, beta) - fill_prob(delta0, alpha, beta)) / h)
    assert abs(analytic - numeric) < 1e-4


def test_fill_intensity_proportional():
    """fill_intensity = lam * fill_prob."""
    delta = np.linspace(0.0, 50.0, 100)
    lam, alpha, beta = 500.0, 2.0, 2.0
    fi = fill_intensity(delta, lam, alpha, beta)
    fp = fill_prob(delta, alpha, beta)
    np.testing.assert_allclose(fi, lam * fp, rtol=1e-12)


def test_riskless_spread_positive():
    """Riskless spread delta_hat_0 must be positive (dealer earns spread)."""
    alpha, beta = 2.0, 1.5
    d0 = riskless_spread(alpha, beta)
    assert d0 > 0.0, f"Expected positive riskless spread, got {d0}"


def test_riskless_spread_foc():
    """At delta_hat_0, the FOC f + delta * f' = 0 holds approximately."""
    alpha, beta = 2.0, 1.5
    d0 = riskless_spread(alpha, beta)
    f   = float(fill_prob(d0, alpha, beta))
    fp  = float(fill_prob_d1(d0, alpha, beta))
    foc = f + d0 * fp
    assert abs(foc) < 1e-6, f"FOC not satisfied: {foc}"


def test_linearization_coefficient_sign():
    """Linearisation coefficient c should be >= 1 (from paper formula)."""
    alpha, beta = 2.0, 1.5
    d0 = riskless_spread(alpha, beta)
    c = linearization_coefficient(d0, alpha, beta)
    assert c >= 0.0, f"c = {c} should be non-negative"


def test_fill_prob_scalar_and_array_equal():
    """Scalar and array inputs give same result."""
    alpha, beta = 1.0, 1.0
    delta_arr = np.array([0.0, 5.0, 10.0])
    for d in delta_arr:
        arr_val = float(fill_prob(np.array([float(d)]), alpha, beta)[0])
        scal_val = float(fill_prob(d, alpha, beta))
        assert abs(arr_val - scal_val) < 1e-14
