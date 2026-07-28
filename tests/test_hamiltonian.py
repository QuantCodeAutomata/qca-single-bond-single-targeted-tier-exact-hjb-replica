"""
Unit tests for Hamiltonian precomputation.
"""

import numpy as np
import pytest
from src.primitives.hamiltonian import compute_hamiltonian, build_tier_hamiltonians
from src.primitives.fill_probability import fill_intensity


def test_hamiltonian_nonnegative():
    """H(p) >= 0 for all p (Hamiltonian is a sup of nonneg quantities)."""
    h = compute_hamiltonian(lam=500.0, alpha=2.0, beta=2.0)
    assert np.all(h.H >= 0.0)


def test_envelope_property():
    """H'(p) ≈ -Lambda(delta_tilde(p)) — envelope theorem."""
    h = compute_hamiltonian(lam=500.0, alpha=2.0, beta=2.0)
    # Check at stored grid points
    lam_at_opt = fill_intensity(h.delta_tilde, 500.0, 2.0, 2.0)
    np.testing.assert_allclose(h.H_prime, -lam_at_opt, rtol=1e-6)


def test_hamiltonian_interpolators_built():
    """HamiltonianInterpolators should have all callables after build."""
    h = compute_hamiltonian(lam=200.0, alpha=1.5, beta=1.5)
    assert callable(h.H_fn)
    assert callable(h.H_prime_fn)
    assert callable(h.H_double_prime_fn)
    assert callable(h.delta_tilde_fn)


def test_H_prime_at_zero_equals_minus_lambda_at_delta_hat():
    """H'(0) = -Lambda(delta_hat_0) where delta_hat_0 = argmax Lambda(d)*d."""
    lam, alpha, beta = 50.0, 1.0, 1.0
    h = compute_hamiltonian(lam=lam, alpha=alpha, beta=beta)
    d0 = float(h.delta_tilde_fn(0.0))
    lambda_at_d0 = float(fill_intensity(d0, lam, alpha, beta))
    Hp_at_0 = float(h.H_prime_fn(0.0))
    assert abs(Hp_at_0 + lambda_at_d0) < 1e-3


def test_build_tier_hamiltonians_keys():
    """TierHamiltonians should have (side, k) entries for all combinations."""
    z = np.array([1.0, 5.0, 20.0])
    lambdas = np.array([500.0, 200.0, 50.0])
    alphas  = np.array([2.0, 1.5, 1.0])
    betas   = np.array([2.0, 1.5, 1.0])
    th = build_tier_hamiltonians("test", lambdas, alphas, betas, z, targeted=True)

    for s in ["bid", "ask"]:
        for k in range(len(z)):
            assert (s, k) in th.hamiltonians, f"Missing key ({s},{k})"


def test_tier_hamiltonians_targeted_flag():
    """targeted flag is stored correctly."""
    z = np.array([1.0, 5.0])
    lams = np.array([100.0, 50.0])
    a = np.array([2.0, 1.5])
    b = np.array([2.0, 1.5])
    th_t = build_tier_hamiltonians("targeted", lams, a, b, z, targeted=True)
    th_u = build_tier_hamiltonians("untargeted", lams, a, b, z, targeted=False)
    assert th_t.targeted is True
    assert th_u.targeted is False


def test_H_double_prime_0_positive():
    """H''(0) must be positive: H(p) = sup_delta Lambda(delta)(delta-p) is convex in p.

    H is the supremum of affine functions in p (each with slope -Lambda < 0),
    so H is convex and H''(p) >= 0 everywhere.  The BEGV Riccati matrix
    D = sum z_k H''(0) must be positive for the stationary solution to exist.
    """
    h = compute_hamiltonian(lam=500.0, alpha=2.0, beta=2.0)
    hpp0 = float(h.H_double_prime_fn(0.0))
    assert hpp0 > 0.0, f"H''(0) = {hpp0} should be positive (H is convex in p)"


def test_delta_hat_0_finite():
    """Riskless spread delta_hat_0 should be a finite positive number."""
    th = build_tier_hamiltonians(
        "test",
        np.array([500.0]),
        np.array([2.0]),
        np.array([2.0]),
        np.array([1.0]),
        targeted=True,
    )
    d0 = th.delta_hat_0[("bid", 0)]
    assert np.isfinite(d0) and d0 > 0.0
