"""
Unit tests for the CTMC forward-law propagation module.
"""

import numpy as np
import pytest
from src.config import make_baseline_scenario
from src.primitives.hamiltonian import build_tier_hamiltonians
from src.approximations.begv import compute_D_scalar, stationary_A, compute_approx_A_policy
from src.ctmc.forward_law import (
    build_generator,
    propagate_forward_law,
    expected_hit_ratio,
    expected_objective,
)
from src.analytics.metrics import compute_hit_ratio


def make_small_ctmc_setup():
    """Build a small CTMC problem for testing speed."""
    q_grid = np.arange(-10.0, 11.0, 1.0)
    cfg = make_baseline_scenario(kappa=10.0)
    z = cfg.z
    lambdas = cfg.tiers[0].lambdas
    alpha = cfg.tiers[0].alpha
    beta  = cfg.tiers[0].beta

    th = build_tier_hamiltonians("tau_star", lambdas, alpha, beta, z, targeted=True)
    tiers = [th]

    D = compute_D_scalar(tiers, z)
    A = stationary_A(D, cfg.phi, cfg.Sigma)
    W_target = float(2.0 * np.dot(z, lambdas))

    pol = compute_approx_A_policy(A, q_grid, z, tiers, cfg.kappa, W_target, cfg.r_star)
    return cfg, tiers, z, q_grid, pol, W_target


def test_generator_row_sums_zero():
    """Generator rows must sum to zero (probability conservation)."""
    cfg, tiers, z, q_grid, pol, _ = make_small_ctmc_setup()
    G = build_generator(q_grid, z, pol["delta"], tiers)
    row_sums = G.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0.0, atol=1e-8,
                               err_msg="Generator rows must sum to zero")


def test_generator_nonneg_off_diagonal():
    """Off-diagonal elements of generator must be non-negative."""
    import scipy.sparse
    cfg, tiers, z, q_grid, pol, _ = make_small_ctmc_setup()
    G = build_generator(q_grid, z, pol["delta"], tiers)
    G_arr = G.toarray() if scipy.sparse.issparse(G) else np.asarray(G)
    n = G_arr.shape[0]
    mask = ~np.eye(n, dtype=bool)
    assert np.all(G_arr[mask] >= -1e-10), "Off-diagonal generator entries must be >= 0"


def test_forward_law_mass_preserved():
    """Total probability mass must equal 1 at all time points."""
    cfg, tiers, z, q_grid, pol, _ = make_small_ctmc_setup()
    G = build_generator(q_grid, z, pol["delta"], tiers)
    t_eval, mu = propagate_forward_law(G, q_grid, T=1.0, q0=0.0, n_eval=10)
    mass = mu.sum(axis=0)  # shape (n_t,)
    np.testing.assert_allclose(mass, 1.0, atol=1e-6,
                               err_msg="Probability mass not preserved")


def test_initial_distribution_delta():
    """At t=0, mu should be concentrated at q0=0."""
    cfg, tiers, z, q_grid, pol, _ = make_small_ctmc_setup()
    G = build_generator(q_grid, z, pol["delta"], tiers)
    t_eval, mu = propagate_forward_law(G, q_grid, T=1.0, q0=0.0, n_eval=10)
    mu0 = mu[:, 0]
    q0_idx = int(np.argmin(np.abs(q_grid - 0.0)))
    assert mu0[q0_idx] > 0.99, f"Initial mass not concentrated at q0: {mu0[q0_idx]:.4f}"


def test_expected_hit_ratio_in_unit_interval():
    """Expected hit ratio must lie in [0, 1]."""
    cfg, tiers, z, q_grid, pol, W_target = make_small_ctmc_setup()
    G = build_generator(q_grid, z, pol["delta"], tiers)
    t_eval, mu = propagate_forward_law(G, q_grid, T=1.0, q0=0.0, n_eval=10)
    r_q = compute_hit_ratio(q_grid, pol["delta"], tiers, z, W_target)
    ehr = expected_hit_ratio(t_eval, mu, q_grid, r_q, T=1.0)
    assert 0.0 <= ehr <= 1.0, f"Expected hit ratio out of [0,1]: {ehr}"


def test_expected_objective_finite():
    """Expected objective must be finite."""
    cfg, tiers, z, q_grid, pol, W_target = make_small_ctmc_setup()
    G = build_generator(q_grid, z, pol["delta"], tiers)
    t_eval, mu = propagate_forward_law(G, q_grid, T=1.0, q0=0.0, n_eval=10)
    obj = expected_objective(
        t_eval, mu, q_grid, pol["delta"], tiers, z,
        W_target, cfg.r_star, cfg.kappa, cfg.phi, cfg.Sigma, cfg.eta, T=1.0,
    )
    assert np.isfinite(obj), f"Expected objective is not finite: {obj}"


def test_spread_in_mu_nonneg():
    """Forward law probabilities must be non-negative."""
    cfg, tiers, z, q_grid, pol, _ = make_small_ctmc_setup()
    G = build_generator(q_grid, z, pol["delta"], tiers)
    t_eval, mu = propagate_forward_law(G, q_grid, T=1.0, q0=0.0, n_eval=10)
    assert np.all(mu >= -1e-9), "Forward-law distribution has negative probabilities"


def test_hit_ratio_monotone_in_kappa():
    """Larger kappa should push expected hit ratio towards r_star=0.1."""
    q_grid = np.arange(-10.0, 11.0, 1.0)
    z = np.array([1.0, 5.0, 20.0])
    alpha = np.array([2.0, 1.5, 1.0])
    beta  = np.array([2.0, 1.5, 1.0])
    lambdas = np.array([500.0, 200.0, 50.0])
    r_star = 0.1

    ehrs = {}
    for kappa in [1.0, 100.0]:
        th = build_tier_hamiltonians("tau_star", lambdas, alpha, beta, z, targeted=True)
        tiers = [th]
        D = compute_D_scalar(tiers, z)
        A = stationary_A(D, 1.0, 1.0)
        W_target = float(2.0 * np.dot(z, lambdas))
        pol = compute_approx_A_policy(A, q_grid, z, tiers, kappa, W_target, r_star)
        G = build_generator(q_grid, z, pol["delta"], tiers)
        t_eval, mu = propagate_forward_law(G, q_grid, T=1.0, q0=0.0, n_eval=10)
        r_q = compute_hit_ratio(q_grid, pol["delta"], tiers, z, W_target)
        ehrs[kappa] = expected_hit_ratio(t_eval, mu, q_grid, r_q, T=1.0)

    assert ehrs[100.0] >= ehrs[1.0] - 0.005, \
        f"Higher kappa should give higher hit ratio: kappa=1->{ehrs[1.0]:.4f}, kappa=100->{ehrs[100.0]:.4f}"
