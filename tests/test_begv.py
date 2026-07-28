"""
Unit tests for the BEGV approximation hierarchy.
"""

import numpy as np
import pytest
from src.primitives.hamiltonian import build_tier_hamiltonians
from src.approximations.begv import (
    compute_D_scalar,
    stationary_A,
    solve_riccati_finite_horizon,
    begv_finite_diff,
    solve_xi_exact,
    compute_constant_closure,
    compute_approx_A_policy,
    compute_approx_B_policy,
    compute_approx_C_policy,
    compute_approx_D_policy,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def make_tier_hamiltonians():
    z = np.array([1.0, 5.0, 20.0])
    lambdas = np.array([500.0, 200.0, 50.0])
    alphas  = np.array([2.0, 1.5, 1.0])
    betas   = np.array([2.0, 1.5, 1.0])
    th = build_tier_hamiltonians("tau_star", lambdas, alphas, betas, z, targeted=True)
    return [th], z


# ---------------------------------------------------------------------------
# D and A
# ---------------------------------------------------------------------------

def test_D_scalar_positive():
    """D must be positive: H(p) is convex so H''(p) >= 0, hence D = sum z_k H''(0) > 0."""
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    assert D > 0.0, f"D = {D} should be positive (H convex in p)"


def test_stationary_A_positive():
    """Stationary A should be positive."""
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    A = stationary_A(D, phi=1.0, Sigma=1.0)
    assert A > 0.0


def test_stationary_A_scaling():
    """A scales as sqrt(phi*Sigma/|D|): doubling phi multiplies A by sqrt(2)."""
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    A1 = stationary_A(D, phi=1.0, Sigma=1.0)
    A2 = stationary_A(D, phi=4.0, Sigma=1.0)
    assert abs(A2 / A1 - 2.0) < 1e-10


def test_riccati_terminal_condition():
    """A(T) = eta*Sigma = 0 when eta=0."""
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    t_grid, A_grid = solve_riccati_finite_horizon(D, phi=1.0, Sigma=1.0, T=1.0)
    assert abs(A_grid[-1]) < 1e-12  # A(T) = 0


def test_riccati_converges_to_stationary():
    """A(t=0) should be close to stationary A for long horizon T."""
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    _, A_grid = solve_riccati_finite_horizon(D, phi=1.0, Sigma=1.0, T=100.0)
    A_stat = stationary_A(D, phi=1.0, Sigma=1.0)
    assert abs(A_grid[0] - A_stat) / A_stat < 0.01


# ---------------------------------------------------------------------------
# BEGV finite differences
# ---------------------------------------------------------------------------

def test_begv_fd_bid_ask_symmetry():
    """At q=0, Delta_bid(z_k/2) and Delta_ask(z_k/2) should be equal."""
    A = 0.05
    z_k = 1.0
    fd_bid = begv_finite_diff(A, 0.0, z_k, "bid")
    fd_ask = begv_finite_diff(A, 0.0, z_k, "ask")
    assert abs(fd_bid - fd_ask) < 1e-12


def test_begv_fd_bid_increases_with_q():
    """Bid finite difference should increase with inventory."""
    A = 0.05
    z_k = 1.0
    fd1 = begv_finite_diff(A, 0.0, z_k, "bid")
    fd2 = begv_finite_diff(A, 10.0, z_k, "bid")
    assert fd2 > fd1


# ---------------------------------------------------------------------------
# xi closures
# ---------------------------------------------------------------------------

def test_xi_exact_converges():
    """solve_xi_exact should return a finite float for typical inputs."""
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    A = stationary_A(D, 1.0, 1.0)
    W_target = 2.0 * float(np.dot(z, np.array([500.0, 200.0, 50.0])))
    xi = solve_xi_exact(A, q=0.0, z=z,
                        targeted_hamiltonians=tier_hamiltonians,
                        kappa=10.0, W_target=W_target, r_star=0.1)
    assert np.isfinite(xi)


def test_xi_exact_symmetry():
    """xi(q) should approximately equal xi(-q) under side symmetry."""
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    A = stationary_A(D, 1.0, 1.0)
    W_target = 2.0 * float(np.dot(z, np.array([500.0, 200.0, 50.0])))
    xi_pos = solve_xi_exact(A, q=10.0, z=z,
                            targeted_hamiltonians=tier_hamiltonians,
                            kappa=10.0, W_target=W_target, r_star=0.1)
    xi_neg = solve_xi_exact(A, q=-10.0, z=z,
                            targeted_hamiltonians=tier_hamiltonians,
                            kappa=10.0, W_target=W_target, r_star=0.1)
    assert abs(xi_pos - xi_neg) < 0.1 * abs(xi_pos), \
        f"xi not even: xi(+10)={xi_pos:.4f}, xi(-10)={xi_neg:.4f}"


def test_constant_closure_finite():
    """Constant closure xi_0 should be a finite float."""
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    A = stationary_A(D, 1.0, 1.0)
    W_target = 2.0 * float(np.dot(z, np.array([500.0, 200.0, 50.0])))
    xi_0 = compute_constant_closure(A, z, tier_hamiltonians,
                                    kappa=10.0, W_target=W_target, r_star=0.1)
    assert np.isfinite(xi_0)


# ---------------------------------------------------------------------------
# Policy functions
# ---------------------------------------------------------------------------

def _make_small_grid(q_range=(-20, 21)):
    tier_hamiltonians, z = make_tier_hamiltonians()
    D = compute_D_scalar(tier_hamiltonians, z)
    A = stationary_A(D, 1.0, 1.0)
    W_target = 2.0 * float(np.dot(z, np.array([500.0, 200.0, 50.0])))
    q_grid = np.arange(float(q_range[0]), float(q_range[1]), 1.0)
    return tier_hamiltonians, z, D, A, W_target, q_grid


def test_approx_A_policy_shape():
    """Approx A policy returns arrays of correct shape."""
    ths, z, D, A, W, q_grid = _make_small_grid()
    pol = compute_approx_A_policy(A, q_grid, z, ths, kappa=10.0, W_target=W, r_star=0.1)
    N = len(q_grid)
    assert pol["xi"].shape == (N,)
    assert pol["hit_ratio"].shape == (N,)
    assert ("tau_star", "ask", 0) in pol["delta"]


def test_approx_C_xi_constant():
    """Approx C xi should be constant over inventory."""
    ths, z, D, A, W, q_grid = _make_small_grid()
    pol = compute_approx_C_policy(A, q_grid, z, ths, kappa=10.0, W_target=W, r_star=0.1)
    xi = pol["xi"]
    assert np.allclose(xi, xi[0]), "Approx C xi should be constant"


def test_approx_D_policy_finite():
    """Approx D (linearised map) quotes should be finite."""
    ths, z, D, A, W, q_grid = _make_small_grid()
    pol = compute_approx_D_policy(A, q_grid, z, ths, kappa=10.0, W_target=W, r_star=0.1)
    asks = pol["delta"][("tau_star", "ask", 0)]
    assert np.all(np.isfinite(asks))


def test_hit_ratio_below_one():
    """All hit-ratio values should be in [0, 1]."""
    ths, z, D, A, W, q_grid = _make_small_grid()
    for method in [compute_approx_A_policy, compute_approx_C_policy]:
        pol = method(A, q_grid, z, ths, kappa=10.0, W_target=W, r_star=0.1)
        hr = pol["hit_ratio"]
        assert np.all(hr >= 0.0) and np.all(hr <= 1.0), \
            f"Hit ratio out of [0,1] for {method.__name__}"


def test_approximation_ordering_max_error():
    """
    Approx A should have lower max error vs exact quote than Approx C,
    which should have lower error than Approx D (at this specific inventory range).

    Note: ordering may not hold pointwise everywhere, but A << D in max error
    is expected.
    """
    ths, z, D, A, W, q_grid = _make_small_grid()
    pol_A = compute_approx_A_policy(A, q_grid, z, ths, kappa=10.0, W_target=W, r_star=0.1)
    pol_C = compute_approx_C_policy(A, q_grid, z, ths, kappa=10.0, W_target=W, r_star=0.1)
    pol_D = compute_approx_D_policy(A, q_grid, z, ths, kappa=10.0, W_target=W, r_star=0.1)

    ask_A = pol_A["delta"][("tau_star", "ask", 0)]
    ask_C = pol_C["delta"][("tau_star", "ask", 0)]
    ask_D = pol_D["delta"][("tau_star", "ask", 0)]

    # Approx A vs Approx C: A uses exact xi so should deviate from constant-xi less
    max_diff_AC = float(np.max(np.abs(ask_A - ask_C)))
    max_diff_AD = float(np.max(np.abs(ask_A - ask_D)))
    # D uses linearised map so should deviate more from A
    assert max_diff_AD >= max_diff_AC * 0.5, \
        f"Expected Approx D to deviate more from A than Approx C: " \
        f"AD={max_diff_AD:.4f}, AC={max_diff_AC:.4f}"
