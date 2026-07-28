"""
BEGV approximation hierarchy for the OTC bond market-making HJB.

Implements four levels of approximation (A through D) built on top of a
quadratic value function u(t,q) ≈ -(1/2) A(t) q^2.

Approximation A: Quadratic u + exact nonlinear H + exact xi(q) closure.
Approximation B: Same quadratic u + exact H + local quadratic xi_0 + xi_2*q^2.
Approximation C: Same quadratic u + exact H + constant closure xi_0.
Approximation D: Quadratic u + linearized quote map (linear delta_tilde) + constant xi.

# Custom — Context7 found no library equivalent (paper BEGV approximation hierarchy)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import brentq

from src.config import ScenarioConfig
from src.primitives.hamiltonian import TierHamiltonians


# ---------------------------------------------------------------------------
# Riccati ODE: A(t) for scalar 1D case
# ---------------------------------------------------------------------------

def compute_D_scalar(
    tier_hamiltonians: List[TierHamiltonians],
    z: np.ndarray,
) -> float:
    """
    Compute D = sum_{tau,s,k} z_k * H''_{tau}^{s,k}(0) for the scalar 1D case.

    This coefficient drives the Riccati ODE A'(t) = A(t)*D*A(t) - phi*Sigma.

    Parameters
    ----------
    tier_hamiltonians : list of TierHamiltonians
    z : np.ndarray

    Returns
    -------
    float
        Scalar D value (units: [day^-1] in bp^-1 million^-1 * million = day^-1 bp^-1).
    """
    D = 0.0
    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(len(z)):
                D += float(z[k]) * th.H_double_prime_0[(s, k)]
    return D


def solve_riccati_finite_horizon(
    D: float,
    phi: float,
    Sigma: float,
    T: float,
    n_t: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the scalar Riccati ODE forward in time-to-maturity s = T - t.

        dA/ds = phi*Sigma - D*A^2,   A(s=0) = eta*Sigma = 0

    Analytic solution:
        a = sqrt(phi*Sigma / D)
        A(s) = a * tanh(a * D * s)

    Returns A on a time grid spanning [0, T] from t=T to t=0.

    Parameters
    ----------
    D : float
        Diffusion coefficient from Hamiltonians.
    phi : float
        Running inventory penalty.
    Sigma : float
        Bond variance.
    T : float
        Horizon.
    n_t : int
        Number of time steps for the time grid.

    Returns
    -------
    t_grid : np.ndarray, shape (n_t+1,)
        Time grid from 0 to T (forward clock).
    A_grid : np.ndarray, shape (n_t+1,)
        A(t) values; A(T)=0, A(0) = a*tanh(a*D*T).
    """
    t_grid = np.linspace(0.0, T, n_t + 1)
    s_grid = T - t_grid  # s = T - t (time-to-maturity, s=0 at t=T)

    a = np.sqrt(phi * Sigma / max(D, 1e-12))
    A_grid = a * np.tanh(a * D * s_grid)

    return t_grid, A_grid


def stationary_A(D: float, phi: float, Sigma: float) -> float:
    """
    Stationary-limit inventory curvature: A = sqrt(phi * Sigma / D).

    In 1D this is the positive root of D*A^2 = phi*Sigma.

    Parameters
    ----------
    D : float
    phi : float
    Sigma : float

    Returns
    -------
    float
    """
    return float(np.sqrt(phi * Sigma / max(D, 1e-12)))


# ---------------------------------------------------------------------------
# BEGV finite differences on quadratic u
# ---------------------------------------------------------------------------

def begv_finite_diff(A: float, q: float, z_k: float, side: str) -> float:
    """
    Compute Delta^{s,k}u for the quadratic value function u = -(1/2)*A*q^2.

        Delta^{b,k}u(q) = [u(q) - u(q+z_k)] / z_k = A*(q + z_k/2)
        Delta^{a,k}u(q) = [u(q) - u(q-z_k)] / z_k = A*(-q + z_k/2)

    Parameters
    ----------
    A : float
        Current value of A(t).
    q : float
        Inventory.
    z_k : float
        Trade size.
    side : str
        'bid' or 'ask'.

    Returns
    -------
    float
        Finite difference value.
    """
    if side == "bid":
        return A * (q + z_k / 2.0)
    else:  # ask
        return A * (-q + z_k / 2.0)


# ---------------------------------------------------------------------------
# Exact xi(q) closure (Approximation A)
# ---------------------------------------------------------------------------

def solve_xi_exact(
    A: float,
    q: float,
    z: np.ndarray,
    targeted_hamiltonians: List[TierHamiltonians],
    kappa: float,
    W_target: float,
    r_star: float,
    xi_init: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 300,
) -> float:
    """
    Solve the scalar fixed-point for xi(q) using the quadratic value function:

        xi = kappa * [r* + W_target^{-1} * sum_{tau in targeted, s, k} z_k * H'^{s,k}(A*(±q+z_k/2) - xi)]

    Parameters
    ----------
    A : float
        Inventory curvature (scalar for 1D).
    q : float
        Inventory level.
    z : np.ndarray
        Size ladder.
    targeted_hamiltonians : list of TierHamiltonians
        Only targeted tiers.
    kappa : float
    W_target : float
    r_star : float
    xi_init : float
    tol : float
    max_iter : int

    Returns
    -------
    float
        Solved xi.
    """
    K = len(z)

    def residual(xi: float) -> float:
        total = 0.0
        for th in targeted_hamiltonians:
            for s in th.sides:
                for k in range(K):
                    du = begv_finite_diff(A, q, float(z[k]), s)
                    p = du - xi
                    total += float(z[k]) * float(th.hamiltonians[(s, k)].H_prime_fn(p))
        return xi - kappa * (r_star + total / W_target)

    # Simple damped iteration
    xi = xi_init
    for _ in range(max_iter):
        total = 0.0
        for th in targeted_hamiltonians:
            for s in th.sides:
                for k in range(K):
                    du = begv_finite_diff(A, q, float(z[k]), s)
                    p = du - xi
                    total += float(z[k]) * float(th.hamiltonians[(s, k)].H_prime_fn(p))
        xi_new = kappa * (r_star + total / W_target)
        if abs(xi_new - xi) < tol:
            return xi_new
        xi = 0.6 * xi + 0.4 * xi_new

    # Brent fallback
    try:
        xi = brentq(residual, -2000.0, 2000.0, xtol=1e-12, maxiter=500)
    except ValueError:
        pass  # use last iterate

    return float(xi)


def compute_approx_A_policy(
    A: float,
    q_grid: np.ndarray,
    z: np.ndarray,
    tier_hamiltonians: List[TierHamiltonians],
    kappa: float,
    W_target: float,
    r_star: float,
) -> Dict:
    """
    Compute Approximation A policy over the inventory grid.

    Uses quadratic u with exact nonlinear H and exact xi(q) closure.

    Returns
    -------
    dict with keys:
        'xi': array of xi(q)
        'delta': dict (tier_name, side, size_idx) -> array of delta*(q)
        'hit_ratio': array of instantaneous hit ratio
    """
    N = len(q_grid)
    targeted_hamiltonians = [th for th in tier_hamiltonians if th.targeted]
    K = len(z)

    xi_arr = np.zeros(N)
    delta_opt = {}
    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(K):
                delta_opt[(th.tier_name, s, k)] = np.full(N, np.nan)

    for i, q in enumerate(q_grid):
        xi = solve_xi_exact(
            A=A, q=q, z=z,
            targeted_hamiltonians=targeted_hamiltonians,
            kappa=kappa, W_target=W_target, r_star=r_star,
            xi_init=xi_arr[i - 1] if i > 0 else 0.0,
        )
        xi_arr[i] = xi

        for th in tier_hamiltonians:
            is_targeted = th.targeted
            for s in th.sides:
                for k in range(K):
                    du = begv_finite_diff(A, q, float(z[k]), s)
                    p = du - (xi if is_targeted else 0.0)
                    dopt = float(th.hamiltonians[(s, k)].delta_tilde_fn(p))
                    delta_opt[(th.tier_name, s, k)][i] = dopt

    # Compute hit ratio (targeted tiers only)
    hit_ratio = _compute_hit_ratio(q_grid, delta_opt, tier_hamiltonians, z, W_target,
                                    targeted_only=True)

    return {"xi": xi_arr, "delta": delta_opt, "hit_ratio": hit_ratio}


# ---------------------------------------------------------------------------
# Constant closure xi_0 (Approximation C)
# ---------------------------------------------------------------------------

def compute_constant_closure(
    A: float,
    z: np.ndarray,
    targeted_hamiltonians: List[TierHamiltonians],
    kappa: float,
    W_target: float,
    r_star: float,
) -> float:
    """
    Compute the constant dual closure xi_0 (Approximation C).

        1/kappa_tilde = 1/kappa + W^{-1} * sum_{targeted, s, k} z_k * H''_{s,k}(0)
        y = r* + W^{-1} * sum z_k H'_{s,k}(0) + (1/(2W)) * sum z_k^2 * A * H''_{s,k}(0)
        xi_0 = kappa_tilde * y

    Parameters
    ----------
    A : float
        Current inventory curvature.
    z : np.ndarray
    targeted_hamiltonians : list of TierHamiltonians (targeted only)
    kappa : float
    W_target : float
    r_star : float

    Returns
    -------
    float
        xi_0 constant closure value.
    """
    K = len(z)
    sum_Hpp = 0.0  # sum z_k H''(0)
    sum_Hp = 0.0   # sum z_k H'(0)
    sum_Hpp_z2 = 0.0  # sum z_k^2 H''(0)

    for th in targeted_hamiltonians:
        for s in th.sides:
            for k in range(K):
                h00 = th.H_double_prime_0[(s, k)]
                hp0 = th.H_prime_0[(s, k)]
                sum_Hpp += float(z[k]) * h00
                sum_Hp += float(z[k]) * hp0
                sum_Hpp_z2 += float(z[k])**2 * h00

    inv_kappa_tilde = 1.0 / kappa + sum_Hpp / W_target
    kappa_tilde = 1.0 / inv_kappa_tilde

    y = r_star + sum_Hp / W_target + A * sum_Hpp_z2 / (2.0 * W_target)

    return kappa_tilde * y


def compute_approx_C_policy(
    A: float,
    q_grid: np.ndarray,
    z: np.ndarray,
    tier_hamiltonians: List[TierHamiltonians],
    kappa: float,
    W_target: float,
    r_star: float,
) -> Dict:
    """
    Compute Approximation C policy: constant xi_0 closure.

    Returns
    -------
    dict with 'xi', 'delta', 'hit_ratio' arrays.
    """
    targeted = [th for th in tier_hamiltonians if th.targeted]
    xi_0 = compute_constant_closure(A, z, targeted, kappa, W_target, r_star)
    xi_arr = np.full(len(q_grid), xi_0)

    K = len(z)
    delta_opt = {}
    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(K):
                delta_opt[(th.tier_name, s, k)] = np.full(len(q_grid), np.nan)

    for i, q in enumerate(q_grid):
        for th in tier_hamiltonians:
            for s in th.sides:
                for k in range(K):
                    du = begv_finite_diff(A, q, float(z[k]), s)
                    p = du - (xi_0 if th.targeted else 0.0)
                    delta_opt[(th.tier_name, s, k)][i] = float(
                        th.hamiltonians[(s, k)].delta_tilde_fn(p)
                    )

    hit_ratio = _compute_hit_ratio(q_grid, delta_opt, tier_hamiltonians, z, W_target,
                                    targeted_only=True)
    return {"xi": xi_arr, "delta": delta_opt, "hit_ratio": hit_ratio}


# ---------------------------------------------------------------------------
# Local quadratic closure (Approximation B)
# ---------------------------------------------------------------------------

def compute_approx_B_policy(
    A: float,
    q_grid: np.ndarray,
    z: np.ndarray,
    tier_hamiltonians: List[TierHamiltonians],
    kappa: float,
    W_target: float,
    r_star: float,
    dq_fd: float = 5.0,
) -> Dict:
    """
    Compute Approximation B: local quadratic closure xi(q) ≈ xi_0 + xi_2 * q^2.

    xi_0 is the constant closure from Approximation C.
    xi_2 is estimated numerically from the exact xi(q) curve (Approximation A):
        xi_2 ≈ (xi_exact(dq) - xi_0) / dq^2

    Parameters
    ----------
    A : float
    q_grid : np.ndarray
    z : np.ndarray
    tier_hamiltonians : list
    kappa : float
    W_target : float
    r_star : float
    dq_fd : float
        Step size for numerical estimation of xi_2 (million notional).

    Returns
    -------
    dict with 'xi', 'delta', 'hit_ratio' arrays.
    """
    targeted = [th for th in tier_hamiltonians if th.targeted]
    xi_0 = compute_constant_closure(A, z, targeted, kappa, W_target, r_star)

    # Estimate xi_2 using exact xi at q = dq_fd
    xi_at_dq = solve_xi_exact(
        A=A, q=dq_fd, z=z,
        targeted_hamiltonians=targeted,
        kappa=kappa, W_target=W_target, r_star=r_star,
        xi_init=xi_0,
    )
    xi_2 = (xi_at_dq - xi_0) / (dq_fd**2)

    K = len(z)
    xi_arr = xi_0 + xi_2 * q_grid**2
    delta_opt = {}
    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(K):
                delta_opt[(th.tier_name, s, k)] = np.full(len(q_grid), np.nan)

    for i, q in enumerate(q_grid):
        xi = xi_arr[i]
        for th in tier_hamiltonians:
            for s in th.sides:
                for k in range(K):
                    du = begv_finite_diff(A, q, float(z[k]), s)
                    p = du - (xi if th.targeted else 0.0)
                    delta_opt[(th.tier_name, s, k)][i] = float(
                        th.hamiltonians[(s, k)].delta_tilde_fn(p)
                    )

    hit_ratio = _compute_hit_ratio(q_grid, delta_opt, tier_hamiltonians, z, W_target,
                                    targeted_only=True)
    return {"xi": xi_arr, "delta": delta_opt, "hit_ratio": hit_ratio}


# ---------------------------------------------------------------------------
# Linearised quote map (Approximation D)
# ---------------------------------------------------------------------------

def compute_approx_D_policy(
    A: float,
    q_grid: np.ndarray,
    z: np.ndarray,
    tier_hamiltonians: List[TierHamiltonians],
    kappa: float,
    W_target: float,
    r_star: float,
) -> Dict:
    """
    Compute Approximation D: linearised quote map + constant closure.

        delta_tilde(p) ≈ delta_hat_0 + p / c^{s,k}

    Uses the constant xi_0 closure.

    Returns
    -------
    dict with 'xi', 'delta', 'hit_ratio' arrays.
    """
    targeted = [th for th in tier_hamiltonians if th.targeted]
    xi_0 = compute_constant_closure(A, z, targeted, kappa, W_target, r_star)
    xi_arr = np.full(len(q_grid), xi_0)

    K = len(z)
    delta_opt = {}
    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(K):
                delta_opt[(th.tier_name, s, k)] = np.full(len(q_grid), np.nan)

    for i, q in enumerate(q_grid):
        for th in tier_hamiltonians:
            for s in th.sides:
                for k in range(K):
                    du = begv_finite_diff(A, q, float(z[k]), s)
                    p = du - (xi_0 if th.targeted else 0.0)
                    d0 = th.delta_hat_0[(s, k)]
                    c = th.c_coeff[(s, k)]
                    # Linearised map
                    dopt = d0 + p / c
                    delta_opt[(th.tier_name, s, k)][i] = dopt

    hit_ratio = _compute_hit_ratio(q_grid, delta_opt, tier_hamiltonians, z, W_target,
                                    targeted_only=True)
    return {"xi": xi_arr, "delta": delta_opt, "hit_ratio": hit_ratio}


# ---------------------------------------------------------------------------
# Hit-ratio helper
# ---------------------------------------------------------------------------

def _compute_hit_ratio(
    q_grid: np.ndarray,
    delta_opt: Dict,
    tier_hamiltonians: List[TierHamiltonians],
    z: np.ndarray,
    W_target: float,
    targeted_only: bool = True,
) -> np.ndarray:
    """
    Compute state-wise instantaneous weighted hit ratio.

        r(q) = W_target^{-1} * sum_{tau in targeted, s, k} z_k * Lambda_tau^{s,k}(delta^{s,k*}(q))

    Parameters
    ----------
    q_grid : np.ndarray
    delta_opt : dict
    tier_hamiltonians : list
    z : np.ndarray
    W_target : float
    targeted_only : bool
        If True, restrict sum to targeted tiers.

    Returns
    -------
    np.ndarray, shape (N,)
    """
    from src.primitives.fill_probability import fill_intensity

    N = len(q_grid)
    K = len(z)
    r_arr = np.zeros(N)

    for th in tier_hamiltonians:
        if targeted_only and not th.targeted:
            continue
        # Get parameters from tier config stored in TierHamiltonians
        lams = np.array([th.hamiltonians[("bid", k)].H_prime_fn(0.0) for k in range(K)])
        # H'(0) = -Lambda(delta_hat_0), so Lambda(delta_hat_0) = -H'(0)
        for s in th.sides:
            for k in range(K):
                d_arr = delta_opt.get((th.tier_name, s, k), None)
                if d_arr is None:
                    continue
                # Retrieve lam from H'(0): H'(0) = -lambda * f(delta_hat_0)
                # Actually we need lambda and fill function... store in TierHamiltonians
                if hasattr(th, '_lambdas'):
                    lam = th._lambdas[k]
                    alpha = th._alphas[k]
                    beta = th._betas[k]
                    for i in range(N):
                        if not np.isnan(d_arr[i]):
                            lam_val = fill_intensity(d_arr[i], lam, alpha, beta)
                            r_arr[i] += float(z[k]) * float(lam_val)

    r_arr /= W_target
    return r_arr
