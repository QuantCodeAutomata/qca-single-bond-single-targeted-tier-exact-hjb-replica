"""
Exact finite-horizon HJB solver for 1D single-bond market-making.

Methodology:
    - Backward explicit Euler in time from t=T to t=0.
    - At each grid point (t, q) solve the implicit scalar fixed-point for xi(t,q)
      then evaluate the separable Hamiltonian contributions.
    - Terminal condition: u(T, q) = -(eta/2) * q^2 * Sigma (= 0 when eta=0).

# Custom — Context7 found no library equivalent (paper HJB Eq. methodology step 6)
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import brentq

from src.config import ScenarioConfig
from src.primitives.hamiltonian import TierHamiltonians
from src.analytics.metrics import compute_hit_ratio


def _finite_diff(u: np.ndarray, q_grid: np.ndarray, z_k: float, side: str) -> np.ndarray:
    """
    Compute the normalised finite-difference Delta^{s,k}u at every grid point.

        Delta^{b,k}u(q) = [u(q) - u(q + z_k)] / z_k   (bid: inventory rises)
        Delta^{a,k}u(q) = [u(q) - u(q - z_k)] / z_k   (ask: inventory falls)

    Grid points where the shifted state is outside [-q_max, q_max] return NaN
    (these transitions are suppressed in the HJB update).

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Value function on q_grid.
    q_grid : np.ndarray, shape (N,)
        Inventory grid.
    z_k : float
        Trade size (million notional).
    side : str
        'bid' or 'ask'.

    Returns
    -------
    np.ndarray, shape (N,)
        Finite differences, NaN where boundary is hit.
    """
    N = len(q_grid)
    dq = q_grid[1] - q_grid[0]
    steps = int(round(z_k / dq))
    fd = np.full(N, np.nan)

    if side == "bid":
        # u(q + z_k): shift right by `steps`
        valid = np.arange(N - steps)
        fd[valid] = (u[valid] - u[valid + steps]) / z_k
    else:  # ask
        # u(q - z_k): shift left by `steps`
        valid = np.arange(steps, N)
        fd[valid] = (u[valid] - u[valid - steps]) / z_k

    return fd


def _solve_xi_fixed_point(
    delta_u_flat: np.ndarray,
    valid_mask: np.ndarray,
    z_vals: np.ndarray,
    H_prime_fns: list,
    kappa: float,
    W: float,
    r_star: float,
    xi_init: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Solve the scalar fixed-point equation for the dual variable xi at one grid point.

        xi = kappa * [r* + W^{-1} * sum_{s,k} z_k * H'^{s,k}(Delta^{s,k}u - xi)]

    Only terms where valid_mask is True are included (boundary transitions excluded).

    Uses simple iteration with damping, falling back to Brent's method if needed.

    Parameters
    ----------
    delta_u_flat : np.ndarray, shape (2*K,)
        Stacked [Delta^{b,0}u, ..., Delta^{b,K-1}u, Delta^{a,0}u, ..., Delta^{a,K-1}u].
    valid_mask : np.ndarray of bool, shape (2*K,)
        True where the transition is inside the grid.
    z_vals : np.ndarray, shape (K,)
        Size ladder.
    H_prime_fns : list of callables, length (2*K,)
        H'(p) interpolators in the same order as delta_u_flat.
    kappa : float
        Hit-ratio penalty weight.
    W : float
        Total notional arrival scale.
    r_star : float
        Target hit ratio.
    xi_init : float
        Initial guess for xi.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum iterations for simple fixed-point iteration.

    Returns
    -------
    float
        Converged xi value.
    """
    K = len(z_vals)
    # Build array of z_k contributions (0 where invalid)
    z_rep = np.concatenate([z_vals, z_vals])  # bid then ask
    z_eff = np.where(valid_mask, z_rep, 0.0)

    def residual(xi: float) -> float:
        """Residual: xi - kappa*[r* + W^{-1} sum z_k H'(delta_u - xi)]"""
        total = 0.0
        for i in range(2 * K):
            if valid_mask[i]:
                p_i = delta_u_flat[i] - xi
                total += z_eff[i] * H_prime_fns[i](p_i)
        return xi - kappa * (r_star + total / W)

    # Simple fixed-point iteration with damping
    xi = xi_init
    for _ in range(max_iter):
        total = 0.0
        for i in range(2 * K):
            if valid_mask[i]:
                p_i = delta_u_flat[i] - xi
                total += z_eff[i] * H_prime_fns[i](p_i)
        xi_new = kappa * (r_star + total / W)
        if abs(xi_new - xi) < tol:
            return xi_new
        xi = 0.7 * xi + 0.3 * xi_new  # damped update

    # Fallback: Brent's method over a reasonable bracket
    try:
        lo, hi = -500.0, 500.0
        f_lo = residual(lo)
        f_hi = residual(hi)
        if f_lo * f_hi > 0:
            # Expand bracket
            for scale in [1e3, 1e4, 1e5]:
                lo, hi = -scale, scale
                f_lo = residual(lo)
                f_hi = residual(hi)
                if f_lo * f_hi <= 0:
                    break
        xi = brentq(residual, lo, hi, xtol=1e-12)
    except ValueError:
        xi = xi_new  # use last iterate

    return float(xi)


def hjb_rhs(
    u: np.ndarray,
    q_grid: np.ndarray,
    z: np.ndarray,
    tier_hamiltonians: List[TierHamiltonians],
    W: float,
    W_target: float,
    kappa: float,
    r_star: float,
    phi: float,
    Sigma: float,
    xi_prev: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the HJB right-hand side at all grid points simultaneously.

    The HJB RHS at each q is:
        -(phi/2)*q^2*Sigma + sum_{tau,s,k} z_k * H_{tau}^{s,k}(p^{s,k})
            + W_target*xi*r* - W_target*xi^2/(2*kappa)

    where p^{s,k} = Delta^{s,k}u - xi for targeted tiers, Delta^{s,k}u otherwise.

    Fully vectorized: all N inventory grid points are processed simultaneously.
    The scalar xi fixed-point at each q is solved via Newton's method applied
    element-wise to an N-vector, exploiting that CubicSpline interpolators accept
    array inputs natively.

    Parameters
    ----------
    u : np.ndarray
        Current value function on q_grid.
    q_grid : np.ndarray
        Inventory grid.
    z : np.ndarray
        Size ladder.
    tier_hamiltonians : list of TierHamiltonians
        All tiers (with .targeted attribute set).
    W : float
        Notional arrival scale for targeted tiers (hit-ratio denominator).
    W_target : float
        Same as W (passed separately for clarity in the xi penalty term).
    kappa : float
        Hit-ratio penalty weight.
    r_star : float
        Target hit ratio.
    phi : float
        Running inventory penalty coefficient.
    Sigma : float
        Bond variance.
    xi_prev : np.ndarray or None
        Previous xi values used as warm-start for Newton iteration.

    Returns
    -------
    rhs : np.ndarray, shape (N,)
        HJB right-hand side at each grid point.
    xi_vals : np.ndarray, shape (N,)
        Solved dual variable xi at each grid point.
    delta_opt : dict
        Optimal quotes delta^{s,k*}(q) for each (tier_name, side, size_idx).
    """
    N = len(q_grid)
    K = len(z)

    # ------------------------------------------------------------------
    # Step 1: precompute all finite differences — shape (N,) per key
    # Boundary transitions (shifted state outside grid) are set to 0
    # and tracked in a boolean valid mask.
    # ------------------------------------------------------------------
    tier_fd: Dict = {}
    tier_valid: Dict = {}
    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(K):
                fd_raw = _finite_diff(u, q_grid, float(z[k]), s)
                valid = ~np.isnan(fd_raw)
                tier_fd[(th.tier_name, s, k)] = np.where(valid, fd_raw, 0.0)
                tier_valid[(th.tier_name, s, k)] = valid

    # ------------------------------------------------------------------
    # Step 2: vectorized Newton solve for xi(q) across all N states.
    #
    # Residual: R(xi) = xi - kappa*[r* + W_target^{-1} sum_{tgt,s,k} z_k H'(fd - xi)]
    # Derivative: dR/dxi = 1 - kappa*W_target^{-1} sum z_k H''(fd - xi)
    #   Since H'' <= 0, dR/dxi >= 1 (unique root, Newton converges fast).
    # ------------------------------------------------------------------
    xi: np.ndarray = np.zeros(N) if xi_prev is None else xi_prev.copy()

    if kappa > 0:
        for _newton in range(60):
            hp_sum = np.zeros(N)
            hpp_sum = np.zeros(N)
            for th in tier_hamiltonians:
                if not th.targeted:
                    continue
                for s in th.sides:
                    for k in range(K):
                        fd = tier_fd[(th.tier_name, s, k)]
                        valid = tier_valid[(th.tier_name, s, k)]
                        p = fd - xi
                        ham = th.hamiltonians[(s, k)]
                        hp = np.asarray(ham.H_prime_fn(p), dtype=float)
                        hpp = np.asarray(ham.H_double_prime_fn(p), dtype=float)
                        zk = float(z[k])
                        hp_sum += zk * np.where(valid, hp, 0.0)
                        hpp_sum += zk * np.where(valid, hpp, 0.0)

            R = xi - kappa * (r_star + hp_sum / W_target)
            dR = 1.0 - kappa * hpp_sum / W_target
            # dR >= 1 theoretically; clip to avoid division by tiny positive values
            dR = np.where(np.abs(dR) < 1e-12, 1e-12, dR)
            step = R / dR
            xi = xi - step
            if float(np.max(np.abs(step))) < 1e-12:
                break

    xi_vals = xi

    # ------------------------------------------------------------------
    # Step 3: evaluate Hamiltonian contributions and extract optimal quotes
    # All CubicSpline evaluations are over the full length-N p-vector.
    # ------------------------------------------------------------------
    rhs = -(phi / 2.0) * q_grid**2 * Sigma
    delta_opt_raw: Dict = {}

    for th in tier_hamiltonians:
        is_targeted = th.targeted
        for s in th.sides:
            for k in range(K):
                fd = tier_fd[(th.tier_name, s, k)]
                valid = tier_valid[(th.tier_name, s, k)]
                p = fd - (xi_vals if is_targeted else 0.0)
                ham = th.hamiltonians[(s, k)]
                H_arr = np.asarray(ham.H_fn(p), dtype=float)
                dopt = np.asarray(ham.delta_tilde_fn(p), dtype=float)
                rhs += float(z[k]) * np.where(valid, H_arr, 0.0)
                delta_opt_raw[(th.tier_name, s, k)] = np.where(valid, dopt, np.nan)

    # xi correction from dualisation of the hit-ratio penalty
    if kappa > 0 and W_target > 0:
        rhs += W_target * r_star * xi_vals - W_target * xi_vals**2 / (2.0 * kappa)

    return rhs, xi_vals, delta_opt_raw


def solve_hjb_exact_1d(
    cfg: ScenarioConfig,
    tier_hamiltonians: List[TierHamiltonians],
    n_time_steps: int = 10000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Solve the exact finite-horizon 1D HJB via backward explicit Euler.

    Time runs from T to 0 in steps of dt = T / n_time_steps.

    Parameters
    ----------
    cfg : ScenarioConfig
        Scenario parameters.
    tier_hamiltonians : list of TierHamiltonians
        Precomputed Hamiltonians.
    n_time_steps : int
        Number of time steps (dt = T / n_time_steps).

    Returns
    -------
    u0 : np.ndarray, shape (N,)
        Value function at t=0.
    xi0 : np.ndarray, shape (N,)
        Dual variable xi at t=0 (last backward step).
    q_grid : np.ndarray, shape (N,)
        Inventory grid.
    delta_opt0 : dict
        Optimal quotes at t=0 for each (tier_name, side, size_idx).
    """
    q_grid = cfg.q_grid
    N = len(q_grid)
    z = cfg.z
    dt = cfg.T / n_time_steps

    W_total = cfg.W_by_tier
    # W = total notional arrival scale across all tiers (for hit-ratio denominator)
    # but since only targeted tiers appear in r(delta) = W_target^{-1} sum z_k Lambda...
    # Follow methodology: W = sum over targeted tiers only
    W = cfg.W_target
    W_target = cfg.W_target

    # Mark targeted attribute on each TierHamiltonians (needed by hjb_rhs)
    for th in tier_hamiltonians:
        tier_cfg = next((t for t in cfg.tiers if t.name == th.tier_name), None)
        if tier_cfg is not None:
            th.targeted = tier_cfg.targeted
        else:
            th.targeted = False

    # Terminal condition: u(T, q) = -(eta/2)*q^2*Sigma
    u = -(cfg.eta / 2.0) * q_grid**2 * cfg.Sigma
    xi_prev = np.zeros(N)

    # Backward Euler
    for step in range(n_time_steps):
        rhs, xi_vals, delta_opt = hjb_rhs(
            u=u,
            q_grid=q_grid,
            z=z,
            tier_hamiltonians=tier_hamiltonians,
            W=W,
            W_target=W_target,
            kappa=cfg.kappa,
            r_star=cfg.r_star,
            phi=cfg.phi,
            Sigma=cfg.Sigma,
            xi_prev=xi_prev,
        )
        u = u + dt * rhs
        xi_prev = xi_vals

    hit_ratio = compute_hit_ratio(q_grid, delta_opt, tier_hamiltonians, z, W_target)
    return {
        "u": u,
        "xi": xi_vals,
        "q_grid": q_grid,
        "delta": delta_opt,
        "hit_ratio": hit_ratio,
    }
