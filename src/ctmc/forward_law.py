"""
Continuous-time Markov chain (CTMC) forward law propagation.

Given a stationary policy (quote maps delta^{s,k*}(q) on the inventory grid),
constructs the generator matrix G and propagates the inventory distribution
forward from an initial point mass at q=0.

# Custom — Context7 found no library equivalent (paper forward-law CTMC)
Uses scipy.integrate.solve_ivp for ODE-based Kolmogorov forward equation.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix, csr_matrix

from src.primitives.fill_probability import fill_intensity
from src.analytics.metrics import compute_hit_ratio


def build_generator(
    q_grid: np.ndarray,
    z: np.ndarray,
    delta_opt: Dict,
    tier_hamiltonians: list,
) -> csr_matrix:
    """
    Build the sparse generator matrix G of the inventory CTMC.

    From state q:
      - Bid fill of size z_k (tier tau): q -> q + z_k at rate Lambda_tau^{b,k}(delta^{b,k*}(q))
      - Ask fill of size z_k (tier tau): q -> q - z_k at rate Lambda_tau^{a,k}(delta^{a,k*}(q))

    Transitions outside [-q_max, q_max] are suppressed.

    Parameters
    ----------
    q_grid : np.ndarray, shape (N,)
        Inventory grid.
    z : np.ndarray, shape (K,)
        Size ladder.
    delta_opt : dict
        (tier_name, side, size_idx) -> array of shape (N,) with optimal quotes.
    tier_hamiltonians : list of TierHamiltonians
        All tiers (for accessing lambda/alpha/beta).

    Returns
    -------
    csr_matrix, shape (N, N)
        Generator matrix (rows sum to zero).
    """
    N = len(q_grid)
    K = len(z)
    dq = q_grid[1] - q_grid[0]

    G = lil_matrix((N, N), dtype=float)

    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(K):
                d_arr = delta_opt.get((th.tier_name, s, k))
                if d_arr is None:
                    continue
                lam = th._lambdas[k]
                alpha = th._alphas[k]
                beta = th._betas[k]
                step = int(round(float(z[k]) / dq))

                for i in range(N):
                    if np.isnan(d_arr[i]):
                        continue
                    rate = float(fill_intensity(d_arr[i], lam, alpha, beta))
                    if rate <= 0:
                        continue

                    if s == "bid":
                        j = i + step  # inventory increases
                    else:
                        j = i - step  # inventory decreases

                    if 0 <= j < N:
                        G[i, j] += rate
                        G[i, i] -= rate

    return G.tocsr()


def propagate_forward_law(
    G: csr_matrix,
    q_grid: np.ndarray,
    T: float,
    q0: float = 0.0,
    n_eval: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate the inventory distribution forward using the Kolmogorov forward equation.

        d mu_t / dt = G^T mu_t

    with initial condition mu_0 = delta_{q0}.

    Uses scipy solve_ivp with 'RK45' method.

    Parameters
    ----------
    G : csr_matrix
        Generator matrix, shape (N, N).
    q_grid : np.ndarray, shape (N,)
        Inventory grid.
    T : float
        Horizon (days).
    q0 : float
        Initial inventory state.
    n_eval : int
        Number of evaluation points in [0, T].

    Returns
    -------
    t_eval : np.ndarray, shape (n_eval,)
        Evaluation times.
    mu_t : np.ndarray, shape (N, n_eval)
        Distribution at each evaluation time (columns sum to 1).
    """
    N = len(q_grid)
    dq = q_grid[1] - q_grid[0]
    q_max = q_grid[-1]

    # Initial condition: point mass at q=q0
    idx0 = int(round((q0 - q_grid[0]) / dq))
    idx0 = max(0, min(N - 1, idx0))
    mu0 = np.zeros(N)
    mu0[idx0] = 1.0

    GT = G.T.toarray()  # dense for ODE (manageable for N~201)

    def ode_rhs(t, mu):
        return GT @ mu

    t_eval = np.linspace(0.0, T, n_eval)
    sol = solve_ivp(
        ode_rhs,
        t_span=(0.0, T),
        y0=mu0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-9,
    )

    mu_t = sol.y  # shape (N, n_eval)

    # Enforce positivity and mass normalisation
    mu_t = np.maximum(mu_t, 0.0)
    col_sums = mu_t.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums > 0, col_sums, 1.0)
    mu_t = mu_t / col_sums

    return t_eval, mu_t


def expected_hit_ratio(
    t_eval: np.ndarray,
    mu_t: np.ndarray,
    q_grid: np.ndarray,
    r_q: np.ndarray,
    T: float,
) -> float:
    """
    Compute the expected realised size-weighted hit ratio over [0, T].

        E[(1/T) integral_0^T r(Q_t) dt] = (1/T) integral_0^T sum_q mu_t(q) r(q) dt

    Parameters
    ----------
    t_eval : np.ndarray, shape (n_eval,)
    mu_t : np.ndarray, shape (N, n_eval)
    q_grid : np.ndarray, shape (N,)
    r_q : np.ndarray, shape (N,)
        State-wise hit ratio.
    T : float

    Returns
    -------
    float
        Expected realised hit ratio.
    """
    # integrand: sum_q mu_t(q, t) * r(q)
    integrand = (mu_t.T @ r_q)  # shape (n_eval,)
    return float(np.trapz(integrand, t_eval)) / T


def _objective_components(
    t_eval: np.ndarray,
    mu_t: np.ndarray,
    q_grid: np.ndarray,
    delta_opt: Dict,
    tier_hamiltonians: list,
    z: np.ndarray,
    W_target: float,
    r_star: float,
    kappa: float,
    phi: float,
    Sigma: float,
    eta: float,
    T: float = 1.0,
) -> Dict:
    """Internal helper: compute all objective component arrays and return a dict."""
    N = len(q_grid)

    # Spread capture per state q: sum_{tier, s, k} z_k * Lambda^{s,k}(delta*(q)) * delta^{s,k}*(q)
    spread_q = np.zeros(N)
    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(len(z)):
                d_arr = delta_opt.get((th.tier_name, s, k))
                if d_arr is None:
                    continue
                lam = th._lambdas[k]
                alpha = th._alphas[k]
                beta = th._betas[k]
                for i in range(N):
                    if not np.isnan(d_arr[i]):
                        lam_val = float(fill_intensity(d_arr[i], lam, alpha, beta))
                        spread_q[i] += float(z[k]) * lam_val * d_arr[i]

    # Inventory penalty per state q
    inv_pen_q = (phi / 2.0) * q_grid**2 * Sigma

    # Hit-ratio penalty: compute r_q internally
    r_q = compute_hit_ratio(q_grid, delta_opt, tier_hamiltonians, z, W_target)
    hr_pen_q = (kappa * W_target / 2.0) * (r_q - r_star) ** 2

    # Integrate running costs over [0, T]
    spread_cap = float(np.trapz(mu_t.T @ spread_q, t_eval))
    inv_pen = float(np.trapz(mu_t.T @ inv_pen_q, t_eval))
    hr_pen = float(np.trapz(mu_t.T @ hr_pen_q, t_eval))

    # Terminal inventory penalty (eta term)
    mu_T = mu_t[:, -1]
    terminal_pen = float(eta / 2.0 * Sigma * np.dot(mu_T, q_grid**2))

    total = spread_cap - inv_pen - hr_pen - terminal_pen
    return {
        "spread_capture": spread_cap,
        "inventory_penalty": inv_pen,
        "hit_ratio_penalty": hr_pen,
        "terminal_penalty": terminal_pen,
        "total": total,
    }


def expected_objective(
    t_eval: np.ndarray,
    mu_t: np.ndarray,
    q_grid: np.ndarray,
    delta_opt: Dict,
    tier_hamiltonians: list,
    z: np.ndarray,
    W_target: float,
    r_star: float,
    kappa: float,
    phi: float,
    Sigma: float,
    eta: float,
    T: float = 1.0,
) -> float:
    """
    Compute the expected total desk objective under forward law mu_t.

    r_q (hit-ratio per state) is computed internally; the caller does not
    need to supply it.  Terminal inventory penalty uses `eta`.

    Parameters
    ----------
    t_eval : np.ndarray, shape (n_eval,)
    mu_t : np.ndarray, shape (N, n_eval)
    q_grid : np.ndarray, shape (N,)
    delta_opt : dict  – optimal quote maps from policy solver
    tier_hamiltonians : list of TierHamiltonians
    z : np.ndarray  – size ladder (million notional)
    W_target : float  – targeted-tier total-arrival scale
    r_star : float  – target hit ratio
    kappa : float  – hit-ratio penalty weight
    phi : float  – running inventory penalty
    Sigma : float  – scalar bond variance
    eta : float  – terminal inventory penalty
    T : float  – horizon in days

    Returns
    -------
    float
        Expected total desk objective (scalar).
    """
    return _objective_components(
        t_eval, mu_t, q_grid, delta_opt, tier_hamiltonians,
        z, W_target, r_star, kappa, phi, Sigma, eta, T,
    )["total"]


def expected_objective_decomposed(
    t_eval: np.ndarray,
    mu_t: np.ndarray,
    q_grid: np.ndarray,
    delta_opt: Dict,
    tier_hamiltonians: list,
    z: np.ndarray,
    W_target: float,
    r_star: float,
    kappa: float,
    phi: float,
    Sigma: float,
    eta: float,
    T: float = 1.0,
) -> Dict:
    """
    Same as expected_objective but returns a full decomposition dict.

    Keys: 'spread_capture', 'inventory_penalty', 'hit_ratio_penalty',
          'terminal_penalty', 'total'.
    """
    return _objective_components(
        t_eval, mu_t, q_grid, delta_opt, tier_hamiltonians,
        z, W_target, r_star, kappa, phi, Sigma, eta, T,
    )
