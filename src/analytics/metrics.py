"""
Analytics and metrics for policy evaluation.

Provides helpers to compute state-wise hit ratio and approximation errors
from the policy output dicts produced by the HJB and BEGV solvers.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np

from src.primitives.fill_probability import fill_intensity


def compute_hit_ratio(
    q_grid: np.ndarray,
    delta_opt: Dict,
    tier_hamiltonians: list,
    z: np.ndarray,
    W_target: float,
    targeted_only: bool = True,
) -> np.ndarray:
    """
    State-wise instantaneous size-weighted hit ratio.

        r(q) = W_target^{-1} * sum_{targeted tiers, s, k} z_k * Lambda_{tau}^{s,k}(delta^{s,k*}(q))

    Parameters
    ----------
    q_grid : np.ndarray
    delta_opt : dict  (tier_name, side, size_idx) -> np.ndarray of shape (N,)
    tier_hamiltonians : list of TierHamiltonians
    z : np.ndarray   size ladder
    W_target : float  targeted-tier notional scale
    targeted_only : bool  if True restrict sum to targeted tiers

    Returns
    -------
    np.ndarray, shape (N,)
    """
    N = len(q_grid)
    K = len(z)
    r_arr = np.zeros(N)

    for th in tier_hamiltonians:
        if targeted_only and not th.targeted:
            continue
        for s in th.sides:
            for k in range(K):
                d_arr = delta_opt.get((th.tier_name, s, k))
                if d_arr is None:
                    continue
                lam = th._lambdas[k]
                alpha = th._alphas[k]
                beta = th._betas[k]
                for i in range(N):
                    if not np.isnan(d_arr[i]):
                        r_arr[i] += float(z[k]) * float(
                            fill_intensity(d_arr[i], lam, alpha, beta)
                        )

    r_arr /= W_target
    return r_arr


def compute_approximation_errors(
    q_grid: np.ndarray,
    delta_exact: np.ndarray,
    delta_approx: np.ndarray,
) -> Dict:
    """
    Pointwise and summary absolute deviations between exact and approximate quotes.

    Parameters
    ----------
    q_grid : np.ndarray
    delta_exact : np.ndarray   exact top-of-book ask quote over q_grid
    delta_approx : np.ndarray  approximation top-of-book ask quote over q_grid

    Returns
    -------
    dict with keys 'pointwise', 'max_abs', 'mean_abs'
    """
    pointwise = np.abs(delta_exact - delta_approx)
    return {
        "pointwise": pointwise,
        "max_abs": float(np.nanmax(pointwise)),
        "mean_abs": float(np.nanmean(pointwise)),
    }
