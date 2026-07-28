"""
Static Hamiltonian precomputation for OTC bond market-making.

For each (side, size) pair, precomputes:
    H(p)       = sup_{delta} Lambda(delta) * (delta - p)
    delta_tilde(p) = argmax_delta Lambda(delta) * (delta - p)
    H'(p)      = -Lambda(delta_tilde(p))         [envelope property]
    H''(p)     = numerical derivative of H'(p)
    H'''(p)    = numerical derivative of H''(p)

# Custom — Context7 found no library equivalent (paper Hamiltonian computation)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

from src.primitives.fill_probability import (
    fill_intensity,
    fill_prob,
    riskless_spread,
    linearization_coefficient,
    fill_prob_d1,
    fill_prob_d2,
)


@dataclass
class HamiltonianInterpolators:
    """
    Precomputed Hamiltonian objects for a single (bond, tier, side, size) tuple.

    All interpolators are defined over a p-grid (in basis points).
    """
    p_grid: np.ndarray         # Conjugate-momentum grid (bp)
    H: np.ndarray              # H(p) values
    H_prime: np.ndarray        # H'(p) = -Lambda(delta_tilde(p))
    H_double_prime: np.ndarray # H''(p)
    H_triple_prime: np.ndarray # H'''(p)
    delta_tilde: np.ndarray    # Optimal quote offset (bp)

    # Callable interpolators (built post-init)
    H_fn: Callable = None
    H_prime_fn: Callable = None
    H_double_prime_fn: Callable = None
    H_triple_prime_fn: Callable = None
    delta_tilde_fn: Callable = None

    def build_interpolators(self) -> None:
        """Build cubic-spline interpolators for all quantities.

        All returned callables clamp their input to [p_min, p_max] before
        evaluating the spline.  This prevents wild cubic extrapolation when
        the HJB finite differences grow beyond the precomputed grid during
        backward Euler integration.
        """
        p_min, p_max = float(self.p_grid[0]), float(self.p_grid[-1])

        def _make_clamped(spline: "CubicSpline") -> "Callable":
            def _fn(p: "np.ndarray | float") -> "np.ndarray | float":
                p_clamp = np.clip(p, p_min, p_max)
                return spline(p_clamp)
            return _fn

        H_sp            = CubicSpline(self.p_grid, self.H,              extrapolate=True)
        H_p_sp          = CubicSpline(self.p_grid, self.H_prime,        extrapolate=True)
        H_pp_sp         = CubicSpline(self.p_grid, self.H_double_prime, extrapolate=True)
        H_ppp_sp        = CubicSpline(self.p_grid, self.H_triple_prime, extrapolate=True)
        delta_tilde_sp  = CubicSpline(self.p_grid, self.delta_tilde,    extrapolate=True)

        self.H_fn              = _make_clamped(H_sp)
        self.H_prime_fn        = _make_clamped(H_p_sp)
        self.H_double_prime_fn = _make_clamped(H_pp_sp)
        self.H_triple_prime_fn = _make_clamped(H_ppp_sp)
        self.delta_tilde_fn    = _make_clamped(delta_tilde_sp)


def compute_hamiltonian(
    lam: float,
    alpha: float,
    beta: float,
    p_min: float = -200.0,
    p_max: float = 200.0,
    n_p: int = 2001,
    delta_min: float = -20.0,
    delta_max: float = 200.0,
    n_delta: int = 5001,
) -> HamiltonianInterpolators:
    """
    Precompute the static Hamiltonian and its derivatives for one (tier, side, size).

    The Hamiltonian is:
        H(p) = sup_{delta in [delta_min, delta_max]} Lambda(delta) * (delta - p)
    where Lambda(delta) = lam * f(delta).

    The supremum is computed numerically by evaluating over a dense delta grid
    and finding the maximizer, then refining with bounded scalar minimisation.

    Envelope property (validated): H'(p) = -Lambda(delta_tilde(p)).

    Parameters
    ----------
    lam : float
        Baseline arrival intensity for this size (day^-1).
    alpha, beta : float
        Fill-probability parameters.
    p_min, p_max : float
        Range for the conjugate-momentum p-grid (bp).
    n_p : int
        Number of points in the p-grid.
    delta_min, delta_max : float
        Bounds for the delta optimisation (bp).
    n_delta : int
        Number of points in the coarse delta grid.

    Returns
    -------
    HamiltonianInterpolators
        Precomputed Hamiltonian objects with interpolators built.
    """
    p_grid = np.linspace(p_min, p_max, n_p)
    delta_coarse = np.linspace(delta_min, delta_max, n_delta)
    Lam_coarse = fill_intensity(delta_coarse, lam, alpha, beta)

    H_vals = np.empty(n_p)
    delta_tilde_vals = np.empty(n_p)

    for i, p in enumerate(p_grid):
        obj_coarse = Lam_coarse * (delta_coarse - p)
        idx0 = int(np.argmax(obj_coarse))
        d_lo = delta_coarse[max(0, idx0 - 1)]
        d_hi = delta_coarse[min(n_delta - 1, idx0 + 1)]

        # Refine with bounded scalar optimisation
        res = minimize_scalar(
            lambda d: -fill_intensity(d, lam, alpha, beta) * (d - p),
            bounds=(d_lo, d_hi),
            method="bounded",
            options={"xatol": 1e-10},
        )
        delta_tilde_vals[i] = res.x
        H_vals[i] = -res.fun

    # H(p) = sup_delta Lambda(delta)*(delta-p) >= 0 by construction; clip numerical noise
    H_vals = np.maximum(H_vals, 0.0)

    # H'(p) = -Lambda(delta_tilde(p))  [envelope theorem]
    H_prime_vals = -fill_intensity(delta_tilde_vals, lam, alpha, beta)

    # H''(p) and H'''(p) via finite differences on H'
    dp = p_grid[1] - p_grid[0]
    H_double_prime_vals = np.gradient(H_prime_vals, dp)
    H_triple_prime_vals = np.gradient(H_double_prime_vals, dp)

    interp = HamiltonianInterpolators(
        p_grid=p_grid,
        H=H_vals,
        H_prime=H_prime_vals,
        H_double_prime=H_double_prime_vals,
        H_triple_prime=H_triple_prime_vals,
        delta_tilde=delta_tilde_vals,
    )
    interp.build_interpolators()
    return interp


@dataclass
class TierHamiltonians:
    """
    Precomputed Hamiltonians for all (side, size) pairs of one tier.

    Sides: 'bid' and 'ask' (side symmetry assumed, so same fill function).
    Sizes: indexed 0..K-1 matching config.z.
    """
    tier_name: str
    sides: list              # ['bid', 'ask']
    sizes: list              # [z_0, z_1, ..., z_{K-1}]
    hamiltonians: dict       # (side, size_idx) -> HamiltonianInterpolators

    # Tier parameters (stored for hit-ratio calculations)
    _lambdas: np.ndarray = None
    _alphas: np.ndarray = None
    _betas: np.ndarray = None
    targeted: bool = False

    # Scalar statistics at p=0
    H0: dict = None          # H(0) by (side, size_idx)
    H_prime_0: dict = None   # H'(0)  = -Lambda(delta_hat_0)
    H_double_prime_0: dict = None  # H''(0)
    H_triple_prime_0: dict = None  # H'''(0)
    delta_hat_0: dict = None # Riskless spread by (side, size_idx)
    c_coeff: dict = None     # Linearisation coefficient c

    def __post_init__(self):
        self.H0 = {}
        self.H_prime_0 = {}
        self.H_double_prime_0 = {}
        self.H_triple_prime_0 = {}
        self.delta_hat_0 = {}
        self.c_coeff = {}
        for key, h in self.hamiltonians.items():
            self.H0[key] = float(h.H_fn(0.0))
            self.H_prime_0[key] = float(h.H_prime_fn(0.0))
            self.H_double_prime_0[key] = float(h.H_double_prime_fn(0.0))
            self.H_triple_prime_0[key] = float(h.H_triple_prime_fn(0.0))
            # delta_hat_0 = delta_tilde at p=0
            self.delta_hat_0[key] = float(h.delta_tilde_fn(0.0))


def build_tier_hamiltonians(
    tier_name: str,
    lambdas: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    z: np.ndarray,
    targeted: bool = False,
    p_min: float = -80.0,
    p_max: float = 80.0,
    n_p: int = 3201,
) -> TierHamiltonians:
    """
    Build Hamiltonian interpolators for all (side, size) pairs of one client tier.

    Side symmetry is assumed: bid and ask have the same fill function,
    so the same Hamiltonian object can be shared.

    Parameters
    ----------
    tier_name : str
        Descriptive name for the tier.
    lambdas : np.ndarray, shape (K,)
        Arrival intensities per size.
    alphas, betas : np.ndarray, shape (K,)
        Fill-probability parameters per size.
    z : np.ndarray, shape (K,)
        Size ladder (million notional).
    p_min, p_max : float
        Conjugate-momentum grid range (bp).
    n_p : int
        Number of p-grid points.

    Returns
    -------
    TierHamiltonians
        Fully precomputed Hamiltonian collection.
    """
    K = len(lambdas)
    assert len(alphas) == K and len(betas) == K and len(z) == K

    sides = ["bid", "ask"]
    hamiltonians = {}
    for s in sides:
        for k in range(K):
            h = compute_hamiltonian(
                lam=float(lambdas[k]),
                alpha=float(alphas[k]),
                beta=float(betas[k]),
                p_min=p_min,
                p_max=p_max,
                n_p=n_p,
            )
            hamiltonians[(s, k)] = h

    th = TierHamiltonians(
        tier_name=tier_name,
        sides=sides,
        sizes=list(z),
        hamiltonians=hamiltonians,
        _lambdas=np.array(lambdas, dtype=float),
        _alphas=np.array(alphas, dtype=float),
        _betas=np.array(betas, dtype=float),
        targeted=targeted,
    )

    # Compute linearisation coefficients c^{s,k}
    th.c_coeff = {}
    for s in sides:
        for k in range(K):
            d0 = th.delta_hat_0[(s, k)]
            th.c_coeff[(s, k)] = linearization_coefficient(
                d0, float(alphas[k]), float(betas[k])
            )

    return th


def compute_D(tier_hamiltonians: list, z: np.ndarray) -> float:
    """
    Compute the scalar D coefficient for the 1D BEGV Riccati equation.

        D = sum_{tau, s, k} z_k * H''_{tau}^{s,k}(0)

    Parameters
    ----------
    tier_hamiltonians : list of TierHamiltonians
        All tiers in the model.
    z : np.ndarray
        Size ladder (million notional).

    Returns
    -------
    float
        Scalar D value.
    """
    D = 0.0
    for th in tier_hamiltonians:
        for s in th.sides:
            for k in range(len(z)):
                D += float(z[k]) * th.H_double_prime_0[(s, k)]
    return D


def compute_W(tier_hamiltonians: list, z: np.ndarray, targeted_only: bool = False) -> float:
    """
    Compute total notional arrival scale W.

        W = sum_{tau, s, k} z_k * lambda_k^tau * f_k^tau(delta=0)... 

    Actually W = sum_{tau, s, k} z_k * lambda_k^tau (unconditional):
    This equals 2 * sum_k z_k * lambda_k summed over all tiers.

    But for the hit-ratio, we use W based on arrival rates (not fill rates).
    From the paper: W = sum_{s in {b,a}, k} z_k * lambda^k.

    Parameters
    ----------
    tier_hamiltonians : list of TierHamiltonians
    z : np.ndarray
    targeted_only : bool
        If True, sum only over targeted tiers.

    Returns
    -------
    float
    """
    raise NotImplementedError("Use ScenarioConfig.W_by_tier or W_target instead.")
