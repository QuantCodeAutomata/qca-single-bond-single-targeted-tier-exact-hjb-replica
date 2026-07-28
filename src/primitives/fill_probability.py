"""
Fill-probability functions for OTC bond market-making.

Replication assumption (documented):
    We use the logistic-type fill function
        f_k(delta) = 1 / (1 + exp(alpha_k + beta_k * delta))
    where delta is the quote offset in basis points (positive = dealer earns spread),
    alpha_k is a dimensionless intercept (controls baseline fill probability at delta=0),
    and beta_k is the sensitivity in bp^-1 (controls how quickly fills decay with delta).

    This is a strictly decreasing sigmoid, approaching 1 as delta -> -inf and 0 as
    delta -> +inf, consistent with economic intuition that larger spreads reduce fill
    probability. The logistic form was chosen for analytical tractability of derivatives
    and consistent with the qualitative description in the paper.

# Custom — Context7 found no library equivalent (paper fill-probability specification)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def fill_prob(delta: ArrayLike, alpha: float, beta: float) -> np.ndarray:
    """
    Logistic fill-probability function.

        f(delta) = 1 / (1 + exp(alpha + beta * delta))

    Parameters
    ----------
    delta : array-like
        Quote offset in basis points.
    alpha : float
        Intercept parameter (dimensionless). Larger alpha -> lower baseline fill prob.
    beta : float
        Sensitivity parameter (bp^-1). Larger beta -> faster decay with spread.

    Returns
    -------
    np.ndarray
        Fill probability in [0, 1].
    """
    delta = np.asarray(delta, dtype=float)
    # Clip argument to avoid overflow in exp
    u = np.clip(alpha + beta * delta, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(u))


def fill_prob_d1(delta: ArrayLike, alpha: float, beta: float) -> np.ndarray:
    """
    First derivative of fill probability with respect to delta.

        f'(delta) = -beta * f(delta) * (1 - f(delta))

    Parameters
    ----------
    delta : array-like
        Quote offset in basis points.
    alpha, beta : float
        Fill-probability parameters.

    Returns
    -------
    np.ndarray
        df/d(delta), strictly non-positive.
    """
    f = fill_prob(delta, alpha, beta)
    return -beta * f * (1.0 - f)


def fill_prob_d2(delta: ArrayLike, alpha: float, beta: float) -> np.ndarray:
    """
    Second derivative of fill probability.

        f''(delta) = -beta^2 * f(delta) * (1 - f(delta)) * (2*f(delta) - 1)

    Derivation:
        f' = -beta * f * (1-f)
        f'' = d/d(delta)[-beta * f*(1-f)]
            = -beta * [f'*(1-f) + f*(-f')]
            = -beta * f' * (1 - 2*f)
            = -beta * (-beta*f*(1-f)) * (1 - 2*f)
            = beta^2 * f*(1-f) * (1 - 2*f)
            = -beta^2 * f*(1-f) * (2*f - 1)

    Parameters
    ----------
    delta : array-like
        Quote offset in basis points.
    alpha, beta : float
        Fill-probability parameters.

    Returns
    -------
    np.ndarray
        d^2f/d(delta)^2.
    """
    f = fill_prob(delta, alpha, beta)
    return -beta**2 * f * (1.0 - f) * (2.0 * f - 1.0)


def fill_prob_d3(delta: ArrayLike, alpha: float, beta: float) -> np.ndarray:
    """
    Third derivative of fill probability.

        f'''(delta) = beta^3 * f*(1-f) * (6*f^2 - 6*f + 1)

    Derivation:
        f'' = -beta^2 * f*(1-f)*(2*f-1)
        Let g(f) = f*(1-f)*(2*f-1) = 2f^3 - 3f^2 + f
        dg/df = 6f^2 - 6f + 1
        f''' = -beta^2 * dg/df * f' = -beta^2 * (6f^2-6f+1) * (-beta*f*(1-f))
             = beta^3 * f*(1-f) * (6*f^2 - 6*f + 1)

    Parameters
    ----------
    delta : array-like
        Quote offset in basis points.
    alpha, beta : float
        Fill-probability parameters.

    Returns
    -------
    np.ndarray
        d^3f/d(delta)^3.
    """
    f = fill_prob(delta, alpha, beta)
    return beta**3 * f * (1.0 - f) * (6.0 * f**2 - 6.0 * f + 1.0)


def fill_intensity(delta: ArrayLike, lam: float, alpha: float, beta: float) -> np.ndarray:
    """
    Fill intensity: Lambda(delta) = lambda * f(delta).

    Parameters
    ----------
    delta : array-like
        Quote offset in basis points.
    lam : float
        Baseline arrival intensity (day^-1).
    alpha, beta : float
        Fill-probability parameters.

    Returns
    -------
    np.ndarray
        Fill intensity in day^-1.
    """
    return lam * fill_prob(delta, alpha, beta)


def riskless_spread(alpha: float, beta: float) -> float:
    """
    Compute delta_hat_0 = argmax_delta { lambda * f(delta) * delta }
    (the riskless optimal spread, independent of lambda).

    From the first-order condition:
        f'(delta)*delta + f(delta) = 0
        -beta*f*(1-f)*delta + f = 0
        delta = 1 / (beta*(1-f(delta)))

    Solved numerically via Brent's method.

    Parameters
    ----------
    alpha, beta : float
        Fill-probability parameters.

    Returns
    -------
    float
        Riskless spread in basis points.
    """
    from scipy.optimize import brentq

    def foc(d: float) -> float:
        """FOC: f'(d)*d + f(d) = 0."""
        return fill_prob_d1(d, alpha, beta) * d + fill_prob(d, alpha, beta)

    # Bracket: at d=0 the objective Lambda*d=0; optimal d must be positive
    # f(d)*d increases initially then decreases; find zero of FOC in (0, large)
    try:
        return brentq(foc, 0.0, 1000.0, xtol=1e-10)
    except ValueError:
        # If no root in range, fallback to bounded minimization
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(
            lambda d: -fill_prob(d, alpha, beta) * d,
            bounds=(0.0, 1000.0),
            method="bounded",
        )
        return float(res.x)


def linearization_coefficient(delta0: float, alpha: float, beta: float) -> float:
    """
    Linearization coefficient for Approximation D.

        c = 2 - f(delta_hat_0) * f''(delta_hat_0) / (f'(delta_hat_0))^2

    From the paper's linearized quote-map approximation.

    Parameters
    ----------
    delta0 : float
        Riskless spread delta_hat_0 (bp).
    alpha, beta : float
        Fill-probability parameters.

    Returns
    -------
    float
        Linearization coefficient c (dimensionless).
    """
    f0 = fill_prob(delta0, alpha, beta)
    fd1 = fill_prob_d1(delta0, alpha, beta)
    fd2 = fill_prob_d2(delta0, alpha, beta)
    return 2.0 - (f0 * fd2) / (fd1**2)
