"""
Experiment 4: Multi-Bond Correlated Background Bond.

Two bonds:
  - Bond 1 (targeted bond): receives targeted-tier flow lambda_target=(50,20,5).
  - Bond 2 (background bond): receives background-tier flow lambda_bg=(500,200,50).
  - Covariance Sigma=[[1,rho],[rho,1]] varies with rho.

Uses multi-dimensional BEGV stationary approximation to obtain a 2x2 inventory
curvature matrix A, then evaluates the targeted-bond ask quote surface over
(q1, q2) and the expected targeted-tier hit ratio via Monte Carlo.

Figures:
  - Figure 8: targeted-bond ask quote surface over (q1, q2) at rho=0.8.
  - Figure 9: expected targeted-tier hit ratio vs rho.

Run from /workspace/project:
    python exp/exp4_multibond.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import brentq

from src.primitives.hamiltonian import build_tier_hamiltonians
from src.analytics.metrics import compute_hit_ratio
from src.plotting.figures import (
    plot_surface_2d,
    plot_metric_vs_kappa,
)


# ---------------------------------------------------------------------------
# Multi-bond BEGV helpers
# ---------------------------------------------------------------------------

def build_D_diag(tier_hamiltonians_bond1: list, tier_hamiltonians_bond2: list,
                 z: np.ndarray) -> np.ndarray:
    """
    Construct the 2x2 diagonal matrix D where d_m = sum_{tau,s,k} z_k H''_{m,tau}^{s,k}(0).

    Parameters
    ----------
    tier_hamiltonians_bond1 : list   all TierHamiltonians active on bond 1
    tier_hamiltonians_bond2 : list   all TierHamiltonians active on bond 2
    z : np.ndarray   size ladder

    Returns
    -------
    np.ndarray, shape (2, 2) – diagonal D matrix
    """
    K = len(z)
    d = np.zeros(2)
    for th in tier_hamiltonians_bond1:
        for s in th.sides:
            for k in range(K):
                d[0] += float(z[k]) * float(th.hamiltonians[(s, k)].H_double_prime_fn(0.0))
    for th in tier_hamiltonians_bond2:
        for s in th.sides:
            for k in range(K):
                d[1] += float(z[k]) * float(th.hamiltonians[(s, k)].H_double_prime_fn(0.0))
    return np.diag(d)


def stationary_A_matrix(D: np.ndarray, phi: float, Sigma: np.ndarray) -> np.ndarray:
    """
    Stationary-limit BEGV inventory curvature matrix.

        A = sqrt(phi) * D^{-1/2} * (D^{1/2} Sigma D^{1/2})^{1/2} * D^{-1/2}

    Parameters
    ----------
    D : np.ndarray, shape (M, M)  diagonal positive matrix
    phi : float   running inventory penalty coefficient
    Sigma : np.ndarray, shape (M, M)  covariance matrix

    Returns
    -------
    np.ndarray, shape (M, M)  symmetric positive definite matrix
    """
    D_sqrt = np.diag(np.sqrt(np.diag(D)))
    D_isqrt = np.diag(1.0 / np.sqrt(np.diag(D)))
    inner = D_sqrt @ Sigma @ D_sqrt
    inner_sqrt = np.real(sqrtm(inner))
    return float(phi) ** 0.5 * D_isqrt @ inner_sqrt @ D_isqrt


def solve_xi_scalar_2d(
    p_flat: np.ndarray,
    valid_mask: np.ndarray,
    z_vals: np.ndarray,
    H_prime_fns: list,
    kappa: float,
    W_target: float,
    r_star: float,
) -> float:
    """
    Scalar xi root solve for the targeted tier in the 2D inventory setting.

    xi = kappa * [r* + W_target^{-1} * sum_{s,k} z_k * H'_target^{s,k}(p^{s,k} - xi)]
    """
    K = len(z_vals)

    def residual(xi: float) -> float:
        total = 0.0
        for i, (valid, Hpfn, zk) in enumerate(zip(valid_mask, H_prime_fns, z_vals)):
            if valid:
                total += float(zk) * float(Hpfn(p_flat[i] - xi))
        return xi - kappa * (r_star + total / W_target)

    # Bracket search
    lo, hi = -1e4, 1e4
    try:
        xi = brentq(residual, lo, hi, xtol=1e-10)
    except ValueError:
        # fallback: simple iteration
        xi = 0.0
        for _ in range(300):
            total = sum(
                float(z_vals[i]) * float(H_prime_fns[i](p_flat[i] - xi))
                for i in range(len(p_flat)) if valid_mask[i]
            )
            xi_new = kappa * (r_star + total / W_target)
            if abs(xi_new - xi) < 1e-10:
                break
            xi = 0.5 * xi + 0.5 * xi_new
    return float(xi)


def compute_targeted_ask_surface(
    A: np.ndarray,
    q1_grid: np.ndarray,
    q2_grid: np.ndarray,
    z: np.ndarray,
    th_target: object,
    kappa: float,
    W_target: float,
    r_star: float,
    size_idx: int = 0,
) -> tuple:
    """
    Evaluate the targeted-bond top-of-book ask quote over a 2D inventory mesh.

    Returns
    -------
    ask_surface : np.ndarray, shape (len(q1_grid), len(q2_grid))
    xi_surface  : np.ndarray, same shape
    """
    K = len(z)
    N1, N2 = len(q1_grid), len(q2_grid)
    ask_surface = np.full((N1, N2), np.nan)
    xi_surface = np.full((N1, N2), np.nan)

    # For each inventory state (q1, q2) compute the 2D BEGV finite-difference
    # arguments and solve the scalar xi closure.
    # From the BEGV quadratic u(q) = -(1/2) q^T A q:
    #   Delta^{b,k}u = [u(q) - u(q+z_k e_m)] / z_k = e_m^T A ( q + (z_k/2) e_m)
    #   Delta^{a,k}u = [u(q) - u(q-z_k e_m)] / z_k = e_m^T A (-q + (z_k/2) e_m)
    # These match the 1D begv_finite_diff convention in src/approximations/begv.py.
    e1 = np.array([1.0, 0.0])

    # Gather H' functions for ask side across sizes (targeted tier, bond 1)
    H_prime_fns_ask = [th_target.hamiltonians[("ask", k)].H_prime_fn for k in range(K)]
    H_prime_fns_bid = [th_target.hamiltonians[("bid", k)].H_prime_fn for k in range(K)]

    for i, q1 in enumerate(q1_grid):
        for j, q2 in enumerate(q2_grid):
            q_vec = np.array([q1, q2])

            # p arguments — correct sign per BEGV finite-difference convention:
            #   ask: p^{1,a,k}(q) = e1^T A (-q + (z_k/2) e1)   [q decreases on ask fill]
            #   bid: p^{1,b,k}(q) = e1^T A ( q + (z_k/2) e1)   [q increases on bid fill]
            p_ask = np.array([
                float(e1 @ (A @ (-q_vec + (z[k] / 2.0) * e1)))
                for k in range(K)
            ])
            p_bid = np.array([
                float(e1 @ (A @ (q_vec + (z[k] / 2.0) * e1)))
                for k in range(K)
            ])
            p_all = np.concatenate([p_bid, p_ask])   # bid then ask
            valid_all = np.ones(2 * K, dtype=bool)   # no boundary for policy eval
            Hpfns_all = H_prime_fns_bid + H_prime_fns_ask
            z_all = np.concatenate([z, z])

            xi = solve_xi_scalar_2d(
                p_all, valid_all, z_all, Hpfns_all, kappa, W_target, r_star
            )
            xi_surface[i, j] = xi

            # Optimal ask quote for size_idx
            p_ask_k = p_ask[size_idx] - xi
            ask_quote = float(th_target.hamiltonians[("ask", size_idx)].delta_tilde_fn(p_ask_k))
            ask_surface[i, j] = ask_quote

    return ask_surface, xi_surface


def mc_expected_hit_ratio_2d(
    A: np.ndarray,
    z: np.ndarray,
    th_target: object,
    th_bg: object,
    kappa: float,
    W_target: float,
    r_star: float,
    T: float = 1.0,
    n_paths: int = 100,
    seed: int = 42,
) -> float:
    """
    Monte Carlo estimate of expected targeted-tier hit ratio over [0, T].

    Simulates inventory CTMC via Poisson thinning:
    at current state q, compute all intensities, sample next jump time from
    total intensity, then sample jump type proportionally.

    Parameters
    ----------
    A : np.ndarray (2,2)  BEGV curvature matrix
    z : np.ndarray
    th_target : TierHamiltonians for targeted tier on bond 1
    th_bg : TierHamiltonians for background tier on bond 2
    kappa, W_target, r_star : scalar floats
    T : float  horizon in days
    n_paths : int  number of MC paths
    seed : int

    Returns
    -------
    float  expected targeted-tier hit ratio
    """
    rng = np.random.default_rng(seed)
    K = len(z)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    q_max = 200.0  # boundary for inventory truncation

    total_hr = 0.0

    for _ in range(n_paths):
        q = np.array([0.0, 0.0], dtype=float)
        t = 0.0
        path_hr_integral = 0.0
        path_time = 0.0

        while t < T:
            # --- Compute all intensities at current state ---
            # Targeted tier, bond 1 — BEGV p-argument sign convention:
            #   ask: p = e1^T A (-q + z_k/2 e1)   (inventory falls by z_k on fill)
            #   bid: p = e1^T A ( q + z_k/2 e1)   (inventory rises by z_k on fill)
            p_target_ask = np.array([
                float(e1 @ (A @ (-q + (z[k] / 2.0) * e1)))
                for k in range(K)
            ])
            p_target_bid = np.array([
                float(e1 @ (A @ (q + (z[k] / 2.0) * e1)))
                for k in range(K)
            ])
            p_all_target = np.concatenate([p_target_bid, p_target_ask])
            valid_target = np.ones(2 * K, dtype=bool)
            Hpfns_t = ([th_target.hamiltonians[("bid", k)].H_prime_fn for k in range(K)] +
                       [th_target.hamiltonians[("ask", k)].H_prime_fn for k in range(K)])
            z_t = np.concatenate([z, z])

            xi = solve_xi_scalar_2d(p_all_target, valid_target, z_t, Hpfns_t,
                                     kappa, W_target, r_star)

            # Optimal deltas for targeted tier
            delta_target_ask = np.array([
                float(th_target.hamiltonians[("ask", k)].delta_tilde_fn(p_target_ask[k] - xi))
                for k in range(K)
            ])
            delta_target_bid = np.array([
                float(th_target.hamiltonians[("bid", k)].delta_tilde_fn(p_target_bid[k] - xi))
                for k in range(K)
            ])

            # Background tier, bond 2 (no xi) — same sign convention
            p_bg_ask = np.array([
                float(e2 @ (A @ (-q + (z[k] / 2.0) * e2)))
                for k in range(K)
            ])
            p_bg_bid = np.array([
                float(e2 @ (A @ (q + (z[k] / 2.0) * e2)))
                for k in range(K)
            ])
            delta_bg_ask = np.array([
                float(th_bg.hamiltonians[("ask", k)].delta_tilde_fn(p_bg_ask[k]))
                for k in range(K)
            ])
            delta_bg_bid = np.array([
                float(th_bg.hamiltonians[("bid", k)].delta_tilde_fn(p_bg_bid[k]))
                for k in range(K)
            ])

            # Intensities
            from src.primitives.fill_probability import fill_intensity as fi
            lam_target_ask = np.array([
                float(fi(delta_target_ask[k], th_target._lambdas[k], th_target._alphas[k], th_target._betas[k]))
                for k in range(K)
            ])
            lam_target_bid = np.array([
                float(fi(delta_target_bid[k], th_target._lambdas[k], th_target._alphas[k], th_target._betas[k]))
                for k in range(K)
            ])
            lam_bg_ask = np.array([
                float(fi(delta_bg_ask[k], th_bg._lambdas[k], th_bg._alphas[k], th_bg._betas[k]))
                for k in range(K)
            ])
            lam_bg_bid = np.array([
                float(fi(delta_bg_bid[k], th_bg._lambdas[k], th_bg._alphas[k], th_bg._betas[k]))
                for k in range(K)
            ])

            # Hit ratio at current state
            r_q = float(np.dot(z, lam_target_ask + lam_target_bid)) / W_target

            # Total intensity
            lam_tot = (lam_target_ask.sum() + lam_target_bid.sum() +
                       lam_bg_ask.sum() + lam_bg_bid.sum())

            if lam_tot < 1e-12:
                # No jumps possible: skip to end
                path_hr_integral += r_q * (T - t)
                path_time += (T - t)
                break

            # Sample next jump time
            dt_jump = rng.exponential(1.0 / lam_tot)
            dt_jump = min(dt_jump, T - t)

            path_hr_integral += r_q * dt_jump
            path_time += dt_jump
            t += dt_jump

            if t >= T:
                break

            # Sample jump type
            all_rates = np.concatenate([
                lam_target_bid * z,
                lam_target_ask * z,
                lam_bg_bid * z,
                lam_bg_ask * z,
            ])
            # jump magnitudes (for inventory update, ignore z-weighting for type selection)
            all_rates_unweighted = np.concatenate([
                lam_target_bid, lam_target_ask, lam_bg_bid, lam_bg_ask
            ])
            cumrates = np.cumsum(all_rates_unweighted)
            u = rng.uniform(0.0, cumrates[-1])
            idx = int(np.searchsorted(cumrates, u))
            idx = min(idx, len(cumrates) - 1)

            # Map idx to inventory change
            if idx < K:                    # target bid
                dq = np.array([z[idx], 0.0])
            elif idx < 2 * K:              # target ask
                dq = np.array([-z[idx - K], 0.0])
            elif idx < 3 * K:              # bg bid
                dq = np.array([0.0, z[idx - 2 * K]])
            else:                          # bg ask
                dq = np.array([0.0, -z[idx - 3 * K]])

            q_new = q + dq
            # Truncation
            if np.abs(q_new[0]) <= q_max and np.abs(q_new[1]) <= q_max:
                q = q_new

        total_hr += path_hr_integral / max(path_time, 1e-12)

    return total_hr / n_paths


def main() -> None:
    print("=" * 60)
    print("Experiment 4: Multi-Bond Correlated Background Bond")
    print("=" * 60)

    z = np.array([1.0, 5.0, 20.0])
    alpha = np.array([2.0, 1.5, 1.0])
    beta  = np.array([2.0, 1.5, 1.0])
    lambda_target = np.array([50.0, 20.0, 5.0])
    lambda_bg     = np.array([500.0, 200.0, 50.0])
    W_target = float(2.0 * np.dot(z, lambda_target))
    kappa = 10.0
    r_star = 0.1
    phi   = 1.0

    # ------------------------------------------------------------------
    # Figure 8: pricing surface at rho=0.8
    # ------------------------------------------------------------------
    rho_main = 0.8
    Sigma_main = np.array([[1.0, rho_main], [rho_main, 1.0]])

    print(f"\n--- Pricing surface at rho={rho_main} ---")
    th_target = build_tier_hamiltonians("tau_star", lambda_target, alpha, beta, z, targeted=True)
    th_bg     = build_tier_hamiltonians("tau_0",    lambda_bg,     alpha, beta, z, targeted=False)

    D_mat = build_D_diag([th_target], [th_bg], z)
    print(f"  D = diag({D_mat[0,0]:.4f}, {D_mat[1,1]:.4f})")
    A_mat = stationary_A_matrix(D_mat, phi, Sigma_main)
    print(f"  A =\n{A_mat}")

    # Evaluate surface on coarser grid for speed
    q1_grid = np.arange(-40.0, 41.0, 2.0)
    q2_grid = np.arange(-40.0, 41.0, 2.0)

    print("  Computing targeted ask quote surface...")
    ask_surface, xi_surface = compute_targeted_ask_surface(
        A_mat, q1_grid, q2_grid, z, th_target,
        kappa, W_target, r_star, size_idx=0,
    )

    print("Generating Figure 8: 2D ask quote surface...")
    plot_surface_2d(
        q1_grid, q2_grid, ask_surface,
        xlabel="Bond 2 inventory q2 (million)",
        ylabel="Bond 1 inventory q1 (million)",
        zlabel="Targeted ask quote offset (bp)",
        title=f"Figure 8: Targeted Bond Ask Quote Surface (rho={rho_main}, kappa={kappa})",
        filename="fig8_ask_quote_surface_rho08.png",
    )

    # ------------------------------------------------------------------
    # Figure 9: expected targeted hit ratio vs rho
    # ------------------------------------------------------------------
    rho_grid = np.array([-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    ehr_vs_rho = []

    print("\nComputing Figure 9: Expected hit ratio vs rho (MC per rho)...")
    for i_rho, rho in enumerate(rho_grid):
        Sigma_rho = np.array([[1.0, rho], [rho, 1.0]])
        th_t = build_tier_hamiltonians("tau_star", lambda_target, alpha, beta, z, targeted=True)
        th_b = build_tier_hamiltonians("tau_0",    lambda_bg,     alpha, beta, z, targeted=False)
        D_r = build_D_diag([th_t], [th_b], z)
        A_r = stationary_A_matrix(D_r, phi, Sigma_rho)

        ehr = mc_expected_hit_ratio_2d(
            A_r, z, th_t, th_b,
            kappa, W_target, r_star, T=1.0,
            n_paths=400, seed=42 + i_rho * 17,
        )
        ehr_vs_rho.append(float(ehr))
        print(f"  rho={rho:.2f}: E[hit ratio] = {ehr:.5f}")

    print("Generating Figure 9: Expected hit ratio vs rho...")
    plot_metric_vs_kappa(
        rho_grid.tolist(),
        {"Expected targeted hit ratio": ehr_vs_rho},
        "Expected targeted hit ratio",
        "Figure 9: Expected Targeted Hit Ratio vs Correlation rho",
        "fig9_hit_ratio_vs_rho.png",
        log_x=False,
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    metrics = {
        "experiment": "exp_4",
        "kappa": kappa,
        "rho_main": rho_main,
        "A_matrix_rho08": A_mat.tolist(),
        "D_diagonal": [D_mat[0, 0], D_mat[1, 1]],
        "ask_surface_center_bp": float(ask_surface[len(q1_grid) // 2, len(q2_grid) // 2]),
        "correlation_study": [
            {"rho": float(r), "expected_hit_ratio": float(e)}
            for r, e in zip(rho_grid, ehr_vs_rho)
        ],
    }
    metrics_path = os.path.join(results_dir, "exp4_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")
    print("\n--- Summary: hit ratio vs rho ---")
    for rho, ehr in zip(rho_grid, ehr_vs_rho):
        print(f"  rho={rho:.2f}  E[hit ratio] = {ehr:.5f}")


if __name__ == "__main__":
    main()
