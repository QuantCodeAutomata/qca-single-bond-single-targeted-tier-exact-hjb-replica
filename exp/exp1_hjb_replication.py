"""
Experiment 1: Single-Bond Single-Targeted-Tier Exact HJB Replication.

Reproduces the paper's core 1D numerical experiment:
  - Solves the exact finite-horizon HJB (backward explicit Euler).
  - Computes the four BEGV approximation hierarchy levels (A, B, C, D).
  - Generates Figures 1, 2, 3 equivalents: quote vs inventory, hit ratio
    vs inventory, dual variable xi vs inventory.

Run from /workspace/project:
    python exp/exp1_hjb_replication.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.config import make_baseline_scenario
from src.primitives.hamiltonian import build_tier_hamiltonians
from src.hjb.exact_1d import solve_hjb_exact_1d
from src.approximations.begv import (
    compute_D_scalar,
    stationary_A,
    compute_approx_A_policy,
    compute_approx_B_policy,
    compute_approx_C_policy,
    compute_approx_D_policy,
)
from src.analytics.metrics import compute_hit_ratio, compute_approximation_errors
from src.plotting.figures import (
    plot_quotes_vs_inventory,
    plot_hit_ratio_vs_inventory,
    plot_xi_vs_inventory,
)


def main() -> None:
    print("=" * 60)
    print("Experiment 1: Exact HJB Replication + Approximation Hierarchy")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Build scenario (kappa=10, single targeted tier)
    # ------------------------------------------------------------------
    cfg = make_baseline_scenario(kappa=10.0)
    tier_cfg = cfg.tiers[0]
    print(f"Scenario: {tier_cfg.name}, kappa={cfg.kappa}, r*={cfg.r_star}")
    print(f"q_grid: {cfg.q_grid[0]:.0f} to {cfg.q_grid[-1]:.0f}, N={len(cfg.q_grid)}")

    th = build_tier_hamiltonians(
        tier_cfg.name,
        tier_cfg.lambdas,
        tier_cfg.alpha,
        tier_cfg.beta,
        cfg.z,
        targeted=True,
    )
    tier_hamiltonians = [th]
    W_target = cfg.W_target
    print(f"W_target = {W_target:.1f} day^-1 million")

    # ------------------------------------------------------------------
    # 2. Exact HJB solve
    # ------------------------------------------------------------------
    print("\nSolving exact HJB (n_time_steps=2000, may take ~30s)...")
    _hjb = solve_hjb_exact_1d(cfg, tier_hamiltonians, n_time_steps=2000)
    u0 = _hjb["u"]
    xi_exact = _hjb["xi"]
    q_grid = _hjb["q_grid"]
    delta_opt_exact = _hjb["delta"]
    print(f"  Exact HJB done. u0 range: [{u0.min():.3f}, {u0.max():.3f}]")

    # Extract top-of-book ask quote (z=1, idx=0)
    ask_exact = delta_opt_exact[("tau_star", "ask", 0)]

    # Hit ratio for exact HJB
    hr_exact = compute_hit_ratio(q_grid, delta_opt_exact, tier_hamiltonians, cfg.z, W_target)

    # ------------------------------------------------------------------
    # 3. BEGV approximations (stationary A)
    # ------------------------------------------------------------------
    D = compute_D_scalar(tier_hamiltonians, cfg.z)
    A = stationary_A(D, cfg.phi, cfg.Sigma)
    print(f"\nBEGV: D={D:.6f}, A_stationary={A:.6f}")

    # Use a smaller grid for approximations ([-100, 100] same as exact)
    q_grid_approx = q_grid

    print("Computing Approx A (BEGV + exact xi)...")
    pol_A = compute_approx_A_policy(A, q_grid_approx, cfg.z, tier_hamiltonians,
                                     cfg.kappa, W_target, cfg.r_star)
    print("Computing Approx B (BEGV + quadratic xi)...")
    pol_B = compute_approx_B_policy(A, q_grid_approx, cfg.z, tier_hamiltonians,
                                     cfg.kappa, W_target, cfg.r_star)
    print("Computing Approx C (BEGV + constant xi)...")
    pol_C = compute_approx_C_policy(A, q_grid_approx, cfg.z, tier_hamiltonians,
                                     cfg.kappa, W_target, cfg.r_star)
    print("Computing Approx D (linearised quote map)...")
    pol_D = compute_approx_D_policy(A, q_grid_approx, cfg.z, tier_hamiltonians,
                                     cfg.kappa, W_target, cfg.r_star)

    # ------------------------------------------------------------------
    # 4. Build plot dictionaries
    # ------------------------------------------------------------------
    quote_dict = {
        "Exact HJB": ask_exact,
        "Approx A (BEGV+exact xi)": pol_A["delta"][("tau_star", "ask", 0)],
        "Approx B (BEGV+quad xi)": pol_B["delta"][("tau_star", "ask", 0)],
        "Approx C (BEGV+const xi)": pol_C["delta"][("tau_star", "ask", 0)],
        "Approx D (linear map)": pol_D["delta"][("tau_star", "ask", 0)],
    }

    hr_dict = {
        "Exact HJB": hr_exact,
        "Approx A (BEGV+exact xi)": pol_A["hit_ratio"],
        "Approx B (BEGV+quad xi)": pol_B["hit_ratio"],
        "Approx C (BEGV+const xi)": pol_C["hit_ratio"],
        "Approx D (linear map)": pol_D["hit_ratio"],
    }

    # xi arrays: Exact HJB uses xi_exact (from the last backward step);
    # approximations store xi directly.
    xi_dict = {
        "Exact HJB": xi_exact,
        "Approx A (BEGV+exact xi)": pol_A["xi"],
        "Approx B (BEGV+quad xi)": pol_B["xi"],
        "Approx C (BEGV+const xi)": pol_C["xi"],
        "Approx D (linear map)": pol_D["xi"],
    }

    # ------------------------------------------------------------------
    # 5. Generate figures
    # ------------------------------------------------------------------
    q_range = [-50.0, 50.0]

    print("\nGenerating Figure 1: Quote vs Inventory...")
    plot_quotes_vs_inventory(
        q_grid, quote_dict,
        "Figure 1: Ask Quote vs Inventory (kappa=10)",
        "fig1_quotes_vs_inventory.png",
        q_range=q_range,
    )

    print("Generating Figure 2: Hit Ratio vs Inventory...")
    plot_hit_ratio_vs_inventory(
        q_grid, hr_dict, cfg.r_star,
        "Figure 2: Instantaneous Hit Ratio vs Inventory (kappa=10)",
        "fig2_hit_ratio_vs_inventory.png",
        q_range=q_range,
    )

    print("Generating Figure 3: Xi vs Inventory...")
    plot_xi_vs_inventory(
        q_grid, xi_dict,
        "Figure 3: Dual Variable xi vs Inventory (kappa=10)",
        "fig3_xi_vs_inventory.png",
        q_range=q_range,
    )

    # ------------------------------------------------------------------
    # 6. Compute approximation errors and diagnostics
    # ------------------------------------------------------------------
    approx_labels = {
        "approx_A": pol_A["delta"][("tau_star", "ask", 0)],
        "approx_B": pol_B["delta"][("tau_star", "ask", 0)],
        "approx_C": pol_C["delta"][("tau_star", "ask", 0)],
        "approx_D": pol_D["delta"][("tau_star", "ask", 0)],
    }
    errors = {}
    for name, approx_ask in approx_labels.items():
        err = compute_approximation_errors(q_grid, ask_exact, approx_ask)
        errors[name] = {"max_abs_bp": err["max_abs"], "mean_abs_bp": err["mean_abs"]}
        print(f"  {name}: max_abs={err['max_abs']:.4f} bp, mean_abs={err['mean_abs']:.4f} bp")

    # Symmetry check: xi_exact should be approximately even in q
    mid = len(q_grid) // 2
    xi_exact_half_pos = xi_exact[mid:]
    xi_exact_half_neg = xi_exact[:mid + 1][::-1]
    n_sym = min(len(xi_exact_half_pos), len(xi_exact_half_neg))
    xi_asymmetry = float(np.mean(np.abs(
        xi_exact_half_pos[:n_sym] - xi_exact_half_neg[:n_sym]
    )))

    # ------------------------------------------------------------------
    # 7. Save metrics JSON
    # ------------------------------------------------------------------
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    metrics = {
        "experiment": "exp_1",
        "kappa": cfg.kappa,
        "r_star": cfg.r_star,
        "W_target": W_target,
        "D": D,
        "A_stationary": A,
        "xi_asymmetry_mean_abs": xi_asymmetry,
        "approximation_errors_bp": errors,
        "exact_hjb_ask_at_q0": float(ask_exact[len(q_grid) // 2]),
        "exact_hjb_hr_at_q0": float(hr_exact[len(q_grid) // 2]),
        "exact_hjb_xi_at_q0": float(xi_exact[len(q_grid) // 2]),
    }

    metrics_path = os.path.join(results_dir, "exp1_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")
    print("\n--- Summary ---")
    print(f"  Exact HJB xi asymmetry (mean |xi(q)-xi(-q)|): {xi_asymmetry:.6f}")
    for name, err in errors.items():
        print(f"  {name}: max error = {err['max_abs_bp']:.4f} bp")
    print("\nApproximation quality ranking (lower error = better):")
    ranked = sorted(errors.items(), key=lambda x: x[1]["max_abs_bp"])
    for name, err in ranked:
        print(f"  {name}: {err['max_abs_bp']:.4f} bp")


if __name__ == "__main__":
    main()
