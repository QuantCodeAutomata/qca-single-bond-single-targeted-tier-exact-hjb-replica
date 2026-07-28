"""
Experiment 2: Sensitivity to Hit-Ratio Penalty kappa and Forward-Law Performance.

Studies how varying kappa in {0.1, 1, 10, 100, 1000} changes:
  - State-dependent optimal quotes (Figure 4 equivalent).
  - Expected realised hit ratio vs kappa (Figure 5 equivalent).
  - Expected desk objective vs kappa.

Uses stationary-A BEGV approximation with exact xi(q) closure plus
CTMC forward-law propagation on a reduced q_grid [-50, 50].

Run from /workspace/project:
    python exp/exp2_kappa_sensitivity.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.config import make_baseline_scenario
from src.primitives.hamiltonian import build_tier_hamiltonians
from src.approximations.begv import (
    compute_D_scalar,
    stationary_A,
    compute_approx_A_policy,
)
from src.ctmc.forward_law import (
    build_generator,
    propagate_forward_law,
    expected_hit_ratio,
    expected_objective_decomposed,
)
from src.analytics.metrics import compute_hit_ratio
from src.plotting.figures import (
    plot_quotes_vs_inventory,
    plot_metric_vs_kappa,
)


def main() -> None:
    print("=" * 60)
    print("Experiment 2: Kappa Sensitivity + Forward-Law Performance")
    print("=" * 60)

    kappa_list = [0.1, 1.0, 10.0, 100.0, 1000.0]

    # Use a reduced grid for CTMC to keep generator size manageable
    q_grid_ctmc = np.arange(-50.0, 51.0, 1.0)  # 101 states
    N_ctmc = len(q_grid_ctmc)
    print(f"CTMC q_grid: {q_grid_ctmc[0]:.0f} to {q_grid_ctmc[-1]:.0f}, N={N_ctmc}")

    # Result containers
    quote_curves = {}           # kappa -> ask quote array
    exp_hit_ratios = []
    exp_objectives = []
    spread_caps = []
    inv_pens = []
    hr_pens = []
    results_list = []

    for kappa in kappa_list:
        print(f"\n--- kappa = {kappa} ---")
        cfg = make_baseline_scenario(kappa=kappa)
        tier_cfg = cfg.tiers[0]

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

        # Stationary BEGV
        D = compute_D_scalar(tier_hamiltonians, cfg.z)
        A = stationary_A(D, cfg.phi, cfg.Sigma)

        # Policy on reduced CTMC grid
        pol = compute_approx_A_policy(
            A, q_grid_ctmc, cfg.z, tier_hamiltonians,
            kappa, W_target, cfg.r_star,
        )
        delta_opt = pol["delta"]

        # Store ask quote curve
        quote_curves[f"kappa={kappa}"] = pol["delta"][("tau_star", "ask", 0)].copy()

        # CTMC generator and forward law
        print(f"  Building CTMC generator (N={N_ctmc})...")
        G = build_generator(q_grid_ctmc, cfg.z, delta_opt, tier_hamiltonians)

        print(f"  Propagating forward law over T={cfg.T} day...")
        t_eval, mu_t = propagate_forward_law(G, q_grid_ctmc, T=cfg.T, q0=0.0, n_eval=50)

        # Hit ratio curve
        r_q = compute_hit_ratio(q_grid_ctmc, delta_opt, tier_hamiltonians, cfg.z, W_target)

        # Expected hit ratio
        ehr = expected_hit_ratio(t_eval, mu_t, q_grid_ctmc, r_q, T=cfg.T)
        exp_hit_ratios.append(ehr)
        print(f"  Expected hit ratio = {ehr:.5f}  (target r* = {cfg.r_star})")

        # Expected objective decomposition
        obj = expected_objective_decomposed(
            t_eval, mu_t, q_grid_ctmc, delta_opt,
            tier_hamiltonians, cfg.z,
            W_target, cfg.r_star, kappa,
            cfg.phi, cfg.Sigma, cfg.eta, T=cfg.T,
        )
        exp_objectives.append(obj["total"])
        spread_caps.append(obj["spread_capture"])
        inv_pens.append(obj["inventory_penalty"])
        hr_pens.append(obj["hit_ratio_penalty"])
        print(f"  Expected objective = {obj['total']:.4f}  "
              f"(spread={obj['spread_capture']:.4f}, "
              f"inv_pen={obj['inventory_penalty']:.4f}, "
              f"hr_pen={obj['hit_ratio_penalty']:.4f})")

        results_list.append({
            "kappa": kappa,
            "expected_hit_ratio": ehr,
            "expected_objective": obj["total"],
            "spread_capture": obj["spread_capture"],
            "inventory_penalty": obj["inventory_penalty"],
            "hit_ratio_penalty": obj["hit_ratio_penalty"],
            "A_stationary": float(A),
            "D": float(D),
        })

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating Figure 4: Quote curves vs inventory for different kappa...")
    plot_quotes_vs_inventory(
        q_grid_ctmc, quote_curves,
        "Figure 4: Ask Quote vs Inventory for Different kappa",
        "fig4_quotes_vs_inventory_kappa.png",
        q_range=[-50.0, 50.0],
    )

    print("Generating Figure 5a: Expected hit ratio vs kappa...")
    plot_metric_vs_kappa(
        kappa_list,
        {"Expected targeted hit ratio": exp_hit_ratios},
        "Expected targeted hit ratio",
        "Figure 5a: Expected Hit Ratio vs kappa",
        "fig5a_expected_hit_ratio_vs_kappa.png",
        log_x=True,
    )

    print("Generating Figure 5b: Expected objective vs kappa...")
    plot_metric_vs_kappa(
        kappa_list,
        {"Expected desk objective": exp_objectives},
        "Expected desk objective",
        "Figure 5b: Expected Objective vs kappa",
        "fig5b_expected_objective_vs_kappa.png",
        log_x=True,
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "exp2_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results_list, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")
    print("\n--- Summary ---")
    print(f"{'kappa':>10} {'exp hit ratio':>16} {'exp objective':>15}")
    print("-" * 45)
    for rec in results_list:
        print(f"{rec['kappa']:>10.1f} {rec['expected_hit_ratio']:>16.5f} "
              f"{rec['expected_objective']:>15.4f}")


if __name__ == "__main__":
    main()
