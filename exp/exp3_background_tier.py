"""
Experiment 3: Single-Bond Targeted Tier + Background Tier.

Demonstrates risk-sharing: a non-targeted background tier improves the
targeted tier's expected realised hit ratio by relieving inventory pressure.

Figures:
  - Figure 6: targeted-tier ask quote vs inventory: targeted-only vs with background.
  - Figure 7: expected targeted-tier hit ratio vs background/targeted intensity ratio.

Run from /workspace/project:
    python exp/exp3_background_tier.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.config import TierConfig, ScenarioConfig, make_baseline_scenario
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
)
from src.analytics.metrics import compute_hit_ratio
from src.plotting.figures import (
    plot_quotes_vs_inventory,
    plot_metric_vs_kappa,
)


def make_two_tier_scenario(
    lambda_bg_scale: float = 1.0,
    kappa: float = 100.0,
) -> ScenarioConfig:
    """
    Build a two-tier scenario: targeted tier + background tier.

    Targeted tier: lambda_target = (50, 20, 5) * each side.
    Background tier: lambda_bg_scale * (500, 200, 50) * each side.
    """
    z = np.array([1.0, 5.0, 20.0])
    alpha = np.array([2.0, 1.5, 1.0])
    beta = np.array([2.0, 1.5, 1.0])

    lambda_target = np.array([50.0, 20.0, 5.0])
    lambda_background = lambda_bg_scale * np.array([500.0, 200.0, 50.0])

    tiers = [
        TierConfig(
            name="tau_star",
            lambdas=lambda_target,
            alpha=alpha,
            beta=beta,
            targeted=True,
        ),
        TierConfig(
            name="tau_0",
            lambdas=lambda_background,
            alpha=alpha,
            beta=beta,
            targeted=False,
        ),
    ]

    # q_max=100, dq=1 yields q_grid=[-100,...,100] via the property
    return ScenarioConfig(
        z=z,
        kappa=kappa,
        r_star=0.1,
        phi=1.0,
        eta=0.0,
        T=1.0,
        Sigma=1.0,
        q_max=100.0,
        dq=1.0,
        tiers=tiers,
    )


def main() -> None:
    print("=" * 60)
    print("Experiment 3: Background Tier Risk-Sharing")
    print("=" * 60)

    kappa = 100.0
    q_range = [-50.0, 50.0]
    q_grid_ctmc = np.arange(-50.0, 51.0, 1.0)

    # ------------------------------------------------------------------
    # Scenario 1: targeted tier only (baseline reduced intensities)
    # ------------------------------------------------------------------
    print("\n--- Targeted tier only ---")
    z = np.array([1.0, 5.0, 20.0])
    alpha = np.array([2.0, 1.5, 1.0])
    beta  = np.array([2.0, 1.5, 1.0])
    lambda_target = np.array([50.0, 20.0, 5.0])
    W_target_single = float(2.0 * np.dot(z, lambda_target))

    th_target = build_tier_hamiltonians(
        "tau_star", lambda_target, alpha, beta, z, targeted=True
    )
    tiers_single = [th_target]

    D_single = compute_D_scalar(tiers_single, z)
    A_single = stationary_A(D_single, 1.0, 1.0)
    print(f"  D={D_single:.6f}, A={A_single:.6f}")

    pol_single = compute_approx_A_policy(
        A_single, q_grid_ctmc, z, tiers_single,
        kappa, W_target_single, 0.1,
    )
    ask_single = pol_single["delta"][("tau_star", "ask", 0)]

    # CTMC for single tier
    G_single = build_generator(q_grid_ctmc, z, pol_single["delta"], tiers_single)
    t_eval, mu_single = propagate_forward_law(G_single, q_grid_ctmc, T=1.0, q0=0.0, n_eval=50)
    r_q_single = compute_hit_ratio(q_grid_ctmc, pol_single["delta"], tiers_single, z, W_target_single)
    ehr_single = expected_hit_ratio(t_eval, mu_single, q_grid_ctmc, r_q_single, T=1.0)
    print(f"  Expected targeted hit ratio (no bg): {ehr_single:.5f}")

    # ------------------------------------------------------------------
    # Scenario 2: targeted + background tier (bg scale = 1.0)
    # ------------------------------------------------------------------
    print("\n--- Targeted + background tier (scale=1.0) ---")
    cfg2 = make_two_tier_scenario(lambda_bg_scale=1.0, kappa=kappa)

    th_target2 = build_tier_hamiltonians("tau_star", lambda_target, alpha, beta, z, targeted=True)
    lambda_bg = np.array([500.0, 200.0, 50.0])
    th_bg2 = build_tier_hamiltonians("tau_0", lambda_bg, alpha, beta, z, targeted=False)
    tiers_two = [th_target2, th_bg2]

    D_two = compute_D_scalar(tiers_two, z)
    A_two = stationary_A(D_two, 1.0, 1.0)
    print(f"  D={D_two:.6f}, A={A_two:.6f}")

    pol_two = compute_approx_A_policy(
        A_two, q_grid_ctmc, z, tiers_two,
        kappa, cfg2.W_target, 0.1,
    )
    ask_two = pol_two["delta"][("tau_star", "ask", 0)]

    # CTMC for two tiers
    G_two = build_generator(q_grid_ctmc, z, pol_two["delta"], tiers_two)
    t_eval2, mu_two = propagate_forward_law(G_two, q_grid_ctmc, T=1.0, q0=0.0, n_eval=50)
    r_q_two = compute_hit_ratio(q_grid_ctmc, pol_two["delta"], tiers_two, z, cfg2.W_target)
    ehr_two = expected_hit_ratio(t_eval2, mu_two, q_grid_ctmc, r_q_two, T=1.0)
    print(f"  Expected targeted hit ratio (with bg): {ehr_two:.5f}")

    # ------------------------------------------------------------------
    # Figure 6: Quote comparison
    # ------------------------------------------------------------------
    quote_dict = {
        "Targeted-only": ask_single,
        "Targeted + Background": ask_two,
    }
    print("\nGenerating Figure 6: Quote comparison...")
    plot_quotes_vs_inventory(
        q_grid_ctmc, quote_dict,
        f"Figure 6: Targeted Ask Quote vs Inventory (kappa={kappa})",
        "fig6_quote_comparison_background.png",
        q_range=q_range,
    )

    # ------------------------------------------------------------------
    # Figure 7: Expected targeted hit ratio vs background/targeted ratio
    # ------------------------------------------------------------------
    print("\nComputing Figure 7: Hit ratio vs bg/target intensity ratio...")
    bg_scales = np.logspace(-1, 2, 15)  # 0.1x to 100x
    ehr_vs_scale = []
    for scale in bg_scales:
        lambda_bg_s = scale * np.array([500.0, 200.0, 50.0])
        th_tgt_s = build_tier_hamiltonians("tau_star", lambda_target, alpha, beta, z, targeted=True)
        th_bg_s  = build_tier_hamiltonians("tau_0", lambda_bg_s, alpha, beta, z, targeted=False)
        tiers_s = [th_tgt_s, th_bg_s]
        W_target_s = float(2.0 * np.dot(z, lambda_target))

        D_s = compute_D_scalar(tiers_s, z)
        A_s = stationary_A(D_s, 1.0, 1.0)

        pol_s = compute_approx_A_policy(A_s, q_grid_ctmc, z, tiers_s, kappa, W_target_s, 0.1)
        G_s = build_generator(q_grid_ctmc, z, pol_s["delta"], tiers_s)
        t_s, mu_s = propagate_forward_law(G_s, q_grid_ctmc, T=1.0, q0=0.0, n_eval=30)
        r_s = compute_hit_ratio(q_grid_ctmc, pol_s["delta"], tiers_s, z, W_target_s)
        ehr_s = expected_hit_ratio(t_s, mu_s, q_grid_ctmc, r_s, T=1.0)
        ehr_vs_scale.append(float(ehr_s))
        print(f"  bg_scale={scale:.3f}: expected hit ratio = {ehr_s:.5f}")

    # x-axis: background intensity / targeted intensity (all equal scale factors so ratio = scale)
    ratio_axis = bg_scales.tolist()
    print("\nGenerating Figure 7: Hit ratio vs bg/target ratio...")
    from src.plotting.figures import plot_metric_vs_kappa
    plot_metric_vs_kappa(
        ratio_axis,
        {"Expected targeted hit ratio": ehr_vs_scale},
        "Expected targeted hit ratio",
        "Figure 7: Expected Targeted Hit Ratio vs Background/Targeted Ratio",
        "fig7_hit_ratio_vs_bg_ratio.png",
        log_x=True,
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
    os.makedirs(results_dir, exist_ok=True)

    metrics = {
        "experiment": "exp_3",
        "kappa": kappa,
        "expected_hit_ratio_no_background": ehr_single,
        "expected_hit_ratio_with_background": ehr_two,
        "background_ratio_study": [
            {"bg_scale": float(s), "expected_hit_ratio": float(e)}
            for s, e in zip(bg_scales, ehr_vs_scale)
        ],
    }
    metrics_path = os.path.join(results_dir, "exp3_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")
    print("\n--- Summary ---")
    print(f"  No background: E[hit ratio] = {ehr_single:.5f}")
    print(f"  With background: E[hit ratio] = {ehr_two:.5f}")
    print(f"  Improvement: {(ehr_two - ehr_single)*100:.2f} pp")


if __name__ == "__main__":
    main()
