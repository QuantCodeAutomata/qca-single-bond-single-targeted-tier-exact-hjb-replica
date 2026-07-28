"""
Unit tests for the scenario configuration module.
"""

import numpy as np
import pytest
from src.config import (
    TierConfig,
    ScenarioConfig,
    make_baseline_scenario,
    make_two_tier_scenario,
    make_targeted_only_scenario,
)


def test_baseline_q_grid_bounds():
    """q_grid must run from -q_max to q_max with step dq."""
    cfg = make_baseline_scenario()
    q = cfg.q_grid
    assert q[0] == pytest.approx(-100.0)
    assert q[-1] == pytest.approx(100.0)
    assert len(q) == 201


def test_baseline_tiers():
    """Baseline scenario has exactly one targeted tier."""
    cfg = make_baseline_scenario()
    assert len(cfg.tiers) == 1
    assert cfg.tiers[0].targeted is True
    assert cfg.tiers[0].name == "tau_star"


def test_w_target_positive():
    """W_target must be strictly positive."""
    cfg = make_baseline_scenario()
    assert cfg.W_target > 0.0


def test_w_target_matches_manual():
    """W_target = 2 * sum(z_k * lambda_k) for the targeted tier."""
    cfg = make_baseline_scenario()
    expected = 2.0 * float(np.dot(cfg.z, cfg.tiers[0].lambdas))
    assert cfg.W_target == pytest.approx(expected)


def test_two_tier_scenario_structure():
    """Two-tier scenario must have one targeted and one non-targeted tier."""
    cfg = make_two_tier_scenario()
    targeted = [t for t in cfg.tiers if t.targeted]
    background = [t for t in cfg.tiers if not t.targeted]
    assert len(targeted) == 1
    assert len(background) == 1


def test_targeted_only_scenario():
    """Targeted-only scenario has exactly one targeted tier."""
    cfg = make_targeted_only_scenario()
    assert len(cfg.tiers) == 1
    assert cfg.tiers[0].targeted is True


def test_default_parameters():
    """Default scalar params match the methodology spec."""
    cfg = make_baseline_scenario()
    assert cfg.phi == pytest.approx(1.0)
    assert cfg.eta == pytest.approx(0.0)
    assert cfg.T == pytest.approx(1.0)
    assert cfg.Sigma == pytest.approx(1.0)
    assert cfg.r_star == pytest.approx(0.1)
    assert cfg.q_max == pytest.approx(100.0)
    assert cfg.dq == pytest.approx(1.0)


def test_scenario_kappa_override():
    """kappa override must propagate into ScenarioConfig."""
    cfg = make_baseline_scenario(kappa=50.0)
    assert cfg.kappa == pytest.approx(50.0)


def test_size_ladder():
    """Size ladder must match (1, 5, 20)."""
    cfg = make_baseline_scenario()
    np.testing.assert_array_equal(cfg.z, [1.0, 5.0, 20.0])


def test_w_by_tier_keys():
    """W_by_tier dictionary must have keys matching tier names."""
    cfg = make_two_tier_scenario()
    tier_names = {t.name for t in cfg.tiers}
    assert set(cfg.W_by_tier.keys()) == tier_names
