"""
Scenario configuration for OTC bond market-making experiments.

All parameter values follow the methodology description directly.
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class TierConfig:
    """Configuration for a single client tier."""
    name: str
    lambdas: np.ndarray        # Arrival intensities by size [day^-1]
    alpha: np.ndarray          # Logistic sigmoid offset params (dimensionless)
    beta: np.ndarray           # Logistic sigmoid slope params [bp^-1]
    targeted: bool = False     # Whether this tier is subject to hit-ratio penalty


@dataclass
class ScenarioConfig:
    """
    Full scenario configuration for a market-making experiment.

    Units:
        delta   : basis points (bp)
        z       : million notional
        lambda  : day^-1
        T       : days
        q_max   : million notional
        Sigma   : dimensionless (normalised covariance)
    """
    # Size ladder (million notional)
    z: np.ndarray = field(default_factory=lambda: np.array([1.0, 5.0, 20.0]))

    # Model parameters
    phi: float = 1.0           # Running inventory penalty coefficient
    eta: float = 0.0           # Terminal inventory penalty coefficient
    T: float = 1.0             # Horizon (days)
    Sigma: float = 1.0         # Single-bond variance normalisation

    # Target hit-ratio parameters
    r_star: float = 0.1        # Target size-weighted hit ratio
    kappa: float = 10.0        # Hit-ratio penalty weight

    # Grid parameters
    q_max: float = 100.0       # Inventory truncation (million)
    dq: float = 1.0            # Inventory grid step (million)

    # Client tiers (set after construction)
    tiers: List[TierConfig] = field(default_factory=list)

    def __post_init__(self):
        self.z = np.asarray(self.z, dtype=float)

    @property
    def q_grid(self) -> np.ndarray:
        """Inventory grid from -q_max to q_max inclusive."""
        return np.arange(-self.q_max, self.q_max + self.dq, self.dq)

    @property
    def n_sizes(self) -> int:
        return len(self.z)

    @property
    def targeted_tiers(self) -> List[TierConfig]:
        return [t for t in self.tiers if t.targeted]

    @property
    def W_by_tier(self) -> dict:
        """
        Total notional arrival scale per tier:
        W_tau = sum_{s in {b,a}, k} z_k * lambda_tau^k
        """
        W = {}
        for tier in self.tiers:
            W[tier.name] = 2.0 * float(np.dot(self.z, tier.lambdas))
        return W

    @property
    def W_target(self) -> float:
        """Notional arrival scale for targeted tiers only."""
        total = 0.0
        for tier in self.targeted_tiers:
            total += 2.0 * float(np.dot(self.z, tier.lambdas))
        return total


# ---------------------------------------------------------------------------
# Canonical single-bond single-tier (baseline) scenario
# ---------------------------------------------------------------------------

def make_baseline_scenario(kappa: float = 10.0) -> ScenarioConfig:
    """
    Baseline single-bond single-targeted-tier scenario.

    Parameters from the methodology:
      z = (1,5,20) million
      lambda = (500,200,50) day^-1  (one targeted tier, both sides)
      alpha = (2,1.5,1), beta = (2,1.5,1) bp^-1
      r* = 0.1, phi = 1, eta = 0, T = 1 day, q_max = 100 million, dq = 1 million
    """
    cfg = ScenarioConfig(kappa=kappa)
    targeted_tier = TierConfig(
        name="tau_star",
        lambdas=np.array([500.0, 200.0, 50.0]),
        alpha=np.array([2.0, 1.5, 1.0]),
        beta=np.array([2.0, 1.5, 1.0]),
        targeted=True,
    )
    cfg.tiers = [targeted_tier]
    return cfg


def make_two_tier_scenario(kappa: float = 100.0) -> ScenarioConfig:
    """
    Two-tier single-bond scenario for exp_3.

    Targeted tier has intensities reduced by factor 10 from baseline.
    Background tier has baseline intensities.
    """
    cfg = ScenarioConfig(kappa=kappa)
    targeted_tier = TierConfig(
        name="tau_star",
        lambdas=np.array([50.0, 20.0, 5.0]),
        alpha=np.array([2.0, 1.5, 1.0]),
        beta=np.array([2.0, 1.5, 1.0]),
        targeted=True,
    )
    background_tier = TierConfig(
        name="tau_0",
        lambdas=np.array([500.0, 200.0, 50.0]),
        alpha=np.array([2.0, 1.5, 1.0]),
        beta=np.array([2.0, 1.5, 1.0]),
        targeted=False,
    )
    cfg.tiers = [targeted_tier, background_tier]
    return cfg


def make_targeted_only_scenario(kappa: float = 100.0) -> ScenarioConfig:
    """Targeted tier only (no background), used as comparison in exp_3."""
    cfg = ScenarioConfig(kappa=kappa)
    targeted_tier = TierConfig(
        name="tau_star",
        lambdas=np.array([50.0, 20.0, 5.0]),
        alpha=np.array([2.0, 1.5, 1.0]),
        beta=np.array([2.0, 1.5, 1.0]),
        targeted=True,
    )
    cfg.tiers = [targeted_tier]
    return cfg
