# OTC Bond Market-Making with Hit-Ratio Targeting — HJB Replication

Numerical replication of a finite-horizon Hamilton-Jacobi-Bellman (HJB) control problem
for OTC bond market-making with size-weighted hit-ratio targeting and a quadratic
hit-ratio penalty, following the BEGV/dual-closure methodology.

All experiments use **purely synthetic model inputs** — no empirical market data are required
or fetched.

---

## Research Experiments

| ID | Title | Key Method |
|----|-------|------------|
| exp_1 | Single-Bond HJB Replication & Approximation Hierarchy | Exact 1D HJB backward PDE + 4-level approximation hierarchy |
| exp_2 | κ Sensitivity & Forward-Law Performance | BEGV stationary + CTMC forward Kolmogorov |
| exp_3 | Two-Tier Background Flow Risk-Sharing | Two-tier BEGV + CTMC with background & targeted tiers |
| exp_4 | Multi-Bond Correlated Background Bond | 2D BEGV matrix Riccati + Monte-Carlo CTMC |

---

## Repository Structure

```
.
├── data/                    # Placeholder (no market data — purely synthetic)
├── src/
│   ├── config.py            # Model scenario parameters
│   ├── primitives/
│   │   ├── fill_probability.py   # Logistic fill function f(δ), f′, f″, f‴
│   │   └── hamiltonian.py        # Hamiltonian H(p), H′(p), H″(p), δ̃(p) interpolators
│   ├── hjb/
│   │   └── exact_1d.py           # Explicit-Euler backward HJB solver (1D)
│   ├── approximations/
│   │   └── begv.py               # BEGV Riccati ODE, exact/quadratic/constant ξ closures
│   ├── ctmc/
│   │   └── forward_law.py        # CTMC generator assembly + forward-law propagation
│   ├── analytics/
│   │   └── metrics.py            # Quotes, hit ratio, objective, xi, approximation errors
│   └── plotting/
│       └── figures.py            # All matplotlib figure generators
├── exp/
│   ├── exp1_hjb_replication.py   # Experiment 1: exact HJB + approximation comparison
│   ├── exp2_kappa_sensitivity.py # Experiment 2: κ sweep + forward-law performance
│   ├── exp3_background_tier.py   # Experiment 3: targeted + background tier
│   └── exp4_multibond.py         # Experiment 4: multi-bond correlation study
├── tests/
│   ├── test_fill_probability.py  # Unit tests: fill function bounds, monotonicity
│   ├── test_hamiltonian.py       # Unit tests: envelope property, H″ positivity
│   ├── test_hjb.py               # Unit tests: HJB terminal condition, symmetry
│   ├── test_begv.py              # Unit tests: Riccati solution, closure formulas
│   ├── test_ctmc.py              # Unit tests: generator row-sums, mass preservation
│   └── test_config.py            # Unit tests: parameter validation
├── results/
│   ├── RESULTS.md               # Consolidated experiment findings
│   ├── exp{1-4}_metrics.json    # Machine-readable metrics
│   └── fig{1-9}_*.png           # Publication-quality figures
└── README.md
```

---

## Quickstart

### 1 — Install dependencies

```bash
pip install numpy scipy matplotlib
# Optional acceleration:
pip install numba
```

All core dependencies (numpy, scipy, matplotlib) ship in the pre-configured environment.

### 2 — Run all experiments

```bash
python exp/exp1_hjb_replication.py
python exp/exp2_kappa_sensitivity.py
python exp/exp3_background_tier.py
python exp/exp4_multibond.py
```

Results (JSON metrics + PNG figures) are written to `results/`.

### 3 — Run tests

```bash
pytest tests/ -v
```

Expected: **58 passed, 0 failed**.

---

## Model Summary

### State and controls
- **Inventory** `Q_t ∈ {−q_max, …, q_max}` Mn notional (single bond), or vector-valued for multi-bond.
- **Controls** `δ^{s,k}_t`: bid/ask quote offsets (basis points) per side `s ∈ {b, a}` and size bucket `k`.

### Objective
```
max E[ X_T + Q_T S_T − (η/2) Q_T² Σ
        − ∫₀ᵀ { (φ/2) Q_t² Σ + (κ W / 2)(r(δ_t) − r*)² } dt ]
```
where `r(δ_t)` is the instantaneous size-weighted hit ratio for the targeted client tier.

### Fill model
Logistic fill probability: `f_k(δ) = 1 / (1 + exp(α_k + β_k δ))`, strictly decreasing in `δ`.

### Solution approach
1. **Dualize** the quadratic hit-ratio penalty via `inf_ξ { ξ(r−r*) + ξ²/(2κ) }`.
2. **Precompute** static Hamiltonians `H^{s,k}(p) = sup_δ Λ^k(δ)(δ − p)` on a dense p-grid.
3. **Solve** the exact 1D HJB backward in time (explicit Euler) with an implicit scalar ξ closure.
4. **Compare** approximation hierarchy: BEGV quadratic value function with (A) exact ξ, (B) local quadratic ξ₀+ξ₂q², (C) constant ξ₀, (D) linearised quote map.

### Baseline parameters

| Symbol | Value | Description |
|--------|-------|-------------|
| z | (1, 5, 20) Mn | Size ladder |
| λ | (500, 200, 50) day⁻¹ | RFQ arrival intensities per side |
| α | (2.0, 1.5, 1.0) | Fill function intercepts |
| β | (2.0, 1.5, 1.0) bp⁻¹ | Fill function slopes |
| r* | 0.10 | Target hit ratio |
| φ | 1.0 | Running inventory penalty |
| η | 0.0 | Terminal inventory penalty |
| T | 1 day | Horizon |
| q_max | 100 Mn | Inventory truncation |

---

## Key Results (Summary)

See `results/RESULTS.md` for full tables and figure descriptions.

- **Exp 1:** Exact HJB reproduced; approximation B (local quadratic ξ closure) has lowest
  worst-case error; ξ(q) is confirmed even in q.
- **Exp 2:** Expected hit ratio rises monotonically 0.071 → 0.100 as κ increases 0.1 → 1000;
  desk objective declines 260 → 231, quantifying the targeting cost.
- **Exp 3:** Background-tier flow raises targeted-tier hit ratio; improvement saturates with
  increasing background/targeted intensity ratio.
- **Exp 4:** Correlation ρ = 0.8 yields the highest targeted-tier hit ratio (0.0764) via
  cross-bond inventory risk sharing.

---

## Citation / Methodology Source

This code reproduces the numerical experiments from the paper on OTC bond market-making
with hit-ratio targeting using a Hamilton-Jacobi-Bellman framework with BEGV quadratic
value-function approximations and dual (Lagrangian) hit-ratio penalty closure.
Custom implementations are used throughout; standard market-making libraries (e.g.,
Avellaneda-Stoikov) do not support the targeted hit-ratio structure.
