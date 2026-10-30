# Analytical and Numerical Validation of Optimal Portfolio Choice Under Green Transition Risk

This repository implements and validates the theoretical continuous-time portfolio model
for an investor facing a one-time green transition shock with uncertain timing and
ambiguity aversion. The framework is a robust HJBI control problem solved via semi-explicit
backward ODEs coupled with scalar root equations.

## Project Structure

```
repo/
├── data/                        # Data directory (empty — theoretical model only)
├── src/                         # Core implementations
│   ├── parameters.py            # Parameter dataclass and two-asset builder
│   ├── merton.py                # Merton benchmarks for power and log utility
│   ├── root_solver.py           # Scalar xi* root solver and control recovery
│   ├── ode_solver.py            # Backward ODE solver for f(t) — power and log
│   ├── verification.py          # Admissibility, ODE residuals, identity checks
│   ├── ambiguity_neutral.py     # Ambiguity-neutral power and log limiting solvers
│   ├── worst_case.py            # Worst-case (ε→∞) benchmark solver
│   ├── sign_conditions.py       # Two-asset hedging direction analysis
│   └── plots.py                 # Shared plotting utilities
├── exp/                         # Experiment scripts
│   ├── exp1_core_validation.py  # Semi-explicit solver + diagnostic plots
│   ├── exp2_merton_benchmark.py # Two-asset baseline Merton benchmark
│   ├── exp3_figure2_comparative_statics.py  # Fig 2: λ, κ_b, κ_g sweeps
│   ├── exp4_figure3_surfaces.py # Fig 3: (ε, η) surface
│   ├── exp5_figure4_negative_sign.py        # Fig 4: negative-sign regime
│   ├── exp6_figure5_positive_sign.py        # Fig 5: positive-sign regime
│   ├── exp7_limiting_cases.py   # Log, ambiguity-neutral, worst-case, λ→0 limits
│   └── exp8_sign_conditions.py  # Hedging-direction analytics
├── tests/                       # pytest test suite (129 tests)
│   ├── test_parameters.py
│   ├── test_merton.py
│   ├── test_root_solver.py
│   ├── test_ode_solver.py
│   ├── test_verification.py
│   ├── test_ambiguity_neutral.py
│   ├── test_worst_case.py
│   └── test_sign_conditions.py
├── results/                     # Figures, JSON metrics, and summary
│   └── RESULTS.md               # Experiment findings summary
├── AGENTS.md                    # Repository-level notes for AI agents
├── .gitignore
└── README.md
```

## Model Summary

**Market:** Two risky assets (green and brown) plus a risk-free asset. A one-time
transition shock arrives at a random time τ with intensity λ, applying multiplicative
jumps κ to risky asset prices.

**Investor:** Power utility (η ≠ 1) or log utility (η = 1) with ambiguity aversion ε
over the transition intensity.

**Solution:** Pre-transition optimal controls are semi-explicit — obtained by backward
integration of a scalar ODE for f(t) coupled with a scalar root equation for ξ*(t).
Post-transition controls are closed-form Merton portfolios.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib pandas

# Run all tests
python -m pytest tests/ -q

# Run a specific experiment
python exp/exp1_core_validation.py
python exp/exp2_merton_benchmark.py
# ... through exp8
```

## Key Results

| Experiment | Highlight |
|---|---|
| Exp 1 | Identity 1 + π*(t)ᵀκ = ξ*(t) holds to 2×10⁻¹⁶ |
| Exp 2 | Merton weights (−0.447, +0.240) vs paper (−0.45, +0.24) ✓ |
| Exp 3 | Green tilt grows with λ and \|κ_b\| as predicted |
| Exp 4 | ε and η have qualitatively distinct effects on allocation |
| Exp 5 | Negative-sign regime: strong green tilt, θ*(t) < 1 for all ε |
| Exp 6 | Positive-sign regime: rapid convergence to Merton as ε↑ |
| Exp 7 | All limiting cases (log, ambiguity-neutral, worst-case, λ→0) verified |
| Exp 8 | Σ⁻¹κ analytical vs numerical: max error 7×10⁻¹⁵ |

All 129 tests pass.

## Dependencies

- numpy, scipy, matplotlib, pandas (core)
- pytest (testing)
