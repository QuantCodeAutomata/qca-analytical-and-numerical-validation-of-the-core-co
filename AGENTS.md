# AGENTS.md — Repository Knowledge for AI Agents

## What This Repository Does

Implements the robust HJBI optimal portfolio model for an investor facing a one-time
green/brown transition shock with uncertain timing (Poisson intensity λ) and ambiguity
aversion (parameter ε). Covers power utility (η ≠ 1) and log utility (η = 1).

## Core Algorithms (in `src/`)

### `parameters.py`
- `ModelParams` dataclass holds all model parameters.
- `build_two_asset_params(**kwargs)` builds a standard two-asset green/brown setup.
- Required fields: `T, r, lam, m, mu_0, mu_1, Sigma_0, Sigma_1, kappa, eta, epsilon`.

### `merton.py`
- `merton_power(mu, r, Sigma, eta)` → Merton weight vector for power utility.
- `merton_log(mu, r, Sigma)` → Merton weight vector for log utility.
- `alpha_power(mu, r, Sigma, eta)` → Certainty-equivalent growth rate for power utility.
- `alpha_log(mu, r, Sigma)` → Certainty-equivalent growth rate for log utility.

### `root_solver.py`
- `solve_xi_star(t, f_val, params)` → solve scalar polynomial for ξ*(t).
  Uses `scipy.optimize.brentq` on a bracketed positive interval.
- `compute_controls(xi_star, t, f_val, params)` → returns `(pi0, theta)`.
  Implements: pi0 = pi_M + c · Σ₀⁻¹κ; theta = exp(-ε[α₁(T-t) - f]) · ξ^(-ε).
- Root equation: `-ξ^(ε+2) + [1 + pi_M^T κ] ξ^(ε+1) + (λ/η) exp(-ε[α₁(T-t) - f]) κ^T Σ₀⁻¹ κ = 0`

### `ode_solver.py`
- `solve_pretransition(params, n_steps=1000)` → dict with keys `t, f, xi, theta, pi0`.
  Uses backward Euler stepping from f(T)=0.
- `solve_pretransition_log(params, n_steps=1000)` → same for log utility (η=1).
- ODE rhs: `f'(t) = -r - λ/ε - pi0^T(μ₀-r1) + (η/2) pi0^T Σ₀ pi0 + (λ/ε) θ`

### `verification.py`
- `check_positive_definite(Sigma)` → Cholesky test.
- `check_xi_positive(xi_arr)` → all > 0.
- `check_theta_positive(theta_arr)` → all > 0.
- `check_identity(pi0, kappa, xi)` → verifies 1 + pi0^T kappa == xi (per time step).
- `ode_residuals(sol, params)` → recomputes f'(t) and compares to finite-difference.
- `root_residuals(sol, params)` → re-evaluates root polynomial at xi*(t).

### `ambiguity_neutral.py`
- `solve_ambiguity_neutral_power(params, n_steps=1000)` → limiting case ε → η−1.
  Uses a different root equation (one lower power) and ODE.
- `solve_ambiguity_neutral_log(params, n_steps=1000)` → closed-form xi* formula for log.
  Formula: xi* = ½[1 + pi_M^T κ] + ½√([1 + pi_M^T κ]² + 4λ κ^T Σ₀⁻¹ κ)

### `worst_case.py`
- `solve_worst_case(params, n_steps=1000)` → ε→∞ limit.
  - If `(pi_M^T kappa) < 0`: solves zeta ODE backward; pi_wc = pi_M + c·Σ⁻¹κ.
  - If `(pi_M^T kappa) >= 0`: worst-case equals Merton benchmark everywhere.

### `sign_conditions.py`
- `sigma_inv_kappa(Sigma, kappa)` → numerical Σ⁻¹κ.
- `sigma_inv_kappa_analytical(sigma_g, sigma_b, rho, kappa_g, kappa_b)` → closed-form.
- `green_condition_threshold(kappa_g, kappa_b, sigma_g, sigma_b)` → ρ threshold for ↑green.
- `brown_condition_threshold(kappa_g, kappa_b, sigma_g, sigma_b)` → ρ threshold for ↓brown.
- `classify_reallocation_direction(Sigma, kappa)` → {green_positive, brown_negative}.

## Key Numerical Details

- **Root solver:** `brentq` with bracket `[1e-10, 50]`; tolerance 1e-10.
- **ODE stepping:** Backward Euler on uniform grid with `n_steps=1000` (default).
- **Identity check:** Max error should be ≤ 2.3e-16 (machine epsilon range).
- **Parameter constraint:** For power utility, must have `ε > η − 1` and `ε ≠ 0`.

## Experiment Scripts (`exp/`)

- `exp1`: Validates both utility types; saves `exp1_power_diagnostics.png`, `exp1_log_diagnostics.png`, `exp1_results.json`.
- `exp2`: Merton benchmark validation; saves `exp2_merton_weights.png`, `exp2_results.json`.
- `exp3`: λ/κ sweeps (Figure 2); saves `exp3_figure2.png`, `exp3_results.json`.
- `exp4`: (ε, η) surfaces (Figure 3); saves `exp4_figure3_surface.png`, `exp4_figure3_slices.png`, `exp4_results.json`.
- `exp5`: Negative-sign regime (Figure 4); saves `exp5_figure4.png`, `exp5_figure4_theta.png`, `exp5_results.json`.
- `exp6`: Positive-sign regime (Figure 5); saves `exp6_figure5.png`, `exp6_figure5_theta.png`, `exp6_results.json`.
- `exp7`: Limiting cases; saves `exp7_limiting_cases.png`, `exp7_results.json`.
- `exp8`: Sign conditions; saves `exp8_sign_conditions_rho.png`, `exp8_phase_diagram.png`, `exp8_results.json`.

## Tests

Run with `python -m pytest tests/ -q`. 129 tests, all pass.

Test files: `test_parameters`, `test_merton`, `test_root_solver`, `test_ode_solver`,
`test_verification`, `test_ambiguity_neutral`, `test_worst_case`, `test_sign_conditions`.

## Parameterization Conventions

- `standard drift`: mu_g=0.015, mu_b=0.035 → `(pi_M)^T kappa < 0` (negative-sign regime).
- `reverse drift`: mu_g=0.035, mu_b=0.015 → `(pi_M)^T kappa > 0` (positive-sign regime).
- Asset order is always (green, brown).
- kappa convention: kappa_g > 0 (green gains on transition), kappa_b < 0 (brown loses).

## Known Issues / Documentation

- Figure 2 in the paper does not report ε; assumed ε = 5.0 in exp3.
- A sentence in the paper incorrectly labels the standard drift case as the positive-sign regime; it is actually negative-sign (verified numerically).
- The exp4 grid masks all (η, ε) pairs where ε ≤ η − 1 (invalid domain).
