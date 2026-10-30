"""
Experiment 7: Validation of Limiting Cases and Boundary Regimes.

Numerically verifies theoretical limiting behavior:
  1. Power utility -> log utility as eta -> 1.
  2. Robust power -> ambiguity-neutral as epsilon -> eta - 1 (from above).
  3. Robust power -> worst-case as epsilon -> infinity.
  4. Robust -> Merton as lambda -> 0.
  5. eta < 1, epsilon -> 0 (special regime).
"""

import sys
import json
import numpy as np

sys.path.insert(0, "/workspace/project")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.parameters import build_two_asset_params
from src.ode_solver import solve_pretransition
from src.worst_case import solve_worst_case
from src.ambiguity_neutral import solve_ambiguity_neutral_power, solve_ambiguity_neutral_log
from src.merton import merton_power, merton_log
from src.sign_conditions import build_two_asset_covariance

RESULTS_DIR = "/workspace/project/results"

BASE = dict(
    mu_g=0.015, mu_b=0.035, sigma_g=0.15, sigma_b=0.30,
    rho=0.7, kappa_g=0.2, kappa_b=-0.5, r=0.02, lam=0.1, T=10.0, n_grid=200,
)


def compute_sup_diff(arr1: np.ndarray, arr2: np.ndarray) -> float:
    """Compute sup-norm of element-wise difference for 2D arrays."""
    return float(np.max(np.abs(arr1 - arr2)))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Log-utility limit (eta -> 1)
# ─────────────────────────────────────────────────────────────────────────────
def test_log_limit() -> dict:
    """Verify that power utility -> log utility as eta -> 1."""
    print("\n--- Test 1: Log-utility limit (eta -> 1) ---")

    eta_seq = [1.5, 1.2, 1.1, 1.05, 1.02, 1.01]
    # Log utility reference
    params_log = build_two_asset_params(**BASE, eta=1.0, epsilon=2.0, utility="log")
    sol_log = solve_pretransition(params_log)
    pi_log = sol_log["pi0_star"]  # shape (n, 2)

    errors = []
    for eta_v in eta_seq:
        eps_v = max(eta_v - 1.0 + 0.2, 0.5)  # ensure epsilon > eta - 1
        try:
            p = build_two_asset_params(**BASE, eta=eta_v, epsilon=eps_v, utility="power")
            sol = solve_pretransition(p)
            # Interpolate log onto same t_grid
            sup_err = compute_sup_diff(sol["pi0_star"], pi_log)
            errors.append(sup_err)
            print(f"  eta={eta_v:.3f}, eps={eps_v:.3f}: sup|pi_power - pi_log| = {sup_err:.4f}")
        except Exception as e:
            print(f"  eta={eta_v:.3f}: FAILED ({e})")
            errors.append(np.nan)

    return {"eta_seq": eta_seq, "errors": errors}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ambiguity-neutral limit (epsilon -> eta - 1 from above)
# ─────────────────────────────────────────────────────────────────────────────
def test_ambiguity_neutral_limit() -> dict:
    """Verify robust power -> ambiguity-neutral as epsilon -> eta - 1."""
    print("\n--- Test 2: Ambiguity-neutral limit (epsilon -> eta - 1) ---")

    eta = 2.0
    eps_seq = [eta - 1 + delta for delta in [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]]

    # Ambiguity-neutral reference
    params_an = build_two_asset_params(**BASE, eta=eta, epsilon=eta - 0.5, utility="power")
    sol_an = solve_ambiguity_neutral_power(params_an)
    pi_an = sol_an["pi0_star"]

    errors = []
    for eps_v in eps_seq:
        try:
            p = build_two_asset_params(**BASE, eta=eta, epsilon=eps_v, utility="power")
            sol = solve_pretransition(p)
            sup_err = compute_sup_diff(sol["pi0_star"], pi_an)
            errors.append(sup_err)
            print(f"  eps={eps_v:.3f}: sup|pi_robust - pi_AN| = {sup_err:.4f}")
        except Exception as e:
            print(f"  eps={eps_v:.3f}: FAILED ({e})")
            errors.append(np.nan)

    return {"eps_seq": eps_seq, "errors": errors, "eta": eta}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Worst-case limit (epsilon -> infinity)
# ─────────────────────────────────────────────────────────────────────────────
def test_worst_case_limit() -> dict:
    """Verify robust -> worst-case as epsilon -> infinity."""
    print("\n--- Test 3: Worst-case limit (epsilon -> infinity) ---")

    eta = 2.0
    eps_seq = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

    # Worst-case reference (epsilon not used by worst_case solver; use any valid value).
    params_wc = build_two_asset_params(**BASE, eta=eta, epsilon=2.0, utility="power")
    sol_wc = solve_worst_case(params_wc)
    pi_wc = sol_wc["pi_wc"]
    sign_regime = sol_wc["sign_regime"]
    print(f"  Sign regime: {sign_regime}")

    errors = []
    for eps_v in eps_seq:
        try:
            p = build_two_asset_params(**BASE, eta=eta, epsilon=eps_v, utility="power")
            sol = solve_pretransition(p)
            sup_err = compute_sup_diff(sol["pi0_star"], pi_wc)
            errors.append(sup_err)
            print(f"  eps={eps_v:.1f}: sup|pi_robust - pi_wc| = {sup_err:.4f}")
        except Exception as e:
            print(f"  eps={eps_v:.1f}: FAILED ({e})")
            errors.append(np.nan)

    return {"eps_seq": eps_seq, "errors": errors, "sign_regime": sign_regime}


# ─────────────────────────────────────────────────────────────────────────────
# 4. No-transition limit (lambda -> 0)
# ─────────────────────────────────────────────────────────────────────────────
def test_no_transition_limit() -> dict:
    """Verify robust -> Merton as lambda -> 0."""
    print("\n--- Test 4: No-transition limit (lambda -> 0) ---")

    eta = 2.0
    epsilon = 2.0
    Sigma = build_two_asset_covariance(0.15, 0.30, 0.7)
    pi_merton = merton_power(np.array([0.015, 0.035]), 0.02, Sigma, eta)

    lam_seq = [0.1, 0.05, 0.02, 0.01, 1e-3]
    errors = []

    for lam_v in lam_seq:
        try:
            p = build_two_asset_params(**{**BASE, "lam": lam_v}, eta=eta, epsilon=epsilon, utility="power")
            sol = solve_pretransition(p)
            pi0_0 = sol["pi0_star"][-1]  # initial allocation at t=0
            err = float(np.max(np.abs(pi0_0 - pi_merton)))
            errors.append(err)
            print(f"  lambda={lam_v:.4f}: |pi_0*(0) - Merton| max = {err:.4f}")
        except Exception as e:
            print(f"  lambda={lam_v:.4f}: FAILED ({e})")
            errors.append(np.nan)

    return {"lam_seq": lam_seq, "errors": errors, "pi_merton": pi_merton.tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Special case: eta < 1, epsilon -> 0
# ─────────────────────────────────────────────────────────────────────────────
def test_eta_less_1_small_epsilon() -> dict:
    """Investigate eta < 1 with epsilon -> 0."""
    print("\n--- Test 5: eta < 1, epsilon -> 0 ---")

    eta = 0.5
    eps_seq = [0.5, 0.3, 0.2, 0.1, 0.05]
    # For eta < 1, need eps > eta - 1 = -0.5 (any eps > 0 works)
    Sigma = build_two_asset_covariance(0.15, 0.30, 0.7)
    pi_merton = merton_power(np.array([0.015, 0.035]), 0.02, Sigma, eta)
    print(f"  Merton portfolio (eta={eta}): {pi_merton}")

    results_eps = []
    for eps_v in eps_seq:
        try:
            p = build_two_asset_params(**BASE, eta=eta, epsilon=eps_v, utility="power")
            sol = solve_pretransition(p)
            pi0_0 = sol["pi0_star"][-1]
            xi_0 = sol["xi"][-1]
            theta_0 = sol["theta"][-1]
            results_eps.append({
                "eps": eps_v,
                "pi_g": float(pi0_0[0]),
                "pi_b": float(pi0_0[1]),
                "xi_0": float(xi_0),
                "theta_0": float(theta_0),
            })
            print(f"  eps={eps_v:.3f}: pi_0*(0)={pi0_0}, xi={xi_0:.4f}, theta={theta_0:.4f}")
        except Exception as e:
            print(f"  eps={eps_v:.3f}: FAILED ({e})")

    return {"eta": eta, "results": results_eps, "pi_merton": pi_merton.tolist()}


def main() -> None:
    """Run Experiment 7: Limiting cases validation."""
    print("=" * 70)
    print("EXPERIMENT 7: VALIDATION OF LIMITING CASES")
    print("=" * 70)

    all_results = {}

    all_results["log_limit"] = test_log_limit()
    all_results["ambiguity_neutral_limit"] = test_ambiguity_neutral_limit()
    all_results["worst_case_limit"] = test_worst_case_limit()
    all_results["no_transition_limit"] = test_no_transition_limit()
    all_results["eta_less_1_small_eps"] = test_eta_less_1_small_epsilon()

    # --- Summary Convergence Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Log-limit
    ax = axes[0, 0]
    err1 = all_results["log_limit"]["errors"]
    eta_seq1 = all_results["log_limit"]["eta_seq"]
    valid1 = [i for i, e in enumerate(err1) if not np.isnan(e)]
    if valid1:
        ax.semilogy([eta_seq1[i] for i in valid1], [err1[i] for i in valid1], "b-o")
    ax.set_xlabel("eta")
    ax.set_ylabel("sup|pi_power - pi_log|")
    ax.set_title("Test 1: Power->Log (eta->1)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # 2. Ambiguity-neutral limit
    ax = axes[0, 1]
    err2 = all_results["ambiguity_neutral_limit"]["errors"]
    eps_seq2 = all_results["ambiguity_neutral_limit"]["eps_seq"]
    valid2 = [i for i, e in enumerate(err2) if not np.isnan(e)]
    if valid2:
        ax.semilogy([eps_seq2[i] for i in valid2], [err2[i] for i in valid2], "g-o")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("sup|pi_robust - pi_AN|")
    ax.set_title("Test 2: Robust->Ambiguity-Neutral (eps->eta-1)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # 3. Worst-case limit
    ax = axes[1, 0]
    err3 = all_results["worst_case_limit"]["errors"]
    eps_seq3 = all_results["worst_case_limit"]["eps_seq"]
    valid3 = [i for i, e in enumerate(err3) if not np.isnan(e)]
    if valid3:
        ax.semilogy([eps_seq3[i] for i in valid3], [err3[i] for i in valid3], "r-o")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("sup|pi_robust - pi_wc|")
    ax.set_title("Test 3: Robust->Worst-case (eps->inf)")
    ax.grid(True, alpha=0.3)

    # 4. No-transition limit
    ax = axes[1, 1]
    err4 = all_results["no_transition_limit"]["errors"]
    lam_seq4 = all_results["no_transition_limit"]["lam_seq"]
    valid4 = [i for i, e in enumerate(err4) if not np.isnan(e)]
    if valid4:
        ax.loglog([lam_seq4[i] for i in valid4], [err4[i] for i in valid4], "purple", marker="o")
    ax.set_xlabel("lambda")
    ax.set_ylabel("|pi_0*(0) - Merton| max")
    ax.set_title("Test 4: Robust->Merton (lambda->0)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Experiment 7: Numerical Convergence of Limiting Cases", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp7_limiting_cases.png", dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {RESULTS_DIR}/exp7_limiting_cases.png")

    # JSON-serializable results
    def make_serializable(d):
        if isinstance(d, dict):
            return {k: make_serializable(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [make_serializable(v) for v in d]
        elif isinstance(d, float) and np.isnan(d):
            return None
        return d

    with open(f"{RESULTS_DIR}/exp7_results.json", "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/exp7_results.json")

    print("\n=== EXPERIMENT 7 SUMMARY ===")
    # Check monotone convergence
    def is_monotone_decreasing(lst):
        valid = [x for x in lst if x is not None and not np.isnan(x)]
        return all(valid[i] >= valid[i+1] for i in range(len(valid)-1)) if len(valid) > 1 else True

    print(f"Log limit (Test 1): errors {['%.3f' % e if not np.isnan(e) else 'NaN' for e in err1]}")
    print(f"AN limit   (Test 2): monotone? {is_monotone_decreasing(err2)}")
    print(f"WC limit   (Test 3): monotone? {is_monotone_decreasing(err3)}")
    print(f"λ->0 limit (Test 4): monotone? {is_monotone_decreasing(err4)}")


if __name__ == "__main__":
    main()
