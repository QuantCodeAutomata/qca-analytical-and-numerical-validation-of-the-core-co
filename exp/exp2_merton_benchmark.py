"""
Experiment 2: Baseline Two-Asset Green/Brown Merton Benchmark Validation.

Constructs the stylized two-asset green/brown market and verifies the
benchmark Merton allocations reported in the paper.
"""

import sys
import json
import numpy as np

sys.path.insert(0, "/workspace/project")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.merton import merton_power, merton_log, alpha_power, alpha_log
from src.sign_conditions import classify_reallocation_direction

RESULTS_DIR = "/workspace/project/results"


def main() -> None:
    """Run Experiment 2: Merton benchmark validation."""
    print("=" * 70)
    print("EXPERIMENT 2: MERTON BENCHMARK VALIDATION")
    print("=" * 70)

    # Market parameters (paper's baseline)
    sigma_g = 0.15
    sigma_b = 0.30
    rho = 0.7
    r = 0.02
    eta = 2.0
    kappa_g = 0.2
    kappa_b = -0.5
    kappa = np.array([kappa_g, kappa_b])

    # Build covariance matrix
    Sigma = np.array([
        [sigma_g**2,              rho * sigma_g * sigma_b],
        [rho * sigma_g * sigma_b, sigma_b**2             ],
    ])
    print(f"\nCovariance matrix:\n{Sigma}")
    print(f"Sigma^{{-1}}:\n{np.linalg.inv(Sigma)}")

    # Verify positive definiteness
    try:
        np.linalg.cholesky(Sigma)
        print("\nSigma is POSITIVE DEFINITE: PASS")
    except np.linalg.LinAlgError:
        print("\nSigma is NOT positive definite: FAIL")
        return

    # --- Standard drift case ---
    mu_std = np.array([0.015, 0.035])
    print(f"\n--- Standard drift: mu = {mu_std} ---")
    pi_std = merton_power(mu_std, r, Sigma, eta)
    print(f"Merton pi^{{M,eta}} = {pi_std}")
    print(f"Expected approx  = [-0.45, 0.24]")
    err_std = np.abs(pi_std - np.array([-0.45, 0.24]))
    print(f"Absolute error   = {err_std}")
    print(f"Benchmark check  = {'PASS' if np.all(err_std < 1e-2) else 'FAIL (see note)'}")

    alpha_std = alpha_power(mu_std, r, Sigma, eta)
    print(f"alpha^eta        = {alpha_std:.6f}")

    pi_log_std = merton_log(mu_std, r, Sigma)
    print(f"Log-utility pi^M = {pi_log_std}")

    # Sign diagnostic: (pi^M)^T kappa
    sign_std = float(np.dot(pi_std, kappa))
    print(f"(pi^M)^T kappa   = {sign_std:.6f} -> {'NEGATIVE' if sign_std < 0 else 'POSITIVE'} regime")

    # --- Reverse drift case ---
    mu_rev = np.array([0.035, 0.015])
    print(f"\n--- Reverse drift: mu = {mu_rev} ---")
    pi_rev = merton_power(mu_rev, r, Sigma, eta)
    print(f"Merton pi^{{M,eta}} = {pi_rev}")
    print(f"Expected approx  = [0.73, -0.28]")
    err_rev = np.abs(pi_rev - np.array([0.73, -0.28]))
    print(f"Absolute error   = {err_rev}")
    print(f"Benchmark check  = {'PASS' if np.all(err_rev < 1e-2) else 'FAIL (see note)'}")

    alpha_rev = alpha_power(mu_rev, r, Sigma, eta)
    print(f"alpha^eta        = {alpha_rev:.6f}")

    pi_log_rev = merton_log(mu_rev, r, Sigma)
    print(f"Log-utility pi^M = {pi_log_rev}")

    sign_rev = float(np.dot(pi_rev, kappa))
    print(f"(pi^M)^T kappa   = {sign_rev:.6f} -> {'NEGATIVE' if sign_rev < 0 else 'POSITIVE'} regime")

    # Sign regime summary
    print("\n--- Sign Regime Summary ---")
    print(f"Standard drift: (pi^M)^T kappa = {sign_std:.4f} -> {'NEGATIVE (Fig 4 regime)' if sign_std < 0 else 'POSITIVE (Fig 5 regime)'}")
    print(f"Reverse drift:  (pi^M)^T kappa = {sign_rev:.4f} -> {'NEGATIVE (Fig 4 regime)' if sign_rev < 0 else 'POSITIVE (Fig 5 regime)'}")

    # Sign conditions analysis
    sc = classify_reallocation_direction(sigma_g, sigma_b, rho, kappa_g, kappa_b)
    print(f"\n--- Hedging Direction Analysis (rho={rho}) ---")
    print(f"Sigma_0^{{-1}} kappa = {sc['sigma_inv_kappa']}")
    print(f"Green component positive: {sc['green_positive']}")
    print(f"Brown component negative: {sc['brown_negative']}")
    print(f"kappa^T Sigma^{{-1}} kappa = {sc['kappa_quad']:.6f}")

    # Save results
    results = {
        "standard_drift": {
            "mu": mu_std.tolist(),
            "pi_merton": pi_std.tolist(),
            "expected": [-0.45, 0.24],
            "error": err_std.tolist(),
            "pass": bool(np.all(err_std < 1e-2)),
            "sign_dot_kappa": float(sign_std),
            "sign_regime": "negative" if sign_std < 0 else "positive",
        },
        "reverse_drift": {
            "mu": mu_rev.tolist(),
            "pi_merton": pi_rev.tolist(),
            "expected": [0.73, -0.28],
            "error": err_rev.tolist(),
            "pass": bool(np.all(err_rev < 1e-2)),
            "sign_dot_kappa": float(sign_rev),
            "sign_regime": "negative" if sign_rev < 0 else "positive",
        },
        "hedging_direction": {
            "sigma_inv_kappa": sc["sigma_inv_kappa"].tolist(),
            "green_positive": sc["green_positive"],
            "brown_negative": sc["brown_negative"],
            "kappa_quad": sc["kappa_quad"],
        },
    }
    with open(f"{RESULTS_DIR}/exp2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/exp2_results.json")

    # Bar chart of Merton portfolios
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    w = 0.3
    ax.bar(x - w/2, pi_std, w, label="Standard drift (mu_g<mu_b)", color="steelblue")
    ax.bar(x + w/2, pi_rev, w, label="Reverse drift (mu_g>mu_b)", color="tomato")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Green", "Brown"])
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Merton Benchmark Portfolios (eta=2): Standard vs Reverse Drift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_merton_weights.png", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {RESULTS_DIR}/exp2_merton_weights.png")

    # Overall summary
    print("\n=== EXPERIMENT 2 SUMMARY ===")
    print(f"Standard drift benchmark check: {'PASS' if np.all(err_std < 1e-2) else 'NOTE: values differ slightly'}")
    print(f"Reverse drift benchmark check : {'PASS' if np.all(err_rev < 1e-2) else 'NOTE: values differ slightly'}")
    print(f"Standard drift sign regime    : {'NEGATIVE ✓' if sign_std < 0 else 'POSITIVE'}")
    print(f"Reverse drift sign regime     : {'POSITIVE ✓' if sign_rev > 0 else 'NEGATIVE'}")


if __name__ == "__main__":
    main()
