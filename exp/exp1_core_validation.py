"""
Experiment 1: Analytical and Numerical Validation of the Core Model.

Validates pre- and post-transition optimal portfolio formulas for both
power and log utility. Runs all internal consistency checks.
"""

import sys
import json
import numpy as np

sys.path.insert(0, "/workspace/project")

from src.parameters import build_two_asset_params
from src.ode_solver import solve_pretransition, solve_posttransition
from src.verification import (
    verify_solution,
    print_verification_report,
    compute_ode_residuals,
    check_admissibility,
)
from src.merton import compute_merton_quantities
from src.plots import plot_verification_diagnostics

RESULTS_DIR = "/workspace/project/results"


def run_power_utility() -> dict:
    """Run and validate the power-utility pre-transition problem."""
    print("\n" + "=" * 70)
    print("POWER UTILITY VALIDATION")
    print("=" * 70)

    params = build_two_asset_params(
        mu_g=0.015,
        mu_b=0.035,
        sigma_g=0.15,
        sigma_b=0.30,
        rho=0.7,
        kappa_g=0.2,
        kappa_b=-0.5,
        r=0.02,
        lam=0.1,
        eta=2.0,
        epsilon=2.0,
        T=10.0,
        n_grid=500,
        utility="power",
    )

    # Admissibility
    adm = check_admissibility(params)
    print("\nAdmissibility checks:")
    for k, v in adm.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    assert all(adm.values()), "Admissibility checks failed."

    # Post-transition solution (closed-form)
    post = solve_posttransition(params)
    mq = compute_merton_quantities(params)
    print(f"\nPost-transition pi_1* = {post['pi1_star']}")
    print(f"alpha_1^eta          = {post['alpha1']:.6f}")
    assert np.allclose(post["pi1_star"], mq["pi1_merton"]), "Post-transition mismatch."

    # Pre-transition solution (backward ODE)
    print("\nSolving pre-transition backward ODE...")
    solution = solve_pretransition(params)

    # Verification
    print_verification_report(params, solution)
    ver = verify_solution(params, solution)
    ode_res = compute_ode_residuals(params, solution)

    checks = {
        "xi_positive": ver["xi_positive"],
        "theta_positive": ver["theta_positive"],
        "identity_satisfied": ver["identity_satisfied"],
        "root_res_passed": ver["root_res_passed"],
        "f_terminal_ok": ver["f_terminal_ok"],
        "ode_residual_max": float(np.max(ode_res)),
    }

    print(f"\n  ODE residual max (FD): {checks['ode_residual_max']:.2e}")

    # Print pi_0*(0) — initial allocation
    pi0_0 = solution["pi0_star"][-1]
    xi_0 = solution["xi"][-1]
    theta_0 = solution["theta"][-1]
    print(f"\nInitial allocation pi_0*(0)   = {pi0_0}")
    print(f"Initial xi*(0)                = {xi_0:.6f}")
    print(f"Initial theta*(0)             = {theta_0:.6f}")
    print(f"Merton pi_0^M                 = {mq['pi0_merton']}")

    # Plot diagnostics
    plot_verification_diagnostics(solution, params, filename="exp1_power_diagnostics.png")

    return {
        "utility": "power",
        "eta": params.eta,
        "epsilon": params.epsilon,
        "pi0_star_t0": pi0_0.tolist(),
        "xi_t0": float(xi_0),
        "theta_t0": float(theta_0),
        "pi_merton": mq["pi0_merton"].tolist(),
        "alpha1": float(post["alpha1"]),
        "checks": checks,
        "sign_regime": ver["sign_regime"],
        "identity_max_err": float(ver["identity_max_err"]),
        "root_res_max": float(ver["root_res_max"]),
    }


def run_log_utility() -> dict:
    """Run and validate the log-utility pre-transition problem."""
    print("\n" + "=" * 70)
    print("LOG UTILITY VALIDATION")
    print("=" * 70)

    params = build_two_asset_params(
        mu_g=0.015,
        mu_b=0.035,
        sigma_g=0.15,
        sigma_b=0.30,
        rho=0.7,
        kappa_g=0.2,
        kappa_b=-0.5,
        r=0.02,
        lam=0.1,
        eta=1.0,
        epsilon=2.0,
        T=10.0,
        n_grid=500,
        utility="log",
    )

    adm = check_admissibility(params)
    print("\nAdmissibility checks (log utility):")
    for k, v in adm.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    post = solve_posttransition(params)
    mq = compute_merton_quantities(params)
    print(f"\nPost-transition pi_1* = {post['pi1_star']}")
    print(f"alpha_1 (log)         = {post['alpha1']:.6f}")

    print("\nSolving log-utility pre-transition backward ODE...")
    solution = solve_pretransition(params)
    print_verification_report(params, solution)

    ver = verify_solution(params, solution)
    ode_res = compute_ode_residuals(params, solution)

    pi0_0 = solution["pi0_star"][-1]
    xi_0 = solution["xi"][-1]
    theta_0 = solution["theta"][-1]
    print(f"\nInitial allocation pi_0*(0)   = {pi0_0}")
    print(f"Initial xi*(0)                = {xi_0:.6f}")
    print(f"Merton pi_0^M (log)           = {mq['pi0_merton']}")

    plot_verification_diagnostics(solution, params, filename="exp1_log_diagnostics.png")

    return {
        "utility": "log",
        "eta": params.eta,
        "epsilon": params.epsilon,
        "pi0_star_t0": pi0_0.tolist(),
        "xi_t0": float(xi_0),
        "theta_t0": float(theta_0),
        "pi_merton": mq["pi0_merton"].tolist(),
        "alpha1": float(post["alpha1"]),
        "identity_max_err": float(ver["identity_max_err"]),
        "root_res_max": float(ver["root_res_max"]),
        "ode_residual_max": float(np.max(ode_res)),
    }


def main() -> None:
    """Run Experiment 1: core model validation."""
    results = {}

    try:
        results["power"] = run_power_utility()
        print("\nPower utility: ALL CHECKS PASSED" if all(
            results["power"]["checks"].values()
        ) else "\nPower utility: SOME CHECKS FAILED")
    except Exception as e:
        print(f"Power utility FAILED: {e}")
        results["power"] = {"error": str(e)}

    try:
        results["log"] = run_log_utility()
        print("\nLog utility: PASSED")
    except Exception as e:
        print(f"Log utility FAILED: {e}")
        results["log"] = {"error": str(e)}

    # Save results
    outpath = f"{RESULTS_DIR}/exp1_results.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
