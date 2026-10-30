"""
Experiment 5: Reproduction of Figure 4 — Negative-Sign Regime Time Profiles.

Computes time-series profiles of pi_0*(t), pi_0_b*(t), and theta*(t) for
multiple ambiguity levels in the regime where (pi^{M,eta})^T kappa < 0.

Standard drift (mu_g=0.015, mu_b=0.035) with kappa=(0.2,-0.5) produces this regime.
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
from src.merton import merton_power
from src.worst_case import solve_worst_case
from src.verification import verify_solution
from src.plots import plot_time_profiles

RESULTS_DIR = "/workspace/project/results"

BASELINE = dict(
    mu_g=0.015, mu_b=0.035,
    sigma_g=0.15, sigma_b=0.30,
    rho=0.7,
    kappa_g=0.2, kappa_b=-0.5,
    r=0.02, lam=0.1, eta=2.0,
    T=10.0, n_grid=300,
)

# Inferred epsilon values from Figure 4 legend.
# Note: epsilon=1.0 is on the inadmissible boundary for eta=2 (requires epsilon > eta-1=1).
# Use epsilon=1.1 as the closest admissible substitute.
EPSILON_VALS = [1.1, 2.0, 5.0, 20.0, 50.0]


def solve_profile(epsilon: float) -> dict:
    """Solve pre-transition robust problem for a given epsilon."""
    params = build_two_asset_params(
        **BASELINE, epsilon=epsilon, utility="power"
    )
    sol = solve_pretransition(params)
    ver = verify_solution(params, sol)
    return {"sol": sol, "params": params, "ver": ver}


def main() -> None:
    """Run Experiment 5: Figure 4 time profiles — negative-sign regime."""
    print("=" * 70)
    print("EXPERIMENT 5: FIGURE 4 — NEGATIVE-SIGN REGIME TIME PROFILES")
    print("=" * 70)

    # Verify sign regime
    from src.sign_conditions import build_two_asset_covariance
    Sigma = build_two_asset_covariance(0.15, 0.30, 0.7)
    mu_std = np.array([0.015, 0.035])
    kappa = np.array([0.2, -0.5])
    pi_m = merton_power(mu_std, 0.02, Sigma, 2.0)
    dot_kappa = float(np.dot(pi_m, kappa))
    print(f"\nMerton benchmark: {pi_m}")
    print(f"(pi^M)^T kappa = {dot_kappa:.6f} -> {'NEGATIVE ✓' if dot_kappa < 0 else 'POSITIVE (unexpected!)'}")
    assert dot_kappa < 0, "Expected negative-sign regime for standard drift!"

    # Solve for each epsilon
    print(f"\nSolving for epsilon values: {EPSILON_VALS}")
    profiles = {}
    pi0_lists = []
    theta_lists = []
    labels = []
    t_grid = None

    for eps in EPSILON_VALS:
        print(f"  epsilon = {eps}...")
        result = solve_profile(eps)
        profiles[eps] = result
        sol = result["sol"]
        if t_grid is None:
            t_grid = sol["t_grid"]
        pi0_lists.append(sol["pi0_star"])
        theta_lists.append(sol["theta"])
        labels.append(f"$\\epsilon={eps}$")

        # Print diagnostic
        ver = result["ver"]
        print(f"    xi > 0: {ver['xi_positive']}, theta > 0: {ver['theta_positive']}, "
              f"identity max err: {ver['identity_max_err']:.2e}")

    # Worst-case benchmark
    print("\nSolving worst-case (epsilon -> inf) benchmark...")
    # Worst-case doesn't use epsilon in its computation; use a valid admissible value.
    wc_params = build_two_asset_params(**BASELINE, epsilon=2.0, utility="power")
    wc_sol = solve_worst_case(wc_params)
    pi_wc = wc_sol["pi_wc"]
    print(f"  Sign regime: {wc_sol['sign_regime']}")
    print(f"  pi_wc at t=0: {pi_wc[-1]}")
    print(f"  pi_wc at t=T: {pi_wc[0]}")

    # Plot time profiles
    plot_path = plot_time_profiles(
        t_grid=t_grid,
        pi0_arr_list=pi0_lists,
        theta_arr_list=theta_lists,
        labels=labels,
        pi_merton=pi_m,
        asset_names=["Green", "Brown"],
        pi_wc=pi_wc,
        title="Figure 4: Time profiles — Negative-sign regime (mu_g=0.015, mu_b=0.035)",
        filename="exp5_figure4.png",
    )
    print(f"\nFigure saved to {plot_path}")

    # Additional: theta profiles only
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(EPSILON_VALS)))
    for eps, theta_arr, color in zip(EPSILON_VALS, theta_lists, colors):
        ax.plot(t_grid, theta_arr, color=color, lw=2, label=f"$\\epsilon={eps}$")
    ax.axhline(1.0, color="gray", linestyle=":", lw=1.5, label=r"$\theta=1$")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\theta^*(t)$")
    ax.set_title("Figure 4: Optimal distortion theta*(t) — Negative-sign regime")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp5_figure4_theta.png", dpi=150)
    plt.close(fig)
    print(f"Theta figure saved to {RESULTS_DIR}/exp5_figure4_theta.png")

    # Qualitative checks
    print("\n--- Qualitative checks ---")
    # As epsilon increases, pi_g*(0) should move toward Merton
    pi_g_t0 = [profiles[eps]["sol"]["pi0_star"][-1][0] for eps in EPSILON_VALS]
    pi_b_t0 = [profiles[eps]["sol"]["pi0_star"][-1][1] for eps in EPSILON_VALS]
    theta_t0 = [profiles[eps]["sol"]["theta"][-1] for eps in EPSILON_VALS]

    print("  epsilon | pi_g*(0) | pi_b*(0) | theta*(0)")
    for eps, pg, pb, th in zip(EPSILON_VALS, pi_g_t0, pi_b_t0, theta_t0):
        print(f"  {eps:6.1f} | {pg:8.4f} | {pb:8.4f} | {th:9.4f}")
    print(f"  Merton  | {pi_m[0]:8.4f} | {pi_m[1]:8.4f} |  (n/a)")

    # theta*(t) below 1 check
    all_theta_below_1 = all(
        np.all(profiles[eps]["sol"]["theta"] <= 1.0 + 1e-4) for eps in EPSILON_VALS
    )
    print(f"\n  theta*(t) <= 1 for all epsilon: {'PASS' if all_theta_below_1 else 'NOTE: some > 1'}")

    # Save results
    results = {
        "sign_regime": "negative",
        "pi_merton": pi_m.tolist(),
        "pi_merton_dot_kappa": dot_kappa,
        "pi_g_t0": {str(eps): pg for eps, pg in zip(EPSILON_VALS, pi_g_t0)},
        "pi_b_t0": {str(eps): pb for eps, pb in zip(EPSILON_VALS, pi_b_t0)},
        "theta_t0": {str(eps): th for eps, th in zip(EPSILON_VALS, theta_t0)},
        "pi_wc_t0": pi_wc[-1].tolist(),
        "all_theta_below_1": all_theta_below_1,
    }
    with open(f"{RESULTS_DIR}/exp5_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/exp5_results.json")

    print("\n=== EXPERIMENT 5 SUMMARY ===")
    print(f"Sign regime correctly identified: {'PASS' if dot_kappa < 0 else 'FAIL'}")
    print(f"All theta*(t) <= 1: {'PASS' if all_theta_below_1 else 'NOTE'}")


if __name__ == "__main__":
    main()
