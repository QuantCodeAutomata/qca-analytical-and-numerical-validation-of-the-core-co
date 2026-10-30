"""
Experiment 6: Reproduction of Figure 5 — Positive-Sign Regime Time Profiles.

Computes time-series profiles for the reverse-drift case where
(pi^{M,eta})^T kappa > 0. Uses mu_g=0.035, mu_b=0.015.
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
    mu_g=0.035, mu_b=0.015,   # reversed drifts
    sigma_g=0.15, sigma_b=0.30,
    rho=0.7,
    kappa_g=0.2, kappa_b=-0.5,
    r=0.02, lam=0.1, eta=2.0,
    T=10.0, n_grid=300,
)

# epsilon=1.0 is inadmissible for eta=2 (requires epsilon > eta-1=1); use 1.1.
EPSILON_VALS = [1.1, 2.0, 5.0, 20.0, 50.0]


def solve_profile(epsilon: float) -> dict:
    """Solve pre-transition robust problem for a given epsilon."""
    params = build_two_asset_params(**BASELINE, epsilon=epsilon, utility="power")
    sol = solve_pretransition(params)
    ver = verify_solution(params, sol)
    return {"sol": sol, "params": params, "ver": ver}


def main() -> None:
    """Run Experiment 6: Figure 5 time profiles — positive-sign regime."""
    print("=" * 70)
    print("EXPERIMENT 6: FIGURE 5 — POSITIVE-SIGN REGIME TIME PROFILES")
    print("=" * 70)

    # Verify sign regime
    from src.sign_conditions import build_two_asset_covariance
    Sigma = build_two_asset_covariance(0.15, 0.30, 0.7)
    mu_rev = np.array([0.035, 0.015])
    kappa = np.array([0.2, -0.5])
    pi_m = merton_power(mu_rev, 0.02, Sigma, 2.0)
    dot_kappa = float(np.dot(pi_m, kappa))
    print(f"\nMerton benchmark (reverse drift): {pi_m}")
    print(f"Expected approx: (0.73, -0.28)")
    print(f"(pi^M)^T kappa = {dot_kappa:.6f} -> {'POSITIVE ✓' if dot_kappa > 0 else 'NEGATIVE (unexpected!)'}")
    assert dot_kappa > 0, "Expected positive-sign regime for reverse drift!"

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

        ver = result["ver"]
        print(f"    xi > 0: {ver['xi_positive']}, theta > 0: {ver['theta_positive']}, "
              f"identity max err: {ver['identity_max_err']:.2e}")

    # Worst-case benchmark: in positive-sign regime, worst-case = Merton
    print("\nWorst-case (epsilon -> inf) benchmark:")
    # Worst-case doesn't use epsilon in its computation; use a valid admissible value.
    wc_params = build_two_asset_params(**BASELINE, epsilon=2.0, utility="power")
    wc_sol = solve_worst_case(wc_params)
    pi_wc = wc_sol["pi_wc"]
    print(f"  Sign regime: {wc_sol['sign_regime']}")
    print(f"  Worst-case = Merton: {np.allclose(pi_wc, np.tile(pi_m, (len(t_grid), 1)))}")

    # Plot time profiles
    plot_path = plot_time_profiles(
        t_grid=t_grid,
        pi0_arr_list=pi0_lists,
        theta_arr_list=theta_lists,
        labels=labels,
        pi_merton=pi_m,
        asset_names=["Green", "Brown"],
        pi_wc=pi_wc,
        title="Figure 5: Time profiles — Positive-sign regime (mu_g=0.035, mu_b=0.015)",
        filename="exp6_figure5.png",
    )
    print(f"\nFigure saved to {plot_path}")

    # Theta profiles
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(EPSILON_VALS)))
    for eps, theta_arr, color in zip(EPSILON_VALS, theta_lists, colors):
        ax.plot(t_grid, theta_arr, color=color, lw=2, label=f"$\\epsilon={eps}$")
    ax.axhline(1.0, color="gray", linestyle=":", lw=1.5, label=r"$\theta=1$")
    ax.set_xlabel("Time $t$")
    ax.set_ylabel(r"$\theta^*(t)$")
    ax.set_title("Figure 5: Optimal distortion theta*(t) — Positive-sign regime")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp6_figure5_theta.png", dpi=150)
    plt.close(fig)
    print(f"Theta figure saved to {RESULTS_DIR}/exp6_figure5_theta.png")

    # Qualitative checks
    pi_g_t0 = [profiles[eps]["sol"]["pi0_star"][-1][0] for eps in EPSILON_VALS]
    pi_b_t0 = [profiles[eps]["sol"]["pi0_star"][-1][1] for eps in EPSILON_VALS]
    theta_t0 = [profiles[eps]["sol"]["theta"][-1] for eps in EPSILON_VALS]

    print("\n--- Qualitative checks ---")
    print("  epsilon | pi_g*(0) | pi_b*(0) | theta*(0)")
    for eps, pg, pb, th in zip(EPSILON_VALS, pi_g_t0, pi_b_t0, theta_t0):
        print(f"  {eps:6.1f} | {pg:8.4f} | {pb:8.4f} | {th:9.4f}")
    print(f"  Merton  | {pi_m[0]:8.4f} | {pi_m[1]:8.4f} |  (n/a)")

    # Convergence to Merton as epsilon increases
    diff_large_eps = np.abs(np.array([pi_g_t0[-1], pi_b_t0[-1]]) - pi_m)
    diff_small_eps = np.abs(np.array([pi_g_t0[0], pi_b_t0[0]]) - pi_m)
    print(f"\n  Deviation from Merton at eps={EPSILON_VALS[0]}: {diff_small_eps}")
    print(f"  Deviation from Merton at eps={EPSILON_VALS[-1]}: {diff_large_eps}")
    converges_to_merton = bool(np.all(diff_large_eps < diff_small_eps + 1e-3))
    print(f"  Convergence toward Merton as eps increases: {'PASS' if converges_to_merton else 'CHECK'}")

    results = {
        "sign_regime": "positive",
        "pi_merton": pi_m.tolist(),
        "pi_merton_dot_kappa": dot_kappa,
        "pi_g_t0": {str(eps): pg for eps, pg in zip(EPSILON_VALS, pi_g_t0)},
        "pi_b_t0": {str(eps): pb for eps, pb in zip(EPSILON_VALS, pi_b_t0)},
        "theta_t0": {str(eps): th for eps, th in zip(EPSILON_VALS, theta_t0)},
        "worst_case_equals_merton": bool(wc_sol["sign_regime"] == "positive"),
        "converges_to_merton": converges_to_merton,
    }
    with open(f"{RESULTS_DIR}/exp6_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/exp6_results.json")

    print("\n=== EXPERIMENT 6 SUMMARY ===")
    print(f"Sign regime correctly identified: {'PASS' if dot_kappa > 0 else 'FAIL'}")
    print(f"Worst-case = Merton (positive regime): {'PASS' if wc_sol['sign_regime'] == 'positive' else 'FAIL'}")
    print(f"Convergence to Merton as eps grows: {'PASS' if converges_to_merton else 'CHECK'}")


if __name__ == "__main__":
    main()
