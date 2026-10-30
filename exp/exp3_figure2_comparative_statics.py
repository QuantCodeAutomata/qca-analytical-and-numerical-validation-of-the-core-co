"""
Experiment 3: Reproduction of Figure 2 Comparative Statics.

Computes initial optimal allocations pi_0*(0) as functions of transition
intensity lambda, brown jump kappa_b, and green jump kappa_g.
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

RESULTS_DIR = "/workspace/project/results"


def solve_initial_allocation(
    mu_g=0.015, mu_b=0.035, sigma_g=0.15, sigma_b=0.30, rho=0.7,
    kappa_g=0.1, kappa_b=-0.2, r=0.02, lam=0.1, eta=2.0, epsilon=2.0,
    T=10.0, n_grid=200,
) -> dict:
    """
    Solve the pre-transition problem and return the initial allocation.

    Returns dict with pi0_g, pi0_b, xi_0, theta_0, diagnostics.
    """
    params = build_two_asset_params(
        mu_g=mu_g, mu_b=mu_b, sigma_g=sigma_g, sigma_b=sigma_b, rho=rho,
        kappa_g=kappa_g, kappa_b=kappa_b, r=r, lam=lam, eta=eta,
        epsilon=epsilon, T=T, n_grid=n_grid, utility="power",
    )
    sol = solve_pretransition(params)
    # t_grid is [0..T], pi0_star[-1] is at t=0 (initial time)
    pi0_0 = sol["pi0_star"][-1]
    xi_0 = float(sol["xi"][-1])
    theta_0 = float(sol["theta"][-1])
    root_res = float(sol["root_residuals"][-1])
    return {
        "pi0_g": float(pi0_0[0]),
        "pi0_b": float(pi0_0[1]),
        "xi_0": xi_0,
        "theta_0": theta_0,
        "root_res": root_res,
    }


def sweep_lambda(
    lambda_grid: np.ndarray,
    epsilon: float = 2.0,
    kappa_g: float = 0.1,
    kappa_b: float = -0.2,
) -> dict:
    """Panel (a): sweep over lambda."""
    pi_g = np.zeros(len(lambda_grid))
    pi_b = np.zeros(len(lambda_grid))

    for i, lam in enumerate(lambda_grid):
        if lam < 1e-6:
            # Limit as lambda -> 0: use Merton benchmark
            from src.sign_conditions import build_two_asset_covariance
            _Sigma = build_two_asset_covariance(0.15, 0.30, 0.7)
            pi_m = merton_power(np.array([0.015, 0.035]), 0.02, _Sigma, 2.0)
            pi_g[i] = float(pi_m[0])
            pi_b[i] = float(pi_m[1])
        else:
            try:
                res = solve_initial_allocation(lam=lam, kappa_g=kappa_g, kappa_b=kappa_b, epsilon=epsilon)
                pi_g[i] = res["pi0_g"]
                pi_b[i] = res["pi0_b"]
            except Exception as e:
                print(f"  lambda={lam:.3f} failed: {e}")
                pi_g[i] = np.nan
                pi_b[i] = np.nan

    return {"pi_g": pi_g, "pi_b": pi_b}


def sweep_kappa_b(
    kappa_b_grid: np.ndarray,
    epsilon: float = 2.0,
    lam: float = 0.1,
) -> dict:
    """Panel (b): sweep over kappa_b (kappa_g = 0)."""
    pi_g = np.zeros(len(kappa_b_grid))
    pi_b = np.zeros(len(kappa_b_grid))

    for i, kb in enumerate(kappa_b_grid):
        try:
            res = solve_initial_allocation(lam=lam, kappa_g=0.0, kappa_b=kb, epsilon=epsilon)
            pi_g[i] = res["pi0_g"]
            pi_b[i] = res["pi0_b"]
        except Exception as e:
            print(f"  kappa_b={kb:.3f} failed: {e}")
            pi_g[i] = np.nan
            pi_b[i] = np.nan

    return {"pi_g": pi_g, "pi_b": pi_b}


def sweep_kappa_g(
    kappa_g_grid: np.ndarray,
    epsilon: float = 2.0,
    lam: float = 0.1,
) -> dict:
    """Panel (c): sweep over kappa_g (kappa_b = 0)."""
    pi_g = np.zeros(len(kappa_g_grid))
    pi_b = np.zeros(len(kappa_g_grid))

    for i, kg in enumerate(kappa_g_grid):
        try:
            res = solve_initial_allocation(lam=lam, kappa_g=kg, kappa_b=0.0, epsilon=epsilon)
            pi_g[i] = res["pi0_g"]
            pi_b[i] = res["pi0_b"]
        except Exception as e:
            print(f"  kappa_g={kg:.3f} failed: {e}")
            pi_g[i] = np.nan
            pi_b[i] = np.nan

    return {"pi_g": pi_g, "pi_b": pi_b}


def main() -> None:
    """Run Experiment 3: Figure 2 comparative statics."""
    print("=" * 70)
    print("EXPERIMENT 3: FIGURE 2 COMPARATIVE STATICS")
    print("=" * 70)

    # Assumed epsilon (not reported in paper for Figure 2)
    epsilon = 2.0
    print(f"\nNote: epsilon = {epsilon} (assumed; not specified in paper for Figure 2)")

    # Merton benchmark (transition-unaware)
    from src.sign_conditions import build_two_asset_covariance
    Sigma = build_two_asset_covariance(0.15, 0.30, 0.7)
    pi_merton = merton_power(np.array([0.015, 0.035]), 0.02, Sigma, 2.0)
    print(f"Merton benchmark: {pi_merton}")

    # --- Panel (a): sweep lambda ---
    print("\nPanel (a): sweeping lambda...")
    lambda_grid = np.linspace(1e-4, 0.5, 40)
    res_a = sweep_lambda(lambda_grid, epsilon=epsilon, kappa_g=0.1, kappa_b=-0.2)

    # --- Panel (b): sweep kappa_b ---
    print("\nPanel (b): sweeping kappa_b (kappa_g=0)...")
    kappa_b_grid = np.linspace(-0.85, -0.01, 40)
    res_b = sweep_kappa_b(kappa_b_grid, epsilon=epsilon, lam=0.1)

    # --- Panel (c): sweep kappa_g ---
    print("\nPanel (c): sweeping kappa_g (kappa_b=0)...")
    kappa_g_grid = np.linspace(0.001, 0.5, 40)
    res_c = sweep_kappa_g(kappa_g_grid, epsilon=epsilon, lam=0.1)

    # Sensitivity across epsilon values
    print("\nEpsilon sensitivity at lambda=0.1, kappa_g=0.1, kappa_b=-0.2:")
    eps_vals = [0.5, 1.0, 2.0, 5.0]
    for eps_v in eps_vals:
        try:
            res = solve_initial_allocation(epsilon=eps_v, kappa_g=0.1, kappa_b=-0.2)
            print(f"  eps={eps_v}: pi_g={res['pi0_g']:.4f}, pi_b={res['pi0_b']:.4f}")
        except Exception as e:
            print(f"  eps={eps_v}: FAILED ({e})")

    # --- Plotting (3-panel figure) ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel a
    ax = axes[0]
    ax.plot(lambda_grid, res_a["pi_g"], "g-o", markersize=3, lw=2, label="Green $\\pi_{0,g}^*(0)$")
    ax.plot(lambda_grid, res_a["pi_b"], "brown", marker="s", markersize=3, lw=2, label="Brown $\\pi_{0,b}^*(0)$")
    ax.axhline(pi_merton[0], color="g", linestyle="--", lw=1.5, label="Merton Green")
    ax.axhline(pi_merton[1], color="brown", linestyle="--", lw=1.5, label="Merton Brown")
    ax.set_xlabel(r"$\lambda$ (transition intensity)")
    ax.set_ylabel("Initial allocation")
    ax.set_title("(a) vs. transition intensity $\\lambda$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel b
    ax = axes[1]
    ax.plot(kappa_b_grid, res_b["pi_g"], "g-o", markersize=3, lw=2, label="Green")
    ax.plot(kappa_b_grid, res_b["pi_b"], "brown", marker="s", markersize=3, lw=2, label="Brown")
    ax.axhline(pi_merton[0], color="g", linestyle="--", lw=1.5, label="Merton Green")
    ax.axhline(pi_merton[1], color="brown", linestyle="--", lw=1.5, label="Merton Brown")
    ax.set_xlabel(r"$\kappa_b$ (brown jump)")
    ax.set_ylabel("Initial allocation")
    ax.set_title("(b) vs. brown jump $\\kappa_b$ ($\\kappa_g=0$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel c
    ax = axes[2]
    ax.plot(kappa_g_grid, res_c["pi_g"], "g-o", markersize=3, lw=2, label="Green")
    ax.plot(kappa_g_grid, res_c["pi_b"], "brown", marker="s", markersize=3, lw=2, label="Brown")
    ax.axhline(pi_merton[0], color="g", linestyle="--", lw=1.5, label="Merton Green")
    ax.axhline(pi_merton[1], color="brown", linestyle="--", lw=1.5, label="Merton Brown")
    ax.set_xlabel(r"$\kappa_g$ (green jump)")
    ax.set_ylabel("Initial allocation")
    ax.set_title("(c) vs. green jump $\\kappa_g$ ($\\kappa_b=0$)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Figure 2: Comparative Statics — Initial Optimal Allocations (eps={epsilon})",
        fontsize=13,
    )
    fig.tight_layout()
    out_fig = f"{RESULTS_DIR}/exp3_figure2.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved to {out_fig}")

    # Save results
    results = {
        "epsilon_assumed": epsilon,
        "pi_merton": pi_merton.tolist(),
        "panel_a": {
            "lambda_grid": lambda_grid.tolist(),
            "pi_g": [x for x in res_a["pi_g"].tolist()],
            "pi_b": [x for x in res_a["pi_b"].tolist()],
        },
        "panel_b": {
            "kappa_b_grid": kappa_b_grid.tolist(),
            "pi_g": [x for x in res_b["pi_g"].tolist()],
            "pi_b": [x for x in res_b["pi_b"].tolist()],
        },
        "panel_c": {
            "kappa_g_grid": kappa_g_grid.tolist(),
            "pi_g": [x for x in res_c["pi_g"].tolist()],
            "pi_b": [x for x in res_c["pi_b"].tolist()],
        },
    }
    with open(f"{RESULTS_DIR}/exp3_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/exp3_results.json")

    print("\n=== EXPERIMENT 3 SUMMARY ===")
    print(f"Panel (a): lambda sweep, final green={res_a['pi_g'][-1]:.4f}, brown={res_a['pi_b'][-1]:.4f}")
    print(f"Panel (b): kappa_b sweep, first green={res_b['pi_g'][0]:.4f}, brown={res_b['pi_b'][0]:.4f}")
    print(f"Panel (c): kappa_g sweep, last green={res_c['pi_g'][-1]:.4f}, brown={res_c['pi_b'][-1]:.4f}")


if __name__ == "__main__":
    main()
