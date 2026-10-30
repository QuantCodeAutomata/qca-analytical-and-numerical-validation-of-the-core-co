"""
Experiment 4: Reproduction of Figure 3 Surfaces over (eta, epsilon).

Computes pi_0*(0) as a joint function of risk aversion eta and ambiguity
aversion epsilon, respecting the admissibility constraint epsilon > eta - 1.
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
from src.plots import plot_surface

RESULTS_DIR = "/workspace/project/results"


def solve_surface_point(
    eta: float,
    epsilon: float,
    lam: float = 0.1,
    kappa_g: float = 0.2,
    kappa_b: float = -0.5,
    mu_g: float = 0.015,
    mu_b: float = 0.035,
    sigma_g: float = 0.15,
    sigma_b: float = 0.30,
    rho: float = 0.7,
    r: float = 0.02,
    T: float = 10.0,
    n_grid: int = 150,
) -> dict:
    """
    Solve for the initial allocation at a single (eta, epsilon) point.

    Returns dict with pi_g, pi_b, xi_0, theta_0 or NaN if failed.
    """
    # Admissibility check (power utility requires eps > eta - 1 and eps != 0)
    if epsilon <= eta - 1.0 or epsilon <= 0 or eta <= 0 or abs(eta - 1.0) < 1e-6:
        return {"pi_g": np.nan, "pi_b": np.nan, "xi_0": np.nan, "theta_0": np.nan}

    try:
        params = build_two_asset_params(
            mu_g=mu_g, mu_b=mu_b, sigma_g=sigma_g, sigma_b=sigma_b, rho=rho,
            kappa_g=kappa_g, kappa_b=kappa_b, r=r, lam=lam, eta=eta,
            epsilon=epsilon, T=T, n_grid=n_grid, utility="power",
        )
        sol = solve_pretransition(params)
        pi0_0 = sol["pi0_star"][-1]
        return {
            "pi_g": float(pi0_0[0]),
            "pi_b": float(pi0_0[1]),
            "xi_0": float(sol["xi"][-1]),
            "theta_0": float(sol["theta"][-1]),
        }
    except Exception as e:
        return {"pi_g": np.nan, "pi_b": np.nan, "xi_0": np.nan, "theta_0": np.nan}


def main() -> None:
    """Run Experiment 4: Figure 3 surface plots."""
    print("=" * 70)
    print("EXPERIMENT 4: FIGURE 3 SURFACES OVER (eta, epsilon)")
    print("=" * 70)

    # Define grid — use coarser grid for speed
    eta_vals = np.linspace(0.3, 5.0, 20)
    eps_vals = np.linspace(0.1, 6.0, 20)

    n_eta = len(eta_vals)
    n_eps = len(eps_vals)

    pi_g_surface = np.full((n_eta, n_eps), np.nan)
    pi_b_surface = np.full((n_eta, n_eps), np.nan)
    valid_count = 0

    print(f"\nGrid: eta x epsilon = {n_eta} x {n_eps} = {n_eta*n_eps} points")
    print("Computing surface (masked invalid region eps <= eta - 1)...\n")

    for i, eta in enumerate(eta_vals):
        for j, eps in enumerate(eps_vals):
            res = solve_surface_point(eta=eta, epsilon=eps)
            if not np.isnan(res["pi_g"]):
                pi_g_surface[i, j] = res["pi_g"]
                pi_b_surface[i, j] = res["pi_b"]
                valid_count += 1

        print(f"  eta={eta:.2f} complete")

    print(f"\nTotal valid points: {valid_count} / {n_eta * n_eps}")
    print(f"Green range: [{np.nanmin(pi_g_surface):.4f}, {np.nanmax(pi_g_surface):.4f}]")
    print(f"Brown range: [{np.nanmin(pi_b_surface):.4f}, {np.nanmax(pi_b_surface):.4f}]")

    # Generate surface plots
    plot_path = plot_surface(
        eta_grid=eta_vals,
        eps_grid=eps_vals,
        pi_g_surface=pi_g_surface,
        pi_b_surface=pi_b_surface,
        filename="exp4_figure3_surface.png",
        title="Figure 3: pi_0*(0) over (eta, epsilon) — lambda=0.1, kappa=(0.2,-0.5)",
    )
    print(f"\nSurface plot saved to {plot_path}")

    # Slices at fixed eta
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    eta_slices = [1.5, 2.0, 3.0, 4.0]
    colors = ["blue", "green", "orange", "red"]

    for ax, label in zip(axes, ["Green", "Brown"]):
        for eta_v, color in zip(eta_slices, colors):
            idx = np.argmin(np.abs(eta_vals - eta_v))
            surface = pi_g_surface if label == "Green" else pi_b_surface
            vals = surface[idx, :]
            valid = ~np.isnan(vals)
            ax.plot(eps_vals[valid], vals[valid], color=color, lw=2,
                    label=f"eta={eta_vals[idx]:.1f}")
        ax.set_xlabel(r"$\epsilon$ (ambiguity aversion)")
        ax.set_ylabel(f"$\\pi_{{0,{label}}}^*(0)$")
        ax.set_title(f"{label} allocation vs. epsilon (fixed eta slices)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 3 slices: Fixed eta, varying epsilon", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp4_figure3_slices.png", dpi=150)
    plt.close(fig)
    print(f"Slice plot saved to {RESULTS_DIR}/exp4_figure3_slices.png")

    # Save results summary (avoid saving full NaN arrays in JSON)
    results = {
        "eta_grid": eta_vals.tolist(),
        "eps_grid": eps_vals.tolist(),
        "valid_count": valid_count,
        "total_points": n_eta * n_eps,
        "pi_g_range": [float(np.nanmin(pi_g_surface)), float(np.nanmax(pi_g_surface))],
        "pi_b_range": [float(np.nanmin(pi_b_surface)), float(np.nanmax(pi_b_surface))],
    }
    with open(f"{RESULTS_DIR}/exp4_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}/exp4_results.json")

    print("\n=== EXPERIMENT 4 SUMMARY ===")
    print(f"Valid grid points : {valid_count} / {n_eta * n_eps}")
    print(f"Green surface range: {np.nanmin(pi_g_surface):.4f} to {np.nanmax(pi_g_surface):.4f}")
    print(f"Brown surface range: {np.nanmin(pi_b_surface):.4f} to {np.nanmax(pi_b_surface):.4f}")
    print("Qualitative check: increasing eta should reduce risky allocations (power CRRA).")


if __name__ == "__main__":
    main()
