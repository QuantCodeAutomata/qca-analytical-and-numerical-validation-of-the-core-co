"""
Experiment 8: Two-Asset Sign-Condition and Hedging-Direction Analysis.

Validates analytical sign conditions determining whether transition awareness
increases green allocation and reduces brown allocation relative to Merton.
"""

import sys
import json
import numpy as np

sys.path.insert(0, "/workspace/project")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.sign_conditions import (
    compute_sigma_inv_kappa,
    classify_reallocation_direction,
    sweep_sign_conditions,
    green_condition_threshold,
    brown_condition_threshold,
    build_two_asset_covariance,
)
from src.merton import merton_power
from src.plots import plot_sign_conditions

RESULTS_DIR = "/workspace/project/results"


def main() -> None:
    """Run Experiment 8: Sign condition analysis."""
    print("=" * 70)
    print("EXPERIMENT 8: SIGN CONDITIONS AND HEDGING DIRECTION ANALYSIS")
    print("=" * 70)

    # Baseline parameters
    sigma_g = 0.15
    sigma_b = 0.30
    rho = 0.7
    kappa_g = 0.2
    kappa_b = -0.5

    # --- 1. Analytical computation ---
    print("\n--- 1. Analytical Sigma^{-1}*kappa ---")
    sik = compute_sigma_inv_kappa(sigma_g, sigma_b, rho, kappa_g, kappa_b)
    print(f"Sigma_0^{{-1}} kappa = {sik}")
    print(f"  Green component: {sik[0]:.6f} -> {'POSITIVE (favor green) ✓' if sik[0] > 0 else 'NEGATIVE'}")
    print(f"  Brown component: {sik[1]:.6f} -> {'NEGATIVE (reduce brown) ✓' if sik[1] < 0 else 'POSITIVE'}")

    # Numerical validation
    Sigma = build_two_asset_covariance(sigma_g, sigma_b, rho)
    kappa = np.array([kappa_g, kappa_b])
    sik_numeric = np.linalg.solve(Sigma, kappa)
    print(f"\nNumerical Sigma^{{-1}} kappa = {sik_numeric}")
    print(f"Analytical vs numerical max err: {np.max(np.abs(sik - sik_numeric)):.2e}")
    assert np.allclose(sik, sik_numeric, atol=1e-10), "Analytical vs numerical mismatch!"
    print("Consistency check: PASS")

    # --- 2. Threshold analysis ---
    print("\n--- 2. Threshold rho values ---")
    rho_thresh_green = green_condition_threshold(sigma_g, sigma_b, kappa_g, kappa_b)
    rho_thresh_brown = brown_condition_threshold(sigma_g, sigma_b, kappa_g, kappa_b)
    print(f"Green condition threshold rho: {rho_thresh_green:.4f}")
    print(f"Brown condition threshold rho: {rho_thresh_brown:.4f}")
    print(f"Both conditions satisfied at rho={rho}: {rho > rho_thresh_green and rho > rho_thresh_brown}")
    print(f"  rho={rho} > green threshold {rho_thresh_green:.4f}: {rho > rho_thresh_green} ✓")
    print(f"  rho={rho} > brown threshold {rho_thresh_brown:.4f}: {rho > rho_thresh_brown} ✓")

    # --- 3. Verify conditions when rho > 0 ---
    print("\n--- 3. Positive correlation verification ---")
    sc = classify_reallocation_direction(sigma_g, sigma_b, rho, kappa_g, kappa_b)
    print(f"rho = {rho} > 0:")
    print(f"  Green component positive: {sc['green_positive']} {'✓' if sc['green_positive'] else '✗'}")
    print(f"  Brown component negative: {sc['brown_negative']} {'✓' if sc['brown_negative'] else '✗'}")
    print(f"  kappa^T Sigma^{{-1}} kappa = {sc['kappa_quad']:.6f}")

    # --- 4. Sweep over correlation rho ---
    print("\n--- 4. Correlation sweep ---")
    rho_grid = np.linspace(-0.99, 0.99, 200)
    sweep = sweep_sign_conditions(sigma_g, sigma_b, kappa_g, kappa_b, rho_grid)

    plot_path = plot_sign_conditions(
        rho_grid=rho_grid,
        sigma_inv_kappa_arr=sweep["sigma_inv_kappa_arr"],
        green_threshold=rho_thresh_green,
        brown_threshold=rho_thresh_brown,
        filename="exp8_sign_conditions_rho.png",
    )
    print(f"Correlation sweep plot saved to {plot_path}")

    # --- 5. Phase diagram over (kappa_g, kappa_b) ---
    print("\n--- 5. Phase diagram over (kappa_g, kappa_b) ---")
    kg_vals = np.linspace(0.01, 0.8, 40)
    kb_vals = np.linspace(-0.9, -0.01, 40)
    KG, KB = np.meshgrid(kg_vals, kb_vals)
    green_region = np.zeros_like(KG)
    brown_region = np.zeros_like(KG)
    both_region = np.zeros_like(KG)

    for i in range(len(kb_vals)):
        for j in range(len(kg_vals)):
            c = classify_reallocation_direction(sigma_g, sigma_b, rho, KG[i,j], KB[i,j])
            green_region[i,j] = 1.0 if c["green_positive"] else 0.0
            brown_region[i,j] = 1.0 if c["brown_negative"] else 0.0
            both_region[i,j] = 1.0 if (c["green_positive"] and c["brown_negative"]) else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    extent = [kg_vals[0], kg_vals[-1], kb_vals[0], kb_vals[-1]]

    axes[0].imshow(green_region, extent=extent, origin="lower", cmap="Greens",
                   aspect="auto", vmin=0, vmax=1)
    axes[0].set_xlabel("kappa_g (green jump)")
    axes[0].set_ylabel("kappa_b (brown jump)")
    axes[0].set_title("Green component positive\n(more green than Merton)")

    axes[1].imshow(brown_region, extent=extent, origin="lower", cmap="Oranges",
                   aspect="auto", vmin=0, vmax=1)
    axes[1].set_xlabel("kappa_g")
    axes[1].set_ylabel("kappa_b")
    axes[1].set_title("Brown component negative\n(less brown than Merton)")

    axes[2].imshow(both_region, extent=extent, origin="lower", cmap="Blues",
                   aspect="auto", vmin=0, vmax=1)
    axes[2].plot(kappa_g, kappa_b, "r*", markersize=15, label="Baseline")
    axes[2].set_xlabel("kappa_g")
    axes[2].set_ylabel("kappa_b")
    axes[2].set_title("Both conditions satisfied\n(green up, brown down)")
    axes[2].legend()

    fig.suptitle(f"Phase diagrams over (kappa_g, kappa_b) at rho={rho}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp8_phase_diagram.png", dpi=150)
    plt.close(fig)
    print(f"Phase diagram saved to {RESULTS_DIR}/exp8_phase_diagram.png")

    # --- 6. Sweep over volatility ratio sigma_b / sigma_g ---
    print("\n--- 6. Volatility ratio sweep ---")
    sig_ratio_vals = np.linspace(0.5, 4.0, 50)
    green_pos_ratio = []
    brown_neg_ratio = []
    for ratio in sig_ratio_vals:
        sb_v = sigma_g * ratio
        c = classify_reallocation_direction(sigma_g, sb_v, rho, kappa_g, kappa_b)
        green_pos_ratio.append(c["green_positive"])
        brown_neg_ratio.append(c["brown_negative"])

    print(f"  sigma_b/sigma_g range: green positive always? {all(green_pos_ratio)}")
    print(f"  sigma_b/sigma_g range: brown negative always? {all(brown_neg_ratio)}")

    # --- 7. Optional: connect to full dynamic model ---
    print("\n--- 7. Connection to full dynamic model (sign check) ---")
    from src.parameters import build_two_asset_params
    from src.ode_solver import solve_pretransition
    params = build_two_asset_params(
        kappa_g=kappa_g, kappa_b=kappa_b, rho=rho, epsilon=2.0, eta=2.0,
        n_grid=100, utility="power",
    )
    sol = solve_pretransition(params)
    pi0_0 = sol["pi0_star"][-1]  # initial allocation at t=0
    pi_m = merton_power(params.mu_0, params.r, Sigma, params.eta)
    correction = pi0_0 - pi_m
    print(f"  pi_0*(0) = {pi0_0}")
    print(f"  Merton   = {pi_m}")
    print(f"  Correction pi_0*(0) - Merton = {correction}")
    print(f"  Green correction > 0: {correction[0] > 0} {'✓' if correction[0] > 0 else '(check c(t) sign)'}")
    print(f"  Brown correction < 0: {correction[1] < 0} {'✓' if correction[1] < 0 else '(check c(t) sign)'}")

    # Save results
    results = {
        "sigma_inv_kappa_analytical": sik.tolist(),
        "sigma_inv_kappa_numerical": sik_numeric.tolist(),
        "consistency_pass": bool(np.allclose(sik, sik_numeric, atol=1e-10)),
        "green_threshold": rho_thresh_green,
        "brown_threshold": rho_thresh_brown,
        "baseline_rho": rho,
        "green_positive_at_baseline": sc["green_positive"],
        "brown_negative_at_baseline": sc["brown_negative"],
        "kappa_quad": sc["kappa_quad"],
        "dynamic_correction": correction.tolist(),
    }
    with open(f"{RESULTS_DIR}/exp8_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/exp8_results.json")

    print("\n=== EXPERIMENT 8 SUMMARY ===")
    print(f"Analytical = Numerical Sigma^{{-1}}kappa: PASS")
    print(f"Green component positive at baseline: {'PASS' if sc['green_positive'] else 'FAIL'}")
    print(f"Brown component negative at baseline: {'PASS' if sc['brown_negative'] else 'FAIL'}")
    print(f"rho > 0 both conditions met: {'PASS' if sc['green_positive'] and sc['brown_negative'] else 'FAIL'}")


if __name__ == "__main__":
    main()
