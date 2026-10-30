"""Plotting utilities for the robust transition-timing portfolio model.

All plots are saved to the results/ directory. Uses Matplotlib with a
consistent styling convention across experiments.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from typing import List, Optional, Dict

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> str:
    """Save figure to results directory."""
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_time_profiles(
    t_grid: np.ndarray,
    pi0_arr_list: List[np.ndarray],
    theta_arr_list: List[np.ndarray],
    labels: List[str],
    pi_merton: np.ndarray,
    asset_names: List[str],
    pi_wc: Optional[np.ndarray] = None,
    title: str = "",
    filename: str = "time_profiles.png",
    colors: Optional[List] = None,
) -> str:
    """
    Plot time profiles of pi_0*(t) per asset and theta*(t).

    Parameters
    ----------
    t_grid : np.ndarray
        Time grid on [0, T].
    pi0_arr_list : list of np.ndarray, each shape (n, m)
        Portfolio paths for each epsilon/scenario.
    theta_arr_list : list of np.ndarray, each shape (n,)
        Distortion paths.
    labels : list of str
        Legend labels for each scenario.
    pi_merton : np.ndarray, shape (m,)
        Transition-unaware Merton benchmark (constant).
    asset_names : list of str
        Names for assets, e.g. ['Green', 'Brown'].
    pi_wc : np.ndarray, shape (n, m), optional
        Worst-case portfolio path.
    title : str
        Overall figure title.
    filename : str
        Output filename.
    colors : list, optional
        Color sequence for lines.

    Returns
    -------
    str
        Path to saved figure.
    """
    m = len(asset_names)
    n_rows = m + 1  # one per asset + theta
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3 * n_rows), sharex=True)

    if colors is None:
        colors = cm.viridis(np.linspace(0.1, 0.9, len(labels)))

    for j, (pi_arr, theta_arr, label, color) in enumerate(
        zip(pi0_arr_list, theta_arr_list, labels, colors)
    ):
        for asset_idx in range(m):
            axes[asset_idx].plot(
                t_grid, pi_arr[:, asset_idx], label=label, color=color, lw=1.8
            )
        axes[m].plot(t_grid, theta_arr, label=label, color=color, lw=1.8)

    # Merton horizontal lines and worst-case
    for asset_idx in range(m):
        axes[asset_idx].axhline(
            pi_merton[asset_idx],
            color="black",
            linestyle="--",
            lw=2,
            label="Merton",
        )
        if pi_wc is not None:
            axes[asset_idx].plot(
                t_grid, pi_wc[:, asset_idx],
                color="red", linestyle="-.", lw=2, label="Worst-case"
            )
        axes[asset_idx].set_ylabel(f"$\\pi_{{0,{asset_names[asset_idx]}}}^*(t)$")
        axes[asset_idx].legend(fontsize=7, ncol=3)
        axes[asset_idx].grid(True, alpha=0.3)

    axes[m].axhline(1.0, color="gray", linestyle=":", lw=1.5, label="theta=1")
    axes[m].set_ylabel(r"$\theta^*(t)$")
    axes[m].set_xlabel("Time $t$")
    axes[m].legend(fontsize=7, ncol=3)
    axes[m].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


def plot_sweep_vs_param(
    param_grid: np.ndarray,
    pi_g_vals: np.ndarray,
    pi_b_vals: np.ndarray,
    pi_merton: np.ndarray,
    param_name: str,
    title: str = "",
    filename: str = "sweep.png",
) -> str:
    """
    Plot initial allocations pi_0*(0) against a single swept parameter.

    Parameters
    ----------
    param_grid : np.ndarray
        Values of the swept parameter.
    pi_g_vals, pi_b_vals : np.ndarray
        Optimal initial allocations for green and brown.
    pi_merton : np.ndarray, shape (2,)
        Merton benchmark weights.
    param_name : str
        Name of swept parameter for x-axis.
    title, filename : str

    Returns
    -------
    str
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, vals, name, merton_val in zip(
        axes,
        [pi_g_vals, pi_b_vals],
        ["Green", "Brown"],
        [pi_merton[0], pi_merton[1]],
    ):
        ax.plot(param_grid, vals, "b-o", markersize=4, lw=2, label=f"Robust $\\pi_{{0,{name}}}^*(0)$")
        ax.axhline(merton_val, color="red", linestyle="--", lw=2, label=f"Merton")
        ax.set_xlabel(param_name)
        ax.set_ylabel(f"$\\pi_{{0,{name}}}^*(0)$")
        ax.set_title(f"{name} allocation vs {param_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


def plot_surface(
    eta_grid: np.ndarray,
    eps_grid: np.ndarray,
    pi_g_surface: np.ndarray,
    pi_b_surface: np.ndarray,
    filename: str = "surface.png",
    title: str = "Portfolio Surface over (eta, epsilon)",
) -> str:
    """
    Plot 2D surface/heatmap of initial allocations over (eta, epsilon) grid.

    Parameters
    ----------
    eta_grid : np.ndarray, shape (n_eta,)
    eps_grid : np.ndarray, shape (n_eps,)
    pi_g_surface : np.ndarray, shape (n_eta, n_eps)
        Green allocation at t=0.
    pi_b_surface : np.ndarray, shape (n_eta, n_eps)
        Brown allocation at t=0.
    filename, title : str

    Returns
    -------
    str
        Path to saved figure.
    """
    ETA, EPS = np.meshgrid(eta_grid, eps_grid, indexing="ij")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, surface, label in zip(
        axes,
        [pi_g_surface, pi_b_surface],
        ["Green $\\pi_{0,g}^*(0)$", "Brown $\\pi_{0,b}^*(0)$"],
    ):
        im = ax.contourf(ETA, EPS, surface, levels=30, cmap="RdBu_r")
        ax.contour(ETA, EPS, surface, levels=10, colors="k", alpha=0.3, linewidths=0.5)

        # Overlay boundary epsilon = eta - 1
        eps_boundary = eta_grid - 1.0
        valid = eps_boundary > 0
        if valid.any():
            ax.plot(
                eta_grid[valid], eps_boundary[valid],
                "k--", lw=2, label="$\\epsilon = \\eta - 1$"
            )

        plt.colorbar(im, ax=ax, label=label)
        ax.set_xlabel(r"$\eta$ (risk aversion)")
        ax.set_ylabel(r"$\epsilon$ (ambiguity aversion)")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return _save(fig, filename)


def plot_sign_conditions(
    rho_grid: np.ndarray,
    sigma_inv_kappa_arr: np.ndarray,
    green_threshold: float,
    brown_threshold: float,
    filename: str = "sign_conditions.png",
) -> str:
    """
    Plot Sigma^{-1}*kappa components vs correlation rho.

    Parameters
    ----------
    rho_grid : np.ndarray
    sigma_inv_kappa_arr : np.ndarray, shape (n, 2)
    green_threshold, brown_threshold : float
    filename : str

    Returns
    -------
    str
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(rho_grid, sigma_inv_kappa_arr[:, 0], "g-", lw=2, label="Green component")
    axes[0].axhline(0, color="k", lw=1)
    axes[0].axvline(green_threshold, color="r", linestyle="--", lw=1.5,
                    label=f"$\\rho$ threshold = {green_threshold:.3f}")
    axes[0].set_xlabel(r"$\rho$ (correlation)")
    axes[0].set_ylabel(r"$(\Sigma_0^{-1}\kappa)_g$")
    axes[0].set_title("Green hedging direction vs correlation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rho_grid, sigma_inv_kappa_arr[:, 1], "brown", lw=2, label="Brown component")
    axes[1].axhline(0, color="k", lw=1)
    axes[1].axvline(brown_threshold, color="r", linestyle="--", lw=1.5,
                    label=f"$\\rho$ threshold = {brown_threshold:.3f}")
    axes[1].set_xlabel(r"$\rho$ (correlation)")
    axes[1].set_ylabel(r"$(\Sigma_0^{-1}\kappa)_b$")
    axes[1].set_title("Brown hedging direction vs correlation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Sign conditions: Sigma_0^{-1} kappa components vs correlation", fontsize=12)
    fig.tight_layout()
    return _save(fig, filename)


def plot_verification_diagnostics(
    solution: dict,
    params,
    filename: str = "verification_diagnostics.png",
) -> str:
    """
    Plot diagnostic time series: xi*(t), theta*(t), f(t), and identity check.

    Parameters
    ----------
    solution : dict
        Output from solve_pretransition.
    params : ModelParams
    filename : str

    Returns
    -------
    str
    """
    t = solution["t_grid"]
    xi = solution["xi"]
    theta = solution["theta"]
    f = solution["f"]
    pi0 = solution["pi0_star"]
    kappa = params.kappa

    identity = 1.0 + pi0 @ kappa - xi

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, f, "b-", lw=2)
    axes[0, 0].set_title("$f(t)$")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(t, xi, "g-", lw=2)
    axes[0, 1].axhline(0, color="k", lw=1, linestyle="--")
    axes[0, 1].set_title(r"$\xi^*(t)$")
    axes[0, 1].set_xlabel("t")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(t, theta, "r-", lw=2)
    axes[1, 0].axhline(0, color="k", lw=1, linestyle="--")
    axes[1, 0].axhline(1.0, color="gray", lw=1, linestyle=":")
    axes[1, 0].set_title(r"$\theta^*(t)$")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(t, np.abs(identity) + 1e-20, "purple", lw=2)
    axes[1, 1].set_title(r"$|1 + \pi_0^*(t)^T\kappa - \xi^*(t)|$")
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("Absolute error")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        f"Verification diagnostics — {params.utility} utility, eta={params.eta}, eps={params.epsilon}",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, filename)
