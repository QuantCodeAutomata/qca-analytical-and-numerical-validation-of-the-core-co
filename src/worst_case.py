"""Worst-case benchmark solver for the epsilon -> infinity limit.

When mu_1 = mu_0 and Sigma_1 = Sigma_0 (unchanged market coefficients
after transition), the worst-case limit is characterized by:

  If (pi_0^{M,eta})^T kappa < 0 (negative-sign regime):
    Solve zeta'(t) = (eta/2) * zeta(t) * (zeta(t) - 1 - (pi_0^{M,eta})^T kappa)^2
                     / (kappa^T Sigma_0^{-1} kappa)
    zeta(T) = 1
    pi_0^{wc}(t) = pi_0^{M,eta} + c(zeta) * Sigma_0^{-1} kappa
    where c(zeta) = (zeta(t) - 1 - (pi_0^{M,eta})^T kappa) / (kappa^T Sigma_0^{-1} kappa)

  If (pi_0^{M,eta})^T kappa >= 0 (positive-sign regime):
    pi_0^{wc}(t) = pi_0^{M,eta} (worst case = Merton)

# Custom — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Dict
from .parameters import ModelParams
from .merton import compute_merton_quantities


def solve_worst_case(params: ModelParams) -> Dict[str, np.ndarray]:
    """
    Solve the worst-case (epsilon -> infinity) portfolio problem.

    Parameters
    ----------
    params : ModelParams
        Model parameters. Assumes mu_0 = mu_1 and Sigma_0 = Sigma_1 for
        the standard worst-case formulas to apply.

    Returns
    -------
    dict with keys:
        t_grid, pi_wc, theta_wc, zeta (zeta = 1 in positive-sign regime),
        sign_regime, pi_merton_dot_kappa
    """
    mq = compute_merton_quantities(params)
    pi0_merton = mq["pi0_merton"]
    kappa_quad = mq["kappa_Sigma0_inv_kappa"]
    Sigma0_inv_kappa = mq["Sigma0_inv_kappa"]
    T = params.T

    pi0_dot_kappa = float(np.dot(pi0_merton, params.kappa))
    n = params.n_grid
    t_grid = np.linspace(0.0, T, n)

    if pi0_dot_kappa >= 0:
        # Positive-sign regime: worst-case = Merton
        pi_wc = np.tile(pi0_merton, (n, 1))
        zeta_arr = np.ones(n)
        theta_wc = np.zeros(n)  # distortion = 0 (neutral)
        return {
            "t_grid": t_grid,
            "pi_wc": pi_wc,
            "theta_wc": theta_wc,
            "zeta": zeta_arr,
            "sign_regime": "positive",
            "pi_merton_dot_kappa": pi0_dot_kappa,
        }

    # Negative-sign regime: solve zeta ODE backward from T
    # zeta'(t) = (eta/2) * zeta(t) * (zeta(t) - 1 - pi0.kappa)^2 / kappa_quad
    # zeta(T) = 1
    # Use substitution s = T - t, phi(s) = zeta(T - s), phi(0) = 1
    # dphi/ds = -(eta/2) * phi(s) * (phi(s) - 1 - pi0.kappa)^2 / kappa_quad

    b = 1.0 + pi0_dot_kappa  # b = 1 + pi0^M . kappa

    def zeta_ode(s, phi):
        c = (phi[0] - b) / kappa_quad
        dphi_ds = -(params.eta / 2.0) * phi[0] * ((phi[0] - b) ** 2) / kappa_quad
        return [dphi_ds]

    s_span = (0.0, T)
    s_eval = np.linspace(0.0, T, n)
    sol = solve_ivp(
        zeta_ode,
        s_span,
        [1.0],
        method="RK45",
        t_eval=s_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    # Convert back: t = T - s, zeta(t) = phi(T - t)
    zeta_s = sol.y[0]  # on s_eval = [0, T]
    zeta_t = zeta_s[::-1]  # now indexed by t = T - s, i.e., t from T to 0, reversed

    # Compute pi_wc(t) and theta_wc(t)
    pi_wc = np.zeros((n, params.m))
    theta_wc = np.zeros(n)

    for i in range(n):
        z = zeta_t[i]
        c = (z - b) / kappa_quad
        pi_wc[i] = pi0_merton + c * Sigma0_inv_kappa
        # theta_wc is lambda-normalized correction (not well-defined in limit, set to 0)
        theta_wc[i] = 0.0

    return {
        "t_grid": t_grid,
        "pi_wc": pi_wc,
        "theta_wc": theta_wc,
        "zeta": zeta_t,
        "sign_regime": "negative",
        "pi_merton_dot_kappa": pi0_dot_kappa,
    }
