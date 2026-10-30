"""Ambiguity-neutral (epsilon -> eta-1) benchmark solver.

When epsilon = eta - 1 (i.e., epsilon -> eta - 1 from above for power utility),
the robust problem converges to the ambiguity-neutral case. The ambiguity-
neutral power-utility formulas involve a different ODE for xi*(t):

  0 = -xi^(eta+1) + [1 + (pi_0^{M,eta})^T kappa] xi^eta
      + (lambda/eta) exp((1-eta)[alpha_1^eta (T-t) - f(t)]) kappa^T Sigma_0^{-1} kappa

  f'(t) = -r + lambda/(1-eta) - pi_0*(t)^T(mu_0 - r*1) + (eta/2) pi_0*(t)^T Sigma_0 pi_0*(t)
          - [lambda/(1-eta)] [xi*(t)]^(1-eta) exp((1-eta)[alpha_1^eta(T-t) - f(t)])

For the log-utility ambiguity-neutral case (eta = 1), the paper gives a
closed-form for xi*:
  xi* = 0.5 * A + 0.5 * sqrt(A^2 + 4*lambda*kappa^T Sigma_0^{-1} kappa)
where A = 1 + (pi_0^M)^T kappa.

# Custom — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
from scipy.optimize import brentq
from typing import Tuple, Dict
from .parameters import ModelParams
from .merton import compute_merton_quantities
from .root_solver import _h_xi


def _xi_ambiguity_neutral_power(
    A: float,
    B_an: float,
    eta: float,
    tol: float = 1e-10,
) -> Tuple[float, float]:
    """
    Find xi* for the ambiguity-neutral power-utility case.

    Root of: h(xi) = -xi^(eta+1) + A*xi^eta + B_an = 0

    Parameters
    ----------
    A : float
        1 + pi_0^{M,eta} . kappa
    B_an : float
        (lambda/eta) * exp((1-eta)*[alpha_1^eta * tau - f_t]) * kappa_quad
    eta : float
        Risk aversion.
    tol : float
        Root tolerance.

    Returns
    -------
    xi_star, residual : floats
    """
    def h(xi):
        return -(xi ** (eta + 1)) + A * (xi ** eta) + B_an

    # h(0+) = B_an > 0, h(large) -> -inf
    high = max(abs(A) + 1.0, 1.0)
    for _ in range(100):
        if h(high) < 0:
            break
        high *= 2.0
    xi_star = brentq(h, 1e-12, high, xtol=tol, rtol=tol)
    return float(xi_star), abs(h(xi_star))


def xi_closed_form_log_neutral(
    A: float,
    kappa_quad: float,
    lam: float,
) -> float:
    """
    Closed-form xi* for the log-utility ambiguity-neutral case.

    xi* = 0.5*A + 0.5*sqrt(A^2 + 4*lambda*kappa_quad)

    Parameters
    ----------
    A : float
        1 + pi_0^M . kappa
    kappa_quad : float
        kappa^T Sigma_0^{-1} kappa
    lam : float
        Transition intensity.

    Returns
    -------
    float
        Closed-form xi*.
    """
    disc = A**2 + 4.0 * lam * kappa_quad
    assert disc >= 0, "Discriminant must be non-negative."
    return 0.5 * A + 0.5 * np.sqrt(disc)


def solve_ambiguity_neutral_power(params: ModelParams) -> Dict[str, np.ndarray]:
    """
    Solve the ambiguity-neutral (no ambiguity aversion) power-utility problem.

    This uses the ambiugity-neutral (AN) ODE which has a different xi* root
    equation than the robust case. The epsilon parameter is ignored; instead
    this corresponds formally to the eps -> eta-1 limit.

    Parameters
    ----------
    params : ModelParams
        Must have utility='power'.

    Returns
    -------
    dict with t_grid, f, xi, pi0_star, root_residuals.
    """
    assert params.utility == "power", "This solver requires power utility."
    mq = compute_merton_quantities(params)
    pi0_merton = mq["pi0_merton"]
    alpha1 = mq["alpha1"]
    kappa_quad = mq["kappa_Sigma0_inv_kappa"]
    Sigma0_inv_kappa = mq["Sigma0_inv_kappa"]
    T = params.T
    n = params.n_grid
    eta = params.eta
    r = params.r
    lam = params.lam

    A = 1.0 + float(np.dot(pi0_merton, params.kappa))

    s_grid = np.linspace(0.0, T, n)
    ds = s_grid[1] - s_grid[0]

    g_arr = np.zeros(n)
    xi_arr = np.zeros(n)
    pi0_arr = np.zeros((n, params.m))
    res_arr = np.zeros(n)

    def compute_at(s, g_val):
        tau = max(s, 1e-8)
        exp_val = np.exp((1.0 - eta) * (alpha1 * tau - g_val))
        B_an = (lam / eta) * exp_val * kappa_quad
        xi_star, resid = _xi_ambiguity_neutral_power(A, B_an, eta)
        c = (xi_star - A) / kappa_quad  # (xi* - 1 - pi0.kappa) / kappa_quad
        pi0_star = pi0_merton + c * Sigma0_inv_kappa

        excess = params.mu_0 - r * np.ones(params.m)
        xi_pw = xi_star ** (1.0 - eta)
        fp = (
            -r
            + lam / (1.0 - eta)
            - np.dot(pi0_star, excess)
            + (eta / 2.0) * np.dot(pi0_star, params.Sigma_0 @ pi0_star)
            - (lam / (1.0 - eta)) * xi_pw * exp_val
        )
        return -fp, xi_star, pi0_star, resid

    _, xi0, pi0_0, res0 = compute_at(1e-8, 0.0)
    xi_arr[0] = xi0
    pi0_arr[0] = pi0_0
    res_arr[0] = res0

    for i in range(n - 1):
        s_i = s_grid[i]
        g_i = g_arr[i]
        k1, xi_i, pi0_i, res_i = compute_at(s_i, g_i)
        k2, _, _, _ = compute_at(s_i + ds / 2, g_i + ds / 2 * k1)
        k3, _, _, _ = compute_at(s_i + ds / 2, g_i + ds / 2 * k2)
        k4, _, _, _ = compute_at(s_i + ds, g_i + ds * k3)
        g_arr[i + 1] = g_i + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        xi_arr[i] = xi_i
        pi0_arr[i] = pi0_i
        res_arr[i] = res_i

    _, xi_N, pi0_N, res_N = compute_at(s_grid[-1], g_arr[-1])
    xi_arr[-1] = xi_N
    pi0_arr[-1] = pi0_N
    res_arr[-1] = res_N

    t_grid = T - s_grid[::-1]
    return {
        "t_grid": t_grid,
        "f": g_arr[::-1],
        "xi": xi_arr[::-1],
        "pi0_star": pi0_arr[::-1],
        "root_residuals": res_arr[::-1],
        "merton_quantities": mq,
    }


def solve_ambiguity_neutral_log(params: ModelParams) -> Dict[str, np.ndarray]:
    """
    Solve the ambiguity-neutral log-utility problem using the closed-form xi*.

    xi* is constant (time-independent) since it equals:
      xi* = 0.5*A + 0.5*sqrt(A^2 + 4*lambda*kappa_quad)

    Parameters
    ----------
    params : ModelParams
        Must have utility='log' (or eta ~= 1).

    Returns
    -------
    dict with t_grid, f, xi, pi0_star.
    """
    mq = compute_merton_quantities(params)
    pi0_merton = mq["pi0_merton"]
    kappa_quad = mq["kappa_Sigma0_inv_kappa"]
    Sigma0_inv_kappa = mq["Sigma0_inv_kappa"]
    T = params.T
    n = params.n_grid

    A = 1.0 + float(np.dot(pi0_merton, params.kappa))
    xi_star = xi_closed_form_log_neutral(A, kappa_quad, params.lam)

    c = (xi_star - A) / kappa_quad
    pi0_star = pi0_merton + c * Sigma0_inv_kappa

    # Constant pi0_star -> f'(t) is constant -> f(t) = f'(t)*(t - T)
    excess = params.mu_0 - params.r * np.ones(params.m)
    # For log ambiguity-neutral, theta* = (1/lambda)*xi**(c/kappa_quad * kappa_quad)... 
    # Actually theta* in AN log case: see paper. For the constant case:
    # We use a simple backward integration.
    fp = (
        -params.r
        - params.lam / params.epsilon
        - np.dot(pi0_star, excess)
        + 0.5 * np.dot(pi0_star, params.Sigma_0 @ pi0_star)
        + (params.lam / params.epsilon) * (1.0 / params.lam) * xi_star * c
    )

    t_grid = np.linspace(0.0, T, n)
    # f(T) = 0, f'(t) = fp (constant), so f(t) = fp * (t - T)
    f_arr = fp * (t_grid - T)

    pi0_arr = np.tile(pi0_star, (n, 1))
    xi_arr = np.full(n, xi_star)

    return {
        "t_grid": t_grid,
        "f": f_arr,
        "xi": xi_arr,
        "pi0_star": pi0_arr,
        "merton_quantities": mq,
    }
