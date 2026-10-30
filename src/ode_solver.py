"""Backward ODE solver for f(t) in the pre-transition robust portfolio problem.

The pre-transition value function uses the ansatz V_0(t,x,z) proportional to
exp((1-eta)*f(t)) for power utility, or directly adds f(t) for log utility.

The ODE for f(t) (power utility) is:
  f'(t) = -r - lambda/eps - pi_0*(t)^T (mu_0 - r*1)
          + (eta/2) pi_0*(t)^T Sigma_0 pi_0*(t) + (lambda/eps) * theta*(t)
  f(T) = 0

We integrate backward from T to 0 via the substitution s = T - t
(so s goes from 0 to T forward in time), setting g(s) = f(T-s).

# Custom — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
from typing import Dict, List, Tuple
from .parameters import ModelParams
from .merton import compute_merton_quantities
from .root_solver import (
    solve_xi,
    build_AB_power,
    build_AB_log,
    compute_controls_power,
    compute_controls_log,
)


def _ode_rhs_g_power(
    s: float,
    g_val: float,
    params: ModelParams,
    mq: dict,
) -> Tuple[float, float, np.ndarray, float]:
    """
    Evaluate dg/ds = -f'(T-s) for power utility.

    tau = s (time-to-go).
    g_val = f(T-s) = f at time t = T - s.

    Returns
    -------
    dgds : float
        Derivative of g at s.
    xi_star : float
        Optimal xi at this time point.
    pi0_star : np.ndarray
        Optimal portfolio at this time point.
    theta_star : float
        Optimal distortion at this time point.
    """
    tau = s  # time-to-go = T - t = s
    A, B = build_AB_power(
        pi0_merton=mq["pi0_merton"],
        kappa=params.kappa,
        lam=params.lam,
        eta=params.eta,
        alpha1_eta=mq["alpha1"],
        tau=tau,
        f_t=g_val,
        eps=params.epsilon,
        kappa_quad=mq["kappa_Sigma0_inv_kappa"],
    )
    xi_star, _ = solve_xi(A, B, params.epsilon)
    pi0_star, theta_star = compute_controls_power(
        xi_star=xi_star,
        pi0_merton=mq["pi0_merton"],
        kappa=params.kappa,
        Sigma0_inv_kappa=mq["Sigma0_inv_kappa"],
        kappa_quad=mq["kappa_Sigma0_inv_kappa"],
        lam=params.lam,
        eta=params.eta,
        alpha1_eta=mq["alpha1"],
        tau=tau,
        f_t=g_val,
        eps=params.epsilon,
    )

    excess_0 = params.mu_0 - params.r * np.ones(params.m)
    # f'(t) = -r - lambda/eps - pi0*(t).excess_0 + (eta/2) pi0*.Sigma0.pi0* + (lambda/eps)*theta*
    fp = (
        -params.r
        - params.lam / params.epsilon
        - np.dot(pi0_star, excess_0)
        + (params.eta / 2.0) * np.dot(pi0_star, params.Sigma_0 @ pi0_star)
        + (params.lam / params.epsilon) * theta_star
    )
    # dg/ds = -f'(T-s) = -fp
    dgds = -fp
    return dgds, xi_star, pi0_star, theta_star


def _ode_rhs_g_log(
    s: float,
    g_val: float,
    params: ModelParams,
    mq: dict,
) -> Tuple[float, float, np.ndarray, float]:
    """
    Evaluate dg/ds = -f'(T-s) for log utility.

    Parameters same as power version.
    """
    tau = s
    A, B = build_AB_log(
        pi0_merton_log=mq["pi0_merton"],
        kappa=params.kappa,
        lam=params.lam,
        alpha1_log=mq["alpha1"],
        tau=tau,
        f_t=g_val,
        eps=params.epsilon,
        kappa_quad=mq["kappa_Sigma0_inv_kappa"],
    )
    xi_star, _ = solve_xi(A, B, params.epsilon)
    pi0_star, theta_star = compute_controls_log(
        xi_star=xi_star,
        pi0_merton_log=mq["pi0_merton"],
        kappa=params.kappa,
        Sigma0_inv_kappa=mq["Sigma0_inv_kappa"],
        kappa_quad=mq["kappa_Sigma0_inv_kappa"],
        lam=params.lam,
    )

    excess_0 = params.mu_0 - params.r * np.ones(params.m)
    fp = (
        -params.r
        - params.lam / params.epsilon
        - np.dot(pi0_star, excess_0)
        + 0.5 * np.dot(pi0_star, params.Sigma_0 @ pi0_star)
        + (params.lam / params.epsilon) * theta_star
    )
    dgds = -fp
    return dgds, xi_star, pi0_star, theta_star


def solve_pretransition(
    params: ModelParams,
) -> Dict[str, np.ndarray]:
    """
    Solve the full pre-transition optimal control problem over [0, T].

    Uses a custom backward time-stepping loop (RK4 in forward s = T-t time)
    where each RHS evaluation requires a scalar xi* root solve.

    Parameters
    ----------
    params : ModelParams
        Full model parameters.

    Returns
    -------
    dict with keys:
        t_grid    : time grid on [0, T], shape (n,)
        f         : f(t) values, shape (n,)
        xi        : xi*(t) values, shape (n,)
        pi0_star  : pi_0*(t) matrix, shape (n, m)
        theta     : theta*(t) values, shape (n,)
        root_residuals : residuals from xi* root solve at each time, shape (n,)
    """
    mq = compute_merton_quantities(params)
    n = params.n_grid
    T = params.T

    # s-grid: s = T - t, so s=0 corresponds to t=T, s=T corresponds to t=0
    s_grid = np.linspace(0.0, T, n)
    ds = s_grid[1] - s_grid[0]

    g_arr = np.zeros(n)     # g(s) = f(T - s), so g(0) = f(T) = 0
    xi_arr = np.zeros(n)
    pi0_arr = np.zeros((n, params.m))
    theta_arr = np.zeros(n)
    residuals = np.zeros(n)

    # At s=0 (t=T): special case — use limiting value as tau->0
    # At tau=0, xi* -> 1 + pi0_merton . kappa (from the root equation in the limit B->0)
    # We handle s=0 by using a tiny positive tau
    def rhs_and_aux(s_val, g_val):
        if params.utility == "power":
            return _ode_rhs_g_power(s_val, g_val, params, mq)
        else:
            return _ode_rhs_g_log(s_val, g_val, params, mq)

    # Compute at s=0: use very small tau to avoid division issues
    eps_tau = 1e-8
    dgds0, xi0, pi0_0, th0 = rhs_and_aux(eps_tau, 0.0)
    g_arr[0] = 0.0
    xi_arr[0] = xi0
    pi0_arr[0] = pi0_0
    theta_arr[0] = th0

    # Validate root at s=0
    if params.utility == "power":
        A0, B0 = build_AB_power(
            mq["pi0_merton"], params.kappa, params.lam, params.eta,
            mq["alpha1"], eps_tau, 0.0, params.epsilon, mq["kappa_Sigma0_inv_kappa"]
        )
    else:
        A0, B0 = build_AB_log(
            mq["pi0_merton"], params.kappa, params.lam,
            mq["alpha1"], eps_tau, 0.0, params.epsilon, mq["kappa_Sigma0_inv_kappa"]
        )
    from .root_solver import _h_xi
    residuals[0] = abs(_h_xi(xi0, A0, B0, params.epsilon))

    # RK4 integration forward in s
    for i in range(n - 1):
        s_i = s_grid[i]
        g_i = g_arr[i]

        # k1
        k1, xi_i, pi0_i, th_i = rhs_and_aux(s_i, g_i)

        # k2
        k2, _, _, _ = rhs_and_aux(s_i + ds / 2, g_i + ds / 2 * k1)

        # k3
        k3, _, _, _ = rhs_and_aux(s_i + ds / 2, g_i + ds / 2 * k2)

        # k4
        k4, _, _, _ = rhs_and_aux(s_i + ds, g_i + ds * k3)

        g_arr[i + 1] = g_i + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Store auxiliary at grid point i (already computed k1)
        xi_arr[i] = xi_i
        pi0_arr[i] = pi0_i
        theta_arr[i] = th_i

        # Root residual
        if params.utility == "power":
            A_i, B_i = build_AB_power(
                mq["pi0_merton"], params.kappa, params.lam, params.eta,
                mq["alpha1"], s_i + eps_tau if s_i == 0 else s_i,
                g_i, params.epsilon, mq["kappa_Sigma0_inv_kappa"]
            )
        else:
            A_i, B_i = build_AB_log(
                mq["pi0_merton"], params.kappa, params.lam,
                mq["alpha1"], s_i + eps_tau if s_i == 0 else s_i,
                g_i, params.epsilon, mq["kappa_Sigma0_inv_kappa"]
            )
        residuals[i] = abs(_h_xi(xi_i, A_i, B_i, params.epsilon))

    # Fill last grid point (s = T, t = 0)
    s_N = s_grid[-1]
    g_N = g_arr[-1]
    _, xi_N, pi0_N, th_N = rhs_and_aux(s_N, g_N)
    xi_arr[-1] = xi_N
    pi0_arr[-1] = pi0_N
    theta_arr[-1] = th_N
    if params.utility == "power":
        A_N, B_N = build_AB_power(
            mq["pi0_merton"], params.kappa, params.lam, params.eta,
            mq["alpha1"], s_N, g_N, params.epsilon, mq["kappa_Sigma0_inv_kappa"]
        )
    else:
        A_N, B_N = build_AB_log(
            mq["pi0_merton"], params.kappa, params.lam,
            mq["alpha1"], s_N, g_N, params.epsilon, mq["kappa_Sigma0_inv_kappa"]
        )
    residuals[-1] = abs(_h_xi(xi_N, A_N, B_N, params.epsilon))

    # Convert back to physical time: t_grid = T - s_grid (reversed)
    t_grid = T - s_grid[::-1]   # t=0 ... T
    f_arr = g_arr[::-1]          # f(t=0) ... f(t=T)
    xi_out = xi_arr[::-1]
    pi0_out = pi0_arr[::-1]
    theta_out = theta_arr[::-1]
    res_out = residuals[::-1]

    return {
        "t_grid": t_grid,
        "f": f_arr,
        "xi": xi_out,
        "pi0_star": pi0_out,
        "theta": theta_out,
        "root_residuals": res_out,
        "merton_quantities": mq,
    }


def solve_posttransition(params: ModelParams) -> dict:
    """
    Return the closed-form post-transition optimal portfolio and growth rate.

    For power utility:
      pi_1* = pi_1^{M,eta}
      alpha_1^eta = r + (1/(2 eta)) (mu_1 - r)^T Sigma_1^{-1} (mu_1 - r)

    For log utility:
      pi_1* = pi_1^M
      alpha_1 = r + (1/2)(mu_1 - r)^T Sigma_1^{-1}(mu_1 - r)

    Parameters
    ----------
    params : ModelParams

    Returns
    -------
    dict with keys: pi1_star, alpha1
    """
    mq = compute_merton_quantities(params)
    return {
        "pi1_star": mq["pi1_merton"],
        "alpha1": mq["alpha1"],
    }
