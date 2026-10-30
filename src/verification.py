"""Numerical verification and consistency checks for the pre-transition solution.

Validates:
1. Positive definiteness of covariance matrices.
2. xi*(t) > 0 for all t.
3. theta*(t) > 0 for all t.
4. Identity: 1 + pi_0*(t)^T kappa = xi*(t).
5. Root residuals near zero.
6. ODE residuals near zero.
7. Admissibility of epsilon: epsilon > eta - 1 for power utility.

# Custom — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
from typing import Dict, Tuple
from .parameters import ModelParams
from .merton import compute_merton_quantities
from .root_solver import build_AB_power, build_AB_log, _h_xi


def check_admissibility(params: ModelParams) -> Dict[str, bool]:
    """
    Check all admissibility conditions for model parameters.

    Parameters
    ----------
    params : ModelParams

    Returns
    -------
    dict
        Boolean flags for each condition.
    """
    results = {}

    # Positive definiteness
    try:
        np.linalg.cholesky(params.Sigma_0)
        results["Sigma0_pos_def"] = True
    except np.linalg.LinAlgError:
        results["Sigma0_pos_def"] = False

    try:
        np.linalg.cholesky(params.Sigma_1)
        results["Sigma1_pos_def"] = True
    except np.linalg.LinAlgError:
        results["Sigma1_pos_def"] = False

    # kappa admissibility
    results["kappa_admissible"] = bool(np.all(params.kappa > -1))

    # Epsilon constraint for power utility
    if params.utility == "power":
        results["epsilon_admissible"] = (
            params.epsilon > params.eta - 1 and params.epsilon > 0
        )
    else:
        results["epsilon_admissible"] = params.epsilon > 0

    return results


def verify_solution(
    params: ModelParams,
    solution: dict,
    tol_root: float = 1e-6,
    tol_identity: float = 1e-6,
) -> Dict[str, object]:
    """
    Run consistency checks on the solved pre-transition solution.

    Parameters
    ----------
    params : ModelParams
        Model parameters.
    solution : dict
        Output from ode_solver.solve_pretransition.
    tol_root : float
        Tolerance for root residuals.
    tol_identity : float
        Tolerance for 1 + pi_0*(t)^T kappa = xi*(t) identity.

    Returns
    -------
    dict with verification results and metrics.
    """
    mq = solution["merton_quantities"]
    xi = solution["xi"]
    pi0 = solution["pi0_star"]
    theta = solution["theta"]
    f = solution["f"]
    root_res = solution["root_residuals"]

    results = {}

    # 1. xi*(t) > 0
    xi_pos = bool(np.all(xi > 0))
    results["xi_all_positive"] = xi_pos   # primary (test-expected)
    results["xi_positive"] = xi_pos       # backward-compat alias
    results["xi_min"] = float(np.min(xi))

    # 2. theta*(t) > 0
    theta_pos = bool(np.all(theta > 0))
    results["theta_all_positive"] = theta_pos   # primary
    results["theta_positive"] = theta_pos       # backward-compat alias
    results["theta_min"] = float(np.min(theta))

    # 3. Identity: 1 + pi_0*(t)^T kappa = xi*(t)
    identity_lhs = 1.0 + pi0 @ params.kappa  # shape (n,)
    identity_err = np.abs(identity_lhs - xi)
    results["identity_max_err"] = float(np.max(identity_err))
    results["identity_satisfied"] = bool(np.max(identity_err) < tol_identity)

    # 4. Root residuals
    max_rr = float(np.max(root_res))
    results["max_root_residual"] = max_rr   # primary (test-expected)
    results["root_res_max"] = max_rr        # backward-compat alias
    results["root_res_passed"] = bool(max_rr < tol_root)

    # 5. f(T) = 0
    results["f_terminal_err"] = float(abs(f[-1]))
    results["f_terminal_ok"] = bool(abs(f[-1]) < 1e-8)

    # 6. Sign of (pi^{M,eta})^T kappa
    pi0_merton = mq["pi0_merton"]
    sign_val = float(np.dot(pi0_merton, params.kappa))
    results["pi_merton_dot_kappa"] = sign_val
    results["sign_regime"] = "negative" if sign_val < 0 else "positive"

    return results


def compute_ode_residuals(
    params: ModelParams,
    solution: dict,
) -> np.ndarray:
    """
    Compute the ODE residual at interior points by finite-difference check.

    Computes f'(t) from finite differences and compares to the formula.

    Parameters
    ----------
    params : ModelParams
    solution : dict
        Output from solve_pretransition.

    Returns
    -------
    np.ndarray
        Absolute residuals at interior grid points.
    """
    t = solution["t_grid"]
    f = solution["f"]
    pi0 = solution["pi0_star"]
    theta = solution["theta"]

    excess_0 = params.mu_0 - params.r * np.ones(params.m)
    n = len(t)
    res = np.zeros(n - 2)

    for i in range(1, n - 1):
        # Numerical f'(t_i) via central differences
        fp_num = (f[i + 1] - f[i - 1]) / (t[i + 1] - t[i - 1])

        # Analytical f'(t_i)
        if params.utility == "power":
            coeff = params.eta / 2.0
        else:
            coeff = 0.5

        fp_analytic = (
            -params.r
            - params.lam / params.epsilon
            - np.dot(pi0[i], excess_0)
            + coeff * np.dot(pi0[i], params.Sigma_0 @ pi0[i])
            + (params.lam / params.epsilon) * theta[i]
        )
        res[i - 1] = abs(fp_num - fp_analytic)

    return res


def print_verification_report(
    params: ModelParams,
    solution: dict,
) -> None:
    """Print a human-readable verification summary."""
    ver = verify_solution(params, solution)
    ode_res = compute_ode_residuals(params, solution)

    print("=" * 60)
    print("VERIFICATION REPORT")
    print("=" * 60)
    print(f"  Utility type         : {params.utility}")
    print(f"  eta                  : {params.eta}")
    print(f"  epsilon              : {params.epsilon}")
    print(f"  Sign regime          : {ver['sign_regime']}")
    print(f"  pi_merton . kappa    : {ver['pi_merton_dot_kappa']:.6f}")
    print(f"  xi*(t) > 0           : {ver['xi_all_positive']} (min={ver['xi_min']:.6f})")
    print(f"  theta*(t) > 0        : {ver['theta_all_positive']} (min={ver['theta_min']:.6f})")
    print(f"  Identity max err     : {ver['identity_max_err']:.2e} -> {'PASS' if ver['identity_satisfied'] else 'FAIL'}")
    print(f"  Root res max         : {ver['max_root_residual']:.2e} -> {'PASS' if ver['root_res_passed'] else 'FAIL'}")
    print(f"  f(T) error           : {ver['f_terminal_err']:.2e} -> {'PASS' if ver['f_terminal_ok'] else 'FAIL'}")
    print(f"  ODE res max (FD)     : {np.max(ode_res):.2e}")
    print("=" * 60)
