"""Scalar root equation solver for the optimal xi*(t) variable.

At each time t, the pre-transition robust portfolio requires finding the
unique positive root xi*(t) of:

  Power utility:
    h(xi) = -xi^(eps+2) + A * xi^(eps+1) + B = 0

  Log utility (same structural form, different A and B):
    h(xi) = -xi^(eps+2) + A * xi^(eps+1) + B = 0

where eps = epsilon, A = 1 + pi_0^{M,eta} . kappa (or pi_0^M . kappa for log),
and B > 0 encodes the transition hazard and preference interaction.

# Custom — Context7 found no library equivalent (methodology from paper).
"""

import numpy as np
from scipy.optimize import brentq
from typing import Tuple


def _h_xi(
    xi: float,
    A: float,
    B: float,
    eps: float,
) -> float:
    """
    Evaluate h(xi) = -xi^(eps+2) + A*xi^(eps+1) + B.

    Parameters
    ----------
    xi : float
        Candidate root value (xi > 0).
    A : float
        Coefficient = 1 + pi_0^{M} . kappa.
    B : float
        Non-negative coefficient encapsulating hazard and preferences.
    eps : float
        Ambiguity aversion epsilon.

    Returns
    -------
    float
        Value of h at xi.
    """
    return -(xi ** (eps + 2)) + A * (xi ** (eps + 1)) + B


def _find_upper_bracket(
    A: float,
    B: float,
    eps: float,
    max_iter: int = 100,
) -> float:
    """
    Find an upper bracket b such that h(b) < 0.

    Parameters
    ----------
    A, B, eps : floats
        Parameters of h.
    max_iter : int
        Maximum iterations to search for upper bracket.

    Returns
    -------
    float
        Upper bracket value.

    Raises
    ------
    RuntimeError
        If upper bracket cannot be found within max_iter iterations.
    """
    b = max(abs(A) + 1.0, 1.0)
    for _ in range(max_iter):
        if _h_xi(b, A, B, eps) < 0:
            return b
        b *= 2.0
    raise RuntimeError(
        f"Could not find upper bracket for xi root: A={A:.4f}, B={B:.4e}, eps={eps:.4f}."
    )


def solve_xi(
    A: float,
    B: float,
    eps: float,
    tol: float = 1e-10,
) -> Tuple[float, float]:
    """
    Solve h(xi) = -xi^(eps+2) + A*xi^(eps+1) + B = 0 for the unique positive root.

    Parameters
    ----------
    A : float
        Coefficient 1 + pi_0^{M} . kappa.
    B : float
        Non-negative hazard coefficient (B > 0).
    eps : float
        Ambiguity aversion epsilon (eps > 0).
    tol : float
        Root-finding tolerance.

    Returns
    -------
    xi_star : float
        Unique positive root.
    residual : float
        Absolute value of h at the returned root (for validation).

    Raises
    ------
    ValueError
        If B <= 0 or eps <= 0.
    RuntimeError
        If brentq fails to converge.
    """
    if B <= 0:
        raise ValueError(f"Coefficient B must be positive, got B={B}.")
    if eps <= 0:
        raise ValueError(f"Epsilon must be positive, got eps={eps}.")

    low = 1e-12
    high = _find_upper_bracket(A, B, eps)

    # h(low) ≈ B > 0, h(high) < 0 by construction
    xi_star = brentq(_h_xi, low, high, args=(A, B, eps), xtol=tol, rtol=tol)
    residual = abs(_h_xi(xi_star, A, B, eps))
    return float(xi_star), float(residual)


def build_AB_power(
    pi0_merton: np.ndarray,
    kappa: np.ndarray,
    lam: float,
    eta: float,
    alpha1_eta: float,
    tau: float,        # time-to-go = T - t
    f_t: float,
    eps: float,
    kappa_quad: float,  # kappa^T Sigma0^{-1} kappa
) -> Tuple[float, float]:
    """
    Construct coefficients A and B for the power-utility xi root equation.

    A = 1 + pi_0^{M,eta} . kappa
    B = (lambda/eta) * exp(-eps * [alpha_1^eta * tau - f_t]) * kappa^T Sigma_0^{-1} kappa

    Parameters
    ----------
    pi0_merton : np.ndarray
        Power-utility Merton portfolio for pre-transition market.
    kappa : np.ndarray
        Jump vector.
    lam : float
        Transition intensity.
    eta : float
        Risk aversion.
    alpha1_eta : float
        Post-transition Merton growth rate (power utility).
    tau : float
        Time-to-go T - t (>= 0).
    f_t : float
        Current value of f(t).
    eps : float
        Epsilon (ambiguity aversion).
    kappa_quad : float
        kappa^T Sigma_0^{-1} kappa.

    Returns
    -------
    A, B : floats
    """
    A = 1.0 + float(np.dot(pi0_merton, kappa))
    exponent = -eps * (alpha1_eta * tau - f_t)
    B = (lam / eta) * np.exp(exponent) * kappa_quad
    return A, B


def build_AB_log(
    pi0_merton_log: np.ndarray,
    kappa: np.ndarray,
    lam: float,
    alpha1_log: float,
    tau: float,
    f_t: float,
    eps: float,
    kappa_quad: float,
) -> Tuple[float, float]:
    """
    Construct coefficients A and B for the log-utility xi root equation.

    A = 1 + pi_0^M . kappa
    B = lambda * exp(-eps * [alpha_1 * tau - f_t]) * kappa^T Sigma_0^{-1} kappa

    Parameters
    ----------
    pi0_merton_log : np.ndarray
        Log-utility Merton portfolio for pre-transition market.
    kappa : np.ndarray
        Jump vector.
    lam : float
        Transition intensity.
    alpha1_log : float
        Post-transition Merton growth rate (log utility).
    tau : float
        Time-to-go T - t.
    f_t : float
        Current value of f(t).
    eps : float
        Epsilon.
    kappa_quad : float
        kappa^T Sigma_0^{-1} kappa.

    Returns
    -------
    A, B : floats
    """
    A = 1.0 + float(np.dot(pi0_merton_log, kappa))
    exponent = -eps * (alpha1_log * tau - f_t)
    B = lam * np.exp(exponent) * kappa_quad
    return A, B


def compute_controls_power(
    xi_star: float,
    pi0_merton: np.ndarray,
    kappa: np.ndarray,
    Sigma0_inv_kappa: np.ndarray,
    kappa_quad: float,
    lam: float,
    eta: float,
    alpha1_eta: float,
    tau: float,
    f_t: float,
    eps: float,
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal portfolio pi_0*(t) and distortion theta*(t) for power utility.

    pi_0*(t) = pi_0^{M,eta} + c(xi*) * Sigma_0^{-1} kappa
    where c(xi*) = (xi* - 1 - pi_0^{M,eta} . kappa) / (kappa^T Sigma_0^{-1} kappa)

    theta*(t) = exp(-eps * [alpha_1^eta * tau - f_t]) * xi*^(-eps)
    (equivalently = (eta/lambda) * xi* * c(xi*))

    Parameters
    ----------
    xi_star : float
        Optimal xi value from root solve.
    pi0_merton : np.ndarray
        Pre-transition power Merton portfolio.
    kappa : np.ndarray
        Jump vector.
    Sigma0_inv_kappa : np.ndarray
        Sigma_0^{-1} kappa.
    kappa_quad : float
        kappa^T Sigma_0^{-1} kappa.
    lam : float
        Transition intensity.
    eta : float
        Risk aversion.
    alpha1_eta : float
        Post-transition Merton growth rate.
    tau : float
        Time-to-go T - t.
    f_t : float
        Current f(t).
    eps : float
        Epsilon.

    Returns
    -------
    pi0_star : np.ndarray, shape (m,)
        Optimal pre-transition portfolio.
    theta_star : float
        Optimal distortion intensity.
    """
    pi0_dot_kappa = float(np.dot(pi0_merton, kappa))
    c = (xi_star - 1.0 - pi0_dot_kappa) / kappa_quad
    pi0_star = pi0_merton + c * Sigma0_inv_kappa

    # theta via exponential formula (numerically stable for large eps)
    exponent = -eps * (alpha1_eta * tau - f_t)
    theta_star = np.exp(exponent) * (xi_star ** (-eps))

    return pi0_star, float(theta_star)


def compute_controls_log(
    xi_star: float,
    pi0_merton_log: np.ndarray,
    kappa: np.ndarray,
    Sigma0_inv_kappa: np.ndarray,
    kappa_quad: float,
    lam: float,
) -> Tuple[np.ndarray, float]:
    """
    Compute optimal portfolio pi_0*(t) and distortion theta*(t) for log utility.

    pi_0*(t) = pi_0^M + c(xi*) * Sigma_0^{-1} kappa
    theta*(t) = (1/lambda) * xi* * c(xi*)

    Parameters
    ----------
    xi_star : float
        Optimal xi.
    pi0_merton_log : np.ndarray
        Log-utility Merton portfolio.
    kappa : np.ndarray
        Jump vector.
    Sigma0_inv_kappa : np.ndarray
        Sigma_0^{-1} kappa.
    kappa_quad : float
        kappa^T Sigma_0^{-1} kappa.
    lam : float
        Transition intensity.

    Returns
    -------
    pi0_star : np.ndarray
        Optimal pre-transition portfolio.
    theta_star : float
        Optimal distortion.
    """
    pi0_dot_kappa = float(np.dot(pi0_merton_log, kappa))
    c = (xi_star - 1.0 - pi0_dot_kappa) / kappa_quad
    pi0_star = pi0_merton_log + c * Sigma0_inv_kappa
    theta_star = (1.0 / lam) * xi_star * c
    return pi0_star, float(theta_star)
