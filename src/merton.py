"""Merton benchmark portfolio computations.

Implements the transition-unaware Merton optimal portfolio and associated
growth-rate (alpha) quantities for both power and log utility.
"""

import numpy as np
from .parameters import ModelParams


def merton_power(
    mu: np.ndarray,
    r: float,
    Sigma: np.ndarray,
    eta: float,
) -> np.ndarray:
    """
    Compute the power-utility Merton portfolio.

    pi^{M,eta} = (1/eta) * Sigma^{-1} * (mu - r*1)

    Parameters
    ----------
    mu : np.ndarray, shape (m,)
        Asset drift vector.
    r : float
        Risk-free rate.
    Sigma : np.ndarray, shape (m, m)
        Covariance matrix.
    eta : float
        Risk aversion parameter (eta > 0, eta != 1).

    Returns
    -------
    np.ndarray, shape (m,)
        Merton optimal portfolio weights.
    """
    excess = mu - r * np.ones(len(mu))
    # Use np.linalg.solve for numerical stability over explicit inversion
    pi_m = np.linalg.solve(Sigma, excess) / eta
    return pi_m


def merton_log(
    mu: np.ndarray,
    r: float,
    Sigma: np.ndarray,
) -> np.ndarray:
    """
    Compute the log-utility Merton portfolio.

    pi^M = Sigma^{-1} * (mu - r*1)

    Parameters
    ----------
    mu : np.ndarray, shape (m,)
        Asset drift vector.
    r : float
        Risk-free rate.
    Sigma : np.ndarray, shape (m, m)
        Covariance matrix.

    Returns
    -------
    np.ndarray, shape (m,)
        Log-utility Merton optimal portfolio weights.
    """
    excess = mu - r * np.ones(len(mu))
    pi_m = np.linalg.solve(Sigma, excess)
    return pi_m


def alpha_power(
    mu: np.ndarray,
    r: float,
    Sigma: np.ndarray,
    eta: float,
) -> float:
    """
    Compute the power-utility Merton growth rate.

    alpha^eta = r + (1/(2*eta)) * (mu - r*1)^T * Sigma^{-1} * (mu - r*1)

    Parameters
    ----------
    mu : np.ndarray, shape (m,)
        Asset drift.
    r : float
        Risk-free rate.
    Sigma : np.ndarray, shape (m, m)
        Covariance matrix.
    eta : float
        Risk aversion.

    Returns
    -------
    float
        Merton growth rate for power utility.
    """
    excess = mu - r * np.ones(len(mu))
    Sigma_inv_excess = np.linalg.solve(Sigma, excess)
    return float(r + (1.0 / (2.0 * eta)) * np.dot(excess, Sigma_inv_excess))


def alpha_log(
    mu: np.ndarray,
    r: float,
    Sigma: np.ndarray,
) -> float:
    """
    Compute the log-utility Merton growth rate.

    alpha = r + (1/2) * (mu - r*1)^T * Sigma^{-1} * (mu - r*1)

    Parameters
    ----------
    mu : np.ndarray, shape (m,)
        Asset drift.
    r : float
        Risk-free rate.
    Sigma : np.ndarray, shape (m, m)
        Covariance matrix.

    Returns
    -------
    float
        Merton growth rate for log utility.
    """
    excess = mu - r * np.ones(len(mu))
    Sigma_inv_excess = np.linalg.solve(Sigma, excess)
    return float(r + 0.5 * np.dot(excess, Sigma_inv_excess))


def compute_merton_quantities(params: ModelParams) -> dict:
    """
    Compute all Merton benchmark quantities for the given model parameters.

    Parameters
    ----------
    params : ModelParams
        Model parameters.

    Returns
    -------
    dict with keys:
        pi0_merton : Merton portfolio for pre-transition market.
        pi1_merton : Merton portfolio for post-transition market.
        alpha0     : Merton growth rate for pre-transition market.
        alpha1     : Merton growth rate for post-transition market.
        Sigma0_inv_kappa : Sigma_0^{-1} kappa vector.
        kappa_Sigma0_inv_kappa : kappa^T Sigma_0^{-1} kappa (scalar).
    """
    if params.utility == "power":
        pi0 = merton_power(params.mu_0, params.r, params.Sigma_0, params.eta)
        pi1 = merton_power(params.mu_1, params.r, params.Sigma_1, params.eta)
        a0 = alpha_power(params.mu_0, params.r, params.Sigma_0, params.eta)
        a1 = alpha_power(params.mu_1, params.r, params.Sigma_1, params.eta)
    else:  # log utility
        pi0 = merton_log(params.mu_0, params.r, params.Sigma_0)
        pi1 = merton_log(params.mu_1, params.r, params.Sigma_1)
        a0 = alpha_log(params.mu_0, params.r, params.Sigma_0)
        a1 = alpha_log(params.mu_1, params.r, params.Sigma_1)

    Sigma0_inv_kappa = np.linalg.solve(params.Sigma_0, params.kappa)
    kappa_quad = float(np.dot(params.kappa, Sigma0_inv_kappa))

    return {
        "pi0_merton": pi0,
        "pi1_merton": pi1,
        "alpha0": a0,
        "alpha1": a1,
        "Sigma0_inv_kappa": Sigma0_inv_kappa,
        "kappa_Sigma0_inv_kappa": kappa_quad,
    }
