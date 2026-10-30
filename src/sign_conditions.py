"""Two-asset sign conditions and hedging-direction analysis.

Validates the analytical conditions under which transition awareness
increases green allocation and decreases brown allocation relative to
the transition-unaware Merton benchmark.

For the two-asset green/brown model:
  Sigma = [[sigma_g^2,          rho*sigma_g*sigma_b],
           [rho*sigma_g*sigma_b, sigma_b^2          ]]

  Sigma^{-1}*kappa = (1/(1-rho^2)) * [
      (kappa_g/sigma_g^2 - rho*kappa_b/(sigma_g*sigma_b)),
      (kappa_b/sigma_b^2 - rho*kappa_g/(sigma_g*sigma_b)),
  ]

The optimal portfolio correction is:
  pi_0*(t) = pi_0^{benchmark} + c(t) * Sigma^{-1}*kappa

where c(t) > 0 (resp. < 0) depending on the sign regime. The sign of
the green (resp. brown) correction is the sign of (Sigma^{-1}*kappa)_g
(resp. (Sigma^{-1}*kappa)_b).

# Custom — Context7 found no library equivalent (paper methodology).
"""

import numpy as np
from typing import Tuple


def build_two_asset_covariance(
    sigma_g: float,
    sigma_b: float,
    rho: float,
) -> np.ndarray:
    """
    Build the two-asset covariance matrix.

    Parameters
    ----------
    sigma_g, sigma_b : float
        Volatilities of green and brown assets.
    rho : float
        Correlation in (-1, 1).

    Returns
    -------
    np.ndarray, shape (2, 2)
    """
    if abs(rho) >= 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho}.")
    return np.array([
        [sigma_g**2,             rho * sigma_g * sigma_b],
        [rho * sigma_g * sigma_b, sigma_b**2             ],
    ])


def compute_sigma_inv_kappa(
    sigma_g: float,
    sigma_b: float,
    rho: float,
    kappa_g: float,
    kappa_b: float,
) -> np.ndarray:
    """
    Compute Sigma^{-1}*kappa analytically for the two-asset case.

    Sigma^{-1}*kappa = 1/(1-rho^2) * [
        (1/sigma_g) * (kappa_g/sigma_g - rho*kappa_b/sigma_b),
        (1/sigma_b) * (kappa_b/sigma_b - rho*kappa_g/sigma_g),
    ]

    Parameters
    ----------
    sigma_g, sigma_b, rho, kappa_g, kappa_b : floats

    Returns
    -------
    np.ndarray, shape (2,)
        The vector Sigma^{-1}*kappa = [green_component, brown_component].
    """
    det_factor = 1.0 - rho**2
    assert abs(det_factor) > 1e-12, "Correlation too close to +-1."

    green_comp = (1.0 / sigma_g) * (kappa_g / sigma_g - rho * kappa_b / sigma_b)
    brown_comp = (1.0 / sigma_b) * (kappa_b / sigma_b - rho * kappa_g / sigma_g)

    return np.array([green_comp, brown_comp]) / det_factor


def green_condition_threshold(
    sigma_g: float,
    sigma_b: float,
    kappa_g: float,
    kappa_b: float,
) -> float:
    """
    Compute the rho threshold above which transition awareness increases green.

    Condition: (Sigma^{-1}*kappa)_green > 0
    iff kappa_g/sigma_g - rho*kappa_b/sigma_b > 0
    iff rho > (kappa_g*sigma_b) / (kappa_b*sigma_g)   [if kappa_b > 0, flip sign]

    When kappa_g > 0 and kappa_b < 0:
      rho_threshold_green = -(kappa_g/|kappa_b|) * (sigma_b/sigma_g)
      condition: rho > rho_threshold_green

    Parameters
    ----------
    sigma_g, sigma_b, kappa_g, kappa_b : floats

    Returns
    -------
    float
        The rho threshold for the green component to be positive.
    """
    # (Sigma^{-1}*kappa)_g > 0 iff kappa_g/sigma_g > rho * kappa_b/sigma_b
    # iff (kappa_g/sigma_g) / (kappa_b/sigma_b) > rho    [kappa_b < 0 means flip]
    # iff rho > (kappa_g*sigma_b)/(kappa_b*sigma_g)
    return (kappa_g * sigma_b) / (kappa_b * sigma_g)


def brown_condition_threshold(
    sigma_g: float,
    sigma_b: float,
    kappa_g: float,
    kappa_b: float,
) -> float:
    """
    Compute the rho threshold above which transition awareness decreases brown.

    Condition: (Sigma^{-1}*kappa)_brown < 0
    iff kappa_b/sigma_b - rho*kappa_g/sigma_g < 0
    iff rho > (kappa_b*sigma_g)/(kappa_g*sigma_b)

    Parameters
    ----------
    sigma_g, sigma_b, kappa_g, kappa_b : floats

    Returns
    -------
    float
        The rho threshold for the brown component to be negative.
    """
    # (Sigma^{-1}*kappa)_b < 0 iff kappa_b/sigma_b < rho*kappa_g/sigma_g
    # iff (kappa_b*sigma_g)/(kappa_g*sigma_b) < rho
    return (kappa_b * sigma_g) / (kappa_g * sigma_b)


def classify_reallocation_direction(
    sigma_g: float,
    sigma_b: float,
    rho: float,
    kappa_g: float,
    kappa_b: float,
) -> dict:
    """
    Classify whether transition awareness increases green and/or decreases brown.

    Parameters
    ----------
    sigma_g, sigma_b, rho, kappa_g, kappa_b : floats

    Returns
    -------
    dict with keys:
        sigma_inv_kappa : np.ndarray — the direction vector
        green_positive  : bool — green component positive
        brown_negative  : bool — brown component negative
        green_threshold : float — rho threshold for green
        brown_threshold : float — rho threshold for brown
        kappa_quad      : float — kappa^T Sigma^{-1} kappa
    """
    sik = compute_sigma_inv_kappa(sigma_g, sigma_b, rho, kappa_g, kappa_b)
    kappa = np.array([kappa_g, kappa_b])
    kappa_quad = float(np.dot(kappa, sik))

    return {
        "sigma_inv_kappa": sik,
        "green_positive": bool(sik[0] > 0),
        "brown_negative": bool(sik[1] < 0),
        "green_threshold": green_condition_threshold(sigma_g, sigma_b, kappa_g, kappa_b),
        "brown_threshold": brown_condition_threshold(sigma_g, sigma_b, kappa_g, kappa_b),
        "kappa_quad": kappa_quad,
    }


def sweep_sign_conditions(
    sigma_g: float = 0.15,
    sigma_b: float = 0.30,
    kappa_g: float = 0.2,
    kappa_b: float = -0.5,
    rho_grid: np.ndarray = None,
) -> dict:
    """
    Sweep over correlation values and evaluate sign conditions.

    Parameters
    ----------
    sigma_g, sigma_b, kappa_g, kappa_b : floats
    rho_grid : np.ndarray
        Grid of rho values in (-1, 1). Defaults to 200-point grid.

    Returns
    -------
    dict with keys:
        rho_grid, green_positive (bool array), brown_negative (bool array),
        green_sign, brown_sign, sigma_inv_kappa_arr, kappa_quad_arr
    """
    if rho_grid is None:
        rho_grid = np.linspace(-0.99, 0.99, 200)

    green_positive = np.zeros(len(rho_grid), dtype=bool)
    brown_negative = np.zeros(len(rho_grid), dtype=bool)
    green_sign = np.zeros(len(rho_grid))
    brown_sign = np.zeros(len(rho_grid))
    sik_arr = np.zeros((len(rho_grid), 2))
    kq_arr = np.zeros(len(rho_grid))

    for i, rho in enumerate(rho_grid):
        res = classify_reallocation_direction(sigma_g, sigma_b, rho, kappa_g, kappa_b)
        sik_arr[i] = res["sigma_inv_kappa"]
        green_positive[i] = bool(res["green_positive"])
        brown_negative[i] = bool(res["brown_negative"])
        green_sign[i] = 1.0 if res["green_positive"] else -1.0
        brown_sign[i] = -1.0 if res["brown_negative"] else 1.0
        kq_arr[i] = res["kappa_quad"]

    return {
        "rho_grid": rho_grid,
        "green_positive": green_positive,
        "brown_negative": brown_negative,
        "green_sign": green_sign,
        "brown_sign": brown_sign,
        "sigma_inv_kappa_arr": sik_arr,
        "kappa_quad_arr": kq_arr,
    }
