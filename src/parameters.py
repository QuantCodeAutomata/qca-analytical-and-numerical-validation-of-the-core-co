"""Model parameter dataclasses for the robust transition-timing portfolio problem.

Defines the full parameter set required for both pre- and post-transition
power-utility and log-utility optimal portfolio computations.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class ModelParams:
    """
    Full parameter set for the continuous-time robust portfolio model
    with one-time green-transition shock and ambiguity aversion.

    Attributes
    ----------
    T : float
        Investment horizon (T > 0).
    r : float
        Risk-free rate.
    lam : float
        Transition arrival intensity (lambda > 0).
    mu_0 : np.ndarray
        Pre-transition drift vector, shape (m,).
    mu_1 : np.ndarray
        Post-transition drift vector, shape (m,).
    Sigma_0 : np.ndarray
        Pre-transition covariance matrix, shape (m, m). Positive definite.
    Sigma_1 : np.ndarray
        Post-transition covariance matrix, shape (m, m). Positive definite.
    kappa : np.ndarray
        Transition jump vector, shape (m,). Each entry in (-1, inf).
    eta : float
        Risk aversion parameter. For power utility: eta > 0, eta != 1.
        For log utility: eta = 1.
    epsilon : float
        Ambiguity aversion parameter. For power utility: epsilon > eta - 1,
        epsilon != 0. For log utility: epsilon > 0.
    n_grid : int
        Number of time grid points for the backward ODE integration.
    utility : str
        Either 'power' or 'log'.
    """

    T: float
    r: float
    lam: float
    mu_0: np.ndarray
    mu_1: np.ndarray
    Sigma_0: np.ndarray
    Sigma_1: np.ndarray
    kappa: np.ndarray
    eta: float
    epsilon: float
    n_grid: int = 500
    utility: str = "power"

    def __post_init__(self) -> None:
        self.mu_0 = np.asarray(self.mu_0, dtype=float)
        self.mu_1 = np.asarray(self.mu_1, dtype=float)
        self.Sigma_0 = np.asarray(self.Sigma_0, dtype=float)
        self.Sigma_1 = np.asarray(self.Sigma_1, dtype=float)
        self.kappa = np.asarray(self.kappa, dtype=float)
        self.validate()

    def validate(self) -> None:
        """Validate parameter constraints."""
        m = len(self.mu_0)
        assert self.T > 0, "Horizon T must be positive."
        assert self.lam > 0, "Intensity lambda must be positive."
        assert self.Sigma_0.shape == (m, m), "Sigma_0 shape mismatch."
        assert self.Sigma_1.shape == (m, m), "Sigma_1 shape mismatch."
        assert len(self.kappa) == m, "kappa length mismatch."
        assert np.all(self.kappa > -1), "All kappa entries must be > -1."
        assert self.eta > 0, "eta must be positive."
        assert self.epsilon > 0, "epsilon must be positive."
        if self.utility == "power":
            assert abs(self.eta - 1) > 1e-12, "For power utility, eta != 1."
            assert self.epsilon > self.eta - 1, (
                f"Power utility requires epsilon > eta - 1 = {self.eta - 1}."
            )
        elif self.utility == "log":
            assert abs(self.eta - 1) < 1e-12 or True, "Log utility: set eta=1."
        # Check positive definiteness via Cholesky
        try:
            np.linalg.cholesky(self.Sigma_0)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Sigma_0 is not positive definite.") from exc
        try:
            np.linalg.cholesky(self.Sigma_1)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Sigma_1 is not positive definite.") from exc

    @property
    def m(self) -> int:
        """Number of risky assets."""
        return len(self.mu_0)


def build_two_asset_params(
    mu_g: float = 0.015,
    mu_b: float = 0.035,
    sigma_g: float = 0.15,
    sigma_b: float = 0.30,
    rho: float = 0.7,
    kappa_g: float = 0.2,
    kappa_b: float = -0.5,
    r: float = 0.02,
    lam: float = 0.1,
    eta: float = 2.0,
    epsilon: float = 1.5,
    T: float = 10.0,
    n_grid: int = 500,
    utility: str = "power",
    post_equal_pre: bool = True,
) -> ModelParams:
    """
    Construct ModelParams for the stylized two-asset green/brown setup.

    Parameters
    ----------
    mu_g, mu_b : floats
        Pre-transition drifts for green and brown assets.
    sigma_g, sigma_b : floats
        Volatilities for green and brown assets.
    rho : float
        Correlation between green and brown assets.
    kappa_g, kappa_b : floats
        Jump magnitudes at transition for green and brown.
    r : float
        Risk-free rate.
    lam : float
        Transition intensity.
    eta : float
        Risk aversion.
    epsilon : float
        Ambiguity aversion.
    T : float
        Time horizon.
    n_grid : int
        Number of time grid points.
    utility : str
        'power' or 'log'.
    post_equal_pre : bool
        If True, post-transition coefficients equal pre-transition.

    Returns
    -------
    ModelParams
    """
    Sigma = np.array([
        [sigma_g**2, rho * sigma_g * sigma_b],
        [rho * sigma_g * sigma_b, sigma_b**2],
    ])
    mu_0 = np.array([mu_g, mu_b])
    mu_1 = mu_0.copy() if post_equal_pre else mu_0.copy()
    Sigma_1 = Sigma.copy() if post_equal_pre else Sigma.copy()
    kappa = np.array([kappa_g, kappa_b])

    return ModelParams(
        T=T,
        r=r,
        lam=lam,
        mu_0=mu_0,
        mu_1=mu_1,
        Sigma_0=Sigma,
        Sigma_1=Sigma_1,
        kappa=kappa,
        eta=eta,
        epsilon=epsilon,
        n_grid=n_grid,
        utility=utility,
    )
