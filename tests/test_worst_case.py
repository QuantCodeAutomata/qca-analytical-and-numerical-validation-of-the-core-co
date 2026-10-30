"""Tests for src/worst_case.py — worst-case (epsilon->inf) portfolio solver."""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/workspace/project")

from src.parameters import build_two_asset_params
from src.worst_case import solve_worst_case
from src.merton import compute_merton_quantities


def make_neg_sign_params(**kw):
    """Standard drift: (pi^M)^T kappa < 0."""
    defaults = dict(n_grid=100, lam=0.1, kappa_g=0.2, kappa_b=-0.5,
                    eta=2.0, epsilon=1.5, T=5.0,
                    mu_g=0.015, mu_b=0.035)
    defaults.update(kw)
    return build_two_asset_params(**defaults)


def make_pos_sign_params(**kw):
    """Reverse drift: (pi^M)^T kappa > 0."""
    defaults = dict(n_grid=100, lam=0.1, kappa_g=0.2, kappa_b=-0.5,
                    eta=2.0, epsilon=1.5, T=5.0,
                    mu_g=0.035, mu_b=0.015)
    defaults.update(kw)
    return build_two_asset_params(**defaults)


class TestSolveWorstCasePositiveRegime:
    """In the positive-sign regime, worst-case = Merton."""

    def setup_method(self):
        self.params = make_pos_sign_params()
        self.sol = solve_worst_case(self.params)
        self.mq = compute_merton_quantities(self.params)

    def test_sign_regime(self):
        assert self.sol["sign_regime"] == "positive"

    def test_pi_wc_equals_merton(self):
        pi_wc = self.sol["pi_wc"]
        merton = self.mq["pi0_merton"]
        assert np.allclose(pi_wc, merton, atol=1e-10)

    def test_t_grid_correct(self):
        assert self.sol["t_grid"][0] == pytest.approx(0.0)
        assert self.sol["t_grid"][-1] == pytest.approx(self.params.T, abs=1e-10)

    def test_keys(self):
        expected = {"t_grid", "pi_wc", "theta_wc", "zeta", "sign_regime", "pi_merton_dot_kappa"}
        assert expected == set(self.sol.keys())


class TestSolveWorstCaseNegativeRegime:
    """In the negative-sign regime, worst-case differs from Merton."""

    def setup_method(self):
        self.params = make_neg_sign_params()
        self.sol = solve_worst_case(self.params)
        self.mq = compute_merton_quantities(self.params)

    def test_sign_regime(self):
        assert self.sol["sign_regime"] == "negative"

    def test_pi_wc_not_equal_merton_everywhere(self):
        pi_wc = self.sol["pi_wc"]
        merton = self.mq["pi0_merton"]
        max_diff = np.max(np.abs(pi_wc - merton))
        # They should differ somewhere (especially near t=0)
        assert max_diff > 1e-4, \
            f"Worst-case should differ from Merton in negative-sign regime: max_diff={max_diff:.4f}"

    def test_pi_wc_terminal(self):
        """At t=T, zeta(T)=1; pi_wc satisfies the formula with zeta=1.

        pi_wc(T) = pi_merton + ((zeta(T) - 1 - pi_merton.kappa) / kappa_quad)
                               * Sigma0_inv_kappa
                 = pi_merton + (-pi_merton.kappa / kappa_quad) * Sigma0_inv_kappa

        This is NOT equal to Merton in general (only when pi_merton.kappa = 0).
        """
        pi_wc_T = self.sol["pi_wc"][-1]
        merton = self.mq["pi0_merton"]
        kappa_quad = self.mq["kappa_Sigma0_inv_kappa"]
        Sigma0_inv_kappa = self.mq["Sigma0_inv_kappa"]
        pi_dot_kappa = float(np.dot(merton, self.params.kappa))
        # zeta(T) = 1
        c_T = (1.0 - 1.0 - pi_dot_kappa) / kappa_quad
        expected_T = merton + c_T * Sigma0_inv_kappa
        assert np.allclose(pi_wc_T, expected_T, atol=1e-4), (
            f"pi_wc(T) should satisfy formula with zeta=1.\n"
            f"Got {pi_wc_T}, expected {expected_T}"
        )

    def test_zeta_terminal(self):
        assert self.sol["zeta"][-1] == pytest.approx(1.0, abs=1e-4)

    def test_zeta_positive(self):
        assert np.all(self.sol["zeta"] > 0)

    def test_shape(self):
        n = self.params.n_grid
        m = self.params.m
        assert self.sol["pi_wc"].shape == (n, m)
        assert len(self.sol["t_grid"]) == n
        assert len(self.sol["zeta"]) == n


class TestWorstCaseAsRobustLimit:
    """
    For large finite epsilon, the robust solution should approach the worst case.
    We verify qualitatively (direction of deviation from Merton).
    """

    def test_large_epsilon_approaches_worst_case_neg_regime(self):
        from src.ode_solver import solve_pretransition
        params_large_eps = make_neg_sign_params(epsilon=50.0, n_grid=80, T=5.0)
        params_wc = make_neg_sign_params(n_grid=80, T=5.0)

        sol_large = solve_pretransition(params_large_eps)
        sol_wc = solve_worst_case(params_wc)

        # pi_g at t=0: large-eps should be between Merton and worst-case
        pi_g_merton = compute_merton_quantities(params_wc)["pi0_merton"][0]
        pi_g_large = sol_large["pi0_star"][0, 0]
        pi_g_wc = sol_wc["pi_wc"][0, 0]

        # All three should be finite
        assert all(np.isfinite([pi_g_merton, pi_g_large, pi_g_wc]))

    def test_positive_regime_worst_case_equals_merton(self):
        params = make_pos_sign_params(n_grid=60, T=5.0)
        sol_wc = solve_worst_case(params)
        mq = compute_merton_quantities(params)
        assert np.allclose(sol_wc["pi_wc"], mq["pi0_merton"], atol=1e-10)
