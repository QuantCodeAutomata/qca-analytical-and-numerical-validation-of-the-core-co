"""Tests for src/ode_solver.py — backward ODE integration and post-transition solution."""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/workspace/project")

from src.parameters import build_two_asset_params
from src.ode_solver import solve_pretransition, solve_posttransition


def make_params(**kw):
    defaults = dict(n_grid=100, lam=0.1, kappa_g=0.2, kappa_b=-0.5,
                    eta=2.0, epsilon=1.5, T=5.0)
    defaults.update(kw)
    return build_two_asset_params(**defaults)


class TestSolvePosttransition:
    def test_returns_keys(self):
        sol = solve_posttransition(make_params())
        assert "pi1_star" in sol
        assert "alpha1" in sol

    def test_alpha1_positive(self):
        sol = solve_posttransition(make_params())
        assert sol["alpha1"] > 0

    def test_pi1_shape(self):
        sol = solve_posttransition(make_params())
        assert sol["pi1_star"].shape == (2,)


class TestSolvePreTransitionPower:
    def setup_method(self):
        self.params = make_params(n_grid=200)
        self.sol = solve_pretransition(self.params)

    def test_return_keys(self):
        expected = {"t_grid", "f", "xi", "pi0_star", "theta", "root_residuals", "merton_quantities"}
        assert expected == set(self.sol.keys())

    def test_t_grid_range(self):
        t = self.sol["t_grid"]
        assert t[0] == pytest.approx(0.0, abs=1e-10)
        assert t[-1] == pytest.approx(self.params.T, abs=1e-10)

    def test_terminal_f_zero(self):
        """f(T) = 0 by boundary condition."""
        assert self.sol["f"][-1] == pytest.approx(0.0, abs=1e-3)

    def test_xi_all_positive(self):
        assert np.all(self.sol["xi"] > 0), "xi*(t) must be positive everywhere."

    def test_theta_all_positive(self):
        assert np.all(self.sol["theta"] > 0), "theta*(t) must be positive everywhere."

    def test_consistency_identity(self):
        """1 + pi_0*(t)^T kappa == xi*(t) at every grid point."""
        pi0 = self.sol["pi0_star"]
        xi = self.sol["xi"]
        kappa = self.params.kappa
        lhs = 1.0 + pi0 @ kappa
        assert np.allclose(lhs, xi, atol=1e-4), \
            f"Identity violated: max error = {np.max(np.abs(lhs - xi)):.2e}"

    def test_root_residuals_small(self):
        residuals = self.sol["root_residuals"]
        assert np.max(residuals) < 1e-6, \
            f"Root residuals too large: max = {np.max(residuals):.2e}"

    def test_pi0_shape(self):
        n = self.params.n_grid
        m = self.params.m
        assert self.sol["pi0_star"].shape == (n, m)

    def test_arrays_length_consistent(self):
        n = self.params.n_grid
        assert len(self.sol["t_grid"]) == n
        assert len(self.sol["f"]) == n
        assert len(self.sol["xi"]) == n
        assert len(self.sol["theta"]) == n
        assert len(self.sol["root_residuals"]) == n


class TestSolvePreTransitionLog:
    def setup_method(self):
        self.params = build_two_asset_params(
            eta=1.0, epsilon=0.5, utility="log", n_grid=100, T=5.0
        )
        self.sol = solve_pretransition(self.params)

    def test_terminal_f_zero(self):
        assert self.sol["f"][-1] == pytest.approx(0.0, abs=1e-3)

    def test_xi_positive(self):
        assert np.all(self.sol["xi"] > 0)

    def test_theta_positive(self):
        assert np.all(self.sol["theta"] > 0)

    def test_identity_log(self):
        pi0 = self.sol["pi0_star"]
        xi = self.sol["xi"]
        kappa = self.params.kappa
        lhs = 1.0 + pi0 @ kappa
        assert np.allclose(lhs, xi, atol=1e-4)

    def test_root_residuals_small(self):
        assert np.max(self.sol["root_residuals"]) < 1e-6


class TestLambdaToZeroLimit:
    """As lambda -> 0, xi*(t) -> A = 1 + pi0_merton . kappa constantly."""

    def test_lambda_small(self):
        params = make_params(lam=1e-4, n_grid=80, T=5.0)
        sol = solve_pretransition(params)
        from src.merton import compute_merton_quantities
        mq = compute_merton_quantities(params)
        A_expected = 1.0 + np.dot(mq["pi0_merton"], params.kappa)
        # xi should be close to A everywhere (B -> 0)
        assert np.allclose(sol["xi"], A_expected, atol=0.05), \
            f"xi far from A at small lambda: {np.mean(sol['xi']):.4f} vs A={A_expected:.4f}"
