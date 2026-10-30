"""Tests for src/ambiguity_neutral.py — ambiguity-neutral limit solvers."""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/workspace/project")

from src.parameters import build_two_asset_params
from src.ambiguity_neutral import (
    xi_closed_form_log_neutral,
    solve_ambiguity_neutral_power,
    solve_ambiguity_neutral_log,
)
from src.merton import compute_merton_quantities


def make_power_params(**kw):
    defaults = dict(n_grid=100, lam=0.1, kappa_g=0.2, kappa_b=-0.5,
                    eta=2.0, epsilon=1.5, T=5.0)
    defaults.update(kw)
    return build_two_asset_params(**defaults)


class TestXiClosedFormLogNeutral:
    def test_positive_result(self):
        xi = xi_closed_form_log_neutral(A=1.2, kappa_quad=0.5, lam=0.1)
        assert xi > 0

    def test_reduces_to_A_when_B_zero(self):
        """When lam=0, xi* -> A (no jump risk)."""
        xi = xi_closed_form_log_neutral(A=1.5, kappa_quad=0.5, lam=1e-10)
        assert xi == pytest.approx(1.5, abs=0.01)

    def test_exact_formula(self):
        A, kappa_quad, lam = 1.2, 0.3, 0.1
        xi = xi_closed_form_log_neutral(A, kappa_quad, lam)
        expected = 0.5 * A + 0.5 * np.sqrt(A**2 + 4.0 * lam * kappa_quad)
        assert xi == pytest.approx(expected, rel=1e-12)

    def test_satisfies_log_neutral_root_eq(self):
        """xi* satisfies -xi^2 + A*xi + lambda*kappa_quad = 0."""
        A, kappa_quad, lam = 1.3, 0.4, 0.15
        xi = xi_closed_form_log_neutral(A, kappa_quad, lam)
        residual = -(xi**2) + A * xi + lam * kappa_quad
        assert abs(residual) < 1e-10


class TestSolveAmbiguityNeutralPower:
    def setup_method(self):
        self.params = make_power_params(n_grid=100, T=5.0)
        self.sol = solve_ambiguity_neutral_power(self.params)

    def test_keys(self):
        expected = {"t_grid", "f", "xi", "pi0_star", "root_residuals"}
        assert expected.issubset(set(self.sol.keys()))

    def test_t_grid_range(self):
        assert self.sol["t_grid"][0] == pytest.approx(0.0)
        assert self.sol["t_grid"][-1] == pytest.approx(self.params.T, abs=1e-10)

    def test_f_T_zero(self):
        assert self.sol["f"][-1] == pytest.approx(0.0, abs=1e-3)

    def test_xi_positive(self):
        assert np.all(self.sol["xi"] > 0)

    def test_root_residuals_small(self):
        assert np.max(self.sol["root_residuals"]) < 1e-6

    def test_identity(self):
        """1 + pi_0^T kappa == xi also in ambiguity-neutral case."""
        pi0 = self.sol["pi0_star"]
        xi = self.sol["xi"]
        kappa = self.params.kappa
        lhs = 1.0 + pi0 @ kappa
        assert np.allclose(lhs, xi, atol=1e-3)


class TestSolveAmbiguityNeutralLog:
    def setup_method(self):
        self.params = build_two_asset_params(
            eta=1.0, epsilon=0.5, utility="log",
            lam=0.1, kappa_g=0.2, kappa_b=-0.5,
            n_grid=100, T=5.0
        )
        self.sol = solve_ambiguity_neutral_log(self.params)

    def test_keys(self):
        assert "t_grid" in self.sol
        assert "xi" in self.sol
        assert "pi0_star" in self.sol

    def test_xi_positive(self):
        assert np.all(self.sol["xi"] > 0)

    def test_xi_constant(self):
        """Log ambiguity-neutral xi* should be constant (closed form, time-invariant)."""
        xi = self.sol["xi"]
        assert np.allclose(xi, xi[0], rtol=1e-10), \
            "Log-neutral xi* should be constant over time."

    def test_identity_log_neutral(self):
        pi0 = self.sol["pi0_star"]
        xi = self.sol["xi"]
        kappa = self.params.kappa
        lhs = 1.0 + pi0 @ kappa
        assert np.allclose(lhs, xi, atol=1e-6)


class TestAmbiguityNeutralAsLimit:
    """
    As epsilon -> eta - 1, the robust power solution should approach the
    ambiguity-neutral power solution.  We verify at a coarse level.
    """

    def test_epsilon_near_eta_minus_1(self):
        from src.ode_solver import solve_pretransition
        eta = 2.0
        eps_limit = eta - 1.0   # = 1.0 (domain boundary)
        eps_near = eps_limit + 0.05   # just above boundary

        params_near = make_power_params(eta=eta, epsilon=eps_near, n_grid=80, T=5.0)
        sol_near = solve_pretransition(params_near)
        sol_an = solve_ambiguity_neutral_power(make_power_params(eta=eta, n_grid=80, T=5.0))

        # The initial green weight should be reasonably close
        diff = abs(sol_near["pi0_star"][0, 0] - sol_an["pi0_star"][0, 0])
        assert diff < 1.0, \
            f"Solutions too far apart near ambiguity-neutral limit: diff={diff:.4f}"
