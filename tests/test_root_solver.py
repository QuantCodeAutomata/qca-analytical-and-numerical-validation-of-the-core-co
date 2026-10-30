"""Tests for src/root_solver.py — xi* root equation and control computation."""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/workspace/project")

from src.root_solver import (
    _h_xi,
    solve_xi,
    build_AB_power,
    build_AB_log,
    compute_controls_power,
    compute_controls_log,
)
from src.parameters import build_two_asset_params
from src.merton import compute_merton_quantities


class TestHXi:
    def test_basic_evaluation(self):
        """h(xi) = -xi^(eps+2) + A*xi^(eps+1) + B."""
        A, B, eps = 1.5, 0.5, 2.0
        xi = 1.0
        val = _h_xi(xi, A, B, eps)
        expected = -(1.0 ** 4) + 1.5 * (1.0 ** 3) + 0.5
        assert val == pytest.approx(expected, abs=1e-14)

    def test_sign_at_zero(self):
        """h(0+) should be positive when B > 0."""
        assert _h_xi(1e-10, 1.5, 0.5, 2.0) > 0

    def test_sign_at_infinity(self):
        """h(large xi) should be negative."""
        assert _h_xi(1e6, 1.5, 0.5, 2.0) < 0


class TestSolveXi:
    def test_positive_root(self):
        """Root should be positive."""
        xi_star, _ = solve_xi(A=1.5, B=0.1, eps=2.0)
        assert xi_star > 0

    def test_root_is_actual_root(self):
        """h(xi*) should be near zero."""
        A, B, eps = 1.3, 0.05, 1.5
        xi_star, residual = solve_xi(A, B, eps)
        assert residual < 1e-8
        assert abs(_h_xi(xi_star, A, B, eps)) < 1e-8

    def test_small_B(self):
        """For small B, xi* should be close to A (dominant balance)."""
        xi_star, _ = solve_xi(A=2.0, B=1e-5, eps=2.0)
        # As B->0, -xi^(eps+2) + A*xi^(eps+1) ≈ 0 => xi ≈ A
        assert xi_star == pytest.approx(2.0, abs=0.05)

    def test_invalid_B_raises(self):
        with pytest.raises(ValueError, match="positive"):
            solve_xi(A=1.5, B=-0.1, eps=2.0)

    def test_invalid_eps_raises(self):
        with pytest.raises(ValueError, match="positive"):
            solve_xi(A=1.5, B=0.1, eps=-1.0)

    def test_residual_within_tolerance(self):
        tol = 1e-10
        xi_star, residual = solve_xi(A=1.2, B=0.2, eps=3.0, tol=tol)
        assert residual <= tol * 100  # brentq residual may be slightly above xtol

    def test_various_eps_values(self):
        """Root should always be positive for different eps."""
        for eps in [0.5, 1.0, 2.0, 5.0, 10.0]:
            xi_star, res = solve_xi(A=1.5, B=0.1, eps=eps)
            assert xi_star > 0
            assert res < 1e-6


class TestBuildABPower:
    def setup_method(self):
        self.params = build_two_asset_params()
        self.mq = compute_merton_quantities(self.params)

    def test_A_value(self):
        """A = 1 + pi0_merton . kappa."""
        A, _ = build_AB_power(
            pi0_merton=self.mq["pi0_merton"],
            kappa=self.params.kappa,
            lam=self.params.lam,
            eta=self.params.eta,
            alpha1_eta=self.mq["alpha1"],
            tau=5.0, f_t=0.0,
            eps=self.params.epsilon,
            kappa_quad=self.mq["kappa_Sigma0_inv_kappa"],
        )
        expected_A = 1.0 + np.dot(self.mq["pi0_merton"], self.params.kappa)
        assert A == pytest.approx(expected_A, rel=1e-12)

    def test_B_positive(self):
        """B must always be positive (lambda, kappa_quad, exp > 0)."""
        _, B = build_AB_power(
            pi0_merton=self.mq["pi0_merton"],
            kappa=self.params.kappa,
            lam=self.params.lam,
            eta=self.params.eta,
            alpha1_eta=self.mq["alpha1"],
            tau=5.0, f_t=0.0,
            eps=self.params.epsilon,
            kappa_quad=self.mq["kappa_Sigma0_inv_kappa"],
        )
        assert B > 0

    def test_tau_zero_B(self):
        """At tau=0, exponent = -eps * (alpha1 * 0 - f) = eps * f."""
        params = self.params
        mq = self.mq
        f_t = 0.5
        A, B = build_AB_power(
            mq["pi0_merton"], params.kappa, params.lam, params.eta,
            mq["alpha1"], tau=0.0, f_t=f_t, eps=params.epsilon,
            kappa_quad=mq["kappa_Sigma0_inv_kappa"],
        )
        expected_B = (params.lam / params.eta) * np.exp(params.epsilon * f_t) * mq["kappa_Sigma0_inv_kappa"]
        assert B == pytest.approx(expected_B, rel=1e-10)


class TestBuildABLog:
    def setup_method(self):
        self.params = build_two_asset_params(eta=1.0, epsilon=0.5, utility="log")
        self.mq = compute_merton_quantities(self.params)

    def test_A_value(self):
        A, _ = build_AB_log(
            pi0_merton_log=self.mq["pi0_merton"],
            kappa=self.params.kappa,
            lam=self.params.lam,
            alpha1_log=self.mq["alpha1"],
            tau=5.0, f_t=0.0,
            eps=self.params.epsilon,
            kappa_quad=self.mq["kappa_Sigma0_inv_kappa"],
        )
        expected_A = 1.0 + np.dot(self.mq["pi0_merton"], self.params.kappa)
        assert A == pytest.approx(expected_A, rel=1e-12)

    def test_B_positive(self):
        _, B = build_AB_log(
            pi0_merton_log=self.mq["pi0_merton"],
            kappa=self.params.kappa,
            lam=self.params.lam,
            alpha1_log=self.mq["alpha1"],
            tau=5.0, f_t=0.0,
            eps=self.params.epsilon,
            kappa_quad=self.mq["kappa_Sigma0_inv_kappa"],
        )
        assert B > 0


class TestComputeControlsPower:
    def setup_method(self):
        self.params = build_two_asset_params()
        self.mq = compute_merton_quantities(self.params)

    def _get_xi_star(self, tau=5.0, f_t=0.0):
        from src.root_solver import build_AB_power, solve_xi
        A, B = build_AB_power(
            self.mq["pi0_merton"], self.params.kappa, self.params.lam,
            self.params.eta, self.mq["alpha1"], tau, f_t, self.params.epsilon,
            self.mq["kappa_Sigma0_inv_kappa"],
        )
        xi_star, _ = solve_xi(A, B, self.params.epsilon)
        return xi_star, A

    def test_identity_1_plus_pi_kappa_equals_xi(self):
        """1 + pi_0*(t)^T kappa == xi*(t) is the fundamental identity."""
        xi_star, _ = self._get_xi_star(tau=5.0)
        pi0_star, _ = compute_controls_power(
            xi_star=xi_star,
            pi0_merton=self.mq["pi0_merton"],
            kappa=self.params.kappa,
            Sigma0_inv_kappa=self.mq["Sigma0_inv_kappa"],
            kappa_quad=self.mq["kappa_Sigma0_inv_kappa"],
            lam=self.params.lam,
            eta=self.params.eta,
            alpha1_eta=self.mq["alpha1"],
            tau=5.0, f_t=0.0,
            eps=self.params.epsilon,
        )
        lhs = 1.0 + np.dot(pi0_star, self.params.kappa)
        assert lhs == pytest.approx(xi_star, abs=1e-6)

    def test_theta_positive(self):
        xi_star, _ = self._get_xi_star()
        _, theta_star = compute_controls_power(
            xi_star, self.mq["pi0_merton"], self.params.kappa,
            self.mq["Sigma0_inv_kappa"], self.mq["kappa_Sigma0_inv_kappa"],
            self.params.lam, self.params.eta, self.mq["alpha1"],
            5.0, 0.0, self.params.epsilon,
        )
        assert theta_star > 0


class TestComputeControlsLog:
    def setup_method(self):
        self.params = build_two_asset_params(eta=1.0, epsilon=0.5, utility="log")
        self.mq = compute_merton_quantities(self.params)

    def _get_xi_star(self, tau=5.0, f_t=0.0):
        from src.root_solver import build_AB_log, solve_xi
        A, B = build_AB_log(
            self.mq["pi0_merton"], self.params.kappa, self.params.lam,
            self.mq["alpha1"], tau, f_t, self.params.epsilon,
            self.mq["kappa_Sigma0_inv_kappa"],
        )
        xi_star, _ = solve_xi(A, B, self.params.epsilon)
        return xi_star

    def test_identity_log(self):
        """1 + pi_0*(t)^T kappa == xi*(t) also holds for log utility."""
        xi_star = self._get_xi_star()
        pi0_star, _ = compute_controls_log(
            xi_star, self.mq["pi0_merton"], self.params.kappa,
            self.mq["Sigma0_inv_kappa"], self.mq["kappa_Sigma0_inv_kappa"],
            self.params.lam,
        )
        lhs = 1.0 + np.dot(pi0_star, self.params.kappa)
        assert lhs == pytest.approx(xi_star, abs=1e-6)

    def test_theta_positive_log(self):
        xi_star = self._get_xi_star()
        _, theta_star = compute_controls_log(
            xi_star, self.mq["pi0_merton"], self.params.kappa,
            self.mq["Sigma0_inv_kappa"], self.mq["kappa_Sigma0_inv_kappa"],
            self.params.lam,
        )
        assert theta_star > 0
