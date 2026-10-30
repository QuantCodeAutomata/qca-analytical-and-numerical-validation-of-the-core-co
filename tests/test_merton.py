"""Tests for src/merton.py — Merton portfolio and alpha computations."""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/workspace/project")

from src.parameters import build_two_asset_params
from src.merton import (
    merton_power,
    merton_log,
    alpha_power,
    alpha_log,
    compute_merton_quantities,
)

# ── shared fixtures ──────────────────────────────────────────────────────────

# Two-asset covariance matrix from paper baseline
SIGMA = np.array([
    [0.15**2,              0.7 * 0.15 * 0.30],
    [0.7 * 0.15 * 0.30,   0.30**2           ],
])
R = 0.02
MU_STD = np.array([0.015, 0.035])   # standard drift case
MU_REV = np.array([0.035, 0.015])   # reverse-drift case
ETA = 2.0


class TestMertonPower:
    def test_standard_drift_approx_paper(self):
        """Paper reports approximately (-0.45, 0.24) for standard drift, eta=2."""
        pi = merton_power(MU_STD, R, SIGMA, ETA)
        assert pi[0] == pytest.approx(-0.45, abs=0.02)
        assert pi[1] == pytest.approx(0.24, abs=0.02)

    def test_reverse_drift_approx_paper(self):
        """Paper reports approximately (0.73, -0.28) for reverse drift, eta=2."""
        pi = merton_power(MU_REV, R, SIGMA, ETA)
        assert pi[0] == pytest.approx(0.73, abs=0.02)
        assert pi[1] == pytest.approx(-0.28, abs=0.02)

    def test_eta_scaling(self):
        """Doubling eta should halve the Merton weights."""
        pi1 = merton_power(MU_STD, R, SIGMA, 1.0)
        pi2 = merton_power(MU_STD, R, SIGMA, 2.0)
        assert np.allclose(pi1, 2.0 * pi2, rtol=1e-10)

    def test_zero_excess_drift(self):
        """If mu == r*1, Merton portfolio is zero."""
        mu_flat = R * np.ones(2)
        pi = merton_power(mu_flat, R, SIGMA, ETA)
        assert np.allclose(pi, 0.0, atol=1e-12)

    def test_output_shape(self):
        pi = merton_power(MU_STD, R, SIGMA, ETA)
        assert pi.shape == (2,)


class TestMertonLog:
    def test_log_is_eta1_limit(self):
        """log-utility = power utility with eta=1."""
        pi_log = merton_log(MU_STD, R, SIGMA)
        pi_pow = merton_power(MU_STD, R, SIGMA, eta=1.0)
        assert np.allclose(pi_log, pi_pow, rtol=1e-12)

    def test_log_scales_power_by_eta(self):
        """merton_log == eta * merton_power(..., eta)."""
        pi_log = merton_log(MU_STD, R, SIGMA)
        pi_pow = merton_power(MU_STD, R, SIGMA, ETA)
        assert np.allclose(pi_log, ETA * pi_pow, rtol=1e-10)


class TestAlpha:
    def test_alpha_power_exceeds_r(self):
        """Growth rate for power utility must be >= r when mu != r*1."""
        a = alpha_power(MU_STD, R, SIGMA, ETA)
        assert a >= R

    def test_alpha_log_exceeds_r(self):
        a = alpha_log(MU_STD, R, SIGMA)
        assert a >= R

    def test_alpha_power_at_merton_r_when_mu_flat(self):
        mu_flat = R * np.ones(2)
        a = alpha_power(mu_flat, R, SIGMA, ETA)
        assert a == pytest.approx(R, abs=1e-12)

    def test_alpha_log_equals_alpha_power_eta1(self):
        a_log = alpha_log(MU_STD, R, SIGMA)
        a_pow = alpha_power(MU_STD, R, SIGMA, eta=1.0)
        assert a_log == pytest.approx(a_pow, rel=1e-10)


class TestComputeMertonQuantities:
    def test_keys_power(self):
        params = build_two_asset_params()
        mq = compute_merton_quantities(params)
        expected_keys = {"pi0_merton", "pi1_merton", "alpha0", "alpha1",
                         "Sigma0_inv_kappa", "kappa_Sigma0_inv_kappa"}
        assert expected_keys == set(mq.keys())

    def test_kappa_quad_positive(self):
        """kappa^T Sigma^{-1} kappa must be positive (positive definite Sigma)."""
        params = build_two_asset_params()
        mq = compute_merton_quantities(params)
        assert mq["kappa_Sigma0_inv_kappa"] > 0

    def test_sign_standard_drift(self):
        """Standard drift: (pi^{M,eta})^T kappa should be negative."""
        params = build_two_asset_params(mu_g=0.015, mu_b=0.035, kappa_g=0.2, kappa_b=-0.5)
        mq = compute_merton_quantities(params)
        sign = np.dot(mq["pi0_merton"], params.kappa)
        assert sign < 0, f"Expected negative sign, got {sign:.4f}"

    def test_sign_reverse_drift(self):
        """Reverse drift: (pi^{M,eta})^T kappa should be positive."""
        params = build_two_asset_params(mu_g=0.035, mu_b=0.015, kappa_g=0.2, kappa_b=-0.5)
        mq = compute_merton_quantities(params)
        sign = np.dot(mq["pi0_merton"], params.kappa)
        assert sign > 0, f"Expected positive sign, got {sign:.4f}"

    def test_log_utility_keys(self):
        params = build_two_asset_params(eta=1.0, epsilon=0.5, utility="log")
        mq = compute_merton_quantities(params)
        assert "pi0_merton" in mq
