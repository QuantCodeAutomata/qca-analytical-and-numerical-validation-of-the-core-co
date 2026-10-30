"""Tests for src/parameters.py — ModelParams construction and validation."""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/workspace/project")

from src.parameters import ModelParams, build_two_asset_params


def make_valid_power_params(**overrides):
    defaults = dict(
        T=10.0, r=0.02, lam=0.1,
        mu_0=[0.015, 0.035], mu_1=[0.015, 0.035],
        Sigma_0=[[0.0225, 0.0315], [0.0315, 0.09]],
        Sigma_1=[[0.0225, 0.0315], [0.0315, 0.09]],
        kappa=[0.2, -0.5], eta=2.0, epsilon=2.0,
        n_grid=50, utility="power",
    )
    defaults.update(overrides)
    return ModelParams(**defaults)


class TestModelParamsConstruction:
    def test_valid_power_params(self):
        p = make_valid_power_params()
        assert p.m == 2
        assert p.T == 10.0
        assert p.utility == "power"

    def test_valid_log_params(self):
        p = build_two_asset_params(eta=1.0, epsilon=0.5, utility="log")
        assert p.utility == "log"

    def test_positive_horizon(self):
        with pytest.raises(AssertionError):
            make_valid_power_params(T=-1.0)

    def test_positive_lambda(self):
        with pytest.raises(AssertionError):
            make_valid_power_params(lam=-0.1)

    def test_kappa_admissibility(self):
        with pytest.raises(AssertionError):
            make_valid_power_params(kappa=[-1.5, 0.1])

    def test_epsilon_power_admissibility(self):
        # epsilon must be > eta - 1 = 1
        with pytest.raises(AssertionError):
            make_valid_power_params(eta=2.0, epsilon=0.5)  # 0.5 <= 2-1=1

    def test_sigma_not_pos_def(self):
        bad_Sigma = [[1.0, 2.0], [2.0, 1.0]]  # not positive definite
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            make_valid_power_params(Sigma_0=bad_Sigma)

    def test_power_eta_not_1(self):
        with pytest.raises(AssertionError):
            make_valid_power_params(eta=1.0, epsilon=2.0, utility="power")


class TestBuildTwoAssetParams:
    def test_default_build(self):
        p = build_two_asset_params()
        assert p.m == 2
        assert p.Sigma_0[0, 0] == pytest.approx(0.15**2)
        assert p.Sigma_0[1, 1] == pytest.approx(0.30**2)
        assert p.Sigma_0[0, 1] == pytest.approx(0.7 * 0.15 * 0.30)

    def test_post_equal_pre(self):
        p = build_two_asset_params(mu_g=0.015, mu_b=0.035, post_equal_pre=True)
        assert np.allclose(p.mu_0, p.mu_1)
        assert np.allclose(p.Sigma_0, p.Sigma_1)

    def test_kappa_vector(self):
        p = build_two_asset_params(kappa_g=0.2, kappa_b=-0.5)
        assert p.kappa[0] == pytest.approx(0.2)
        assert p.kappa[1] == pytest.approx(-0.5)

    def test_positive_definite_auto(self):
        # Default correlation 0.7 gives positive definite Sigma
        p = build_two_asset_params()
        eigenvalues = np.linalg.eigvalsh(p.Sigma_0)
        assert np.all(eigenvalues > 0)
