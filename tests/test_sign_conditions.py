"""Tests for src/sign_conditions.py — hedging direction and sign conditions."""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/workspace/project")

from src.sign_conditions import (
    build_two_asset_covariance,
    compute_sigma_inv_kappa,
    green_condition_threshold,
    brown_condition_threshold,
    classify_reallocation_direction,
    sweep_sign_conditions,
)

# Paper baseline
SIGMA_G, SIGMA_B, RHO = 0.15, 0.30, 0.7
KAPPA_G, KAPPA_B = 0.2, -0.5


class TestBuildTwoAssetCovariance:
    def test_shape(self):
        S = build_two_asset_covariance(SIGMA_G, SIGMA_B, RHO)
        assert S.shape == (2, 2)

    def test_symmetry(self):
        S = build_two_asset_covariance(SIGMA_G, SIGMA_B, RHO)
        assert np.allclose(S, S.T)

    def test_diagonal_entries(self):
        S = build_two_asset_covariance(SIGMA_G, SIGMA_B, RHO)
        assert S[0, 0] == pytest.approx(SIGMA_G**2)
        assert S[1, 1] == pytest.approx(SIGMA_B**2)

    def test_off_diagonal(self):
        S = build_two_asset_covariance(SIGMA_G, SIGMA_B, RHO)
        assert S[0, 1] == pytest.approx(RHO * SIGMA_G * SIGMA_B)

    def test_positive_definite(self):
        S = build_two_asset_covariance(SIGMA_G, SIGMA_B, RHO)
        eigvals = np.linalg.eigvalsh(S)
        assert np.all(eigvals > 0)

    def test_rho_boundary_raises(self):
        with pytest.raises((ValueError, AssertionError)):
            build_two_asset_covariance(SIGMA_G, SIGMA_B, rho=1.0)


class TestComputeSigmaInvKappa:
    def test_shape(self):
        sik = compute_sigma_inv_kappa(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert sik.shape == (2,)

    def test_green_component_positive_at_baseline(self):
        """With paper baseline parameters and rho=0.7, green component > 0."""
        sik = compute_sigma_inv_kappa(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert sik[0] > 0, f"Green component should be positive, got {sik[0]:.4f}"

    def test_brown_component_negative_at_baseline(self):
        """With paper baseline parameters and rho=0.7, brown component < 0."""
        sik = compute_sigma_inv_kappa(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert sik[1] < 0, f"Brown component should be negative, got {sik[1]:.4f}"

    def test_consistency_with_matrix_inverse(self):
        """Explicit formula should match matrix inversion."""
        S = build_two_asset_covariance(SIGMA_G, SIGMA_B, RHO)
        kappa = np.array([KAPPA_G, KAPPA_B])
        S_inv_kappa_direct = np.linalg.solve(S, kappa)
        sik = compute_sigma_inv_kappa(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert np.allclose(sik, S_inv_kappa_direct, rtol=1e-8)

    def test_negative_rho_changes_signs(self):
        """With strongly negative rho, sign conditions may change."""
        sik = compute_sigma_inv_kappa(SIGMA_G, SIGMA_B, rho=-0.9, kappa_g=KAPPA_G, kappa_b=KAPPA_B)
        # Just verify it runs and returns finite values
        assert np.all(np.isfinite(sik))


class TestGreenConditionThreshold:
    def test_formula(self):
        """Threshold = (kappa_g * sigma_b) / (kappa_b * sigma_g)."""
        threshold = green_condition_threshold(SIGMA_G, SIGMA_B, KAPPA_G, KAPPA_B)
        expected = (KAPPA_G * SIGMA_B) / (KAPPA_B * SIGMA_G)
        assert threshold == pytest.approx(expected, rel=1e-12)

    def test_baseline_rho_exceeds_threshold(self):
        """With rho=0.7, the baseline should exceed the green threshold (i.e. green > 0)."""
        threshold = green_condition_threshold(SIGMA_G, SIGMA_B, KAPPA_G, KAPPA_B)
        assert RHO > threshold, \
            f"rho={RHO} should exceed green threshold={threshold:.4f}"


class TestBrownConditionThreshold:
    def test_formula(self):
        threshold = brown_condition_threshold(SIGMA_G, SIGMA_B, KAPPA_G, KAPPA_B)
        expected = (KAPPA_B * SIGMA_G) / (KAPPA_G * SIGMA_B)
        assert threshold == pytest.approx(expected, rel=1e-12)

    def test_baseline_rho_exceeds_threshold(self):
        threshold = brown_condition_threshold(SIGMA_G, SIGMA_B, KAPPA_G, KAPPA_B)
        assert RHO > threshold, \
            f"rho={RHO} should exceed brown threshold={threshold:.4f}"


class TestClassifyReallocationDirection:
    def test_baseline_green_positive(self):
        result = classify_reallocation_direction(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert result["green_positive"]

    def test_baseline_brown_negative(self):
        result = classify_reallocation_direction(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert result["brown_negative"]

    def test_kappa_quad_positive(self):
        result = classify_reallocation_direction(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert result["kappa_quad"] > 0

    def test_sigma_inv_kappa_shape(self):
        result = classify_reallocation_direction(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert result["sigma_inv_kappa"].shape == (2,)

    def test_thresholds_both_negative_for_paper_params(self):
        """Both thresholds should be negative for kappa_g>0, kappa_b<0."""
        result = classify_reallocation_direction(SIGMA_G, SIGMA_B, RHO, KAPPA_G, KAPPA_B)
        assert result["green_threshold"] < 0
        assert result["brown_threshold"] < 0


class TestSweepSignConditions:
    def test_default_run(self):
        out = sweep_sign_conditions()
        assert "rho_grid" in out
        assert "green_positive" in out
        assert "brown_negative" in out

    def test_output_length_matches_grid(self):
        rho_grid = np.linspace(-0.9, 0.9, 15)
        out = sweep_sign_conditions(rho_grid=rho_grid)
        assert len(out["green_positive"]) == 15
        assert len(out["brown_negative"]) == 15

    def test_positive_rho_gives_both_conditions(self):
        """All positive rho values in baseline should give green=True, brown=True."""
        rho_grid = np.linspace(0.1, 0.9, 9)
        out = sweep_sign_conditions(kappa_g=KAPPA_G, kappa_b=KAPPA_B, rho_grid=rho_grid)
        assert all(out["green_positive"])
        assert all(out["brown_negative"])
