"""Tests for src/verification.py — admissibility, solution consistency, ODE residuals."""

import pytest
import numpy as np
import sys

sys.path.insert(0, "/workspace/project")

from src.parameters import build_two_asset_params
from src.ode_solver import solve_pretransition
from src.verification import check_admissibility, verify_solution, compute_ode_residuals


def make_params(**kw):
    defaults = dict(n_grid=100, lam=0.1, kappa_g=0.2, kappa_b=-0.5,
                    eta=2.0, epsilon=1.5, T=5.0)
    defaults.update(kw)
    return build_two_asset_params(**defaults)


class TestCheckAdmissibility:
    def test_valid_params_passes(self):
        params = make_params()
        result = check_admissibility(params)
        for key, val in result.items():
            assert val, f"Admissibility check {key!r} failed unexpectedly."

    def test_sigma0_positive_definite(self):
        params = make_params()
        result = check_admissibility(params)
        assert result.get("Sigma0_pos_def", True)

    def test_sigma1_positive_definite(self):
        params = make_params()
        result = check_admissibility(params)
        assert result.get("Sigma1_pos_def", True)

    def test_epsilon_domain_power(self):
        params = make_params(eta=2.0, epsilon=1.5)  # epsilon > eta-1 = 1.0
        result = check_admissibility(params)
        assert result.get("epsilon_domain", True)

    def test_log_utility_admissibility(self):
        params = build_two_asset_params(eta=1.0, epsilon=0.5, utility="log")
        result = check_admissibility(params)
        for key, val in result.items():
            assert val, f"Log admissibility check {key!r} failed."


class TestVerifySolution:
    def setup_method(self):
        self.params = make_params(n_grid=100)
        self.sol = solve_pretransition(self.params)

    def test_returns_dict(self):
        result = verify_solution(self.params, self.sol)
        assert isinstance(result, dict)

    def test_xi_positive_key(self):
        result = verify_solution(self.params, self.sol)
        assert result.get("xi_all_positive", True)

    def test_theta_positive_key(self):
        result = verify_solution(self.params, self.sol)
        assert result.get("theta_all_positive", True)

    def test_identity_satisfied(self):
        result = verify_solution(self.params, self.sol)
        assert result.get("identity_max_err", 1.0) < 1e-3, \
            "Identity 1 + pi^T kappa = xi should hold with low error."

    def test_root_residuals_small(self):
        result = verify_solution(self.params, self.sol)
        assert result.get("max_root_residual", 1.0) < 1e-6

    def test_all_checks_pass(self):
        result = verify_solution(self.params, self.sol)
        for key, val in result.items():
            if isinstance(val, bool):
                assert val, f"Verification check {key!r} failed."


class TestComputeODEResiduals:
    def setup_method(self):
        self.params = make_params(n_grid=200)
        self.sol = solve_pretransition(self.params)

    def test_returns_array(self):
        residuals = compute_ode_residuals(self.params, self.sol)
        assert isinstance(residuals, np.ndarray)

    def test_residuals_shape(self):
        residuals = compute_ode_residuals(self.params, self.sol)
        # interior points only (excluding first/last)
        assert residuals.ndim == 1

    def test_residuals_small(self):
        residuals = compute_ode_residuals(self.params, self.sol)
        assert np.max(np.abs(residuals)) < 0.05, \
            f"ODE residuals too large: max = {np.max(np.abs(residuals)):.4f}"
