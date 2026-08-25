"""Tests of the bucket-validation entry point (offline synthetic mode)."""

from __future__ import annotations

import numpy as np

from postprocessing.metrics import validate_experiment
from postprocessing.validate_bucket import _synthetic_experiments


class TestSyntheticValidation:
    def test_synthetic_experiments_build(self) -> None:
        experiments = _synthetic_experiments()
        assert set(experiments) == {
            "synthetic_new_convention",
            "synthetic_legacy_convention",
        }

    def test_new_convention_is_physically_valid(self) -> None:
        for name, exp in _synthetic_experiments().items():
            report = validate_experiment(name, exp)
            assert report.passed, report

    def test_synthetic_matrices_are_stochastic(self) -> None:
        experiments = _synthetic_experiments()
        P_new = experiments["synthetic_new_convention"]["species"]["small"]["P"]
        P_legacy = experiments["synthetic_legacy_convention"]["species"]["small"]["P"]

        # New convention: rows sum to 1.
        np.testing.assert_allclose(P_new.sum(axis=1), 1.0)
        # Legacy storage is the transpose: columns sum to 1.
        np.testing.assert_allclose(P_legacy.sum(axis=0), 1.0)
        np.testing.assert_allclose(P_legacy, P_new.T)
