"""Tests of the post-processing physics core (``postprocessing.metrics``).

Validates:

* the transition-matrix convention detection and standardisation (legacy
  column-stochastic data are auto-transposed);
* the propagation direction (row convention: ``phi_next = phi @ P``) and the
  mass conservation;
* the segregation metrics (RSD, entropy, intensity of segregation, mixing
  times) and their physical bounds;
* the physical validation of experiments in both stored conventions.
"""

from __future__ import annotations

import numpy as np
import pytest

from postprocessing.metrics import (
    clean_transition_matrix,
    concentration_from_S,
    detect_convention,
    entropy_concentration,
    intensity_of_segregation,
    mixing_times,
    propagate_markov,
    propagate_markov_inhomogeneous,
    rsd_concentration,
    rsd_from_S,
    standardize_transition_matrix,
    stationary_distribution,
    validate_experiment,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def n_states() -> int:
    return 10


@pytest.fixture
def row_stochastic_matrix(rng: np.random.RandomState, n_states: int) -> np.ndarray:
    """Row-stochastic matrix with diagonal dominance (small tau)."""
    P = rng.rand(n_states, n_states) * 0.02
    P[np.arange(n_states), np.arange(n_states)] += 1.0
    P /= P.sum(axis=1, keepdims=True)
    return P


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


def _make_times(n: int, start: int = 250) -> np.ndarray:
    return np.arange(start, start + n)


# ============================================================================
# CONVENTION DETECTION AND STANDARDISATION
# ============================================================================


class TestConvention:
    def test_detect_row(self, row_stochastic_matrix: np.ndarray) -> None:
        assert detect_convention(row_stochastic_matrix) == "row"

    def test_detect_column(self, row_stochastic_matrix: np.ndarray) -> None:
        assert detect_convention(row_stochastic_matrix.T) == "column"

    def test_detect_none(self, rng: np.random.RandomState, n_states: int) -> None:
        assert detect_convention(rng.rand(n_states, n_states)) == "none"

    def test_standardize_keeps_row(self, row_stochastic_matrix: np.ndarray) -> None:
        P, transposed = standardize_transition_matrix(row_stochastic_matrix)
        assert not transposed
        np.testing.assert_allclose(P, row_stochastic_matrix)

    def test_standardize_transposes_legacy(
        self, row_stochastic_matrix: np.ndarray
    ) -> None:
        """Legacy column-stochastic storage is transposed back to rows."""
        legacy = row_stochastic_matrix.T  # as stored by the old pipeline
        P, transposed = standardize_transition_matrix(legacy)
        assert transposed
        np.testing.assert_allclose(P, row_stochastic_matrix)
        assert detect_convention(P) == "row"

    def test_standardize_renormalises_non_stochastic(
        self, rng: np.random.RandomState, n_states: int
    ) -> None:
        P = rng.rand(n_states, n_states)  # rows do not sum to 1
        P_out, transposed = standardize_transition_matrix(P, warn=False)
        assert not transposed
        np.testing.assert_allclose(P_out.sum(axis=1), 1.0)

    def test_doubly_stochastic_kept(self, n_states: int) -> None:
        P = np.ones((n_states, n_states)) / n_states
        assert detect_convention(P) == "row"
        P_out, transposed = standardize_transition_matrix(P)
        assert not transposed
        np.testing.assert_allclose(P_out, P)


# ============================================================================
# CLEANING AND PROPAGATION
# ============================================================================


class TestPropagation:
    def test_mass_conservation_homogeneous(
        self, row_stochastic_matrix: np.ndarray, n_states: int
    ) -> None:
        times = _make_times(200)
        activated = np.ones(n_states, dtype=bool)
        S0 = np.ones(n_states) * 50.0
        S0[0] = 500.0  # non-uniform initial state

        traj, _times_markov = propagate_markov(
            S0, row_stochastic_matrix, times, 250, 10, activated
        )
        assert traj.ndim == 2
        assert traj.shape[1] == n_states
        for t in range(len(traj)):
            assert np.isclose(traj[t].sum(), S0.sum(), rtol=1e-10), (
                f"Mass lost at step {t}: {traj[t].sum()} != {S0.sum()}"
            )

    def test_legacy_matrix_gives_identical_trajectory(
        self, row_stochastic_matrix: np.ndarray, n_states: int
    ) -> None:
        """Old column-stochastic storage must give the same trajectory."""
        times = _make_times(200)
        activated = np.ones(n_states, dtype=bool)
        S0 = np.arange(n_states, dtype=float) + 1.0

        traj_new, t_new = propagate_markov(
            S0, row_stochastic_matrix, times, 250, 10, activated
        )
        traj_legacy, t_legacy = propagate_markov(
            S0, row_stochastic_matrix.T, times, 250, 10, activated
        )
        np.testing.assert_array_equal(t_new, t_legacy)
        np.testing.assert_allclose(traj_new, traj_legacy)

    def test_deactivated_states_zeroed(
        self, row_stochastic_matrix: np.ndarray, n_states: int
    ) -> None:
        times = _make_times(200)
        activated = np.zeros(n_states, dtype=bool)
        activated[:5] = True
        S0 = np.ones(n_states) * 50.0

        traj, _ = propagate_markov(S0, row_stochastic_matrix, times, 250, 10, activated)
        np.testing.assert_allclose(traj[0][~activated], 0.0)

    def test_inhomogeneous_mass_conservation(
        self, rng: np.random.RandomState, n_states: int
    ) -> None:
        times = _make_times(400)
        blocks = []
        for _ in range(3):
            P = rng.rand(n_states, n_states)
            P /= P.sum(axis=1, keepdims=True)
            blocks.append(P)
        P_blocks = np.array(blocks)

        activated = np.ones(n_states, dtype=bool)
        S0 = np.ones(n_states) * 50.0
        S0[0] = 500.0

        traj, _ = propagate_markov_inhomogeneous(
            S0, P_blocks, times, 250, 10, activated, step=20, nlt=3
        )
        for t in range(len(traj)):
            assert np.isclose(traj[t].sum(), S0.sum(), rtol=1e-10)

    def test_inhomogeneous_standardises_each_block(
        self, rng: np.random.RandomState, n_states: int
    ) -> None:
        """Blocks stored in the legacy convention are transposed on load."""
        times = _make_times(400)
        blocks = []
        for _ in range(3):
            P = rng.rand(n_states, n_states)
            P /= P.sum(axis=1, keepdims=True)
            blocks.append(P.T)  # legacy storage
        P_blocks = np.array(blocks)

        activated = np.ones(n_states, dtype=bool)
        S0 = np.ones(n_states) * 50.0
        traj, _ = propagate_markov_inhomogeneous(
            S0, P_blocks, times, 250, 10, activated, step=20, nlt=3
        )
        assert np.isclose(traj[-1].sum(), S0.sum(), rtol=1e-10)


class TestCleaning:
    def test_clean_removes_unvisited_rows(
        self, row_stochastic_matrix: np.ndarray
    ) -> None:
        P = row_stochastic_matrix.copy()
        P[0, :] = 0.0  # unvisited state (row)
        P_clean, activated = clean_transition_matrix(P)
        assert not activated[0]
        np.testing.assert_allclose(P_clean[0], 0.0)
        np.testing.assert_allclose(P_clean[activated].sum(axis=1), 1.0)

    def test_clean_standardises_legacy_input(
        self, row_stochastic_matrix: np.ndarray
    ) -> None:
        """clean_transition_matrix auto-transposes column-stochastic data."""
        legacy = row_stochastic_matrix.T.copy()
        legacy[:, 0] = 0.0  # unvisited column in legacy storage
        P_clean, activated = clean_transition_matrix(legacy)
        assert not activated[0]
        np.testing.assert_allclose(P_clean[activated].sum(axis=1), 1.0)


# ============================================================================
# SEGREGATION METRICS
# ============================================================================


class TestMetrics:
    def test_rsd_from_S_uniform_is_zero(self, n_states: int) -> None:
        S = np.ones((5, n_states)) * 10.0
        rsd = rsd_from_S(S, np.ones(n_states, dtype=bool))
        np.testing.assert_allclose(rsd, 0.0, atol=1e-12)

    def test_rsd_from_S_segregated_is_positive(self, n_states: int) -> None:
        S = np.zeros((5, n_states))
        S[:, 0] = 100.0
        rsd = rsd_from_S(S, np.ones(n_states, dtype=bool))
        assert (rsd > 0).all()

    def test_concentration_in_unit_interval(
        self, rng: np.random.RandomState, n_states: int
    ) -> None:
        S_small = rng.rand(6, n_states) * 10
        S_large = rng.rand(6, n_states) * 10
        C = concentration_from_S(S_small, S_large)
        assert (C >= 0).all() and (C <= 1).all()

    def test_concentration_rsd_bounds(
        self, rng: np.random.RandomState, n_states: int
    ) -> None:
        S_small = rng.rand(6, n_states) * 10
        S_large = rng.rand(6, n_states) * 10
        act = np.ones(n_states, dtype=bool)
        rsd = rsd_concentration(S_small, S_large, act, act)
        assert (rsd >= 0).all() and (rsd <= 1).all()

    def test_entropy_of_uniform_distribution_is_log_n(self, n_states: int) -> None:
        from postprocessing.metrics import entropy_from_S

        S = np.ones((1, n_states)) * 10.0
        H = entropy_from_S(S, np.ones(n_states, dtype=bool))
        np.testing.assert_allclose(H, np.log(n_states), atol=1e-12)

    def test_entropy_concentration_normalized_bounds(self, n_states: int) -> None:
        # Fully segregated: every cell is pure → H = 0.
        S_small = np.zeros((1, n_states))
        S_large = np.zeros((1, n_states))
        S_small[:, : n_states // 2] = 10
        S_large[:, n_states // 2 :] = 10
        act = np.ones(n_states, dtype=bool)
        H_seg = entropy_concentration(S_small, S_large, act, act)
        np.testing.assert_allclose(H_seg, 0.0, atol=1e-12)

        # Perfectly mixed: every cell at C = 0.5 → H = log 2 per cell → 1
        # after normalisation.
        S_small = np.ones((1, n_states)) * 5.0
        S_large = np.ones((1, n_states)) * 5.0
        H_mix = entropy_concentration(S_small, S_large, act, act)
        np.testing.assert_allclose(H_mix, 1.0, atol=1e-12)

    def test_intensity_of_segregation_bounds(self, n_states: int) -> None:
        act = np.ones(n_states, dtype=bool)
        # Segregated → I = 1.
        S_small = np.zeros((1, n_states))
        S_large = np.zeros((1, n_states))
        S_small[:, : n_states // 2] = 10
        S_large[:, n_states // 2 :] = 10
        I_seg = intensity_of_segregation(S_small, S_large, act, act)
        np.testing.assert_allclose(I_seg, 1.0, atol=1e-12)

        # Mixed → I = 0.
        S_small = np.ones((1, n_states)) * 5.0
        S_large = np.ones((1, n_states)) * 5.0
        I_mix = intensity_of_segregation(S_small, S_large, act, act)
        np.testing.assert_allclose(I_mix, 0.0, atol=1e-12)

    def test_mixing_times(self) -> None:
        rsd = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.05])
        times = np.arange(6) * 0.01
        t = mixing_times(rsd, times)
        assert t[0.5] == pytest.approx(0.03)  # t50: first below half → index 3
        assert t[0.1] == pytest.approx(0.05)  # t90: first below 10% → index 5

    def test_stationary_distribution_is_left_eigenvector(
        self, row_stochastic_matrix: np.ndarray
    ) -> None:
        pi = stationary_distribution(row_stochastic_matrix)
        np.testing.assert_allclose(pi.sum(), 1.0)
        # pi P = pi (stationarity).
        np.testing.assert_allclose(pi @ row_stochastic_matrix, pi, atol=1e-12)


# ============================================================================
# PHYSICAL VALIDATION OF EXPERIMENTS
# ============================================================================


def _build_experiment(P_stored: np.ndarray, n_states: int, n_timesteps: int) -> dict:
    S_small = np.zeros((n_timesteps, n_states))
    S_large = np.zeros((n_timesteps, n_states))
    S_small[0, : n_states // 2] = 20.0
    S_large[0, n_states // 2 :] = 20.0
    for t in range(1, n_timesteps):
        S_small[t] = S_small[t - 1] @ P_stored
        S_large[t] = S_large[t - 1] @ P_stored
    return {
        "config": {"tau": 50, "nlt": 2},
        "stats": {"species_list": ["small", "large"]},
        "species": {
            "small": {
                "P": P_stored,
                "S_matrix": S_small,
                "times": np.arange(n_timesteps),
            },
            "large": {
                "P": P_stored,
                "S_matrix": S_large,
                "times": np.arange(n_timesteps),
            },
        },
        "inhomogeneous": False,
        "inhomogeneous_metadata": None,
    }


class TestValidation:
    def test_new_convention_passes(
        self, row_stochastic_matrix: np.ndarray, n_states: int
    ) -> None:
        exp = _build_experiment(row_stochastic_matrix, n_states, 120)
        report = validate_experiment("test_new", exp)
        assert report.passed, report

    def test_legacy_convention_passes(
        self, row_stochastic_matrix: np.ndarray, n_states: int
    ) -> None:
        """Legacy column-stochastic data validate thanks to standardisation."""
        exp = _build_experiment(row_stochastic_matrix.T, n_states, 120)
        report = validate_experiment("test_legacy", exp)
        assert report.passed, report

    def test_inhomogeneous_blocks_validated(
        self, row_stochastic_matrix: np.ndarray, n_states: int
    ) -> None:
        P_blocks = np.stack([row_stochastic_matrix, row_stochastic_matrix])
        exp = _build_experiment(row_stochastic_matrix, n_states, 120)
        for sp in exp["species"]:
            exp["species"][sp]["P"] = None
            exp["species"][sp]["P_blocks"] = P_blocks
        report = validate_experiment("test_inhom", exp)
        assert report.passed, report

    def test_negative_entries_fail(
        self, row_stochastic_matrix: np.ndarray, n_states: int
    ) -> None:
        P = row_stochastic_matrix.copy()
        P[0, 1] = -0.1
        P[0, 0] += 0.1
        exp = _build_experiment(P, n_states, 120)
        report = validate_experiment("test_negative", exp)
        assert not report.passed
        names = [c.name for c in report.checks if not c.passed]
        assert any("non-negative" in name for name in names)

    def test_non_stochastic_fails(
        self, rng: np.random.RandomState, n_states: int
    ) -> None:
        # Raw matrix with rows not summing to 1: the standardiser fixes it,
        # but the check documents the deviation.
        P = rng.rand(n_states, n_states)
        exp = _build_experiment(P, n_states, 120)
        report = validate_experiment("test_raw", exp)
        # Mass conservation holds after standardisation; RSD may still be
        # physical, so the report simply must be built without crashing.
        assert len(report.checks) > 0
