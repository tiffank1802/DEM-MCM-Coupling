"""
tests/test_markov_math.py.
=========================
Unit tests for Markov mathematical operations.

Tests verify:
- Matrix analysis functions
- RSD and entropy computations
- Normalization validation
- Trajectory comparisons
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.Markov.markov_math import (
    analyze_transition_matrix,
    compare_trajectories,
    compute_entropy,
    compute_rsd,
    validate_normalization,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def stochastic_matrix():
    """Create a valid stochastic matrix."""
    M = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ],
        dtype=np.float64,
    )
    return M


@pytest.fixture
def markov_trajectory():
    """Create a sample Markov trajectory."""
    n_timesteps = 100
    n_states = 5
    trajectory = np.random.rand(n_timesteps, n_states)
    # Normalize to stochastic
    trajectory = (
        trajectory / trajectory.sum(axis=1, keepdims=True) * 100
    )  # Scale to 100 particles
    return trajectory


@pytest.fixture
def segregated_trajectory():
    """Create a segregated trajectory (non-mixed)."""
    n_timesteps = 100
    n_states = 5
    trajectory = np.zeros((n_timesteps, n_states))
    # Put all particles in first state initially
    trajectory[:, 0] = 100
    # Gradually spread
    for t in range(n_timesteps):
        fraction = t / n_timesteps
        trajectory[t, 0] = 100 * (1 - fraction)
        trajectory[t, 1:] = 100 * fraction / (n_states - 1)
    return trajectory


@pytest.fixture
def mixed_trajectory():
    """Create a well-mixed trajectory."""
    n_timesteps = 100
    n_states = 5
    # All states equally populated at all times
    return np.ones((n_timesteps, n_states)) * (100 / n_states)


# ============================================================================
# TEST: analyze_transition_matrix
# ============================================================================


class TestAnalyzeTransitionMatrix:
    """Test matrix analysis functions."""

    def test_analyze_basic_matrix(self, stochastic_matrix) -> None:
        """Test analyzing a basic stochastic matrix."""
        props = analyze_transition_matrix(stochastic_matrix)

        assert "largest_eigenvalue" in props
        assert "spectral_gap" in props
        assert "condition_number" in props

    def test_largest_eigenvalue_one(self, stochastic_matrix) -> None:
        """Test that largest eigenvalue is 1 for stochastic matrix."""
        props = analyze_transition_matrix(stochastic_matrix)

        assert np.isclose(props["largest_eigenvalue"], 1.0, atol=1e-10)

    def test_spectral_gap_positive(self, stochastic_matrix) -> None:
        """Test that spectral gap is positive (mixing)."""
        props = analyze_transition_matrix(stochastic_matrix)

        assert props["spectral_gap"] > 0
        assert props["spectral_gap"] < 1

    def test_condition_number_positive(self, stochastic_matrix) -> None:
        """Test that condition number is positive."""
        props = analyze_transition_matrix(stochastic_matrix)

        assert props["condition_number"] > 0

    def test_identity_matrix(self) -> None:
        """Test identity matrix (no mixing)."""
        I = np.eye(5)
        props = analyze_transition_matrix(I)

        # Spectral gap should be 0 (no mixing)
        assert np.isclose(props["spectral_gap"], 0.0, atol=1e-10)


# ============================================================================
# TEST: RSD Computation
# ============================================================================


class TestComputeRSD:
    """Test RSD (relative standard deviation) computation."""

    def test_uniform_distribution_rsd(self) -> None:
        """Test RSD for uniform distribution (well-mixed)."""
        phi = np.ones(10) * 10  # Uniform
        rsd = compute_rsd(phi)

        # Should be near 0 for uniform distribution
        assert rsd < 0.01

    def test_segregated_distribution_rsd(self) -> None:
        """Test RSD for segregated distribution."""
        phi = np.zeros(10)
        phi[0] = 100  # All particles in first state
        rsd = compute_rsd(phi)

        # Should be high for segregated
        assert rsd > 1.0

    def test_rsd_positive(self) -> None:
        """Test that RSD is always non-negative."""
        for _ in range(10):
            phi = np.random.exponential(1, 10)
            rsd = compute_rsd(phi)
            assert rsd >= 0

    def test_empty_states_handling(self) -> None:
        """Test RSD with empty states."""
        phi = np.array([10, 0, 10, 0, 10])
        rsd = compute_rsd(phi)

        # Should handle empty states gracefully
        assert np.isfinite(rsd)


# ============================================================================
# TEST: Normalization Validation
# ============================================================================


class TestValidateNormalization:
    """Test normalization checking."""

    def test_valid_trajectory(self, markov_trajectory) -> None:
        """Test trajectory with proper normalization."""
        total_particles = markov_trajectory.sum(axis=1)

        # Should all be close to 100
        result = validate_normalization(markov_trajectory, total_particles[0])

        assert "is_valid" in result

    def test_invalid_trajectory(self) -> None:
        """Test trajectory with drift in normalization."""
        trajectory = np.linspace(100, 90, 100).reshape(-1, 1).repeat(5, axis=1)
        trajectory = trajectory / trajectory.sum(axis=1, keepdims=True) * 100
        trajectory[:, 0] += np.linspace(0, 10, 100)  # Introduce drift

        result = validate_normalization(trajectory, 100)

        # Should detect deviation
        assert result["max_deviation"] > 0

    def test_perfectly_conserved(self) -> None:
        """Test perfectly conserved trajectory."""
        n_particles = 100
        trajectory = np.ones((50, 10)) * (n_particles / 10)

        result = validate_normalization(trajectory, n_particles)

        assert result["is_valid"]
        assert result["max_deviation"] < 1e-10


# ============================================================================
# TEST: Entropy Computation
# ============================================================================


class TestComputeEntropy:
    """Test entropy calculations."""

    def test_uniform_entropy_maximum(self) -> None:
        """Test that uniform distribution has maximum entropy."""
        phi_uniform = np.ones(10) * 10
        entropy_uniform = compute_entropy(phi_uniform)

        phi_segregated = np.zeros(10)
        phi_segregated[0] = 100
        entropy_seg = compute_entropy(phi_segregated)

        assert entropy_uniform > entropy_seg

    def test_entropy_bounds(self) -> None:
        """Test that entropy is bounded."""
        for _ in range(10):
            phi = np.random.exponential(10, 20)
            entropy = compute_entropy(phi)

            # Typically 0 to 1 for well-normalized distributions
            assert 0 <= entropy <= 1.01 or np.isnan(entropy)

    def test_zero_entropy_degenerate(self) -> None:
        """Test degenerate case (all in one state)."""
        phi = np.zeros(10)
        phi[0] = 100

        entropy = compute_entropy(phi)

        # Degenerate case should give low entropy
        assert entropy < 0.1 or np.isnan(entropy)


# ============================================================================
# TEST: Trajectory Comparison
# ============================================================================


class TestCompareTrajectories:
    """Test trajectory comparison metrics."""

    def test_identical_trajectories(self, markov_trajectory) -> None:
        """Test comparison of identical trajectories."""
        result = compare_trajectories(markov_trajectory, markov_trajectory)

        # Distance should be near 0
        assert "distances" in result
        assert np.allclose(result["distances"], 0)

    def test_different_trajectories(
        self, segregated_trajectory, mixed_trajectory
    ) -> None:
        """Test comparison of different trajectories."""
        result = compare_trajectories(segregated_trajectory, mixed_trajectory)

        # Distance should be significant
        assert result["mean_distance"] > 0

    def test_distance_symmetry(self, markov_trajectory) -> None:
        """Test that distance is symmetric."""
        traj1 = markov_trajectory[:50]
        traj2 = markov_trajectory[50:]

        result1 = compare_trajectories(traj1, traj2)
        result2 = compare_trajectories(traj2, traj1)

        # Distances should be similar (symmetric)
        assert np.isclose(result1["mean_distance"], result2["mean_distance"], rtol=0.1)


# ============================================================================
# TEST: Integration with Trajectories
# ============================================================================


class TestTrajectoryIntegration:
    """Test integration of math functions with trajectories."""

    def test_segregated_to_mixed_analysis(self, segregated_trajectory) -> None:
        """Test analyzing segregation evolution."""
        # Compute RSD at each timestep
        rsd_evolution = np.array(
            [
                compute_rsd(segregated_trajectory[t])
                for t in range(len(segregated_trajectory))
            ]
        )

        # RSD should decrease over time
        assert rsd_evolution[0] > rsd_evolution[-1]

    def test_entropy_increase_over_time(self, segregated_trajectory) -> None:
        """Test that entropy increases with mixing."""
        entropy_evolution = np.array(
            [
                compute_entropy(segregated_trajectory[t])
                for t in range(len(segregated_trajectory))
            ]
        )

        # Entropy should generally increase
        assert entropy_evolution[-1] > entropy_evolution[0]

    def test_convergence_to_steady_state(self, mixed_trajectory) -> None:
        """Test that well-mixed trajectory stays mixed."""
        rsd_evolution = np.array(
            [compute_rsd(mixed_trajectory[t]) for t in range(len(mixed_trajectory))]
        )

        # RSD should be low and stable
        assert np.all(rsd_evolution < 0.1)
        assert np.std(rsd_evolution) < 0.02


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_very_large_matrix(self) -> None:
        """Test with large transition matrix."""
        n = 500
        M = np.eye(n) * 0.999 + np.ones((n, n)) * 0.001 / n

        props = analyze_transition_matrix(M)

        assert np.isfinite(props["largest_eigenvalue"])
        assert np.isfinite(props["condition_number"])

    def test_very_small_values(self) -> None:
        """Test with very small population values."""
        phi = np.array([1e-10, 1e-10, 1e-10, 1e-10, 1e-10])

        rsd = compute_rsd(phi)

        # Should handle gracefully
        assert np.isfinite(rsd)

    def test_single_state_trajectory(self) -> None:
        """Test trajectory with single state."""
        trajectory = np.ones((100, 1)) * 100

        rsd = compute_rsd(trajectory[0])
        entropy = compute_entropy(trajectory[0])

        # Should be defined (though degenerate)
        assert np.isfinite(rsd) or rsd == 0
        assert np.isfinite(entropy) or entropy == 0 or np.isnan(entropy)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
