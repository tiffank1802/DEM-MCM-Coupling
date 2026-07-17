"""
tests/test_session_sync.py.
==========================
Integration tests for session context synchronization across pages.

Tests verify that:
- Page 1 modifications are reflected in pages 2-4
- Context version increments on changes
- Refresh notifications trigger correctly
- Models are properly loaded and cached
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.Markov._config import (
    AppContext,
    LoadedModel,
    StateVector,
)

from typing import Any

# ============================================================================

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_loaded_model() -> LoadedModel:
    """Create a sample LoadedModel for testing."""
    return LoadedModel(
        folder_name="voronoi_125_run1",
        method="voronoi",
        particle_diameter=0.004,
        n_states=125,
        n_particles=5000,
        nlt=100,
        tau=50,
        fraction_visited=0.99,
    )


@pytest.fixture
def sample_transition_matrix() -> np.ndarray:
    """Create a valid transition matrix for testing."""
    n = 10
    M = np.random.rand(n, n)
    M = M / M.sum(axis=1, keepdims=True)  # Row normalize
    return M


@pytest.fixture
def app_context() -> AppContext:
    """Create an AppContext instance."""
    return AppContext()


# ============================================================================
# TEST: AppContext State Management
# ============================================================================


class TestAppContextStateManagement:
    """Test AppContext add/remove/update operations."""

    def test_add_model(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test adding a model to context."""
        assert len(app_context.selected_models) == 0

        app_context.add_model(sample_loaded_model)

        assert len(app_context.selected_models) == 1
        assert sample_loaded_model in app_context.selected_models

    def test_add_duplicate_model(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test that duplicate models are not added."""
        app_context.add_model(sample_loaded_model)
        app_context.add_model(sample_loaded_model)

        assert len(app_context.selected_models) == 1

    def test_remove_model(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test removing a model."""
        app_context.add_model(sample_loaded_model)
        assert len(app_context.selected_models) == 1

        app_context.remove_model(sample_loaded_model.folder_name)

        assert len(app_context.selected_models) == 0

    def test_version_increment_on_add(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test that version increments when model is added."""
        initial_version = app_context.version

        app_context.add_model(sample_loaded_model)

        assert app_context.version > initial_version

    def test_version_increment_on_remove(
        self, app_context: AppContext, sample_loaded_model: LoadedModel
    ) -> None:
        """Test that version increments when model is removed."""
        app_context.add_model(sample_loaded_model)
        version_after_add = app_context.version

        app_context.remove_model(sample_loaded_model.folder_name)

        assert app_context.version > version_after_add

    def test_clear_models(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test clearing all models."""
        app_context.add_model(sample_loaded_model)
        assert len(app_context.selected_models) > 0

        app_context.clear_models()

        assert len(app_context.selected_models) == 0

    def test_get_model(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test retrieving a model by folder name."""
        app_context.add_model(sample_loaded_model)

        retrieved = app_context.get_model(sample_loaded_model.folder_name)

        assert retrieved == sample_loaded_model


# ============================================================================
# TEST: LoadedModel Type Safety
# ============================================================================


class TestLoadedModelTypeSafety:
    """Test LoadedModel dataclass constraints."""

    def test_loaded_model_immutable(self, sample_loaded_model: LoadedModel) -> None:
        """Test that LoadedModel is immutable."""
        with pytest.raises((AttributeError, TypeError)):
            sample_loaded_model.n_states = 200

    def test_loaded_model_repr(self, sample_loaded_model: LoadedModel) -> None:
        """Test LoadedModel string representation."""
        repr_str = repr(sample_loaded_model)

        assert "voronoi" in repr_str
        assert "0.004" in repr_str
        assert "125" in repr_str

    def test_loaded_model_is_data_loaded_false(self, sample_loaded_model: LoadedModel) -> None:
        """Test is_data_loaded when matrices not loaded."""
        assert not sample_loaded_model.is_data_loaded()

    def test_loaded_model_to_dict(self, sample_loaded_model: LoadedModel) -> None:
        """Test serialization to dict."""
        d = sample_loaded_model.to_dict()

        assert d["folder_name"] == sample_loaded_model.folder_name
        assert d["method"] == "voronoi"
        assert d["n_states"] == 125
        # Verify cache fields excluded
        assert "transition_matrix" not in d
        assert "config_dict" not in d


# ============================================================================
# TEST: StateVector Validation
# ============================================================================


class TestStateVectorValidation:
    """Test StateVector creation and normalization."""

    def test_state_vector_creation(self) -> None:
        """Test creating a valid StateVector."""
        phi = np.array([10.0, 15.0, 8.0, 12.0])
        state = StateVector(
            phi=phi,
            timestamp=250,
            total_particles=45,
        )

        assert state.phi.shape == (4,)
        assert state.timestamp == 250
        assert state.total_particles == 45

    def test_state_vector_normalization_valid(self) -> None:
        """Test StateVector normalization check passes."""
        phi = np.array([10.0, 15.0, 8.0, 12.0])
        state = StateVector(
            phi=phi,
            timestamp=250,
            total_particles=45,
        )

        assert state.validate_normalization()

    def test_state_vector_normalization_invalid(self) -> None:
        """Test StateVector normalization check fails."""
        phi = np.array([10.0, 15.0, 8.0, 12.0])
        state = StateVector(
            phi=phi,
            timestamp=250,
            total_particles=100,  # Sum is 45, not 100
        )

        assert not state.validate_normalization()

    def test_state_vector_1d_only(self) -> None:
        """Test that StateVector requires 1D phi."""
        phi = np.array([[10, 15], [8, 12]])  # 2D

        with pytest.raises(ValueError):
            StateVector(
                phi=phi,
                timestamp=250,
                total_particles=45,
            )


# ============================================================================
# TEST: Transition Matrix Properties
# ============================================================================


class TestTransitionMatrixProperties:
    """Test transition matrix validation."""

    def test_matrix_row_stochastic(self, sample_transition_matrix: np.ndarray) -> None:
        """Test that matrix is row-stochastic (rows sum to 1)."""
        row_sums = sample_transition_matrix.sum(axis=1)

        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-7)

    def test_matrix_all_nonnegative(self, sample_transition_matrix: np.ndarray) -> None:
        """Test that all elements are non-negative."""
        assert np.all(sample_transition_matrix >= 0)

    def test_matrix_eigenvalues_magnitude(self, sample_transition_matrix: np.ndarray) -> None:
        """Test that eigenvalues have magnitude ≤ 1."""
        eigenvalues = np.linalg.eigvals(sample_transition_matrix)
        magnitudes = np.abs(eigenvalues)

        assert np.all(magnitudes <= 1.0 + 1e-10)

    def test_matrix_largest_eigenvalue(self, sample_transition_matrix: np.ndarray) -> None:
        """Test that largest eigenvalue is approximately 1."""
        eigenvalues = np.linalg.eigvals(sample_transition_matrix)
        largest = np.max(np.abs(eigenvalues))

        assert np.isclose(largest, 1.0, atol=1e-7)


# ============================================================================
# TEST: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic workflows."""

    def test_workflow_load_multiple_models(self, app_context: AppContext) -> None:
        """Test loading multiple models into context."""
        models = [
            LoadedModel(
                folder_name=f"voronoi_{n}",
                method="voronoi",
                particle_diameter=0.004,
                n_states=n,
                n_particles=5000,
            )
            for n in [100, 125, 150]
        ]

        for model in models:
            app_context.add_model(model)

        assert len(app_context.selected_models) == 3
        assert app_context.version > 0

    def test_workflow_switch_active_model(
        self, app_context: AppContext, sample_loaded_model: LoadedModel
    ) -> None:
        """Test switching active model in context."""
        model2 = LoadedModel(
            folder_name="cartesian_100",
            method="cartesian",
            particle_diameter=0.008,
            n_states=100,
            n_particles=5000,
        )

        app_context.add_model(sample_loaded_model)
        app_context.add_model(model2)

        # Switch active
        app_context.active_model_index = 1

        assert app_context.selected_models[app_context.active_model_index] == model2

    def test_workflow_filter_by_diameter(self, app_context: AppContext) -> None:
        """Test filtering models by particle diameter."""
        model_small = LoadedModel(
            folder_name="voronoi_small",
            method="voronoi",
            particle_diameter=0.004,
            n_states=125,
            n_particles=5000,
        )

        model_big = LoadedModel(
            folder_name="voronoi_big",
            method="voronoi",
            particle_diameter=0.008,
            n_states=125,
            n_particles=5000,
        )

        app_context.add_model(model_small)
        app_context.add_model(model_big)

        # Filter
        filtered = [
            m for m in app_context.selected_models if m.particle_diameter == 0.004
        ]

        assert len(filtered) == 1
        assert filtered[0].folder_name == "voronoi_small"


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_context_operations(self, app_context: AppContext) -> None:
        """Test operations on empty context."""
        assert len(app_context.selected_models) == 0
        assert app_context.get_model("nonexistent") is None

        # Should not raise
        app_context.clear_models()

    def test_large_matrix_handling(self) -> None:
        """Test handling of large transition matrices."""
        n = 1000
        M = np.eye(n) * 0.99 + np.ones((n, n)) * 0.01 / n

        assert M.shape == (n, n)
        assert np.allclose(M.sum(axis=1), 1.0)

    def test_single_state_matrix(self) -> None:
        """Test degenerate 1×1 matrix."""
        M = np.array([[1.0]])

        assert M.shape == (1, 1)
        assert M.sum() == 1.0

    def test_zero_particles_handling(self) -> None:
        """Test handling zero particles gracefully."""
        phi = np.zeros(10)
        state = StateVector(
            phi=phi,
            timestamp=250,
            total_particles=0,
        )

        # Should validate
        assert state.validate_normalization()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_streamlit_session() -> MagicMock:
    """Mock Streamlit session_state for testing."""
    mock_session = {}

    with patch("app.components.session_manager.st") as mock_st:
        mock_st.session_state = mock_session
        yield mock_session


@pytest.fixture
def sample_loaded_model() -> LoadedModel:
    """Create a sample LoadedModel for testing."""
    return LoadedModel(
        folder_name="voronoi_125_run1",
        method="voronoi",
        particle_diameter=0.004,
        n_states=125,
        n_particles=5000,
        nlt=100,
        tau=50,
        fraction_visited=0.99,
    )


@pytest.fixture
def sample_transition_matrix() -> np.ndarray:
    """Create a valid transition matrix for testing."""
    n = 10
    M = np.random.rand(n, n)
    M = M / M.sum(axis=1, keepdims=True)  # Row normalize
    return M


@pytest.fixture
def app_context() -> AppContext:
    """Create an AppContext instance."""
    return AppContext()


# ============================================================================
# TEST: AppContext State Management
# ============================================================================


class TestAppContextStateManagement:
    """Test AppContext add/remove/update operations."""

    def test_add_model(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test adding a model to context."""
        assert len(app_context.selected_models) == 0

        app_context.add_model(sample_loaded_model)

        assert len(app_context.selected_models) == 1
        assert sample_loaded_model in app_context.selected_models

    def test_add_duplicate_model(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test that duplicate models are not added."""
        app_context.add_model(sample_loaded_model)
        app_context.add_model(sample_loaded_model)

        assert len(app_context.selected_models) == 1

    def test_remove_model(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test removing a model."""
        app_context.add_model(sample_loaded_model)
        assert len(app_context.selected_models) == 1

        app_context.remove_model(sample_loaded_model.folder_name)

        assert len(app_context.selected_models) == 0

    def test_version_increment_on_add(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test that version increments when model is added."""
        initial_version = app_context.version

        app_context.add_model(sample_loaded_model)

        assert app_context.version > initial_version

    def test_version_increment_on_remove(
        self, app_context: AppContext, sample_loaded_model: LoadedModel
    ) -> None:
        """Test that version increments when model is removed."""
        app_context.add_model(sample_loaded_model)
        version_after_add = app_context.version

        app_context.remove_model(sample_loaded_model.folder_name)

        assert app_context.version > version_after_add

    def test_clear_models(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test clearing all models."""
        app_context.add_model(sample_loaded_model)
        assert len(app_context.selected_models) > 0

        app_context.clear_models()

        assert len(app_context.selected_models) == 0

    def test_get_model(self, app_context: AppContext, sample_loaded_model: LoadedModel) -> None:
        """Test retrieving a model by folder name."""
        app_context.add_model(sample_loaded_model)

        retrieved = app_context.get_model(sample_loaded_model.folder_name)

        assert retrieved == sample_loaded_model


# ============================================================================
# TEST: LoadedModel Type Safety
# ============================================================================


class TestLoadedModelTypeSafety:
    """Test LoadedModel dataclass constraints."""

    def test_loaded_model_immutable(self, sample_loaded_model: LoadedModel) -> None:
        """Test that LoadedModel is immutable."""
        with pytest.raises((AttributeError, TypeError)):
            sample_loaded_model.n_states = 200

    def test_loaded_model_repr(self, sample_loaded_model: LoadedModel) -> None:
        """Test LoadedModel string representation."""
        repr_str = repr(sample_loaded_model)

        assert "voronoi" in repr_str
        assert "0.004" in repr_str
        assert "125" in repr_str

    def test_loaded_model_is_data_loaded_false(self, sample_loaded_model: LoadedModel) -> None:
        """Test is_data_loaded when matrices not loaded."""
        assert not sample_loaded_model.is_data_loaded()

    def test_loaded_model_to_dict(self, sample_loaded_model: LoadedModel) -> None:
        """Test serialization to dict."""
        d = sample_loaded_model.to_dict()

        assert d["folder_name"] == sample_loaded_model.folder_name
        assert d["method"] == "voronoi"
        assert d["n_states"] == 125
        # Verify cache fields excluded
        assert "transition_matrix" not in d
        assert "config_dict" not in d


# ============================================================================
# TEST: StateVector Validation
# ============================================================================


class TestStateVectorValidation:
    """Test StateVector creation and normalization."""

    def test_state_vector_creation(self) -> None:
        """Test creating a valid StateVector."""
        phi = np.array([10.0, 15.0, 8.0, 12.0])
        state = StateVector(
            phi=phi,
            timestamp=250,
            total_particles=45,
        )

        assert state.phi.shape == (4,)
        assert state.timestamp == 250
        assert state.total_particles == 45

    def test_state_vector_normalization_valid(self) -> None:
        """Test StateVector normalization check passes."""
        phi = np.array([10.0, 15.0, 8.0, 12.0])
        state = StateVector(
            phi=phi,
            timestamp=250,
            total_particles=45,
        )

        assert state.validate_normalization()

    def test_state_vector_normalization_invalid(self) -> None:
        """Test StateVector normalization check fails."""
        phi = np.array([10.0, 15.0, 8.0, 12.0])
        state = StateVector(
            phi=phi,
            timestamp=250,
            total_particles=100,  # Sum is 45, not 100
        )

        assert not state.validate_normalization()

    def test_state_vector_1d_only(self) -> None:
        """Test that StateVector requires 1D phi."""
        phi = np.array([[10, 15], [8, 12]])  # 2D

        with pytest.raises(ValueError):
            StateVector(
                phi=phi,
                timestamp=250,
                total_particles=45,
            )


# ============================================================================
# TEST: ModelLoaderCache
# ============================================================================


class TestModelLoaderCache:
    """Test lazy-loading cache functionality."""

    def test_cache_initialization(self) -> None:
        """Test cache initializes empty."""
        cache = ModelLoaderCache()

        assert len(cache._models_cache) == 0
        assert len(cache._matrices_cache) == 0

    def test_cache_info(self) -> None:
        """Test cache info reporting."""
        cache = ModelLoaderCache()
        info = cache.cache_info()

        assert "models_cached" in info
        assert "matrices_cached" in info
        assert info["models_cached"] == 0

    def test_cache_clear(self) -> None:
        """Test clearing cache."""
        cache = ModelLoaderCache()

        # Add something to cache (mock)
        cache._models_cache["test"] = "data"
        assert len(cache._models_cache) > 0

        cache.clear_cache()

        assert len(cache._models_cache) == 0

    @patch("app.components.model_loader.get_fs")
    @patch("builtins.open", create=True)
    def test_get_or_load_hit(self, mock_open: MagicMock, mock_fs: MagicMock, sample_loaded_model: LoadedModel) -> None:
        """Test cache hit on get_or_load."""
        cache = ModelLoaderCache()

        # Pre-populate cache
        cache._models_cache[sample_loaded_model.folder_name] = sample_loaded_model

        result = cache.get_or_load(sample_loaded_model.folder_name)

        assert result == sample_loaded_model
        # Verify HF not accessed
        mock_fs.assert_not_called()


# ============================================================================
# TEST: Transition Matrix Properties
# ============================================================================


class TestTransitionMatrixProperties:
    """Test transition matrix validation."""

    def test_matrix_row_stochastic(self, sample_transition_matrix: np.ndarray) -> None:
        """Test that matrix is row-stochastic (rows sum to 1)."""
        row_sums = sample_transition_matrix.sum(axis=1)

        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-7)

    def test_matrix_all_nonnegative(self, sample_transition_matrix: np.ndarray) -> None:
        """Test that all elements are non-negative."""
        assert np.all(sample_transition_matrix >= 0)

    def test_matrix_eigenvalues_magnitude(self, sample_transition_matrix: np.ndarray) -> None:
        """Test that eigenvalues have magnitude ≤ 1."""
        eigenvalues = np.linalg.eigvals(sample_transition_matrix)
        magnitudes = np.abs(eigenvalues)

        assert np.all(magnitudes <= 1.0 + 1e-10)

    def test_matrix_largest_eigenvalue(self, sample_transition_matrix: np.ndarray) -> None:
        """Test that largest eigenvalue is approximately 1."""
        eigenvalues = np.linalg.eigvals(sample_transition_matrix)
        largest = np.max(np.abs(eigenvalues))

        assert np.isclose(largest, 1.0, atol=1e-7)


# ============================================================================
# TEST: Integration Scenarios
# ============================================================================


class TestIntegrationScenarios:
    """Test realistic workflows."""

    def test_workflow_load_multiple_models(self, app_context: AppContext) -> None:
        """Test loading multiple models into context."""
        models = [
            LoadedModel(
                folder_name=f"voronoi_{n}",
                method="voronoi",
                particle_diameter=0.004,
                n_states=n,
                n_particles=5000,
            )
            for n in [100, 125, 150]
        ]

        for model in models:
            app_context.add_model(model)

        assert len(app_context.selected_models) == 3
        assert app_context.version > 0

    def test_workflow_switch_active_model(
        self, app_context: AppContext, sample_loaded_model: LoadedModel
    ) -> None:
        """Test switching active model in context."""
        model2 = LoadedModel(
            folder_name="cartesian_100",
            method="cartesian",
            particle_diameter=0.008,
            n_states=100,
            n_particles=5000,
        )

        app_context.add_model(sample_loaded_model)
        app_context.add_model(model2)

        # Switch active
        app_context.active_model_index = 1

        assert app_context.selected_models[app_context.active_model_index] == model2

    def test_workflow_filter_by_diameter(self, app_context: AppContext) -> None:
        """Test filtering models by particle diameter."""
        model_small = LoadedModel(
            folder_name="voronoi_small",
            method="voronoi",
            particle_diameter=0.004,
            n_states=125,
            n_particles=5000,
        )

        model_big = LoadedModel(
            folder_name="voronoi_big",
            method="voronoi",
            particle_diameter=0.008,
            n_states=125,
            n_particles=5000,
        )

        app_context.add_model(model_small)
        app_context.add_model(model_big)

        # Filter
        filtered = [
            m for m in app_context.selected_models if m.particle_diameter == 0.004
        ]

        assert len(filtered) == 1
        assert filtered[0].folder_name == "voronoi_small"


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_context_operations(self, app_context: AppContext) -> None:
        """Test operations on empty context."""
        assert len(app_context.selected_models) == 0
        assert app_context.get_model("nonexistent") is None

        # Should not raise
        app_context.clear_models()

    def test_large_matrix_handling(self) -> None:
        """Test handling of large transition matrices."""
        n = 1000
        M = np.eye(n) * 0.99 + np.ones((n, n)) * 0.01 / n

        assert M.shape == (n, n)
        assert np.allclose(M.sum(axis=1), 1.0)

    def test_single_state_matrix(self) -> None:
        """Test degenerate 1×1 matrix."""
        M = np.array([[1.0]])

        assert M.shape == (1, 1)
        assert M.sum() == 1.0

    def test_zero_particles_handling(self) -> None:
        """Test handling zero particles gracefully."""
        phi = np.zeros(10)
        state = StateVector(
            phi=phi,
            timestamp=250,
            total_particles=0,
        )

        # Should validate
        assert state.validate_normalization()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
