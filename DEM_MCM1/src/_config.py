"""
Configuration and type definitions for DEM_MCM1 Markov chain models.

This module centralizes all type definitions, constants, and validation functions
used across the package. It replaces the previous scattered constants and provides
proper typing for all public APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np

# =============================================================================
# TYPE ALIASES
# =============================================================================

# Partitioning method identifiers
PartitioningMethod = Literal[
    "cartesian",
    "cylindrical",
    "voronoi",
    "quantile",
    "octree",
    "physics",
    "gmm",
    "spectral",
    "adaptive",
    "multizone",
    "single",
    "dbscan",
]

# Particle diameter in meters
ParticleDiameter = Literal[0.004, 0.008]

# Type aliases for arrays
Array1D = np.ndarray  # Shape (n,)
Array2D = np.ndarray  # Shape (m, n)
Array3D = np.ndarray  # Shape (m, n, p)

# State vector: probability/particle count per partition cell
StateVector = Array1D

# State trajectory: sequence of state vectors over time
StateTrajectory = Array2D  # Shape (n_steps, n_states)

# Transition matrix: row-stochastic matrix
TransitionMatrix = Array2D  # Shape (n_states, n_states)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True, slots=True)
class PartitionerConfig:
    """Configuration for a partitioner instance.

    Attributes:
        method: Type of partitioning method.
        method_kwargs: Keyword arguments for the partitioner constructor.
        n_cells: Number of partition cells (computed from method_kwargs).
        label: Human-readable identifier for this configuration.
    """

    method: PartitioningMethod
    method_kwargs: dict[str, Any]
    n_cells: int
    label: str


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Metadata for a loaded Markov model from the bucket.

    Attributes:
        folder_name: Unique folder name in the bucket.
        method: Partitioning method used.
        particle_diameter: Diameter of particles in the experiment.
        n_states: Number of partition cells.
        n_particles: Total number of particles.
        nlt: Number of learning timesteps.
        tau: Time step between snapshots.
        fraction_visited: Fraction of cells visited during learning.
    """

    folder_name: str
    method: PartitioningMethod
    particle_diameter: ParticleDiameter | None
    n_states: int
    n_particles: int
    nlt: int
    tau: int
    fraction_visited: float

    def is_data_loaded(self) -> bool:
        """Check if transition matrices are loaded."""
        return hasattr(self, "_matrices_loaded") and self._matrices_loaded

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for API responses."""
        return {
            "folder_name": self.folder_name,
            "method": self.method,
            "particle_diameter": self.particle_diameter,
            "n_states": self.n_states,
            "n_particles": self.n_particles,
            "nlt": self.nlt,
            "tau": self.tau,
            "fraction_visited": self.fraction_visited,
        }


@dataclass(frozen=True, slots=True)
class AppContext:
    """Application context for session state synchronization across pages.

    This is a singleton-like context that maintains the selected models and
    notifies pages of changes via version increments.
    """

    selected_models: list[LoadedModel]
    version: int = 0

    def add_model(self, model: LoadedModel) -> None:
        """Add a model to the selection if not already present."""
        if model not in self.selected_models:
            self.selected_models.append(model)
            self.version += 1

    def remove_model(self, folder_name: str) -> bool:
        """Remove a model by folder name. Returns True if removed."""
        for i, model in enumerate(self.selected_models):
            if model.folder_name == folder_name:
                self.selected_models.pop(i)
                self.version += 1
                return True
        return False

    def get_model(self, folder_name: str) -> LoadedModel | None:
        """Get a model by folder name."""
        for model in self.selected_models:
            if model.folder_name == folder_name:
                return model
        return None

    def clear_models(self) -> None:
        """Clear all selected models."""
        if self.selected_models:
            self.selected_models.clear()
            self.version += 1


# =============================================================================
# CONSTANTS
# =============================================================================

# HuggingFace bucket configuration
BUCKET_ID: str = "ktongue/DEM_MCM"
BUCKET_PREFIXES: dict[ParticleDiameter | None, str] = {
    0.004: "_Good/SMALL",
    0.008: "_Good/BIG",
    None: "_Good/Experiment",
}

# Time conversion
TIMESTEP_TO_SECONDS: float = 0.01  # 1 timestep = 0.01 seconds

# Default bucket prefix for general experiments
DEFAULT_BUCKET_PREFIX: str = BUCKET_PREFIXES[None]


def get_bucket_prefix(
    particle_diameter: ParticleDiameter | None = None,
) -> str:
    """Get the bucket prefix for a given particle diameter.

    Args:
        particle_diameter: Diameter of particles (0.004, 0.008, or None).

    Returns:
        Bucket prefix string (e.g., "_Good/SMALL", "_Good/BIG").
    """
    return BUCKET_PREFIXES.get(particle_diameter, DEFAULT_BUCKET_PREFIX)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_partitioning_method(method: str) -> PartitioningMethod:
    """Validate and return a partitioning method.

    Args:
        method: String identifier for the partitioning method.

    Returns:
        Validated PartitioningMethod literal.

    Raises:
        ValueError: If method is not recognized.
    """
    valid_methods: set[PartitioningMethod] = {
        "cartesian",
        "cylindrical",
        "voronoi",
        "quantile",
        "octree",
        "physics",
        "gmm",
        "spectral",
        "adaptive",
        "multizone",
        "single",
        "dbscan",
    }

    if method not in valid_methods:
        raise ValueError(
            f"Unknown partitioning method: '{method}'. "
            f"Valid methods: {', '.join(sorted(valid_methods))}"
        )
    return method  # type: ignore[return-value]


def get_bucket_prefix(particle_diameter: ParticleDiameter | None = None) -> str:
    """Get the bucket prefix for a given particle diameter.

    Args:
        particle_diameter: Particle diameter in meters (0.004, 0.008, or None).

    Returns:
        Bucket prefix string (e.g., "_Good/Experiment", "_Good/SMALL").
    """
    return BUCKET_PREFIXES.get(particle_diameter, DEFAULT_BUCKET_PREFIX)


# =============================================================================
# TYPED DICTS FOR COMPLEX STRUCTURES
# =============================================================================


class SpeciesData(TypedDict):
    """Data for a single particle species."""

    P_raw: TransitionMatrix
    S_matrix: StateTrajectory
    times: Array1D


class ExperimentData(TypedDict):
    """Complete experiment data loaded from bucket."""

    config: dict[str, Any]
    stats: dict[str, Any]
    species: dict[str, SpeciesData]
    matrix: StateTrajectory | None


class InhomogeneousExperimentData(TypedDict):
    """Inhomogeneous experiment data with multiple transition matrices (one per NLT block).

    Attributes:
        P_blocks: 3D array of transition matrices, shape (n_blocks, n_states, n_states).
        S_matrix: State trajectory matrix, shape (n_timesteps, n_states).
        times: Timestep indices.
    """

    P_blocks: TransitionMatrix  # (n_blocks, n_states, n_states)
    S_matrix: StateTrajectory
    times: Array1D
