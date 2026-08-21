"""Configuration, constants and shared types for :mod:`dem_mcm_coupling`.

This module centralises every type definition, constant and small helper
used across the package, so that the public API is typed consistently and
the Hugging Face bucket layout is described in a single place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import numpy as np

# =============================================================================
# PUBLIC TYPE ALIASES
# =============================================================================

#: Identifiers of the partitioning methods available in the registry.
PartitioningMethod = Literal[
    "cartesian",
    "cylindrical",
    "voronoi",
    "quantile",
    "octree",
    "physics",
    "physics_full_vel",
    "gmm",
    "spectral",
    "spectral_biclustering",
    "adaptive",
    "multizone",
    "single",
    "dbscan",
]

#: Particle diameters (in metres) used in the DEM experiments.
#: (A plain float alias: PEP 586 Literal does not accept float values.)
ParticleDiameter = float

#: Array of shape ``(n,)``.
Array1D = np.ndarray
#: Array of shape ``(m, n)``.
Array2D = np.ndarray
#: Array of shape ``(m, n, p)``.
Array3D = np.ndarray

#: Particle counts (or probabilities) per partition cell, shape ``(n_states,)``.
StateVectorArray = Array1D

#: Sequence of state vectors over time, shape ``(n_steps, n_states)``.
StateTrajectoryArray = Array2D

#: Row-stochastic transition matrix, shape ``(n_states, n_states)``.
#
# Convention used everywhere in this package:
#
# * ``P[i, j]`` is the probability to jump from state ``i`` to state ``j``;
# * rows therefore sum to one (``P.sum(axis=1) == 1``);
# * a state vector ``phi`` evolves as ``phi_next = phi @ P``.
TransitionMatrix = Array2D


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True, slots=True)
class PartitionerConfig:
    """Configuration of a single partitioner instance.

    Attributes:
        method: Identifier of the partitioning method.
        method_kwargs: Keyword arguments forwarded to the partitioner.
        n_cells: Number of partition cells (Markov states).
        label: Human-readable identifier of the configuration.
    """

    method: PartitioningMethod
    method_kwargs: dict[str, Any]
    n_cells: int
    label: str


@dataclass(frozen=True, slots=True)
class StateVectorData:
    """An initial Markov state vector ``phi(0)`` and its metadata.

    Attributes:
        phi: Particle counts per partition cell, shape ``(n_states,)``.
        timestamp: Index of the DEM timestep the vector was built from.
        total_particles: Total number of particles represented by ``phi``.
        description: Human-readable description of the vector.
    """

    phi: np.ndarray
    timestamp: int
    total_particles: int
    description: str = ""

    def validate_normalization(self, rtol: float = 1e-3) -> bool:
        """Check that ``phi.sum()`` matches ``total_particles``.

        Args:
            rtol: Relative tolerance used for the comparison.

        Returns:
            ``True`` when the vector is correctly normalised.
        """
        return bool(np.isclose(self.phi.sum(), self.total_particles, rtol=rtol))


@dataclass(frozen=True, slots=True)
class MarkovTrajectory:
    """Trajectory produced by a Markov propagation.

    Attributes:
        states: State vectors over time, shape ``(n_steps + 1, n_states)``.
        times: Timestep indices, shape ``(n_steps + 1,)``.
        times_seconds: Times in seconds (``times * TIMESTEP_TO_SECONDS``).
        method: Partitioning method used for the simulation.
        description: Human-readable description of the trajectory.
    """

    states: np.ndarray
    times: np.ndarray
    times_seconds: np.ndarray
    method: str
    description: str = ""


# =============================================================================
# CONSTANTS
# =============================================================================

#: Identifier of the Hugging Face dataset repository holding DEM simulations
#: and pre-computed Markov results.
BUCKET_ID: str = "ktongue/DEM_MCM"

#: Bucket prefixes by particle diameter.
BUCKET_PREFIXES: dict[ParticleDiameter | None, str] = {
    0.004: "_Good/SMALL",
    0.008: "_Good/BIG",
    None: "_Good/Experiment",
}

#: Conversion factor between a DEM timestep index and seconds.
TIMESTEP_TO_SECONDS: float = 0.01

#: Default bucket prefix (all particles, both diameters).
DEFAULT_BUCKET_PREFIX: str = BUCKET_PREFIXES[None]

#: Parquet file name of the complete DEM simulation inside each bucket prefix.
SIMULATION_PARQUET_NAME: str = "simulation_complete.parquet"


def get_bucket_prefix(particle_diameter: ParticleDiameter | None = None) -> str:
    """Return the bucket prefix associated to a particle diameter.

    Args:
        particle_diameter: Particle diameter in metres (``0.004``, ``0.008``)
            or ``None`` for experiments mixing both diameters.

    Returns:
        The bucket prefix, e.g. ``"_Good/SMALL"``.
    """
    return BUCKET_PREFIXES.get(particle_diameter, DEFAULT_BUCKET_PREFIX)


# =============================================================================
# VALIDATION
# =============================================================================

_VALID_PARTITIONING_METHODS: frozenset[str] = frozenset(
    {
        "cartesian",
        "cylindrical",
        "voronoi",
        "quantile",
        "octree",
        "physics",
        "physics_full_vel",
        "gmm",
        "spectral",
        "spectral_biclustering",
        "adaptive",
        "multizone",
        "single",
        "dbscan",
    }
)


def validate_partitioning_method(method: str) -> PartitioningMethod:
    """Validate a partitioning method identifier.

    Args:
        method: String identifier of the partitioning method.

    Returns:
        The validated method literal.

    Raises:
        ValueError: If ``method`` is not a known partitioning method.
    """
    if method not in _VALID_PARTITIONING_METHODS:
        raise ValueError(
            f"Unknown partitioning method: {method!r}. "
            f"Valid methods: {', '.join(sorted(_VALID_PARTITIONING_METHODS))}"
        )
    return method  # type: ignore[return-value]


# =============================================================================
# TYPED DICTS (bucket I/O structures)
# =============================================================================


class SpeciesData(TypedDict):
    """Per-species data of a homogeneous experiment.

    Attributes:
        P_raw: Row-stochastic transition matrix, shape
            ``(n_states, n_states)``.
        S_matrix: State trajectories, shape ``(n_timesteps, n_states)``.
        times: Timestep indices, shape ``(n_timesteps,)``.
    """

    P_raw: TransitionMatrix
    S_matrix: StateTrajectoryArray
    times: Array1D


class InhomogeneousSpeciesData(TypedDict):
    """Per-species data of an inhomogeneous experiment.

    Attributes:
        P_blocks: One transition matrix per NLT block, shape
            ``(n_blocks, n_states, n_states)``.
        S_matrix: State trajectories, shape ``(n_timesteps, n_states)``.
        times: Timestep indices, shape ``(n_timesteps,)``.
    """

    P_blocks: TransitionMatrix
    S_matrix: StateTrajectoryArray
    times: Array1D


class ExperimentData(TypedDict):
    """Complete experiment data loaded from a bucket.

    Attributes:
        config: Configuration dictionary of the experiment.
        stats: Statistics dictionary of the experiment.
        species: Mapping ``species name -> per-species data``.
        matrix: Optional raw state matrix, shape
            ``(n_timesteps, n_particles)``.
    """

    config: dict[str, Any]
    stats: dict[str, Any]
    species: dict[str, SpeciesData]
    matrix: StateTrajectoryArray | None
