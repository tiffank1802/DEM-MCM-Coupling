"""Base classes of the data-source abstraction.

:class:`DataSource` is the single interface the rest of the library relies on
to obtain DEM particle data and to persist/restore Markov experiments. Every
backend (Hugging Face, local directory, in-memory) implements this interface,
so the Markov model and the sweep pipelines can be coupled to any data origin
without modification.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


class DataSourceError(Exception):
    """Base error raised when a data source cannot serve the requested data."""


@dataclass(slots=True)
class DemSnapshot:
    """Particle data at a single DEM timestep.

    Attributes:
        timestep: Index of the timestep in the DEM simulation.
        coordinates: Particle positions, shape ``(n_particles, 3)``.
        velocities: Particle velocities, shape ``(n_particles, 3)``
            (empty when unavailable).
        diameters: Particle diameters in metres, shape ``(n_particles,)``
            (empty when unavailable).
        particle_ids: Unique particle identifiers, shape ``(n_particles,)``
            (empty when unavailable).
        dataframe: The raw row-per-particle data frame, when the backend
            provides it.
    """

    timestep: int
    coordinates: np.ndarray
    velocities: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    diameters: np.ndarray = field(default_factory=lambda: np.empty(0))
    particle_ids: np.ndarray = field(default_factory=lambda: np.empty(0))
    dataframe: pd.DataFrame | None = None

    @property
    def n_particles(self) -> int:
        """Number of particles in the snapshot."""
        return int(self.coordinates.shape[0])


class DataSource(ABC):
    """Abstract interface of a DEM/Markov data source.

    Implementations must provide:

    * :meth:`read_timesteps` — the row-per-particle DEM data, keyed by
      timestep index;
    * :meth:`read_experiment` / :meth:`write_experiment` /
      :meth:`list_experiments` — persistence of pre-computed Markov
      experiments.

    The default helpers (:meth:`snapshot`, :meth:`sample_coordinates`) are
    shared by every backend.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def read_timesteps(
        self, timestep_indices: list[int] | None = None
    ) -> dict[int, pd.DataFrame]:
        """Load DEM particle data keyed by timestep index.

        Args:
            timestep_indices: Optional subset of timesteps to load. When
                ``None``, every available timestep is loaded.

        Returns:
            Mapping ``timestep_index -> DataFrame`` where each row describes
            one particle. Required columns: ``coordinates:0..2`` and
            ``Velocity:0..2``; optional columns: ``Diameter``,
            ``Particle_ID``, ``Fichier_Source``.

        Raises:
            DataSourceError: If the data cannot be read.
        """

    @abstractmethod
    def list_experiments(self, prefix: str | None = None) -> list[str]:
        """List the names of the stored Markov experiments.

        Args:
            prefix: Optional path prefix inside the backend.

        Returns:
            Sorted list of experiment (folder) names.
        """

    @abstractmethod
    def read_experiment(
        self, folder_name: str, prefix: str | None = None
    ) -> dict[str, Any]:
        """Load a pre-computed Markov experiment.

        Args:
            folder_name: Name of the experiment folder.
            prefix: Optional path prefix inside the backend.

        Returns:
            Dictionary with ``"species"`` (per-species data), ``"stats"``,
            ``"config"`` and, for inhomogeneous chains,
            ``"inhomogeneous"``/``"inhomogeneous_metadata"`` entries.

        Raises:
            DataSourceError: If the experiment cannot be found or read.
        """

    @abstractmethod
    def write_experiment(
        self,
        folder_name: str,
        stats: dict[str, Any],
        config: dict[str, Any],
        species_data: dict[str, Any] | None = None,
        partitioner_data: dict[str, Any] | None = None,
        image_data: dict[str, bytes] | None = None,
        prefix: str | None = None,
        inhomogeneous_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a Markov experiment.

        Args:
            folder_name: Name of the experiment folder.
            stats: Statistics dictionary (``stats.json``).
            config: Configuration dictionary (``config.json``).
            species_data: Mapping ``array_name -> ndarray`` saved as ``.npy``.
            partitioner_data: Partitioner arrays/dicts saved in
                ``partitioner/``.
            image_data: Mapping ``image_name -> bytes`` saved in ``images/``.
            prefix: Optional path prefix inside the backend.
            inhomogeneous_metadata: Optional metadata of an inhomogeneous
                experiment (``inhomogeneous_metadata.json``).

        Raises:
            DataSourceError: If the data cannot be written.
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def list_timesteps(self) -> list[int]:
        """Return the sorted list of available timestep indices."""
        return sorted(self.read_timesteps())

    def snapshot(self, timestep: int) -> DemSnapshot:
        """Return a single timestep as a :class:`DemSnapshot`.

        Args:
            timestep: Index of the DEM timestep.

        Returns:
            The snapshot.

        Raises:
            DataSourceError: If the timestep is unavailable.
        """
        timestep_dict = self.read_timesteps([timestep])
        if timestep not in timestep_dict:
            raise DataSourceError(
                f"Timestep {timestep} not found in data source "
                f"(available: {self.list_timesteps()})"
            )
        return _dataframe_to_snapshot(timestep, timestep_dict[timestep])

    def sample_coordinates(
        self, timestep_indices: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stack coordinates, velocities and diameters over timesteps.

        Args:
            timestep_indices: Optional subset of timesteps to include.

        Returns:
            Tuple ``(coordinates, velocities, diameters)`` where coordinates
            and velocities have shape ``(n_points, 3)`` and diameters shape
            ``(n_points,)``.
        """
        all_coords: list[np.ndarray] = []
        all_velocities: list[np.ndarray] = []
        all_diameters: list[np.ndarray] = []

        for idx in sorted(self.read_timesteps(timestep_indices)):
            df = self.read_timesteps([idx])[idx]
            all_coords.append(
                np.column_stack(
                    (
                        df["coordinates:0"].to_numpy(),
                        df["coordinates:1"].to_numpy(),
                        df["coordinates:2"].to_numpy(),
                    )
                )
            )
            all_velocities.append(
                np.column_stack(
                    (
                        df["Velocity:0"].to_numpy(),
                        df["Velocity:1"].to_numpy(),
                        df["Velocity:2"].to_numpy(),
                    )
                )
            )
            if "Diameter" in df.columns:
                all_diameters.append(df["Diameter"].to_numpy())

        return (
            np.vstack(all_coords),
            np.vstack(all_velocities),
            np.concatenate(all_diameters) if all_diameters else np.empty(0),
        )


def _dataframe_to_snapshot(timestep: int, df: pd.DataFrame) -> DemSnapshot:
    """Convert a raw particle data frame into a :class:`DemSnapshot`."""
    coordinates = (
        df[["coordinates:0", "coordinates:1", "coordinates:2"]].to_numpy(
            dtype=np.float32
        )
        if "coordinates:0" in df.columns
        else np.empty((len(df), 3), dtype=np.float32)
    )
    velocity_cols = ["Velocity:0", "Velocity:1", "Velocity:2"]
    velocities = (
        df[velocity_cols].to_numpy(dtype=np.float32)
        if all(col in df.columns for col in velocity_cols)
        else np.empty((len(df), 3), dtype=np.float32)
    )
    diameters = df["Diameter"].to_numpy() if "Diameter" in df.columns else np.empty(0)
    particle_ids = (
        df["Particle_ID"].to_numpy() if "Particle_ID" in df.columns else np.empty(0)
    )
    return DemSnapshot(
        timestep=timestep,
        coordinates=coordinates,
        velocities=velocities,
        diameters=diameters,
        particle_ids=particle_ids,
        dataframe=df,
    )
