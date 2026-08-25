"""In-memory implementation of :class:`DataSource`.

Mainly intended for tests, notebooks and prototyping: feed the Markov model
with data that never touches disk.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd

from dem_mcm_coupling.data.base import DataSource, DataSourceError


class InMemoryDataSource(DataSource):
    """DEM/Markov data held in memory.

    Args:
        timesteps: Optional mapping ``timestep_index -> DataFrame`` of DEM
            particle data.
        experiments: Optional mapping ``folder_name -> experiment dict`` of
            pre-computed experiments.
    """

    def __init__(
        self,
        timesteps: dict[int, pd.DataFrame] | None = None,
        experiments: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._timesteps: dict[int, pd.DataFrame] = dict(timesteps or {})
        self._experiments: dict[str, dict[str, Any]] = dict(experiments or {})

    def add_timestep(self, timestep: int, df: pd.DataFrame) -> None:
        """Register the data frame of one timestep."""
        self._timesteps[timestep] = df.copy()

    def add_experiment(self, folder_name: str, experiment: dict[str, Any]) -> None:
        """Register a pre-computed experiment."""
        self._experiments[folder_name] = deepcopy(experiment)

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def read_timesteps(
        self, timestep_indices: list[int] | None = None
    ) -> dict[int, pd.DataFrame]:
        """Return the registered timesteps (optionally filtered)."""
        if timestep_indices is None:
            return {idx: df.copy() for idx, df in sorted(self._timesteps.items())}
        return {
            idx: self._timesteps[idx].copy()
            for idx in timestep_indices
            if idx in self._timesteps
        }

    def list_experiments(self, prefix: str | None = None) -> list[str]:
        """Return the names of the registered experiments."""
        return sorted(self._experiments)

    def read_experiment(
        self, folder_name: str, prefix: str | None = None
    ) -> dict[str, Any]:
        """Return a registered experiment (deep-copied)."""
        if folder_name not in self._experiments:
            raise DataSourceError(f"Unknown experiment: {folder_name!r}")
        return deepcopy(self._experiments[folder_name])

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
        """Store an experiment built from the given parts.

        numpy arrays are kept as-is; no serialisation is performed.
        """
        experiment: dict[str, Any] = {
            "species": dict(species_data or {}),
            "stats": dict(stats),
            "config": dict(config),
            "inhomogeneous": inhomogeneous_metadata is not None,
            "inhomogeneous_metadata": (
                dict(inhomogeneous_metadata) if inhomogeneous_metadata else None
            ),
        }
        if partitioner_data:
            experiment["partitioner"] = deepcopy(partitioner_data)
        if image_data:
            experiment["images"] = dict(image_data)
        self._experiments[folder_name] = experiment
