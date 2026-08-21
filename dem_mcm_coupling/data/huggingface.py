"""Hugging Face Hub implementation of :class:`DataSource`.

Reads DEM snapshots and Markov experiments from the ``ktongue/DEM_MCM``
dataset repository (see :mod:`dem_mcm_coupling.bucket_io` for the low-level
I/O helpers). No file is downloaded locally when reading: data is streamed
directly from the Hub.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from dem_mcm_coupling import bucket_io
from dem_mcm_coupling._config import (
    BUCKET_ID,
    SIMULATION_PARQUET_NAME,
    ParticleDiameter,
    get_bucket_prefix,
)
from dem_mcm_coupling.data.base import DataSource, DataSourceError
from dem_mcm_coupling.utils import load_parquet_as_timestep_dict


class HuggingFaceDataSource(DataSource):
    """DEM/Markov data stored on the Hugging Face Hub.

    Args:
        particle_diameter: Optional particle-diameter filter (``0.004``,
            ``0.008`` or ``None`` for both). This selects the bucket prefix
            (``_Good/SMALL``, ``_Good/BIG``, ``_Good/Experiment``).
        repo_id: Identifier of the dataset repository.
    """

    def __init__(
        self,
        particle_diameter: ParticleDiameter | None = None,
        repo_id: str = BUCKET_ID,
    ) -> None:
        self.particle_diameter = particle_diameter
        self.repo_id = repo_id
        self.prefix = get_bucket_prefix(particle_diameter)

    @property
    def parquet_path(self) -> str:
        """Full ``hf://`` path of the DEM simulation parquet file."""
        return f"hf://buckets/{self.repo_id}/{self.prefix}/{SIMULATION_PARQUET_NAME}"

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def read_timesteps(
        self, timestep_indices: list[int] | None = None
    ) -> dict[int, pd.DataFrame]:
        """Load every DEM timestep from the simulation parquet file.

        Args:
            timestep_indices: Optional subset of timestep indices to return.
                The full parquet file is streamed either way; only the
                returned mapping is filtered.

        Returns:
            Mapping ``timestep_index -> DataFrame``.

        Raises:
            DataSourceError: If the Hub file cannot be read.
        """
        try:
            timesteps = load_parquet_as_timestep_dict(
                parquet_path=self.parquet_path,
                fs=bucket_io.get_fs(),
            )
        except Exception as exc:
            raise DataSourceError(f"Failed to read {self.parquet_path}: {exc}") from exc
        if timestep_indices is not None:
            return {idx: timesteps[idx] for idx in timestep_indices if idx in timesteps}
        return timesteps

    def list_experiments(self, prefix: str | None = None) -> list[str]:
        """List stored Markov experiments (see :func:`bucket_io.list_experiments`)."""
        return bucket_io.list_experiments(bucket_prefix=prefix or self.prefix)

    def read_experiment(
        self, folder_name: str, prefix: str | None = None
    ) -> dict[str, Any]:
        """Load a pre-computed Markov experiment from the Hub.

        Args:
            folder_name: Name of the experiment folder.
            prefix: Optional bucket prefix overriding the instance default.

        Returns:
            Experiment dictionary (``species``, ``stats``, ``config``, ...).

        Raises:
            DataSourceError: If the experiment cannot be found or read.
        """
        try:
            return bucket_io.load_experiment_from_bucket(
                folder_name, bucket_prefix=prefix or self.prefix
            )
        except FileNotFoundError as exc:
            raise DataSourceError(str(exc)) from exc

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
        """Upload a Markov experiment to the Hub."""
        # ``save_experiment_to_bucket`` derives the prefix from the particle
        # diameter stored in ``stats``; ``prefix`` is accepted for interface
        # compatibility.
        bucket_io.save_experiment_to_bucket(
            folder_name=folder_name,
            stats=stats,
            config=config,
            species_data=species_data,
            partitioner_data=partitioner_data,
            image_data=image_data,
            particle_diameter=stats.get("particle_diameter"),
            inhomogeneous_metadata=inhomogeneous_metadata,
        )
