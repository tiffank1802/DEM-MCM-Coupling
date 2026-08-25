"""Data-source abstraction for the DEM/MCM coupling library.

This subpackage lets the Markov model and the sweep pipelines read DEM
simulations and pre-computed experiments from *different* backends through a
single interface:

* :class:`~dem_mcm_coupling.data.huggingface.HuggingFaceDataSource` — reads
  and writes the ``ktongue/DEM_MCM`` dataset on the Hugging Face Hub;
* :class:`~dem_mcm_coupling.data.local.LocalDataSource` — reads a local
  directory holding parquet/csv snapshots and experiment folders;
* :class:`~dem_mcm_coupling.data.memory.InMemoryDataSource` — serves
  in-memory data (tests, prototyping).

Example:
    >>> from dem_mcm_coupling.data import HuggingFaceDataSource
    >>> source = HuggingFaceDataSource(particle_diameter=0.004)
    >>> timesteps = source.read_timesteps()  # {timestep_index: DataFrame}
"""

from dem_mcm_coupling.data.base import (
    DataSource,
    DataSourceError,
    DemSnapshot,
)
from dem_mcm_coupling.data.huggingface import HuggingFaceDataSource
from dem_mcm_coupling.data.local import LocalDataSource
from dem_mcm_coupling.data.memory import InMemoryDataSource

__all__ = [
    "DataSource",
    "DataSourceError",
    "DemSnapshot",
    "HuggingFaceDataSource",
    "InMemoryDataSource",
    "LocalDataSource",
]
