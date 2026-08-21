"""dem_mcm_coupling — Markov chain models for DEM/MCM coupling.

A library to couple Discrete Element Method (DEM) simulations of granular
mixers with Markov chain models:

* spatial partitioning of the mixer domain
  (:mod:`dem_mcm_coupling.partitioners`);
* transition-matrix construction and state propagation
  (:mod:`dem_mcm_coupling.markov_core`, :mod:`dem_mcm_coupling.run_sweep`);
* pluggable data sources — Hugging Face Hub, local directories, in-memory
  (:mod:`dem_mcm_coupling.data`);
* result analysis (RSD vs tau, mixing times, ...)
  (:mod:`dem_mcm_coupling.analyze_results`).

Example:
    >>> from dem_mcm_coupling.data import InMemoryDataSource
    >>> from dem_mcm_coupling.markov_core import Markov
    >>> source = InMemoryDataSource(timesteps={250: particles_df})
    >>> model = Markov(method="voronoi", method_kwargs={"n_cells": 125},
    ...                data_source=source)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dem-mcm-coupling")
except PackageNotFoundError:  # pragma: no cover — package not installed
    __version__ = "0.0.0"

from dem_mcm_coupling.data import (
    DataSource,
    DataSourceError,
    DemSnapshot,
    HuggingFaceDataSource,
    InMemoryDataSource,
    LocalDataSource,
)
from dem_mcm_coupling.markov_core import Markov
from dem_mcm_coupling.partitioners import REGISTRY, create_partitioner

__all__ = [
    "REGISTRY",
    "DataSource",
    "DataSourceError",
    "DemSnapshot",
    "HuggingFaceDataSource",
    "InMemoryDataSource",
    "LocalDataSource",
    "Markov",
    "__version__",
    "create_partitioner",
]
