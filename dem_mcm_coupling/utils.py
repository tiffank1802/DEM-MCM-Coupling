"""General-purpose helpers for the :mod:`dem_mcm_coupling` package."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from dem_mcm_coupling.data.base import DataSource


def load_parquet_as_timestep_dict(
    parquet_path: str,
    fs: Any | None = None,
) -> dict[int, pd.DataFrame]:
    """Load a parquet file and index its rows by DEM timestep.

    The timestep index is recovered from the ``Fichier_Source`` column
    (``"data_42.csv"`` -> ``42``). Each group of rows sharing the same source
    becomes one data frame in the returned mapping.

    Args:
        parquet_path: Path of the parquet file. Any path accepted by
            :class:`huggingface_hub.HfFileSystem` (e.g. ``hf://...``) or a
            local path.
        fs: Pre-instantiated :class:`~huggingface_hub.HfFileSystem` to stream
            remote files. When ``None`` and ``parquet_path`` starts with
            ``hf://``, a default instance is created.

    Returns:
        Mapping ``timestep_index -> DataFrame`` with one entry per available
        timestep.

    Raises:
        FileNotFoundError: If the parquet file cannot be opened.
        ValueError: If the ``Fichier_Source`` column is missing.
    """
    if fs is not None:
        with fs.open(parquet_path, "rb") as fh:
            df_full = _read_parquet(fh)
    elif parquet_path.startswith("hf://"):
        from huggingface_hub import HfFileSystem

        with HfFileSystem().open(parquet_path, "rb") as fh:
            df_full = _read_parquet(fh)
    else:
        with open(parquet_path, "rb") as fh:
            df_full = _read_parquet(fh)

    if "Fichier_Source" not in df_full.columns:
        raise ValueError(
            f"{parquet_path} lacks the 'Fichier_Source' column: "
            "cannot recover timestep indices"
        )

    timestep_dict: dict[int, pd.DataFrame] = {}
    for source, group_df in df_full.groupby("Fichier_Source", sort=False):
        # "data_42.csv" → 42
        idx = int(str(source).replace("data_", "").replace(".csv", ""))
        timestep_dict[idx] = group_df.reset_index(drop=True)

    if not timestep_dict:
        raise ValueError(f"{parquet_path} contains no timestep data")

    print(
        f"   📦 {len(timestep_dict)} timesteps indexed "
        f"(index {min(timestep_dict)} → {max(timestep_dict)})"
    )
    return dict(sorted(timestep_dict.items()))


def _read_parquet(file_handle: Any) -> pd.DataFrame:
    """Stream a parquet file from any binary file-like object.

    Row groups are read one by one so that very large remote files never have
    to be downloaded in full.
    """
    parquet_file = pq.ParquetFile(file_handle)
    list_dfs: list[pd.DataFrame] = []
    with tqdm(
        total=parquet_file.num_row_groups,
        desc="   Chargement parquet",
        unit="bloc",
    ) as bar:
        for i in range(parquet_file.num_row_groups):
            list_dfs.append(parquet_file.read_row_group(i).to_pandas())
            bar.update(1)
    return pd.concat(list_dfs, ignore_index=True)


def apply_species_mask(
    states: np.ndarray,
    species_labels: np.ndarray | None,
) -> np.ndarray:
    """Filter a state vector to keep only particles of the given species.

    The mask is assumed to be aligned with the particles; when ``states``
    contains several stacked copies of the particle set (one per timestep),
    the mask is tiled to match its length.

    Args:
        states: Assigned partition states, shape ``(n_particles * k,)``.
        species_labels: Boolean species mask, shape ``(n_particles,)``, or
            ``None`` to keep every particle.

    Returns:
        The filtered state array, or ``states`` unchanged when
        ``species_labels`` is ``None``.
    """
    if species_labels is None:
        return states

    mask = np.asarray(species_labels, dtype=bool)
    n_repeats = len(states) // len(mask)
    tiled = np.tile(mask, n_repeats)

    # Handle the case where len(states) is not an exact multiple of len(mask).
    if len(tiled) < len(states):
        remaining = len(states) - len(tiled)
        tiled = np.concatenate((tiled, mask[:remaining]))

    return states[tiled]


def data_source_from_uri(uri: str) -> DataSource:
    """Build a :class:`~dem_mcm_coupling.data.base.DataSource` from a URI.

    Supported URI schemes:

    * ``hf://`` — Hugging Face dataset repository
      (``hf://ktongue/DEM_MCM[/<prefix>]``);
    * ``memory://`` — empty in-memory source (tests);
    * anything else — local directory.

    Args:
        uri: URI of the data source.

    Returns:
        The matching data source instance.
    """
    from dem_mcm_coupling.data.huggingface import HuggingFaceDataSource
    from dem_mcm_coupling.data.local import LocalDataSource
    from dem_mcm_coupling.data.memory import InMemoryDataSource

    if uri.startswith("hf://"):
        parts = uri.removeprefix("hf://").split("/")
        repo_id = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        return HuggingFaceDataSource(repo_id=repo_id)
    if uri.startswith("memory://"):
        return InMemoryDataSource()
    return LocalDataSource(uri)
