"""Local-directory implementation of :class:`DataSource`.

Useful to couple the Markov model to DEM data that lives on the local file
system instead of the Hugging Face Hub.

Expected directory layout::

    root/
    └── simulation_complete.parquet     (or one parquet/csv per timestep)
    └── <experiment_folder>/
        ├── stats.json
        ├── config.json
        └── ...npy arrays
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dem_mcm_coupling._config import SIMULATION_PARQUET_NAME
from dem_mcm_coupling.data.base import DataSource, DataSourceError


class LocalDataSource(DataSource):
    """DEM/Markov data stored in a local directory.

    Args:
        root: Root directory of the data.
        parquet_name: Name of the parquet file holding the DEM snapshots
            (defaults to ``simulation_complete.parquet``).
    """

    def __init__(
        self, root: str | Path, parquet_name: str = SIMULATION_PARQUET_NAME
    ) -> None:
        self.root = Path(root)
        self.parquet_name = parquet_name

    @property
    def parquet_path(self) -> Path:
        """Path of the DEM simulation parquet file."""
        return self.root / self.parquet_name

    def read_timesteps(
        self, timestep_indices: list[int] | None = None
    ) -> dict[int, pd.DataFrame]:
        """Load DEM timesteps from the local parquet file.

        Timestep indices are extracted from the ``Fichier_Source`` column
        (``"data_42.csv" -> 42``); each group of rows with the same source
        becomes one timestep.

        Args:
            timestep_indices: Optional subset of timestep indices to return.

        Returns:
            Mapping ``timestep_index -> DataFrame``.

        Raises:
            DataSourceError: If the parquet file is missing or unreadable.
        """
        if not self.parquet_path.exists():
            raise DataSourceError(f"Parquet file not found: {self.parquet_path}")

        df_full = pd.read_parquet(self.parquet_path)
        if "Fichier_Source" not in df_full.columns:
            raise DataSourceError(
                f"{self.parquet_path} lacks the 'Fichier_Source' column: "
                "cannot recover timestep indices"
            )

        timesteps: dict[int, pd.DataFrame] = {}
        for source, group_df in df_full.groupby("Fichier_Source", sort=False):
            # "data_42.csv" -> 42
            idx = int(str(source).replace("data_", "").replace(".csv", ""))
            timesteps[idx] = group_df.reset_index(drop=True)

        if timestep_indices is not None:
            return {idx: timesteps[idx] for idx in timestep_indices if idx in timesteps}
        return dict(sorted(timesteps.items()))

    def list_experiments(self, prefix: str | None = None) -> list[str]:
        """List experiment folders stored under ``root``."""
        names = [p.name for p in self.root.iterdir() if p.is_dir()]
        return sorted(names)

    def read_experiment(
        self, folder_name: str, prefix: str | None = None
    ) -> dict[str, Any]:
        """Load an experiment folder from disk.

        Returns:
            Experiment dictionary (``species``, ``stats``, ``config``, ...).

        Raises:
            DataSourceError: If the folder is missing or malformed.
        """
        folder = self.root / folder_name
        if not folder.is_dir():
            raise DataSourceError(f"Experiment folder not found: {folder}")

        stats = self._read_json(folder / "stats.json")
        config = self._read_json(folder / "config.json")

        inhomogeneous = (folder / "inhomogeneous_metadata.json").exists()
        inhomogeneous_metadata = (
            self._read_json(folder / "inhomogeneous_metadata.json")
            if inhomogeneous
            else None
        )

        species_list = stats.get("species_list", ["small", "large"])
        species: dict[str, dict[str, np.ndarray]] = {}
        for sp in species_list:
            if inhomogeneous:
                species[sp] = {
                    "P_blocks": self._read_npy(folder / f"P_blocks_{sp}.npy"),
                    "S_matrix": self._read_npy(folder / f"S_matrix_{sp}.npy"),
                    "times": self._read_npy(folder / f"times_{sp}.npy"),
                }
            else:
                species[sp] = {
                    "P": self._read_npy(folder / f"transitionmatrix_{sp}.npy"),
                    "S_matrix": self._read_npy(folder / f"S_matrix_{sp}.npy"),
                    "times": self._read_npy(folder / f"times_{sp}.npy"),
                }

        matrix_path = folder / "states_matrix.npy"
        matrix = self._read_npy(matrix_path) if matrix_path.exists() else None

        return {
            "species": species,
            "stats": stats,
            "config": config,
            "matrix": matrix,
            "inhomogeneous": inhomogeneous,
            "inhomogeneous_metadata": inhomogeneous_metadata,
        }

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
        """Write an experiment folder to disk.

        ``species_data`` values that are numpy arrays are stored as ``.npy``,
        everything else is stored as JSON (partitioner sub-dictionaries).
        """
        folder = self.root / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        self._write_json(folder / "stats.json", stats)
        self._write_json(folder / "config.json", config)

        if species_data:
            for name, value in species_data.items():
                if isinstance(value, np.ndarray):
                    np.save(folder / f"{name}.npy", value)
                else:
                    self._write_json(folder / f"{name}.json", value)

        if partitioner_data:
            part_dir = folder / "partitioner"
            part_dir.mkdir(exist_ok=True)
            for key, value in partitioner_data.items():
                if isinstance(value, np.ndarray):
                    np.save(part_dir / f"{key}.npy", value)
                else:
                    self._write_json(part_dir / f"{key}.json", value)

        if image_data:
            images_dir = folder / "images"
            images_dir.mkdir(exist_ok=True)
            for name, img_bytes in image_data.items():
                (images_dir / name).write_bytes(img_bytes)

        if inhomogeneous_metadata is not None:
            self._write_json(
                folder / "inhomogeneous_metadata.json", inhomogeneous_metadata
            )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open() as fh:
            return json.load(fh)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w") as fh:
            json.dump(payload, fh, indent=2)

    @staticmethod
    def _read_npy(path: Path) -> np.ndarray:
        with path.open("rb") as fh:
            return np.load(fh)
