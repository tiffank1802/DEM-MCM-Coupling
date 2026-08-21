"""Markovian analysis — loading and comparison of RSD vs tau curves.

Loads the experiments of a given method (voronoi, cartesian, cylindrical,
quantile, octree, physics, ...) from a data source and compares the RSD vs
tau curves against the DEM reference.

Usage::

    from dem_mcm_coupling.analyze_results import MarkovAnalyzer

    analyzer = MarkovAnalyzer()
    analyzer.load_method("physics")
    analyzer.plot_rsd_vs_tau_comparison(...)
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from collections import defaultdict
from typing import Any

import numpy as np

from dem_mcm_coupling import bucket_io as b_io
from dem_mcm_coupling._config import BUCKET_ID, DEFAULT_BUCKET_PREFIX, get_bucket_prefix
from dem_mcm_coupling.data.base import DataSource
from dem_mcm_coupling.utils import load_parquet_as_timestep_dict

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BUCKET_PREFIX = DEFAULT_BUCKET_PREFIX
BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"

#: Bucket bases explored when no specific prefix is requested.
ALL_BUCKET_BASES = [
    f"hf://buckets/{BUCKET_ID}/{get_bucket_prefix(d)}" for d in (0.004, 0.008, None)
]

#: Name prefixes used to detect the method of an experiment folder.
METHOD_PREFIXES: dict[str, list[str]] = {
    "cartesian": ["cartesian_", "NLT_"],
    "cylindrical": ["cylindrical_"],
    "voronoi": ["voronoi_"],
    "quantile": ["quantile_"],
    "octree": ["octree_"],
    "physics": ["physics_"],
    "adaptive": ["adaptive_"],
    "multizone": ["multizone_"],
    "single": ["single_"],
}

#: Default colour per method (matplotlib hex codes).
METHOD_COLORS: dict[str, str] = {
    "cartesian": "#1f77b4",
    "cylindrical": "#ff7f0e",
    "voronoi": "#2ca02c",
    "quantile": "#d62728",
    "octree": "#9467bd",
    "physics": "#8c564b",
    "adaptive": "#af6c3c",
    "multizone": "#4b5d4c",
    "single": "#2b3e4ba8",
    "unknown": "#7f7f7f",
}


# =============================================================================
# MAIN CLASS
# =============================================================================


class MarkovAnalyzer:
    """Loader and analyser of Markovian experiment results."""

    def __init__(self, data_source: DataSource | None = None) -> None:
        from dem_mcm_coupling.bucket_io import get_fs

        self.data_source = data_source
        self.fs = get_fs()
        self.results: dict[str, dict[str, Any]] = {}
        self.by_method: defaultdict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

        # DEM data.
        self.dem_snapshots: list[dict[str, Any]] = []
        self.dem_file_indices: list[int] = []
        self.n_particles = 0
        self.dem_diameters: np.ndarray | None = None
        self.dem_velocities: np.ndarray | None = None
        self.species_labels: np.ndarray | None = None

        self.current_partitioner: Any | None = None
        self.partitioners: dict[str, Any] = {}

        self.dem_rsd_results: dict[str, dict[str, Any]] = {}
        self.markov_rsd_results: dict[str, Any] = {}

        self.initial_time = 250
        self.C0: np.ndarray | None = None
        self.phi_A_0: np.ndarray | None = None
        self.phi_total_0: np.ndarray | None = None

    # ─────────────────────────────────────────────────────────────────────
    # METHOD DETECTION
    # ─────────────────────────────────────────────────────────────────────

    def _detect_method(
        self, folder_name: str, params: dict[str, Any] | None = None
    ) -> str:
        """Detect the partitioning method from a folder name/parameters.

        Args:
            folder_name: Experiment folder name.
            params: Optional experiment parameters (``method`` wins over the
                name prefixes).

        Returns:
            The detected method (``"unknown"`` when undetermined).
        """
        if params:
            if "method" in params:
                return str(params["method"])
            if "nx" in params:
                return "cartesian"

        # Experiment variants carry a leading prefix that is not part of the
        # method name (inhomogeneous chains, no-species pipeline).
        for variant_prefix in ("inhomogeneous_", "nospecies_"):
            if folder_name.startswith(variant_prefix):
                folder_name = folder_name[len(variant_prefix) :]
                break

        for method, prefixes in METHOD_PREFIXES.items():
            for prefix in prefixes:
                if folder_name.startswith(prefix):
                    return method
        return "unknown"

    def _parse_experiment_info(
        self, folder_name: str, params: dict[str, Any], stats: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract human-readable experiment metadata.

        Args:
            folder_name: Experiment folder name.
            params: Experiment parameters.
            stats: Experiment statistics.

        Returns:
            Dictionary with ``folder``, ``n_states``, ``nlt``,
            ``step_size``, ``start_index`` and ``description``.
        """
        info: dict[str, Any] = {
            "folder": folder_name,
            "n_states": None,
            "nlt": None,
            "step_size": None,
            "start_index": None,
            "description": "",
        }
        if stats:
            info["n_states"] = stats.get("n_states")
            info["nlt"] = stats.get("n_timesteps_used")

        if params:
            if "method_kwargs" in params:
                info["description"] = str(params["method_kwargs"])
            info["nlt"] = info["nlt"] or params.get("nlt") or params.get("NLT")
            info["step_size"] = params.get("step_size")
            info["start_index"] = params.get("start_index")

            if "nx" in params and "method" not in params:
                nx = params.get("nx", "?")
                ny = params.get("ny", "?")
                nz = params.get("nz", "?")
                info["description"] = f"nx={nx}, ny={ny}, nz={nz}"
                if info["n_states"] is None:
                    with contextlib.suppress(Exception):
                        info["n_states"] = int(nx) * int(ny) * int(nz)
        return info

    # ─────────────────────────────────────────────────────────────────────
    # LOW-LEVEL LOADING
    # ─────────────────────────────────────────────────────────────────────

    def _load_npy(self, full_path: str) -> np.ndarray:
        """Load a numpy array from a ``hf://`` path."""
        with self.fs.open(full_path, "rb") as fh:
            return np.load(io.BytesIO(fh.read()))

    def _load_json(self, full_path: str) -> dict[str, Any]:
        """Load a JSON dictionary from a ``hf://`` path."""
        with self.fs.open(full_path, "r") as fh:
            return json.load(fh)

    def _load_partitioner_data(self, partitioner_path: str) -> dict[str, Any]:
        """Load the partitioner metadata of an experiment folder.

        Args:
            partitioner_path: Path of the ``partitioner`` sub-folder.

        Returns:
            Dictionary with ``type``, ``label``, ``n_cells`` and, when
            available, the method-specific arrays.
        """
        meta = self._load_json(f"{partitioner_path}/partitioner_meta.json")
        partitioner_data: dict[str, Any] = {
            "type": meta.get("type"),
            "label": meta.get("label"),
            "n_cells": meta.get("n_cells"),
        }

        partitioner_type = meta.get("type")
        if partitioner_type == "CylindricalPartitioner":
            try:
                cyl_params = self._load_json(
                    f"{partitioner_path}/cylindrical_params.json"
                )
                partitioner_data.update(cyl_params)
                partitioner_data["r_edges"] = self._load_npy(
                    f"{partitioner_path}/r_edges.npy"
                )
            except Exception as exc:
                logger.warning("Could not load cylindrical data: %s", exc)

        elif partitioner_type == "CartesianPartitioner":
            try:
                cart_params = self._load_json(
                    f"{partitioner_path}/cartesian_params.json"
                )
                partitioner_data.update(cart_params)
            except Exception as exc:
                logger.warning("Could not load cartesian data: %s", exc)

        elif partitioner_type == "VoronoiPartitioner":
            try:
                vor_params = self._load_json(f"{partitioner_path}/voronoi_params.json")
                partitioner_data.update(vor_params)
                partitioner_data["centroids"] = self._load_npy(
                    f"{partitioner_path}/centroids.npy"
                )
            except Exception as exc:
                logger.warning("Could not load Voronoi data: %s", exc)

        return partitioner_data

    def _list_folders(self, base_path: str = BUCKET_BASE) -> list[str]:
        """List the experiment folders under a bucket base path.

        Args:
            base_path: Bucket base path.

        Returns:
            Sorted folder names (empty list when the path is unreachable).
        """
        try:
            items = self.fs.ls(base_path)
        except FileNotFoundError:
            return []
        names = []
        for item in items:
            if isinstance(item, dict):
                if item.get("type") == "directory":
                    names.append(str(item["name"]).split("/")[-1])
            else:
                stripped = item.rstrip("/")
                if stripped:
                    names.append(stripped.split("/")[-1])
        return sorted(names)

    def _load_experiment(
        self, base_path: str = BUCKET_BASE, folder_name: str | None = None
    ) -> dict[str, Any]:
        """Load one experiment, trying several bucket layouts in order.

        Args:
            base_path: Bucket base path. When ``folder_name`` is ``None``,
                this argument is interpreted as the folder name and the
                default base path is used (single-argument form).
            folder_name: Experiment folder name.

        Returns:
            Dictionary with ``matrix``, ``params``, ``stats``, ``method``,
            ``info``, ``centroids``, ``partitioner_data`` and the
            inhomogeneous flags.

        Raises:
            FileNotFoundError: If the experiment cannot be loaded from any
                bucket.
        """
        if folder_name is None:
            # Single-argument form: _load_experiment(folder_name).
            base_path, folder_name = BUCKET_BASE, base_path
        prefix = f"{base_path}/{folder_name}"

        stats: dict[str, Any] = {}
        try:
            stats = self._load_json(f"{prefix}/stats.json")
            particle_diameter = stats.get("particle_diameter")
            if particle_diameter is not None:
                correct_base_path = (
                    f"hf://buckets/{BUCKET_ID}/{get_bucket_prefix(particle_diameter)}"
                )
                if correct_base_path != base_path:
                    logger.info(
                        "Bucket %s → reloading from %s (particle_diameter=%s)",
                        base_path,
                        correct_base_path,
                        particle_diameter,
                    )
                    base_path = correct_base_path
                    prefix = f"{base_path}/{folder_name}"
        except Exception:
            pass

        buckets_to_try = [base_path] + [b for b in ALL_BUCKET_BASES if b != base_path]

        last_error: Exception | None = None
        inhomogeneous = False
        inhomogeneous_metadata: dict[str, Any] | None = None

        for attempt_path in buckets_to_try:
            prefix = f"{attempt_path}/{folder_name}"
            try:
                # Inhomogeneous-format detection.
                inhomogeneous = False
                inhomogeneous_metadata = None
                try:
                    inhomogeneous_metadata = self._load_json(
                        f"{prefix}/inhomogeneous_metadata.json"
                    )
                    inhomogeneous = True
                except Exception:
                    pass

                if inhomogeneous:
                    # Load P_blocks (all matrices per species).
                    stats = self._load_json(f"{prefix}/stats.json")
                    species_list = stats.get("species_list", ["small"])
                    first_sp = species_list[0]
                    P_blocks = self._load_npy(f"{prefix}/P_blocks_{first_sp}.npy")
                    matrix = P_blocks[0]  # first matrix for compatibility
                else:
                    matrix = self._load_npy(f"{prefix}/transitionmatrix.npy")

                params: dict[str, Any] = {}
                for fname in ("config.json", "params.json"):
                    try:
                        params = self._load_json(f"{prefix}/{fname}")
                        break
                    except Exception:
                        continue

                if not stats:
                    with contextlib.suppress(Exception):
                        stats = self._load_json(f"{prefix}/stats.json")

                centroids = None
                with contextlib.suppress(Exception):
                    centroids = self._load_npy(f"{prefix}/centroids.npy")

                partitioner_data = None
                with contextlib.suppress(Exception):
                    partitioner_data = self._load_partitioner_data(
                        f"{prefix}/partitioner"
                    )

                method = self._detect_method(folder_name, params)
                info = self._parse_experiment_info(folder_name, params, stats)
                if info["n_states"] is None:
                    info["n_states"] = matrix.shape[0]

                return {
                    "matrix": matrix,
                    "params": params,
                    "stats": stats,
                    "method": method,
                    "info": info,
                    "centroids": centroids,
                    "partitioner_data": partitioner_data,
                    "inhomogeneous": inhomogeneous,
                    "inhomogeneous_metadata": inhomogeneous_metadata,
                }
            except Exception as exc:
                last_error = exc
                continue

        buckets_str = ", ".join(
            b.replace(f"hf://buckets/{BUCKET_ID}/", "") for b in buckets_to_try
        )
        raise FileNotFoundError(
            f"Could not load {folder_name} from buckets: {buckets_str}. "
            f"Error: {last_error}"
        )

    def load_method(self, method: str) -> None:
        """Load every experiment of a method.

        Args:
            method: Partitioning method (see :data:`METHOD_PREFIXES`).
        """
        self.results = {}
        self.by_method = defaultdict(dict)
        loaded_folders: set[str] = set()

        for base_path in ALL_BUCKET_BASES:
            bucket_name = base_path.replace(f"hf://buckets/{BUCKET_ID}/", "")
            try:
                folders = self._list_folders(base_path)
            except Exception as exc:
                logger.warning("Could not list %s: %s", bucket_name, exc)
                continue

            for folder in folders:
                if folder in loaded_folders:
                    continue
                if self._detect_method(folder) != method:
                    continue
                try:
                    data = self._load_experiment(base_path, folder)
                    self.results[folder] = data
                    self.by_method[method][folder] = data
                    loaded_folders.add(folder)
                    print(f"   ✅ {folder}: shape={data['matrix'].shape}")
                except Exception as exc:
                    print(f"   ⚠️  {folder}: {exc}")

        print(f"\n{len(self.results)} {method} experiments loaded")

    def get_matrix(self, folder_name: str) -> np.ndarray:
        """Return the transition matrix of a loaded experiment.

        Args:
            folder_name: Experiment folder name.

        Returns:
            The transition matrix.

        Raises:
            KeyError: If the experiment was not loaded.
        """
        return self.results[folder_name]["matrix"]

    # ─────────────────────────────────────────────────────────────────────
    # SPECIES LABELLING
    # ─────────────────────────────────────────────────────────────────────

    def label_species(self, criterion: str = "small") -> np.ndarray:
        """Label particles as species A (True) or species B (False).

        Args:
            criterion: Labelling criterion — ``"small"`` (diameter < 0.006 m),
                ``"first_half"`` (index-based) or ``"spatial_bottom"``
                (bottom half by z-coordinate).

        Returns:
            Boolean array of shape ``(n_particles,)``.

        Raises:
            ValueError: If no DEM snapshot is loaded or the criterion is
                unknown.
        """
        if not self.dem_snapshots:
            raise ValueError("No DEM snapshot loaded. Call load_dem_snapshots() first.")

        snap_0 = self.dem_snapshots[0]
        df_0 = snap_0.get("df")

        if df_0 is None:
            logger.warning("DataFrame unavailable in snapshot — all particles True")
            self.species_labels = np.ones(self.n_particles, dtype=bool)
            return self.species_labels

        if criterion == "small":
            if "Diameter" in df_0.columns:
                self.species_labels = df_0["Diameter"].to_numpy() < 0.006
            else:
                logger.warning("'Diameter' column not found — using first_half")
                self.species_labels = (
                    np.arange(self.n_particles) < self.n_particles // 2
                )
        elif criterion == "first_half":
            self.species_labels = np.arange(self.n_particles) < self.n_particles // 2
        elif criterion == "spatial_bottom":
            z_coords = snap_0["coords"][:, 2]
            self.species_labels = z_coords < np.median(z_coords)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        logger.info(
            "Species labelled: %s → %d / %d particles",
            criterion,
            int(self.species_labels.sum()),
            len(self.species_labels),
        )
        return self.species_labels

    # ─────────────────────────────────────────────────────────────────────
    # DEM DATA
    # ─────────────────────────────────────────────────────────────────────

    def load_dem_snapshots(
        self,
        file_indices: list[int] | None = None,
        sample_every: int = 1,
        particle_diameter: float | None = None,
        data_source: DataSource | None = None,
    ) -> list[dict[str, Any]]:
        """Load DEM snapshots from the parquet file of a data source.

        Converts the parquet data (mapping ``timestep -> DataFrame``) into a
        list of dictionaries ``{"t", "coords", "df"}``.

        Args:
            file_indices: Timestep indices to load (default: ``250, 300, ...,
                6000``).
            sample_every: Sample every Nth timestep.
            particle_diameter: Optional diameter filter. Only used to pick
                the bucket prefix of the default Hugging Face source.
            data_source: Optional explicit data source.

        Returns:
            List of snapshot dictionaries.

        Raises:
            ValueError: If ``file_indices`` is empty.
            DataSourceError: If the data cannot be read.
        """
        if file_indices is None:
            file_indices = list(range(250, 6001, 50))  # 250 to 6000, every 50
        if not file_indices:
            raise ValueError("file_indices cannot be empty")
        if sample_every > 1:
            file_indices = file_indices[::sample_every]

        # Infer the diameter from the loaded stats when not given.
        if particle_diameter is None and self.results:
            for folder_data in self.results.values():
                stats = folder_data.get("stats") or {}
                particle_diameter = stats.get("particle_diameter")
                if particle_diameter:
                    break

        logger.info(
            "Loading DEM snapshots: indices=%s-%s",
            file_indices[0],
            file_indices[-1],
        )

        if data_source is not None:
            timestep_dict = data_source.read_timesteps(file_indices)
        else:
            parquet_path = f"hf://buckets/{BUCKET_ID}/simulation_complete.parquet"
            timestep_dict = load_parquet_as_timestep_dict(
                parquet_path=parquet_path, fs=self.fs
            )

        dem_snapshots: list[dict[str, Any]] = []
        missing_indices: list[int] = []
        for idx in file_indices:
            if idx not in timestep_dict:
                missing_indices.append(idx)
                continue
            df = timestep_dict[idx]
            coords = np.column_stack(
                (
                    df["coordinates:0"].to_numpy(),
                    df["coordinates:1"].to_numpy(),
                    df["coordinates:2"].to_numpy(),
                )
            )
            dem_snapshots.append({"t": idx, "coords": coords, "df": df})

        if missing_indices:
            logger.warning(
                "%d timesteps not found in the parquet: %s",
                len(missing_indices),
                missing_indices[:5],
            )

        self.dem_snapshots = dem_snapshots
        self.dem_file_indices = file_indices
        if dem_snapshots:
            self.n_particles = dem_snapshots[0]["coords"].shape[0]

        logger.info(
            "%d snapshots loaded (N=%d particles)", len(dem_snapshots), self.n_particles
        )
        return dem_snapshots

    def list_available_models(
        self,
        method: str | None = None,
        particle_diameter: float | None = None,
        fraction_visited_threshold: float = 0.95,
    ) -> list[dict[str, Any]]:
        """List the available models with a fraction_visited filter.

        **Critical filter**: only models with
        ``fraction_visited >= threshold`` are kept, to guarantee the DEM data
        cover the domain.

        Args:
            method: Optional method filter.
            particle_diameter: Optional diameter filter.
            fraction_visited_threshold: Minimum ``fraction_visited`` from
                ``stats.json`` (default 0.95).

        Returns:
            List of model dictionaries (``folder_name``, ``method``,
            ``n_states``, ``particle_diameter``, ``fraction_visited``,
            ``stats``, ``info``).
        """
        if particle_diameter is not None:
            buckets = [
                f"hf://buckets/{BUCKET_ID}/{get_bucket_prefix(particle_diameter)}"
            ]
        else:
            buckets = ALL_BUCKET_BASES

        logger.info(
            "Listing models: method=%s, diameter=%s, fraction_visited >= %s",
            method,
            particle_diameter,
            fraction_visited_threshold,
        )

        available_models: list[dict[str, Any]] = []
        for bucket_base in buckets:
            try:
                folders = self._list_folders(bucket_base)
            except Exception as exc:
                logger.warning("Error listing %s: %s", bucket_base, exc)
                continue

            for folder_name in folders:
                if method is not None and self._detect_method(folder_name) != method:
                    continue
                try:
                    data = self._load_experiment(bucket_base, folder_name)
                except Exception as exc:
                    logger.debug("%s: %s", folder_name, exc)
                    continue

                stats = data.get("stats", {})
                info = data.get("info", {})

                fraction_visited = stats.get("fraction_visited", 1.0)
                if fraction_visited < fraction_visited_threshold:
                    logger.debug(
                        "%s: skipped (fraction_visited=%.2f < %.2f)",
                        folder_name,
                        fraction_visited,
                        fraction_visited_threshold,
                    )
                    continue

                available_models.append(
                    {
                        "folder_name": folder_name,
                        "method": data.get("method"),
                        "n_states": data["matrix"].shape[0],
                        "particle_diameter": stats.get("particle_diameter"),
                        "fraction_visited": fraction_visited,
                        "stats": stats,
                        "info": info,
                    }
                )

        logger.info("%d models found", len(available_models))
        return available_models

    # ─────────────────────────────────────────────────────────────────────
    # RSD COMPUTATION
    # ─────────────────────────────────────────────────────────────────────

    def compute_dem_rsd(
        self,
        partitioner: Any,
        species_labels: np.ndarray | None = None,
        partitioner_name: str | None = None,
    ) -> dict[str, Any]:
        """Compute the RSD of the DEM snapshots with a given partitioner.

        Also computes the segregation entropy and the intensity of
        segregation at every snapshot, the mixing times ``t50``/``t90`` and
        the initial concentration field ``C0``.

        Args:
            partitioner: A fitted partitioner.
            species_labels: Optional boolean species mask; auto-labelled
                when ``None``.
            partitioner_name: Optional result key (``partitioner.label`` by
                default).

        Returns:
            Dictionary with ``times``, ``rsd``, ``rsd_percent``,
            ``concentrations``, ``populations``, ``entropy``,
            ``intensity_of_segregation``, mixing times and metadata.

        Raises:
            ValueError: If ``partitioner`` is ``None`` or no snapshot is
                available.
        """
        if partitioner is None:
            raise ValueError("partitioner is required for compute_dem_rsd()")

        n_states = partitioner.n_cells

        if species_labels is None:
            if self.species_labels is None:
                logger.info("species_labels not provided — calling label_species()")
                self.label_species()
            species_labels = self.species_labels

        if not self.dem_snapshots:
            logger.info("No DEM snapshot loaded — loading automatically...")
            self.load_dem_snapshots(file_indices=list(range(250, 6000, 50)))

        n_snaps = len(self.dem_snapshots)
        if n_snaps == 0:
            raise ValueError("No DEM snapshot available after loading")
        if species_labels is None:  # pragma: no cover — defensive
            raise ValueError("species_labels unavailable")

        if partitioner_name is None:
            partitioner_name = partitioner.label

        print("\n" + "═" * 70)
        print("📊 DEM RSD COMPUTATION")
        print("═" * 70)
        print(f"Partitioner : {partitioner_name}")
        print(f"n_states    : {n_states}")
        print(
            f"DEM snaps   : {n_snaps} "
            f"(t={self.dem_snapshots[0]['t']} → {self.dem_snapshots[-1]['t']})"
        )
        print(
            f"Species A   : {species_labels.sum()} particles / {len(species_labels)} total"
        )
        print("─" * 70)

        times = np.zeros(n_snaps)
        rsd = np.zeros(n_snaps)
        entropy = np.zeros(n_snaps)
        intensity_seg = np.zeros(n_snaps)
        concentrations: list[np.ndarray] = []
        populations: list[np.ndarray] = []

        for k, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            times[k] = snap["t"]

            states = partitioner.compute_states(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )

            n_total = np.bincount(states, minlength=n_states).astype(float)
            n_A = np.bincount(states[species_labels], minlength=n_states).astype(float)

            concentration = np.zeros(n_states)
            mask = n_total > 0
            concentration[mask] = n_A[mask] / n_total[mask]
            concentrations.append(concentration.copy())
            populations.append(n_total.copy())

            c_active = concentration[mask]
            if len(c_active) > 1 and c_active.mean() > 0:
                rsd[k] = c_active.std() / c_active.mean()

            if len(c_active) > 0:
                c_clip = np.clip(c_active, 1e-10, 1 - 1e-10)
                h = -np.mean(
                    c_clip * np.log(c_clip) + (1 - c_clip) * np.log(1 - c_clip)
                )
                entropy[k] = h / np.log(2)  # normalised by the max entropy

            c_bar = c_active.mean()
            if 0 < c_bar < 1 and len(c_active) > 1:
                intensity_seg[k] = c_active.var() / (c_bar * (1 - c_bar))

            if (k + 1) % 10 == 0 or k == 0 or k == n_snaps - 1:
                print(
                    f"   [{k + 1:4d}/{n_snaps}] t={int(times[k]):5d} | "
                    f"RSD={rsd[k] * 100:6.2f}% | "
                    f"Entropy={entropy[k]:.4f} | "
                    f"Active cells={mask.sum():3d}/{n_states}"
                )

        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
        mixing_time_50: int | None = None
        mixing_time_90: int | None = None
        for k in range(n_snaps):
            if mixing_time_50 is None and rsd[k] < 0.5 * rsd_0:
                mixing_time_50 = int(times[k])
            if mixing_time_90 is None and rsd[k] < 0.1 * rsd_0:
                mixing_time_90 = int(times[k])

        coords0 = self.dem_snapshots[0]["coords"]
        states0 = partitioner.compute_states(
            coords0[:, 0], coords0[:, 1], coords0[:, 2]
        )
        self.phi_total_0 = np.bincount(states0, minlength=n_states).astype(float)
        self.phi_A_0 = np.bincount(states0[species_labels], minlength=n_states).astype(
            float
        )

        mask0 = self.phi_total_0 > 0
        self.C0 = np.zeros(n_states)
        self.C0[mask0] = self.phi_A_0[mask0] / self.phi_total_0[mask0]
        self.initial_time = int(times[0])

        result = {
            "times": times,
            "rsd": rsd,
            "rsd_percent": rsd * 100,
            "concentrations": np.array(concentrations),
            "populations": populations,
            "entropy": entropy,
            "intensity_of_segregation": intensity_seg,
            "rsd_initial": float(rsd[0]),
            "rsd_final": float(rsd[-1]),
            "mixing_time_50": mixing_time_50,
            "mixing_time_90": mixing_time_90,
            "n_states": n_states,
            "source": "DEM",
            "partitioner_name": partitioner_name,
            "n_snapshots": n_snaps,
        }

        self.dem_rsd_results[partitioner_name] = result
        self.partitioners[partitioner_name] = partitioner
        self.current_partitioner = partitioner

        print("─" * 70)
        print("✅ DEM RSD RESULTS")
        print("─" * 70)
        print(f"RSD initial     : {result['rsd_initial'] * 100:6.2f}%")
        print(f"RSD final       : {result['rsd_final'] * 100:6.2f}%")
        print(
            "RSD reduction   : "
            f"{(1 - result['rsd_final'] / max(result['rsd_initial'], 1e-10)) * 100:6.2f}%"
        )
        print(f"Final entropy   : {entropy[-1]:.4f} / 1.000 (max)")
        print(f"t50 (RSD ÷ 2)   : {mixing_time_50 or 'Not reached'}")
        print(f"t90 (RSD ÷ 10)  : {mixing_time_90 or 'Not reached'}")
        print("─" * 70)
        print(f"Stored in       : self.dem_rsd_results['{partitioner_name}']")
        print(
            f"Initial cond.   : self.C0 (shape={self.C0.shape}) at t={self.initial_time}"
        )
        print("═" * 70 + "\n")

        return result

    # ─────────────────────────────────────────────────────────────────────
    # RSD VS TAU COMPARISON
    # ─────────────────────────────────────────────────────────────────────

    def plot_rsd_vs_tau_comparison(
        self,
        partitioner: Any,
        method: str,
        folder_name_template: str,
        tau_list: list[int] | None = None,
        max_time_seconds: int = 60,
        figsize: tuple[int, int] = (14, 8),
        save_name: str | None = None,
        species_criterion: str = "small",
    ) -> tuple[Any, Any]:
        """Plot the DEM RSD against Markov RSDs for several tau values.

        Args:
            partitioner: Fitted partitioner used for both curves.
            method: Method name (title only).
            folder_name_template: Format string with a ``{tau}`` placeholder
                for the experiment folder names.
            tau_list: List of tau values (default ``[50, 100, 200, 500,
                1000]``).
            max_time_seconds: Upper bound of the x axis.
            figsize: Figure size.
            save_name: Optional output file name.
            species_criterion: Species labelling criterion.

        Returns:
            Tuple ``(fig, ax)`` of the matplotlib figure.
        """
        import matplotlib.pyplot as plt

        if tau_list is None:
            tau_list = [50, 100, 200, 500, 1000]

        fig, ax = plt.subplots(figsize=figsize)

        start_file = 250
        total_files = 5999
        file_indices = list(range(start_file, total_files + 1, 50))

        print("\n📊 DEM RSD computation...")
        self.load_dem_snapshots(file_indices=file_indices)
        if self.species_labels is None:
            self.label_species(criterion=species_criterion)

        species_labels = self.species_labels
        assert species_labels is not None

        all_coords = np.vstack([s["coords"] for s in self.dem_snapshots])
        partitioner.fit(all_coords)

        n_states = partitioner.n_cells
        n_snaps = len(self.dem_snapshots)
        rsd_dem = np.zeros(n_snaps)
        times_dem_files = np.array([s["t"] for s in self.dem_snapshots])

        for i, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            states = partitioner.compute_states(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )
            concentration = np.zeros(n_states)
            for sid in range(n_states):
                mask = states == sid
                if mask.sum() > 0:
                    concentration[sid] = species_labels[mask].sum() / mask.sum()
            mask_active = concentration > 0
            if mask_active.sum() > 1:
                rsd_dem[i] = (
                    concentration[mask_active].std() / concentration[mask_active].mean()
                )

        t_dem_seconds = times_dem_files * 0.01
        print(
            f"   DEM: {n_snaps} points from {t_dem_seconds[0]:.2f}s to "
            f"{t_dem_seconds[-1]:.2f}s"
        )
        ax.plot(
            t_dem_seconds,
            rsd_dem * 100,
            color="black",
            marker="o",
            linewidth=3,
            markersize=8,
            label="RSD DEM (real)",
            zorder=10,
            alpha=0.9,
        )

        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tau_list)))
        print(f"\n📊 Markov RSD computation for {len(tau_list)} tau...")

        for tau_idx, tau in enumerate(tau_list):
            folder_name = folder_name_template.format(tau=tau)
            dt_markov = tau * 0.01
            print(f"\n   ── tau = {tau} ({dt_markov:.3f}s per step) ──")

            try:
                M = self.get_matrix(folder_name)
            except Exception as exc:
                print(f"   ⚠️  Folder {folder_name} not found: {exc}")
                continue

            snap0 = self.dem_snapshots[0]
            coords0 = snap0["coords"]
            states0 = partitioner.compute_states(
                coords0[:, 0], coords0[:, 1], coords0[:, 2]
            )

            phi_A_0 = np.zeros(n_states, dtype=float)
            phi_total_0 = np.zeros(n_states, dtype=float)
            for sid in range(n_states):
                mask = states0 == sid
                phi_total_0[sid] = mask.sum()
                phi_A_0[sid] = species_labels[mask].sum()

            mask_active = phi_total_0 > 0
            n_steps_markov = (total_files - start_file) // tau
            phi_A = phi_A_0.copy()
            phi_total = phi_total_0.copy()
            rsd_markov = np.zeros(n_steps_markov + 1)

            c_t0 = np.zeros(n_states)
            c_t0[mask_active] = phi_A[mask_active] / phi_total[mask_active]
            if mask_active.sum() > 1 and c_t0[mask_active].mean() > 0:
                rsd_markov[0] = c_t0[mask_active].std() / c_t0[mask_active].mean()

            for t in range(1, n_steps_markov + 1):
                phi_A = phi_A @ M
                phi_total = phi_total @ M
                c_t = np.zeros(n_states)
                c_t[mask_active] = phi_A[mask_active] / phi_total[mask_active]
                if mask_active.sum() > 1 and c_t[mask_active].mean() > 0:
                    rsd_markov[t] = c_t[mask_active].std() / c_t[mask_active].mean()

            t_markov_seconds = (start_file + np.arange(n_steps_markov + 1) * tau) * 0.01
            ax.plot(
                t_markov_seconds,
                rsd_markov * 100,
                color=colors[tau_idx],
                linewidth=2.5,
                linestyle="-",
                label=f"Markov tau={tau} ({n_steps_markov + 1} pts)",
                zorder=5,
                alpha=0.8,
            )

            print(
                f"   ✅ {n_steps_markov + 1} points from {t_markov_seconds[0]:.2f}s "
                f"to {t_markov_seconds[-1]:.2f}s (incl. t=0)"
            )

        ax.set_xlabel("Time (s)", fontsize=13, fontweight="bold")
        ax.set_ylabel("RSD (%)", fontsize=13, fontweight="bold")
        ax.set_title(
            "Influence of the Markov time step (tau) on the mixing kinetics\n"
            f"{method.upper()} | {partitioner.label} | {n_states} cells",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.legend(fontsize=10, loc="best", framealpha=0.95, edgecolor="black", ncol=2)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)
        ax.set_xlim(t_dem_seconds[0], max_time_seconds)
        ax.set_ylim(bottom=0)
        ax.minorticks_on()
        ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.5)

        plt.tight_layout()

        if save_name is None:
            save_name = f"rsd_tau_comparison_{method}_{n_states}cells.png"
        plt.savefig(save_name, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"\n✅ Figure saved: {save_name}")
        plt.show()

        return fig, ax

    # ─────────────────────────────────────────────────────────────────────
    # INHOMOGENEOUS METHODS
    # ─────────────────────────────────────────────────────────────────────

    def load_inhomogeneous_method(self, method: str) -> dict[str, dict[str, Any]]:
        """Load the inhomogeneous experiments of a method.

        Sibling of :meth:`load_method` — lists the folders of
        ``Inhomogènes/`` and of every category, filters by method and loads
        with inhomogeneous-format detection.

        Args:
            method: Method name (``"all"`` loads every method).

        Returns:
            Mapping ``folder_name -> experiment data``.
        """
        self.results = {}
        self.by_method = defaultdict(dict)
        loaded_folders: set[str] = set()

        for base_path in ALL_BUCKET_BASES:
            # Search in Inhomogènes/ first, then in the usual categories.
            search_paths = [
                f"{base_path}/Inhomogènes",
                *[f"{base_path}/{cat}" for cat in b_io.ALL_CATEGORIES],
            ]

            for search_base in search_paths:
                try:
                    folders = self._list_folders(search_base)
                except Exception as exc:
                    logger.warning("Could not list %s: %s", search_base, exc)
                    continue

                for folder in folders:
                    if folder in loaded_folders:
                        continue
                    if self._detect_method(folder) != method and method != "all":
                        continue
                    try:
                        data = self._load_experiment(search_base, folder)
                    except Exception as exc:
                        print(f"   ⚠️  {folder}: {exc}")
                        continue
                    if data.get("inhomogeneous", False):
                        self.results[folder] = data
                        self.by_method[method][folder] = data
                        loaded_folders.add(folder)
                        n_blocks = data.get("inhomogeneous_metadata", {}).get(
                            "n_blocks", "?"
                        )
                        print(
                            f"   ✅ {folder}: {n_blocks} blocks, shape={data['matrix'].shape}"
                        )
                    else:
                        print(f"   ⏭️  {folder}: ignored (homogeneous)")

        print(f"\n{len(self.results)} inhomogeneous {method} experiments loaded")
        return self.results

    def visualize_P_blocks_evolution(
        self,
        folder_name: str,
        species: str = "small",
        figsize: tuple[int, int] = (16, 4),
    ) -> tuple[Any, Any] | tuple[None, None]:
        """Visualise the evolution of the transition matrices ``P_k``.

        Heatmaps side by side for every block, plus the differences between
        consecutive blocks.

        Args:
            folder_name: Inhomogeneous experiment folder name.
            species: Species to visualise (``"small"`` or ``"large"``).
            figsize: Figure size.

        Returns:
            Tuple ``(fig, axes)``, or ``(None, None)`` when the data are
            unavailable.
        """
        import matplotlib.pyplot as plt

        data = self._load_experiment(BUCKET_BASE, folder_name)
        if not data.get("inhomogeneous", False):
            print(f"⚠️  {folder_name} is not an inhomogeneous experiment")
            return None, None

        # Look for P_blocks of the requested species.
        P_blocks: np.ndarray | None = None
        prefix = f"{BUCKET_BASE}/{folder_name}"
        try:
            P_blocks = self._load_npy(f"{prefix}/P_blocks_{species}.npy")
        except Exception:
            for base in ALL_BUCKET_BASES:
                for cat in ["Inhomogènes", *b_io.ALL_CATEGORIES]:
                    try:
                        P_blocks = self._load_npy(
                            f"{base}/{cat}/{folder_name}/P_blocks_{species}.npy"
                        )
                        break
                    except Exception:
                        continue
                if P_blocks is not None:
                    break

        if P_blocks is None:
            print(f"❌ P_blocks_{species}.npy not found for {folder_name}")
            return None, None

        n_blocks = P_blocks.shape[0]
        n_states = P_blocks.shape[1]

        fig, axes = plt.subplots(2, max(2, n_blocks), figsize=figsize, squeeze=False)
        fig.suptitle(
            f"Transition matrices evolution P_k — {species}\n"
            f"{folder_name} ({n_blocks} blocks, {n_states} states)",
            fontweight="bold",
            fontsize=14,
        )

        # Row 1: P_k heatmaps.
        for k in range(n_blocks):
            ax = axes[0, k]
            vmax = (
                np.percentile(P_blocks[k][P_blocks[k] > 0], 95)
                if (P_blocks[k] > 0).any()
                else 1.0
            )
            im = ax.imshow(
                P_blocks[k],
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(f"P_{k} (block {k + 1}/{n_blocks})")
            ax.set_xlabel("Destination")
            ax.set_ylabel("Source")
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Hide the unused axes.
        for k in range(n_blocks, axes.shape[1]):
            axes[0, k].set_visible(False)
            axes[1, k].set_visible(False)

        # Row 2: differences between consecutive blocks.
        for k in range(1, n_blocks):
            ax = axes[1, k]
            diff = np.abs(P_blocks[k] - P_blocks[k - 1])
            vmax_diff = np.percentile(diff[diff > 0], 95) if (diff > 0).any() else 0.1
            im = ax.imshow(
                diff,
                aspect="auto",
                cmap="Reds",
                vmin=0,
                vmax=vmax_diff,
                interpolation="nearest",
            )
            ax.set_title(f"|P_{k} - P_{k - 1}| (norm={diff.sum():.4f})")
            ax.set_xlabel("Destination")
            ax.set_ylabel("Source")
            plt.colorbar(im, ax=ax, fraction=0.046)

        axes[1, 0].set_visible(False)
        plt.tight_layout()
        return fig, axes

    def compute_inhomogeneous_rsd(
        self,
        folder_name: str,
        partitioner: Any,
        species_labels: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Compute the RSD of an inhomogeneous chain (time-varying matrices).

        Loads ``P_blocks`` and propagates the state with matrices changing at
        each block; compares with the reference DEM RSD.

        Args:
            folder_name: Inhomogeneous experiment folder name.
            partitioner: Fitted partitioner.
            species_labels: Optional boolean species mask.

        Returns:
            Per-species dictionary with ``times``, ``rsd_markov``,
            ``rsd_dem`` and ``times_dem`` — or ``None`` when the experiment
            is homogeneous.
        """
        from dem_mcm_coupling.bucket_io import load_experiment_from_bucket

        data = load_experiment_from_bucket(folder_name)
        if not data.get("inhomogeneous", False):
            print(f"⚠️  {folder_name} is not inhomogeneous")
            return None

        inhom_meta = data.get("inhomogeneous_metadata", {})
        n_blocks = inhom_meta.get("n_blocks", 1)
        species_list = inhom_meta.get("species_list", ["small", "large"])

        print("\n" + "═" * 70)
        print(f"📊 INHOMOGENEOUS RSD — {folder_name}")
        print("═" * 70)
        print(f"Blocks: {n_blocks}")
        print(f"Species: {species_list}")

        if species_labels is None:
            if self.species_labels is None:
                logger.info("species_labels not provided — calling label_species()")
                self.label_species()
            species_labels = self.species_labels

        if not self.dem_snapshots:
            logger.info("No DEM snapshot loaded — loading automatically...")
            self.load_dem_snapshots(file_indices=list(range(250, 6000, 50)))
        if species_labels is None:  # pragma: no cover — defensive
            raise ValueError("species_labels unavailable")

        n_states = partitioner.n_cells
        n_snaps = len(self.dem_snapshots)
        times = np.array([s["t"] for s in self.dem_snapshots])

        # Reference DEM RSD.
        rsd_dem = np.zeros(n_snaps)
        for i, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            states = partitioner.compute_states(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )
            concentration = np.zeros(n_states)
            for sid in range(n_states):
                mask = states == sid
                if mask.sum() > 0:
                    concentration[sid] = species_labels[mask].sum() / mask.sum()
            mask_active = concentration > 0
            if mask_active.sum() > 1:
                rsd_dem[i] = (
                    concentration[mask_active].std() / concentration[mask_active].mean()
                )

        # Inhomogeneous propagation for each species (row-vector convention:
        # phi_next = phi @ P_k).
        results: dict[str, Any] = {}
        for sp in species_list:
            P_blocks = data["species"][sp]["P_blocks"]  # (n_blocks, n_states, n_states)
            S0 = data["species"][sp]["S_matrix"][0].astype(float)

            n_steps_markov = 200
            block_size = max(1, n_steps_markov // n_blocks)
            phi = S0.copy()
            rsd_markov = np.zeros(n_steps_markov + 1)

            mask_active = phi > 0
            if mask_active.sum() > 1 and phi[mask_active].mean() > 0:
                rsd_markov[0] = phi[mask_active].std() / phi[mask_active].mean()

            for t in range(1, n_steps_markov + 1):
                block_idx = min((t - 1) // block_size, n_blocks - 1)
                phi = phi @ P_blocks[block_idx]
                c_t = np.zeros(n_states)
                c_t[mask_active] = phi[mask_active] / S0[mask_active]
                if mask_active.sum() > 1 and c_t[mask_active].mean() > 0:
                    rsd_markov[t] = c_t[mask_active].std() / c_t[mask_active].mean()

            results[sp] = {
                "times": np.arange(n_steps_markov + 1)
                * 0.01
                * data["stats"].get("tau", 50),
                "rsd_markov": rsd_markov,
                "rsd_dem": rsd_dem,
                "times_dem": times * 0.01,
            }
            print(
                f"   ✅ {sp}: RSD initial={rsd_markov[0] * 100:.2f}%, "
                f"final={rsd_markov[-1] * 100:.2f}%"
            )

        self.markov_rsd_results[folder_name] = results
        return results
