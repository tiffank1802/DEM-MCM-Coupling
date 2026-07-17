"""
===================================================================================
ANALYSE MARKOVIENNE — Chargement et visualisation depuis le bucket HuggingFace.
===================================================================================

Charge les expériences de la méthode spécifiée (voronoi, cartesian, cylindrical,
quantile, octree, physics) et compare les courbes RSD vs tau.

Usage:
    from analyze_results import MarkovAnalyzer
    analyzer = MarkovAnalyzer()
    analyzer.load_method("physics")
    analyzer.plot_rsd_vs_tau_comparison(...)
===================================================================================
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import HfFileSystem

try:
    from . import bucket_io as b_io
    from .bucket_io import ALL_CATEGORIES, CATEGORY_MAP
    from .utils import apply_species_mask, load_parquet_as_timestep_dict
except ImportError:
    import bucket_io as b_io
    from bucket_io import ALL_CATEGORIES
    from utils import load_parquet_as_timestep_dict

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BUCKET_ID = b_io.BUCKET_ID
BUCKET_PREFIX = b_io.BUCKET_PREFIX
BUCKET_BASE = b_io.BUCKET_BASE


def _get_bucket_prefix_from_particle_diameter(particle_diameter: float) -> str:
    if particle_diameter == 0.008:
        return "BIG"
    elif particle_diameter == 0.004:
        return "SMALL"
    else:
        return "Experiments"


ALL_BUCKET_PREFIXES = ["Experiments", "SMALL", "BIG"]

OLD_BUCKET_PREFIX = "Experiments"
OLD_BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{OLD_BUCKET_PREFIX}"

ALL_BUCKET_BASES = [
    f"hf://buckets/{BUCKET_ID}/{prefix}" for prefix in ALL_BUCKET_PREFIXES
]
if OLD_BUCKET_BASE not in ALL_BUCKET_BASES:
    ALL_BUCKET_BASES.append(OLD_BUCKET_BASE)


METHOD_PREFIXES = {
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

METHOD_COLORS = {
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
# CLASSE PRINCIPALE
# =============================================================================


class MarkovAnalyzer:
    """Chargeur et analyseur de résultats Markoviens."""

    def __init__(self: MarkovAnalyzer) -> None:
        self.fs = HfFileSystem()
        self.results: dict = {}
        self.by_method = defaultdict(dict)

        self.dem_snapshots: list = []
        self.dem_file_indices: list = []
        self.n_particles: int = 0
        self.dem_diameters: np.ndarray = None  # type: ignore
        self.dem_velocities: np.ndarray = None  # type: ignore
        self.dem_angular_velocities: np.ndarray = None  # type: ignore
        self.species_labels: np.ndarray = None  # type: ignore

        self.current_partitioner = None
        self.partitioners = {}

        self.dem_rsd_results = {}
        self.markov_rsd_results = {}

        self.initial_time = 250
        self.C0 = None
        self.phi_A_0 = None
        self.phi_total_0 = None

    # ─────────────────────────────────────────────────────────────────────
    # DÉTECTION DE MÉTHODE
    # ─────────────────────────────────────────────────────────────────────

    def _detect_method(self, folder_name: str, params: dict | None = None) -> str:
        if params:
            if "method" in params:
                return params["method"]
            if "nx" in params and "method" not in params:
                return "cartesian"

        for method, prefixes in METHOD_PREFIXES.items():
            for prefix in prefixes:
                if folder_name.startswith(prefix):
                    return method
        return "unknown"

    def _parse_experiment_info(self, folder_name: str, params: dict, stats: dict) -> dict:
        info = {
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
                    with contextlib.suppress(BaseException):
                        info["n_states"] = int(nx) * int(ny) * int(nz)
        return info

    # ─────────────────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────────────────

    def _load_npy(self, full_path: str) -> np.ndarray:
        with self.fs.open(full_path, "rb") as f:
            return np.load(io.BytesIO(f.read()))

    def _load_json(self, full_path: str) -> dict:
        with self.fs.open(full_path, "r") as f:
            return json.load(f)

    def _load_partitioner_data(self, partitioner_path: str) -> dict:
        meta_file = f"{partitioner_path}/partitioner_meta.json"
        meta = self._load_json(meta_file)

        partitioner_data = {
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
                r_edges = self._load_npy(f"{partitioner_path}/r_edges.npy")
                partitioner_data["r_edges"] = r_edges
            except Exception as e:
                print(f"⚠️  Impossible de charger les données cylindriques: {e}")

        elif partitioner_type == "CartesianPartitioner":
            try:
                cart_params = self._load_json(
                    f"{partitioner_path}/cartesian_params.json"
                )
                partitioner_data.update(cart_params)
            except Exception as e:
                print(f"⚠️  Impossible de charger les données cartésiennes: {e}")

        elif partitioner_type == "VoronoiPartitioner":
            try:
                vor_params = self._load_json(f"{partitioner_path}/voronoi_params.json")
                partitioner_data.update(vor_params)
                centroids = self._load_npy(f"{partitioner_path}/centroids.npy")
                partitioner_data["centroids"] = centroids
            except Exception as e:
                print(f"⚠️  Impossible de charger les données Voronoï: {e}")

        return partitioner_data

    def _list_folders(self, base_path: str = BUCKET_BASE) -> list:
        try:
            items = self.fs.ls(base_path)
            return sorted(
                [
                    item["name"].split("/")[-1]
                    for item in items
                    if item["type"] == "directory"
                ]
            )
        except FileNotFoundError:
            return []

    def _load_experiment(self, base_path: str = BUCKET_BASE, folder_name: str | None = None) -> dict:
        prefix = f"{base_path}/{folder_name}"

        stats = {}
        try:
            stats = self._load_json(f"{prefix}/stats.json")
            particle_diameter = stats.get("particle_diameter")
            if particle_diameter is not None:
                correct_bucket_prefix = _get_bucket_prefix_from_particle_diameter(
                    particle_diameter
                )
                correct_base_path = f"hf://buckets/{BUCKET_ID}/{correct_bucket_prefix}"
                if correct_base_path != base_path:
                    print(
                        f"     ℹ️  bucket fourni {base_path} → rechargement depuis {correct_bucket_prefix} (particle_diameter={particle_diameter})"
                    )
                    base_path = correct_base_path
                    prefix = f"{base_path}/{folder_name}"
        except:
            pass

        buckets_to_try = [base_path] + [b for b in ALL_BUCKET_BASES if b != base_path]
        direct_base = base_path.replace("/markov_results", "")
        if direct_base not in buckets_to_try:
            buckets_to_try.append(direct_base)

        loaded = False
        last_error = None
        inhomogeneous = False
        inhomogeneous_metadata = None

        for attempt_path in buckets_to_try:
            prefix = f"{attempt_path}/{folder_name}"
            try:
                # Détection du format inhomogène
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
                    # Chargement inhomogène : P_blocks (toutes les matrices par espèce)
                    stats = self._load_json(f"{prefix}/stats.json")
                    species_list = stats.get("species_list", ["small"])
                    # Prendre la première espèce comme matrice de référence
                    first_sp = species_list[0]
                    P_blocks = self._load_npy(f"{prefix}/P_blocks_{first_sp}.npy")
                    matrix = P_blocks[0]  # 1ère matrice pour compatibilité
                else:
                    matrix = self._load_npy(f"{prefix}/transitionmatrix.npy")

                params = {}
                for fname in ["config.json", "params.json"]:
                    try:
                        params = self._load_json(f"{prefix}/{fname}")
                        break
                    except:
                        continue

                if not stats:
                    with contextlib.suppress(BaseException):
                        stats = self._load_json(f"{prefix}/stats.json")

                centroids = None
                with contextlib.suppress(BaseException):
                    centroids = self._load_npy(f"{prefix}/centroids.npy")

                partitioner_data = None
                with contextlib.suppress(BaseException):
                    partitioner_data = self._load_partitioner_data(
                        f"{prefix}/partitioner"
                    )

                method = self._detect_method(folder_name, params)
                info = self._parse_experiment_info(folder_name, params, stats)
                if info["n_states"] is None:
                    info["n_states"] = matrix.shape[0]

                loaded = True
                break

            except Exception as e:
                last_error = e
                continue

        if loaded:
            return {
                "matrix": matrix,
                "params": params,
                "stats": stats,
                "method": method,
                "info": info,
                "centroids": centroids,
                "partitioner_data": partitioner_data,
                "inhomogeneous": inhomogeneous if loaded else False,
                "inhomogeneous_metadata": inhomogeneous_metadata if loaded else None,
            }
        else:
            buckets_str = ", ".join(
                [b.replace("hf://buckets/ktongue/DEM_MCM/", "") for b in buckets_to_try]
            )
            raise Exception(
                f"Impossible de charger {folder_name} depuis les buckets: {buckets_str}. Erreur: {last_error}"
            )

    def load_method(self, method: str) -> None:
        self.results = {}
        self.by_method = defaultdict(dict)
        loaded_folders = set()

        for base_path in ALL_BUCKET_BASES:
            bucket_name = base_path.replace("hf://buckets/ktongue/DEM_MCM/", "")

            try:
                folders = self._list_folders(base_path)
                for folder in folders:
                    if folder in loaded_folders:
                        continue

                    detected = self._detect_method(folder)
                    if detected == method:
                        try:
                            data = self._load_experiment(base_path, folder)
                            self.results[folder] = data
                            self.by_method[method][folder] = data
                            loaded_folders.add(folder)
                            print(f"   ✅ {folder}: shape={data['matrix'].shape}")
                        except Exception as e:
                            print(f"   ⚠️  {folder}: {e}")
            except Exception as e:
                print(f"   ⚠️  Impossible de lister {bucket_name}: {e}")

        print(f"\n{len(self.results)} expériences {method} chargées")

    def get_matrix(self, folder_name: str) -> np.ndarray:
        return self.results[folder_name]["matrix"]

    # ─────────────────────────────────────────────────────────────────────
    # SPECIES LABELING
    # ─────────────────────────────────────────────────────────────────────

    def label_species(self, criterion: str = "small") -> np.ndarray:
        """
        Label particles as "species A" (True) or "species B" (False).

        Args:
            criterion: Labeling criterion
                - "small": Diameter < 0.006 m (small particles)
                - "first_half": First half of particles (index-based)
                - "spatial_bottom": Bottom half (z-coordinate)

        Returns:
            Boolean array, shape (N_particles,)
        """
        if not self.dem_snapshots:
            raise ValueError(
                "❌ No DEM snapshots loaded. Call load_dem_snapshots() first."
            )

        snap_0 = self.dem_snapshots[0]
        df_0 = snap_0.get("df")

        if df_0 is None:
            logger.warning(
                "⚠️  DataFrame not available in snapshot, cannot label species"
            )
            self.species_labels = np.ones(self.n_particles, dtype=bool)
            return self.species_labels

        if criterion == "small":
            # Label small particles (diameter < 0.006 m)
            if "Diameter" in df_0.columns:
                self.species_labels = df_0["Diameter"].values < 0.006
            else:
                logger.warning(
                    "⚠️  'Diameter' column not found, using first_half criterion"
                )
                self.species_labels = (
                    np.arange(self.n_particles) < self.n_particles // 2
                )

        elif criterion == "first_half":
            # Label first half of particles
            self.species_labels = np.arange(self.n_particles) < self.n_particles // 2

        elif criterion == "spatial_bottom":
            # Label bottom half by z-coordinate
            z_coords = snap_0["coords"][:, 2]
            self.species_labels = z_coords < np.median(z_coords)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        logger.info(
            f"✅ Species labeled: {criterion} → "
            f"{self.species_labels.sum()} / {len(self.species_labels)} particles"
        )

        return self.species_labels

    # ─────────────────────────────────────────────────────────────────────
    # DONNÉES DEM
    # ─────────────────────────────────────────────────────────────────────

    def load_dem_snapshots(
        self,
        file_indices: list[int] | None = None,
        sample_every: int = 1,
        particle_diameter: float | None = None,
    ) -> list[dict[str, any]]:
        """
        Charger les snapshots DEM depuis HuggingFace.

        Convertit le parquet HF (Dict[timestep, DataFrame]) en liste de dicts
        avec format {t: timestep_index, coords: (N, 3) array}.

        Args:
            file_indices: List of timestep indices to load (e.g., [250, 300, 350, ...]).
                         If None, defaults to [250, 300, ..., 6000] (50-step intervals)
            sample_every: Sample every Nth timestep (default=1, no sampling)
            particle_diameter: Filter by diameter (0.004, 0.008, None for all).
                              If None, inferred from stats.json if available.

        Returns:
            List of dicts with structure:
            [
                {"t": 250, "coords": (N, 3) array},
                {"t": 300, "coords": (N, 3) array},
                ...
            ]

        Raises:
            FileNotFoundError: If HF bucket not accessible
            ValueError: If file_indices is empty or invalid

        Examples:
            >>> analyzer = MarkovAnalyzer()
            >>> # Load standard timesteps
            >>> snapshots = analyzer.load_dem_snapshots()
            >>> print(f"Loaded {len(snapshots)} snapshots")
            >>>
            >>> # Load specific timesteps, filter diameter
            >>> snapshots = analyzer.load_dem_snapshots(
            ...     file_indices=[250, 500, 1000],
            ...     particle_diameter=0.004
            ... )
        """
        # Default timesteps if not provided
        if file_indices is None:
            file_indices = list(range(250, 6001, 50))  # 250 to 6000, every 50

        if not file_indices:
            raise ValueError("❌ file_indices cannot be empty")

        # Apply sampling
        if sample_every > 1:
            file_indices = file_indices[::sample_every]

        # Determine bucket prefix from particle diameter
        if particle_diameter is None:
            # Try to infer from stats if available
            try:
                if self.results:
                    for folder_data in self.results.values():
                        if folder_data.get("stats"):
                            particle_diameter = folder_data["stats"].get(
                                "particle_diameter"
                            )
                            if particle_diameter:
                                break
            except:
                pass

        bucket_prefix = _get_bucket_prefix_from_particle_diameter(particle_diameter)
        parquet_path = f"hf://buckets/{BUCKET_ID}/simulation_complete.parquet"

        logger.info(
            f"📦 Chargement DEM snapshots: "
            f"indices={file_indices[0]}-{file_indices[-1]}, "
            f"prefix={bucket_prefix}"
        )

        try:
            # Load timestep dict from HF
            timestep_dict = load_parquet_as_timestep_dict(
                parquet_path=parquet_path, fs=self.fs
            )

            # Convert to dem_snapshots format
            dem_snapshots = []
            missing_indices = []

            for idx in file_indices:
                if idx in timestep_dict:
                    df = timestep_dict[idx]
                    # Extract coordinates (columns: 'coordinates:0', 'coordinates:1', 'coordinates:2')
                    coords = np.column_stack(
                        [
                            df["coordinates:0"].to_numpy(),
                            df["coordinates:1"].to_numpy(),
                            df["coordinates:2"].to_numpy(),
                        ]
                    )
                    dem_snapshots.append(
                        {
                            "t": idx,
                            "coords": coords,
                            "df": df,  # Keep DataFrame for later access if needed
                        }
                    )
                else:
                    missing_indices.append(idx)

            # Log warnings for missing timesteps
            if missing_indices:
                logger.warning(
                    f"⚠️  {len(missing_indices)} timesteps not found in parquet: "
                    f"{missing_indices[:5]}{'...' if len(missing_indices) > 5 else ''}"
                )

            # Store metadata
            self.dem_snapshots = dem_snapshots
            self.dem_file_indices = file_indices
            if dem_snapshots:
                self.n_particles = dem_snapshots[0]["coords"].shape[0]

            logger.info(
                f"✅ {len(dem_snapshots)} snapshots chargés "
                f"(N={self.n_particles} particules)"
            )

            return dem_snapshots

        except Exception as e:
            logger.error(f"❌ Erreur chargement DEM snapshots: {e}")
            raise

    def list_available_models(
        self,
        method: str | None = None,
        particle_diameter: float | None = None,
        fraction_visited_threshold: float = 0.95,
    ) -> list[dict[str, any]]:
        """
        Lister les modèles disponibles sur HuggingFace avec filtrage.

        **FILTRE CRITIQUE**: Garde seulement `fraction_visited >= threshold`
        pour garantir que les données DEM couvrent bien le domaine.

        Args:
            method: Filter by partitioning method (e.g., "voronoi", "cartesian").
                   If None, returns all methods.
            particle_diameter: Filter by diameter (0.004, 0.008, None for all).
            fraction_visited_threshold: Min fraction_visited in stats.json.
                                       Default=0.95 (HF standard).

        Returns:
            List of dicts with structure:
            [
                {
                    "folder_name": "voronoi_125_run1",
                    "method": "voronoi",
                    "n_states": 125,
                    "particle_diameter": 0.004,
                    "fraction_visited": 0.98,
                    "stats": {...},
                    "info": {...},
                },
                ...
            ]

        Examples:
            >>> analyzer = MarkovAnalyzer()
            >>> # All models with good fraction_visited
            >>> models = analyzer.list_available_models()
            >>>
            >>> # Only Voronoi with small particles
            >>> models = analyzer.list_available_models(
            ...     method="voronoi",
            ...     particle_diameter=0.004
            ... )
            >>> for m in models:
            ...     print(f"{m['folder_name']}: {m['n_states']} states, "
            ...           f"fraction_visited={m['fraction_visited']:.2f}")
        """
        available_models = []

        # Determine buckets to search
        if particle_diameter is not None:
            buckets = [
                f"hf://buckets/{BUCKET_ID}/{_get_bucket_prefix_from_particle_diameter(particle_diameter)}"
            ]
        else:
            buckets = ALL_BUCKET_BASES

        logger.info(
            f"🔍 Listage modèles: method={method}, "
            f"diameter={particle_diameter}, "
            f"fraction_visited >= {fraction_visited_threshold}"
        )

        for bucket_base in buckets:
            try:
                folders = self._list_folders(bucket_base)

                for folder_name in folders:
                    # Filter by method if specified
                    if method is not None:
                        detected = self._detect_method(folder_name)
                        if detected != method:
                            continue

                    try:
                        # Load experiment (lightweight: just stats + matrix shape)
                        data = self._load_experiment(bucket_base, folder_name)

                        # Extract stats
                        stats = data.get("stats", {})
                        info = data.get("info", {})

                        # **CRITICAL FILTER**: fraction_visited
                        fv = stats.get("fraction_visited", 1.0)
                        if fv < fraction_visited_threshold:
                            logger.debug(
                                f"   ⏭️  {folder_name}: skipped "
                                f"(fraction_visited={fv:.2f} < {fraction_visited_threshold})"
                            )
                            continue

                        model_info = {
                            "folder_name": folder_name,
                            "method": data.get("method"),
                            "n_states": data["matrix"].shape[0],
                            "particle_diameter": stats.get("particle_diameter"),
                            "fraction_visited": fv,
                            "stats": stats,
                            "info": info,
                        }

                        available_models.append(model_info)
                        logger.debug(
                            f"   ✅ {folder_name} ({model_info['n_states']} states)"
                        )

                    except Exception as e:
                        logger.debug(f"   ⚠️  {folder_name}: {e}")
                        continue

            except Exception as e:
                logger.warning(f"⚠️  Error listing {bucket_base}: {e}")
                continue

        logger.info(f"✅ {len(available_models)} models found")
        return available_models

    def get_model_lazy(self, folder_name: str) -> dict:
        """
        Charger un modèle avec lazy loading des matrices.

        Retourne un LoadedModel-like dict avec matrices chargées on-demand.

        Args:
            folder_name: Name of experiment folder

        Returns:
            Dict with structure:
            {
                "folder_name": str,
                "method": PartitioningMethod,
                "matrix": np.ndarray (lazy-loaded),
                "stats": dict,
                "config": dict,
            }
        """
        if folder_name not in self.results:
            # Try to load it
            try:
                data = self._load_experiment(BUCKET_BASE, folder_name)
                self.results[folder_name] = data
            except Exception as e:
                logger.error(f"❌ Could not load {folder_name}: {e}")
                raise

        return self.results[folder_name]

    def compute_dem_rsd(self, partitioner: str, species_labels: list | None = None, partitioner_name: str | None = None) -> dict:
        if partitioner is None:
            raise ValueError("❌ partitioner est obligatoire pour compute_dem_rsd()")

        n_states = partitioner.n_cells

        if species_labels is None:
            if self.species_labels is None:
                print(
                    "⚠️  species_labels non fourni, appel automatique de label_species()"
                )
                self.label_species()
            species_labels = self.species_labels

        if not hasattr(self, "dem_snapshots") or not self.dem_snapshots:
            print("⚠️  Aucun snapshot DEM chargé, chargement automatique...")
            self.load_dem_snapshots(file_indices=list(range(250, 6000, 50)))

        n_snaps = len(self.dem_snapshots)
        if n_snaps == 0:
            raise ValueError("❌ Aucun snapshot DEM disponible après chargement")

        if partitioner_name is None:
            partitioner_name = partitioner.label

        print(f"\n{'═' * 70}")
        print("📊 CALCUL DU RSD DEM")
        print(f"{'═' * 70}")
        print(f"Partitionneur   : {partitioner_name}")
        print(f"Nombre d'états  : {n_states}")
        print(
            f"Snapshots DEM   : {n_snaps} (t={self.dem_snapshots[0]['t']} → {self.dem_snapshots[-1]['t']})"
        )
        print(
            f"Espèce A        : {species_labels.sum()} particules / {len(species_labels)} total"
        )
        print(f"{'─' * 70}")

        times = np.zeros(n_snaps)
        rsd = np.zeros(n_snaps)
        entropy = np.zeros(n_snaps)
        intensity_seg = np.zeros(n_snaps)
        concentrations = []
        populations = []

        for k, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            times[k] = snap["t"]

            states = partitioner.compute_states(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )

            n_total = np.bincount(states, minlength=n_states).astype(float)
            n_A = np.bincount(states[species_labels], minlength=n_states).astype(float)

            C = np.zeros(n_states)
            mask = n_total > 0
            C[mask] = n_A[mask]
            concentrations.append(C.copy())
            populations.append(n_total.copy())

            C_active = C[mask]
            if len(C_active) > 1 and C_active.mean() > 0:
                rsd[k] = C_active.std() / C_active.mean()
            else:
                rsd[k] = 0

            if len(C_active) > 0:
                C_clip = np.clip(C_active, 1e-10, 1 - 1e-10)
                H = -np.mean(
                    C_clip * np.log(C_clip) + (1 - C_clip) * np.log(1 - C_clip)
                )
                H_max = np.log(2)
                entropy[k] = H / H_max if H_max > 0 else 0
            else:
                entropy[k] = 0

            C_bar = C_active.mean()
            if 0 < C_bar < 1 and len(C_active) > 1:
                intensity_seg[k] = C_active.var() / (C_bar * (1 - C_bar))
            else:
                intensity_seg[k] = 0

            if (k + 1) % 10 == 0 or k == 0 or k == n_snaps - 1:
                print(
                    f"   [{k + 1:4d}/{n_snaps}] t={int(times[k]):5d} | "
                    f"RSD={rsd[k] * 100:6.2f}% | "
                    f"Entropy={entropy[k]:.4f} | "
                    f"Cellules actives={mask.sum():3d}/{n_states}"
                )

        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
        mixing_time_50 = None
        mixing_time_90 = None
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

        print(f"{'─' * 70}")
        print("✅ RÉSULTATS DU CALCUL RSD DEM")
        print(f"{'─' * 70}")
        print(f"RSD initial     : {result['rsd_initial'] * 100:6.2f}%")
        print(f"RSD final       : {result['rsd_final'] * 100:6.2f}%")
        print(
            f"Réduction RSD   : {(1 - result['rsd_final'] / max(result['rsd_initial'], 1e-10)) * 100:6.2f}%"
        )
        print(f"Entropie finale : {entropy[-1]:.4f} / 1.000 (max)")
        print(f"t₅₀ (RSD ÷ 2)  : {mixing_time_50 or 'Non atteint'}")
        print(f"t₉₀ (RSD ÷ 10) : {mixing_time_90 or 'Non atteint'}")
        print(f"{'─' * 70}")
        print(f"Stocké dans     : self.dem_rsd_results['{partitioner_name}']")
        print(
            f"Conditions init : self.C0 (shape={self.C0.shape}) à t={self.initial_time}"
        )
        print(f"{'═' * 70}\n")

        return result

    def plot_rsd_vs_tau_comparison(
        self,
        partitioner: str,
        method: str,
        folder_name_template: str,
        tau_list: list | None = None,
        max_time_seconds: int = 60,
        figsize: tuple = (14, 8),
        save_name: str | None = None,
        species_criterion: str = "small",
    ) -> None:

        if tau_list is None:
            tau_list = [50, 100, 200, 500, 1000]

        fig, ax = plt.subplots(figsize=figsize)

        start_file = 250
        total_files = 5999
        file_indices = list(range(start_file, total_files + 1, 50))

        print("\n📊 Calcul RSD DEM...")
        self.load_dem_snapshots(file_indices=file_indices)
        if self.species_labels is None:
            self.label_species(criterion=species_criterion)

        all_coords = np.vstack([s["coords"] for s in self.dem_snapshots])
        partitioner.fit(all_coords)

        n_states = partitioner.n_cells
        species_labels = self.species_labels
        n_snaps = len(self.dem_snapshots)
        rsd_dem = np.zeros(n_snaps)
        times_dem_files = np.array([s["t"] for s in self.dem_snapshots])

        for i, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            states = partitioner.compute_states(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )
            C_i = np.zeros(n_states)
            for sid in range(n_states):
                mask = states == sid
                if mask.sum() > 0:
                    C_i[sid] = species_labels[mask].sum() / mask.sum()
            mask_active = C_i > 0
            if mask_active.sum() > 1:
                rsd_dem[i] = C_i[mask_active].std() / C_i[mask_active].mean()

        t_dem_seconds = times_dem_files * 0.01
        print(
            f"   DEM: {n_snaps} points de {t_dem_seconds[0]:.2f}s à {t_dem_seconds[-1]:.2f}s"
        )

        ax.plot(
            t_dem_seconds,
            rsd_dem * 100,
            color="black",
            marker="o",
            linewidth=3,
            markersize=8,
            label="RSD DEM (réel)",
            zorder=10,
            alpha=0.9,
        )

        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tau_list)))

        print(f"\n📊 Calcul RSD Markov pour {len(tau_list)} tau...")

        for tau_idx, tau in enumerate(tau_list):
            folder_name = folder_name_template.format(tau=tau)
            dt_markov = tau * 0.01

            print(f"\n   ── tau = {tau} ({dt_markov:.3f}s par pas) ──")

            try:
                M = self.get_matrix(folder_name)
            except Exception as e:
                print(f"   ⚠️  Folder {folder_name} non trouvé: {e}")
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

            C_t0 = np.zeros(n_states)
            C_t0[mask_active] = phi_A[mask_active] / phi_total[mask_active]
            if mask_active.sum() > 1 and C_t0[mask_active].mean() > 0:
                rsd_markov[0] = C_t0[mask_active].std() / C_t0[mask_active].mean()

            for t in range(1, n_steps_markov + 1):
                phi_A = phi_A @ M
                phi_total = phi_total @ M
                C_t = np.zeros(n_states)
                C_t[mask_active] = phi_A[mask_active] / phi_total[mask_active]
                if mask_active.sum() > 1:
                    rsd_markov[t] = (
                        C_t[mask_active].std() / C_t[mask_active].mean()
                        if C_t[mask_active].mean() > 0
                        else 0
                    )

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
                f"   ✅ {n_steps_markov + 1} points de {t_markov_seconds[0]:.2f}s à {t_markov_seconds[-1]:.2f}s (incl. t=0)"
            )

        ax.set_xlabel("Temps (s)", fontsize=13, fontweight="bold")
        ax.set_ylabel("RSD (%)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Influence du pas de temps Markov (tau) sur la cinétique de mélange\n"
            f"{method.upper()} | {partitioner.label} | {n_states} cellules",
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
        print(f"\n✅ Figure sauvegardée: {save_name}")
        plt.show()

        return fig, ax

    # ═════════════════════════════════════════════════════════════════════
    # NOUVELLES MÉTHODES INHOMOGÈNES
    # ═════════════════════════════════════════════════════════════════════

    def load_inhomogeneous_method(self, method: str) -> dict:
        """
        Charge les expériences inhomogènes d'une méthode donnée depuis Inhomogènes/.

        Sibling de load_method() — liste les dossiers dans Inhomogènes/,
        filtre par méthode, et charge avec détection du format inhomogène.
        """
        self.results = {}
        self.by_method = defaultdict(dict)
        loaded_folders = set()

        for base_path in ALL_BUCKET_BASES:
            base_path.replace("hf://buckets/ktongue/DEM_MCM/", "")

            # Chercher dans Inhomogènes/ puis dans les catégories habituelles
            search_paths = [f"{base_path}/Inhomogènes"]
            for cat in ALL_CATEGORIES:
                search_paths.append(f"{base_path}/{cat}")

            for search_base in search_paths:
                try:
                    folders = self._list_folders(search_base)
                    for folder in folders:
                        if folder in loaded_folders:
                            continue

                        detected = self._detect_method(folder)
                        if detected == method or method == "all":
                            try:
                                data = self._load_experiment(search_base, folder)
                                if data.get("inhomogeneous", False):
                                    self.results[folder] = data
                                    self.by_method[method][folder] = data
                                    loaded_folders.add(folder)
                                    n_blocks = data.get(
                                        "inhomogeneous_metadata", {}
                                    ).get("n_blocks", "?")
                                    print(
                                        f"   ✅ {folder}: {n_blocks} blocs, shape={data['matrix'].shape}"
                                    )
                                else:
                                    print(f"   ⏭️  {folder}: ignoré (homogène)")
                            except Exception as e:
                                print(f"   ⚠️  {folder}: {e}")
                except Exception as e:
                    print(f"   ⚠️  Impossible de lister {search_base}: {e}")

        print(f"\n{len(self.results)} expériences inhomogènes {method} chargées")
        return self.results

    def visualize_P_blocks_evolution(
        self, folder_name: str, species: str = "small", figsize: tuple = (16, 4)
    ) -> dict:
        """
        Visualise l'évolution des matrices de transition P_k (une par NLT).

        Affiche les heatmaps côte à côte pour tous les blocs, plus une heatmap
        de la différence entre blocs consécutifs.

        Args:
            folder_name: Nom du dossier de l'expérience inhomogène.
            species: Espèce à visualiser ("small" ou "large").
            figsize: Taille de la figure.

        Returns:
            fig, axes: Figure et axes matplotlib.
        """
        data = self._load_experiment(BUCKET_BASE, folder_name)

        if not data.get("inhomogeneous", False):
            print(f"⚠️  {folder_name} n'est pas une expérience inhomogène")
            return None, None

        # Chercher les P_blocks pour l'espèce demandée
        prefix = f"{BUCKET_BASE}/{folder_name}"
        P_blocks_path = f"{prefix}/P_blocks_{species}.npy"

        try:
            P_blocks = self._load_npy(P_blocks_path)
        except Exception:
            # Essayer de trouver dans Inhomogènes/
            for base in ALL_BUCKET_BASES:
                for cat in ["Inhomogènes", *ALL_CATEGORIES]:
                    try:
                        path = f"{base}/{cat}/{folder_name}/P_blocks_{species}.npy"
                        P_blocks = self._load_npy(path)
                        break
                    except Exception:
                        continue
                else:
                    continue
                break
            else:
                print(f"❌ P_blocks_{species}.npy introuvable pour {folder_name}")
                return None, None

        n_blocks = P_blocks.shape[0]
        n_states = P_blocks.shape[1]

        fig, axes = plt.subplots(2, max(2, n_blocks), figsize=figsize, squeeze=False)

        fig.suptitle(
            f"Évolution des matrices de transition P_k — {species}\n"
            f"{folder_name} ({n_blocks} blocs, {n_states} états)",
            fontweight="bold",
            fontsize=14,
        )

        # Ligne 1 : heatmaps des P_k
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
            ax.set_title(f"P_{k} (bloc {k + 1}/{n_blocks})")
            ax.set_xlabel("Source")
            ax.set_ylabel("Destination")
            plt.colorbar(im, ax=ax, fraction=0.046)

        # Masquer les axes vides
        for k in range(n_blocks, axes.shape[1]):
            axes[0, k].set_visible(False)
            axes[1, k].set_visible(False)

        # Ligne 2 : différences entre blocs consécutifs
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
            ax.set_title(f"|P_{k} - P_{k - 1}| (norme={diff.sum():.4f})")
            ax.set_xlabel("Source")
            ax.set_ylabel("Destination")
            plt.colorbar(im, ax=ax, fraction=0.046)

        axes[1, 0].set_visible(False)

        plt.tight_layout()
        return fig, axes

    def compute_inhomogeneous_rsd(self, folder_name: str, partitioner: str, species_labels: list | None = None) -> dict:
        """
        Calcule le RSD pour une chaîne inhomogène (matrices variables dans le temps).

        Charge P_blocks et propage l'état avec des matrices qui changent à chaque bloc.
        Compare avec le RSD DEM de référence.

        Args:
            folder_name: Nom du dossier de l'expérience inhomogène.
            partitioner: Instance du partitionneur entraîné.
            species_labels: Masque booléen (n_particles,) pour l'espèce.

        Returns:
            dict avec times_markov, rsd_markov, et les métriques.
        """
        from .bucket_io import load_experiment_from_bucket

        data = load_experiment_from_bucket(folder_name)
        if not data.get("inhomogeneous", False):
            print(f"⚠️  {folder_name} n'est pas inhomogène")
            return None

        inhom_meta = data.get("inhomogeneous_metadata", {})
        n_blocks = inhom_meta.get("n_blocks", 1)
        species_list = inhom_meta.get("species_list", ["small", "large"])

        print(f"\n{'═' * 70}")
        print(f"📊 RSD INHOMOGÈNE — {folder_name}")
        print(f"{'═' * 70}")
        print(f"Nombre de blocs: {n_blocks}")
        print(f"Espèces: {species_list}")

        if species_labels is None:
            if self.species_labels is None:
                print(
                    "⚠️  species_labels non fourni, appel automatique de label_species()"
                )
                self.label_species()
            species_labels = self.species_labels

        if not hasattr(self, "dem_snapshots") or not self.dem_snapshots:
            print("⚠️  Aucun snapshot DEM chargé, chargement automatique...")
            self.load_dem_snapshots(file_indices=list(range(250, 6000, 50)))

        n_states = partitioner.n_cells
        n_snaps = len(self.dem_snapshots)
        times = np.array([s["t"] for s in self.dem_snapshots])

        # RSD DEM de référence
        rsd_dem = np.zeros(n_snaps)
        for i, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            states = partitioner.compute_states(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )
            C_i = np.zeros(n_states)
            for sid in range(n_states):
                mask = states == sid
                if mask.sum() > 0:
                    C_i[sid] = species_labels[mask].sum() / mask.sum()
            mask_active = C_i > 0
            if mask_active.sum() > 1:
                rsd_dem[i] = C_i[mask_active].std() / C_i[mask_active].mean()

        # Propagation inhomogène pour chaque espèce
        results = {}
        for sp in species_list:
            P_blocks = data["species"][sp]["P_blocks"]  # (n_blocks, n_states, n_states)
            S0 = data["species"][sp]["S_matrix"][0].astype(float)

            # Propagation avec matrices variables
            n_steps_markov = 200
            block_size = max(1, n_steps_markov // n_blocks)
            phi = S0.copy()
            phi_total = S0.copy()
            rsd_markov = np.zeros(n_steps_markov + 1)

            # État initial
            mask_active = phi > 0
            if mask_active.sum() > 1 and phi[mask_active].mean() > 0:
                rsd_markov[0] = phi[mask_active].std() / phi[mask_active].mean()

            for t in range(1, n_steps_markov + 1):
                block_idx = min((t - 1) // block_size, n_blocks - 1)
                phi = phi @ P_blocks[block_idx]
                phi_total = phi_total @ P_blocks[block_idx]
                C_t = np.zeros(n_states)
                C_t[mask_active] = phi[mask_active] / phi_total[mask_active]
                if mask_active.sum() > 1 and C_t[mask_active].mean() > 0:
                    rsd_markov[t] = C_t[mask_active].std() / C_t[mask_active].mean()

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
