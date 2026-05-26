"""
===================================================================================
ANALYSE MARKOVIENNE — Chargement et visualisation depuis le bucket HuggingFace
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
import numpy as np
import matplotlib.pyplot as plt
import json
import io
import logging
from collections import defaultdict
from huggingface_hub import HfFileSystem

try:
    from .import bucket_io as b_io
    from .utils import apply_species_mask,load_parquet_as_timestep_dict
except ImportError:
    import bucket_io as b_io
    from utils import apply_species_mask,load_parquet_as_timestep_dict

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

BUCKET_ID = b_io.BUCKET_ID
BUCKET_PREFIX = b_io.BUCKET_PREFIX
BUCKET_BASE = b_io.BUCKET_BASE


def _get_bucket_prefix_from_particle_diameter(particle_diameter:float)-> str:
    if particle_diameter == 0.008:
        return "BIG"
    elif particle_diameter == 0.004:
        return "SMALL"
    else:
        return "Experiments"


ALL_BUCKET_PREFIXES = ["Experiments", "SMALL", "BIG"]

OLD_BUCKET_PREFIX = "Experiments"
OLD_BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{OLD_BUCKET_PREFIX}"

ALL_BUCKET_BASES = [f"hf://buckets/{BUCKET_ID}/{prefix}" for prefix in ALL_BUCKET_PREFIXES]
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

    def __init__(self:MarkovAnalyzer)->None:
        self.fs = HfFileSystem()
        self.results:dict = {}
        self.by_method = defaultdict(dict)

        self.dem_snapshots:list = []
        self.dem_file_indices:list = []
        self.n_particles:int = 0
        self.dem_diameters:np.ndarray = None #type:ignore
        self.dem_velocities:np.ndarray = None #type:ignore
        self.dem_angular_velocities:np.ndarray = None #type:ignore
        self.species_labels:np.ndarray = None #type:ignore

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

    def _detect_method(self, folder_name, params=None):
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

    def _parse_experiment_info(self, folder_name, params, stats):
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
                    try:
                        info["n_states"] = int(nx) * int(ny) * int(nz)
                    except:
                        pass
        return info

    # ─────────────────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────────────────

    def _load_npy(self, full_path):
        with self.fs.open(full_path, "rb") as f:
            return np.load(io.BytesIO(f.read()))

    def _load_json(self, full_path):
        with self.fs.open(full_path, "r") as f:
            return json.load(f)

    def _load_partitioner_data(self, partitioner_path):
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
                cyl_params = self._load_json(f"{partitioner_path}/cylindrical_params.json")
                partitioner_data.update(cyl_params)
                r_edges = self._load_npy(f"{partitioner_path}/r_edges.npy")
                partitioner_data["r_edges"] = r_edges
            except Exception as e:
                print(f"⚠️  Impossible de charger les données cylindriques: {e}")

        elif partitioner_type == "CartesianPartitioner":
            try:
                cart_params = self._load_json(f"{partitioner_path}/cartesian_params.json")
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

    def _list_folders(self, base_path=BUCKET_BASE):
        try:
            items = self.fs.ls(base_path)
            return sorted([
                item["name"].split("/")[-1]
                for item in items
                if item["type"] == "directory"
            ])
        except FileNotFoundError:
            return []

    def _load_experiment(self, base_path=BUCKET_BASE, folder_name=None):
        prefix = f"{base_path}/{folder_name}"

        stats = {}
        try:
            stats = self._load_json(f"{prefix}/stats.json")
            particle_diameter = stats.get("particle_diameter")
            if particle_diameter is not None:
                correct_bucket_prefix = _get_bucket_prefix_from_particle_diameter(particle_diameter)
                correct_base_path = f"hf://buckets/{BUCKET_ID}/{correct_bucket_prefix}"
                if correct_base_path != base_path:
                    print(f"     ℹ️  bucket fourni {base_path} → rechargement depuis {correct_bucket_prefix} (particle_diameter={particle_diameter})")
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

        for attempt_path in buckets_to_try:
            prefix = f"{attempt_path}/{folder_name}"
            try:
                matrix = self._load_npy(f"{prefix}/transitionmatrix.npy")

                params = {}
                for fname in ["config.json", "params.json"]:
                    try:
                        params = self._load_json(f"{prefix}/{fname}")
                        break
                    except:
                        continue

                if not stats:
                    try:
                        stats = self._load_json(f"{prefix}/stats.json")
                    except:
                        pass

                centroids = None
                try:
                    centroids = self._load_npy(f"{prefix}/centroids.npy")
                except:
                    pass

                partitioner_data = None
                try:
                    partitioner_data = self._load_partitioner_data(f"{prefix}/partitioner")
                except:
                    pass

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
            }
        else:
            buckets_str = ", ".join([b.replace("hf://buckets/ktongue/DEM_MCM/", "") for b in buckets_to_try])
            raise Exception(f"Impossible de charger {folder_name} depuis les buckets: {buckets_str}. Erreur: {last_error}")

    def load_method(self, method):
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

    def get_matrix(self, folder_name):
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
            raise ValueError("❌ No DEM snapshots loaded. Call load_dem_snapshots() first.")
        
        snap_0 = self.dem_snapshots[0]
        df_0 = snap_0.get("df")
        
        if df_0 is None:
            logger.warning("⚠️  DataFrame not available in snapshot, cannot label species")
            self.species_labels = np.ones(self.n_particles, dtype=bool)
            return self.species_labels
        
        if criterion == "small":
            # Label small particles (diameter < 0.006 m)
            if "Diameter" in df_0.columns:
                self.species_labels = df_0["Diameter"].values < 0.006
            else:
                logger.warning("⚠️  'Diameter' column not found, using first_half criterion")
                self.species_labels = np.arange(self.n_particles) < self.n_particles // 2
                
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
        file_indices: list | None = None,
        sample_every: int = 1,
        particle_diameter: float | None = None,
    ) -> list[dict]:
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
                        if "stats" in folder_data and folder_data["stats"]:
                            particle_diameter = folder_data["stats"].get("particle_diameter")
                            if particle_diameter:
                                break
            except:
                pass
        
        bucket_prefix = _get_bucket_prefix_from_particle_diameter(particle_diameter)
        bucket_base = f"hf://buckets/{BUCKET_ID}/{bucket_prefix}"
        parquet_path = f"{bucket_base}/simulation_complete.parquet"
        
        logger.info(
            f"📦 Chargement DEM snapshots: "
            f"indices={file_indices[0]}-{file_indices[-1]}, "
            f"prefix={bucket_prefix}"
        )
        
        try:
            # Load timestep dict from HF
            timestep_dict = load_parquet_as_timestep_dict(
                parquet_path=parquet_path,
                fs=self.fs
            )
            
            # Convert to dem_snapshots format
            dem_snapshots = []
            missing_indices = []
            
            for idx in file_indices:
                if idx in timestep_dict:
                    df = timestep_dict[idx]
                    # Extract coordinates (columns: 'coordinates:0', 'coordinates:1', 'coordinates:2')
                    coords = np.column_stack([
                        df['coordinates:0'].values,
                        df['coordinates:1'].values,
                        df['coordinates:2'].values,
                    ])
                    dem_snapshots.append({
                        "t": idx,
                        "coords": coords,
                        "df": df,  # Keep DataFrame for later access if needed
                    })
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
    ) -> list[dict]:
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
                        logger.debug(f"   ✅ {folder_name} ({model_info['n_states']} states)")
                        
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

    def compute_dem_rsd(self, partitioner, species_labels=None, partitioner_name=None):
        if partitioner is None:
            raise ValueError("❌ partitioner est obligatoire pour compute_dem_rsd()")

        n_states = partitioner.n_cells

        if species_labels is None:
            if self.species_labels is None:
                print("⚠️  species_labels non fourni, appel automatique de label_species()")
                self.label_species()
            species_labels = self.species_labels

        if not hasattr(self, 'dem_snapshots') or not self.dem_snapshots:
            print("⚠️  Aucun snapshot DEM chargé, chargement automatique...")
            self.load_dem_snapshots(file_indices=list(range(250, 6000, 50)))

        n_snaps = len(self.dem_snapshots)
        if n_snaps == 0:
            raise ValueError("❌ Aucun snapshot DEM disponible après chargement")

        if partitioner_name is None:
            partitioner_name = partitioner.label

        print(f"\n{'═'*70}")
        print(f"📊 CALCUL DU RSD DEM")
        print(f"{'═'*70}")
        print(f"Partitionneur   : {partitioner_name}")
        print(f"Nombre d'états  : {n_states}")
        print(f"Snapshots DEM   : {n_snaps} (t={self.dem_snapshots[0]['t']} → {self.dem_snapshots[-1]['t']})")
        print(f"Espèce A        : {species_labels.sum()} particules / {len(species_labels)} total")
        print(f"{'─'*70}")

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
            # C[mask] = n_A[mask] / n_total[mask]

            concentrations.append(C.copy())
            populations.append(n_total.copy())

            C_active = C[mask]
            if len(C_active) > 1 and C_active.mean() > 0:
                rsd[k] = C_active.std() / C_active.mean()
            else:
                rsd[k] = 0

            if len(C_active) > 0:
                C_clip = np.clip(C_active, 1e-10, 1 - 1e-10)
                H = -np.mean(C_clip * np.log(C_clip) + (1 - C_clip) * np.log(1 - C_clip))
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
                print(f"   [{k+1:4d}/{n_snaps}] t={int(times[k]):5d} | "
                      f"RSD={rsd[k]*100:6.2f}% | "
                      f"Entropy={entropy[k]:.4f} | "
                      f"Cellules actives={mask.sum():3d}/{n_states}")

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
        self.phi_A_0 = np.bincount(states0[species_labels], minlength=n_states).astype(float)

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

        print(f"{'─'*70}")
        print(f"✅ RÉSULTATS DU CALCUL RSD DEM")
        print(f"{'─'*70}")
        print(f"RSD initial     : {result['rsd_initial']*100:6.2f}%")
        print(f"RSD final       : {result['rsd_final']*100:6.2f}%")
        print(f"Réduction RSD   : {(1 - result['rsd_final']/max(result['rsd_initial'], 1e-10))*100:6.2f}%")
        print(f"Entropie finale : {entropy[-1]:.4f} / 1.000 (max)")
        print(f"t₅₀ (RSD ÷ 2)  : {mixing_time_50 or 'Non atteint'}")
        print(f"t₉₀ (RSD ÷ 10) : {mixing_time_90 or 'Non atteint'}")
        print(f"{'─'*70}")
        print(f"Stocké dans     : self.dem_rsd_results['{partitioner_name}']")
        print(f"Conditions init : self.C0 (shape={self.C0.shape}) à t={self.initial_time}")
        print(f"{'═'*70}\n")

        return result

    def plot_rsd_vs_tau_comparison(self, partitioner, method, folder_name_template,
                                    tau_list=None, max_time_seconds=60, figsize=(14, 8), save_name=None,
                                    species_criterion="small"):
        import re

        if tau_list is None:
            tau_list = [50, 100, 200, 500, 1000]

        fig, ax = plt.subplots(figsize=figsize)

        start_file = 250
        total_files = 5999
        file_indices = list(range(start_file, total_files + 1, 50))

        print(f"\n📊 Calcul RSD DEM...")
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
            states = partitioner.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])
            C_i = np.zeros(n_states)
            for sid in range(n_states):
                mask = states == sid
                if mask.sum() > 0:
                    C_i[sid] = species_labels[mask].sum() / mask.sum()
            mask_active = C_i > 0
            if mask_active.sum() > 1:
                rsd_dem[i] = C_i[mask_active].std() / C_i[mask_active].mean()

        t_dem_seconds = times_dem_files * 0.01
        print(f"   DEM: {n_snaps} points de {t_dem_seconds[0]:.2f}s à {t_dem_seconds[-1]:.2f}s")

        ax.plot(t_dem_seconds, rsd_dem * 100,
               color="black", marker='o', linewidth=3, markersize=8,
               label="RSD DEM (réel)", zorder=10, alpha=0.9)

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
            states0 = partitioner.compute_states(coords0[:, 0], coords0[:, 1], coords0[:, 2])

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
                    rsd_markov[t] = C_t[mask_active].std() / C_t[mask_active].mean() if C_t[mask_active].mean() > 0 else 0

            t_markov_seconds = (start_file + np.arange(n_steps_markov + 1) * tau) * 0.01

            ax.plot(t_markov_seconds, rsd_markov * 100,
                   color=colors[tau_idx], linewidth=2.5, linestyle='-',
                   label=f"Markov tau={tau} ({n_steps_markov+1} pts)", zorder=5, alpha=0.8)

            print(f"   ✅ {n_steps_markov+1} points de {t_markov_seconds[0]:.2f}s à {t_markov_seconds[-1]:.2f}s (incl. t=0)")

        ax.set_xlabel("Temps (s)", fontsize=13, fontweight='bold')
        ax.set_ylabel("RSD (%)", fontsize=13, fontweight='bold')
        ax.set_title(
            f"Influence du pas de temps Markov (tau) sur la cinétique de mélange\n"
            f"{method.upper()} | {partitioner.label} | {n_states} cellules",
            fontsize=14, fontweight='bold', pad=15
        )

        ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black', ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_xlim(t_dem_seconds[0], max_time_seconds)
        ax.set_ylim(bottom=0)
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)

        plt.tight_layout()

        if save_name is None:
            save_name = f"rsd_tau_comparison_{method}_{n_states}cells.png"

        plt.savefig(save_name, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Figure sauvegardée: {save_name}")
        plt.show()

        return fig, ax
    

