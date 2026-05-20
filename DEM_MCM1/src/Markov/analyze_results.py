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
from collections import defaultdict
from huggingface_hub import HfFileSystem

try:
    from .import bucket_io as b_io
    from .utils import apply_species_mask
except ImportError:
    import bucket_io as b_io
    from utils import apply_species_mask

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
    # DONNÉES DEM
    # ─────────────────────────────────────────────────────────────────────

    def load_dem_snapshots(self, file_indices=None, sample_every=1):
        import polars as pl

        if not hasattr(self, '_dem_fs'):
            self._dem_fs = HfFileSystem()
            self._dem_files = sorted(
                self._dem_fs.glob("hf://buckets/ktongue/DEM_MCM/Output Paraview/*.csv")
            )
            print(f"{len(self._dem_files)} fichiers DEM disponibles")

        if file_indices is None:
            file_indices = list(range(0, min(len(self._dem_files), 500), 10))

        self.dem_snapshots = []
        self.dem_file_indices = file_indices

        print(f"📂 Chargement de {len(file_indices)} snapshots DEM...")
        for i, idx in enumerate(file_indices):
            with self._dem_fs.open(self._dem_files[idx], "rb") as f:
                df = pl.read_csv(f)
                coords = np.column_stack([
                    df["coordinates:0"].to_numpy(),
                    df["coordinates:1"].to_numpy(),
                    df["coordinates:2"].to_numpy(),
                ])[::sample_every]

                if i == 0:
                    self.dem_diameters = df["Diameter"].to_numpy()[::sample_every]
                    print(f"   Diamètres chargés : {len(self.dem_diameters)} particules")
                    self.dem_velocities = np.column_stack([
                        df["Velocity:0"].to_numpy(),
                        df["Velocity:1"].to_numpy(),
                        df["Velocity:2"].to_numpy(),
                    ])[::sample_every]
                    print(f"   Vitesses chargées : {self.dem_velocities.shape}")
                    if "Angular_velocity:0" in df.columns:
                        self.dem_angular_velocities = np.column_stack([
                            df["Angular_velocity:0"].to_numpy(),
                            df["Angular_velocity:1"].to_numpy(),
                            df["Angular_velocity:2"].to_numpy(),
                        ])[::sample_every]
                        print(f"   Vitesses angulaires chargées : {self.dem_angular_velocities.shape}")

                self.dem_snapshots.append({"t": idx, "coords": coords})

            if (i + 1) % 10 == 0 or i == len(file_indices) - 1:
                print(f"   [{i+1}/{len(file_indices)}] t={idx}: {len(coords)} particules")

        self.n_particles = len(self.dem_snapshots[0]["coords"])
        print(f"✅ {len(self.dem_snapshots)} snapshots | {self.n_particles} particules/snapshot")
        return self.dem_snapshots

    def label_species(self, criterion="small", custom_labels=None):
        if custom_labels is not None:
            self.species_labels = np.asarray(custom_labels, dtype=bool)
            n_a = self.species_labels.sum()
            print(f"✅ Labels custom: {n_a} A / {len(self.species_labels) - n_a} B")
            return self.species_labels

        diameters = self.dem_diameters

        unique_vals = np.unique(diameters)
        if len(unique_vals) != 2:
            print(f"⚠️ Attention : {len(unique_vals)} diamètres différents trouvés (attendu 2).")
            small_val, large_val = unique_vals[0], unique_vals[-1]
        else:
            small_val, large_val = unique_vals[0], unique_vals[1]

        print(f"📏 Diamètres détectés : {small_val:.4f} m et {large_val:.4f} m")

        if criterion == "large":
            labels = diameters == large_val
        elif criterion == "small":
            labels = diameters == small_val
        elif criterion == "auto":
            labels = diameters == large_val
        else:
            raise ValueError(f"Critère '{criterion}' non reconnu. Utilisez 'large', 'small' ou 'auto'.")

        self.species_labels = labels
        n_a = labels.sum()
        print(f"✅ Espèces ({criterion}): {n_a} particules A / {len(labels) - n_a} particules B")
        return self.species_labels

    def create_partitioner_for_comparison(self, method, method_kwargs):
        from .partitioners import create_partitioner

        all_coords = np.vstack([s["coords"] for s in self.dem_snapshots])
        part = create_partitioner(method, **method_kwargs)
        part.fit(all_coords)

        diag = part.diagnostics(all_coords)
        print(f"🔧 {part.label}: {part.n_cells} cellules | "
              f"{diag['n_visited']} visitées | "
              f"pop μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}")
        return part

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
            C[mask] = n_A[mask] / n_total[mask]

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
