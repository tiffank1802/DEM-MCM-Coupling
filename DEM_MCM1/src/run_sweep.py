"""
===================================================================================
SWEEP MARKOVIEN — Lance les calculs pour un type de partitionnement donné.
===================================================================================

Usage:
    python run_sweep.py --method voronoi
    python run_sweep.py --method cartesian
    python run_sweep.py --method all
    python run_sweep.py --method voronoi --list   # liste les configs sans lancer

Depuis Python:
    from run_sweep import run_markov_sweep
    run_markov_sweep("cylindrical")
===================================================================================
"""

import argparse
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfFileSystem
from tqdm import tqdm

# Imports relatifs (notebooks) vs absolus (script direct)
try:
    from . import partitioners as part
    from .bucket_io import BUCKET_BASE, save_experiment_to_bucket
    from .partitioners import REGISTRY, create_partitioner
    from .utils import apply_species_mask, load_parquet_as_timestep_dict
except ImportError:
    # Imports absolus quand lancé directement comme script
    import partitioners as part
    from bucket_io import save_experiment_to_bucket
    from partitioners import REGISTRY, create_partitioner
    from utils import load_parquet_as_timestep_dict


# =============================================================================
# CONFIGURATION GÉNÉRALE
# =============================================================================

BASE_OUTPUT_DIR = "RaffinageTemporel"
HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/simulation_complete.parquet"
SAMPLE_RATE = 50  # pour le fit des partitionneurs


# =============================================================================
# DATACLASS EXPÉRIENCE
# =============================================================================
@dataclass
class ExperimentConfig:
    """Configuration d'une expérience."""

    method: str = "cartesian"
    method_kwargs: dict = field(default_factory=dict)
    nlt: int = 2
    tau: int = 157  # Écart entre start et end pour chaque paire
    step: int = 157  # Distance entre 2 starts principaux (quand NLT > 1)
    dt: int | None = None  # Raffinage temporel à l'intérieur de chaque step
    start_index: int = 157
    particle_diameter: float | None = None
    inhomogeneous: bool = False  # Si True, construit une matrice P par NLT

    def __post_init__(self) -> None:
        if self.method_kwargs is None:
            self.method_kwargs = {}
        if self.dt is None:
            # Raffinage par défaut : 5 apprentissages par step
            self.dt = max(1, self.step // 100)

    def output_folder(self, sample_coords: np.ndarray | None = None) -> str:
        p = create_partitioner(self.method, **self.method_kwargs)
        if sample_coords is not None:
            p.fit(sample_coords)

        # Retourner seulement le nom du dossier, pas un chemin
        folder_name = (
            f"{p.label}_NLT{self.nlt}_step{self.step}_"
            f"dt{self.dt}_tau{self.tau}_start{self.start_index}"
        )

        # Ajouter le suffixe particle_diameter si spécifié
        if self.particle_diameter is not None:
            diameter_str = str(self.particle_diameter).replace(".", "")
            folder_name += f"_d{diameter_str}"

        # NOUVEAU: préfixe inhomogeneous_ pour les chaînes inhomogènes
        if self.inhomogeneous:
            folder_name = f"inhomogeneous_{folder_name}"

        return folder_name  # Retourne juste le nom


# =============================================================================
# CONFIGURATIONS PAR MÉTHODE
# =============================================================================


def get_configs(
    method: str, particle_diameter: float | None = None
) -> list[ExperimentConfig]:
    """Retourne la liste de configs pour une méthode donnée."""
    configs = []

    print(f"   🔍 Génération des configs pour {method}...")
    if particle_diameter is not None:
        print(f"   🎯 Filtre diamètre: {particle_diameter}")

    # ══════════════════════════════════════════════════════════════════════
    # 1. SWEEP DE DISCRÉTISATION SPATIALE (avec paramètres temporels par défaut)
    # ══════════════════════════════════════════════════════════════════════

    if method == "cartesian":
        for n in [2, 3, 4, 5]:
            configs.append(
                ExperimentConfig(
                    method="cartesian",
                    method_kwargs={"nx": n, "ny": n, "nz": n},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "cylindrical":
        for nr in [3, 4, 5, 6]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": nr,
                        "ntheta": 3,
                        "nz": 3,
                        "radial_mode": "equal_area",
                    },
                    particle_diameter=particle_diameter,
                )
            )
    elif method == "dbscan":
        for nr in [3, 4, 5, 6]:
            configs.append(
                ExperimentConfig(
                    method="dbscan",
                    method_kwargs={
                        "min_samples": nr,
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nth in [1, 2, 3, 4]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 1,
                        "ntheta": nth,
                        "nz": 1,
                        "radial_mode": "equal_area",
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nz in [1, 2]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 2,
                        "ntheta": 2,
                        "nz": nz,
                        "radial_mode": "equal_area",
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for mode in ["equal_dr", "equal_area"]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 2,
                        "ntheta": 2,
                        "nz": 1,
                        "radial_mode": mode,
                    },
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "voronoi":
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="voronoi",
                    method_kwargs={"n_cells": nc},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "quantile":
        for n in [2, 3, 4, 5, 6]:
            configs.append(
                ExperimentConfig(
                    method="quantile",
                    method_kwargs={"nx": n, "ny": n, "nz": 1},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "octree":
        for mp in [20, 40, 80, 16, 32, 64, 28, 50, 60, 70, 100]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": mp, "max_depth": 2},
                    particle_diameter=particle_diameter,
                )
            )
        for md in [2, 1]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": 100, "max_depth": md},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "physics":
        # Sweep n_cells avec velocity_weight par défaut
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="physics",
                    method_kwargs={"n_cells": nc, "velocity_weight": 0.5},
                    particle_diameter=particle_diameter,
                )
            )
        # Sweep velocity_weight (importance de la vitesse dans le clustering)
        for vw in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            configs.append(
                ExperimentConfig(
                    method="physics",
                    method_kwargs={"n_cells": 30, "velocity_weight": vw},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "physics_full_vel":
        # Sweep n_cells avec velocity_weight par défaut (vecteur vitesse complet)
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="physics_full_vel",
                    method_kwargs={"n_cells": nc, "velocity_weight": 0.5},
                    particle_diameter=particle_diameter,
                )
            )
        # Sweep velocity_weight (importance du vecteur vitesse)
        for vw in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="physics_full_vel",
                    method_kwargs={"n_cells": 30, "velocity_weight": vw},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "spectral":
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={
                        "n_cells": nc,
                        "velocity_weight": 0.5,
                        "n_neighbors": 15,
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for vw in [0.1, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={
                        "n_cells": 20,
                        "velocity_weight": vw,
                        "n_neighbors": 15,
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for k in [5, 10, 15, 20, 30]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={
                        "n_cells": 20,
                        "velocity_weight": 0.5,
                        "n_neighbors": k,
                    },
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "gmm":
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="gmm",
                    method_kwargs={"n_cells": nc, "velocity_weight": 0.5},
                    particle_diameter=particle_diameter,
                )
            )
        for vw in [0.1, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="gmm",
                    method_kwargs={"n_cells": 20, "velocity_weight": vw},
                    particle_diameter=particle_diameter,
                )
            )
    elif method == "spectral_biclustering":
        for n_cells, n_col in [(20, 3), (30, 3), (40, 4), (50, 4)]:
            configs.append(
                ExperimentConfig(
                    method="spectral_biclustering",
                    method_kwargs={
                        "n_cells": n_cells,
                        "n_col_clusters": n_col,
                        "method": "log",
                        "velocity_weight": 0.6,
                    },
                    particle_diameter=particle_diameter,
                )
            )
    elif method == "adaptive":
        for y_q in [0.5, 0.6, 0.7, 0.8, 0.9]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": y_q,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 1,
                            "ntheta": 36,
                            "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nr in [2, 3, 1]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": nr,
                            "ntheta": 30,
                            "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nz in [1, 2]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 1,
                            "ntheta": 30,
                            "nz": nz,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nth in [
            1,
            4,
            8,
            12,
            16,
            20,
            30,
            21,
            23,
            22,
            35,
            37,
            39,
            40,
            50,
            60,
            70,
            80,
            90,
            10,
            12,
            23,
            40,
        ]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 1,
                            "ntheta": nth,
                            "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for n_top in [1, 2, 3, 4]:
            top_method = "single" if n_top == 1 else "cylindrical"
            top_kwargs = (
                {}
                if n_top == 1
                else {
                    "nr": 1,
                    "ntheta": n_top,
                    "nz": 1,
                    "radial_mode": "equal_area",
                }
            )
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": n_top,
                        "top_method": top_method,
                        "top_kwargs": top_kwargs,
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 1,
                            "ntheta": 30,
                            "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nc in [10, 20, 30, 50, 64, 15, 35, 25, 50]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "voronoi",
                        "bottom_kwargs": {"n_cells": nc},
                    },
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "multizone":
        pass

    elif method == "single":
        configs.append(
            ExperimentConfig(
                method="single",
                method_kwargs={},
                particle_diameter=particle_diameter,
            )
        )

    else:
        raise ValueError(f"Méthode inconnue: {method}")

    spatial_count = len(configs)
    print(f"   📊 Configs spatiales pour {method}: {spatial_count}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. SWEEP TEMPOREL (avec paramètres spatiaux par défaut)
    # ══════════════════════════════════════════════════════════════════════

    default_spatial_kwargs = _get_default_kwargs(method)
    print(f"   🕒 Ajout des sweeps temporels avec: {default_spatial_kwargs}")

    temporal_configs = []

    for nlt in [1, 2, 3, 5]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                nlt=nlt,
                particle_diameter=particle_diameter,
            )
        )

    for step in [20, 30, 40]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                step=step,
                particle_diameter=particle_diameter,
            )
        )

    step_ref = 20
    for dt in [1, 2, 3, 4]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                step=step_ref,
                dt=dt,
                particle_diameter=particle_diameter,
            )
        )

    for tau in [20, 50, 100, 200]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                tau=tau,
                particle_diameter=particle_diameter,
            )
        )

    recommended_configs = [
        ExperimentConfig(
            method=method,
            method_kwargs=default_spatial_kwargs,
            nlt=3,
            step=100,
            dt=1,
            tau=50,
            particle_diameter=particle_diameter,
        ),
        ExperimentConfig(
            method=method,
            method_kwargs=default_spatial_kwargs,
            nlt=5,
            step=20,
            dt=2,
            tau=100,
            particle_diameter=particle_diameter,
        ),
        ExperimentConfig(
            method=method,
            method_kwargs=default_spatial_kwargs,
            nlt=2,
            step=20,
            dt=1,
            tau=100,
            particle_diameter=particle_diameter,
        ),
    ]

    temporal_configs.extend(recommended_configs)

    print(f"   🕒 Configs temporelles générées: {len(temporal_configs)}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. COMBINAISON ET DÉDOUBLONNAGE INTELLIGENT
    # ══════════════════════════════════════════════════════════════════════

    all_configs = configs + temporal_configs
    print(
        f"   🔗 Total avant dédoublonnage: {len(all_configs)} ({spatial_count} spatiales + {len(temporal_configs)} temporelles)"
    )

    seen = set()
    unique = []
    duplicates = 0

    for c in all_configs:
        if c.method in ["adaptive", "multizone"]:
            key = f"{c.method}_{c.method_kwargs}_NLT{c.nlt}_step{c.step}_dt{c.dt}_tau{c.tau}_start{c.start_index}"
        else:
            key = c.output_folder()

        if key not in seen:
            seen.add(key)
            unique.append(c)
        else:
            duplicates += 1

    print(
        f"   🔄 Dédoublonnage: {len(all_configs)} → {len(unique)} ({duplicates} doublons supprimés)"
    )
    return unique


def _get_default_kwargs(method: str) -> dict:
    """Paramètres de discrétisation par défaut pour les sweeps temporels."""
    defaults = {
        "cartesian": {"nx": 5, "ny": 5, "nz": 5},
        "cylindrical": {
            "nr": 3,
            "ntheta": 8,
            "nz": 1,
            "radial_mode": "equal_area",
        },
        "voronoi": {"n_cells": 40},
        "quantile": {"nx": 5, "ny": 5, "nz": 5},
        "octree": {"max_particles": 100, "max_depth": 1},
        "physics": {"n_cells": 30, "velocity_weight": 0.5},
        "physics_full_vel": {"n_cells": 30, "velocity_weight": 0.5},
        "spectral": {
            "n_cells": 20,
            "velocity_weight": 0.5,
            "n_neighbors": 15,
            "max_samples": 5000,
        },
        "gmm": {"n_cells": 20, "velocity_weight": 0.5},
        "spectral_biclustering": {
            "n_cells": 30,
            "n_col_clusters": 3,
            "method": "log",
            "velocity_weight": 0.6,
        },
        "adaptive": {
            "y_split": 0.90,
            "y_split_mode": "quantile",
            "n_cells_top": 1,
            "top_method": "single",
            "top_kwargs": {},
            "bottom_method": "voronoi",
            "bottom_kwargs": {
                "n_cells": 100,
            },
        },
        "multizone": {
            "y_mode": "quantile",
            "zones": [
                {
                    "y_min": 0.0,
                    "y_max": 0.75,
                    "method": "cylindrical",
                    "kwargs": {
                        "nr": 2,
                        "ntheta": 2,
                        "nz": 1,
                        "radial_mode": "equal_area",
                    },
                },
                {
                    "y_min": 0.75,
                    "y_max": 1.0,
                    "method": "single",
                    "kwargs": {},
                },
            ],
        },
        "single": {},
    }
    return defaults.get(method, {})


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================


def compute_P_matrix_torch(
    states_prev: np.ndarray | torch.Tensor,
    states_curr: np.ndarray | torch.Tensor,
    n_states: int,
    device: str = "cpu",
    species_labels: np.ndarray | None = None,
) -> torch.Tensor:
    """
    Calcule P_n pour un timestep - version entièrement vectorisée.
    P[j,i] = probabilité de transition de l'état i vers l'état j.
    """
    if isinstance(states_prev, np.ndarray):
        states_prev = torch.from_numpy(states_prev)
    if isinstance(states_curr, np.ndarray):
        states_curr = torch.from_numpy(states_curr)

    s_prev = states_prev.to(device).long()
    s_curr = states_curr.to(device).long()

    n = min(len(s_prev), len(s_curr))
    s_prev = s_prev[:n]
    s_curr = s_curr[:n]

    phi_prev = (s_prev.unsqueeze(1) == torch.arange(n_states, device=device)).float()
    phi_curr = (s_curr.unsqueeze(1) == torch.arange(n_states, device=device)).float()

    transitions = phi_prev.T @ phi_curr
    denominator = phi_prev.sum(dim=0)

    P = transitions.T / denominator
    P[denominator == 0] = 0.0

    return P.to(torch.float64)


def save_results(
    config: ExperimentConfig, partitioner: part.BasePartitioner, results: dict, stats: dict, image_data: dict | None = None, folder_name: str | None = None
) -> None:
    """Sauvegarde les résultats par espèce dans le bucket HuggingFace."""
    if folder_name is None:
        folder_name = config.output_folder()

    partitioner_data = {}
    if hasattr(partitioner, "centroids") and partitioner.centroids is not None:
        partitioner_data["centroids"] = partitioner.centroids
    if hasattr(partitioner, "_r_edges") and partitioner._r_edges is not None:
        partitioner_data["r_edges"] = partitioner._r_edges
    if hasattr(partitioner, "_leaves") and partitioner._leaves:
        partitioner_data["leaves"] = np.array(partitioner._leaves)
    if hasattr(partitioner, "_x_edges") and partitioner._x_edges is not None:
        partitioner_data["x_edges"] = partitioner._x_edges
        partitioner_data["y_edges"] = partitioner._y_edges
        partitioner_data["z_edges"] = partitioner._z_edges
    if isinstance(partitioner, part.PhysicsAwarePartitioner):
        if partitioner._mean is not None:
            partitioner_data["mean"] = partitioner._mean
        if partitioner._std is not None:
            partitioner_data["std"] = partitioner._std
        partitioner_data["physics_params"] = {
            "n_features": partitioner._n_features,
            "velocity_weight": partitioner.velocity_weight,
            "velocity_mode": partitioner.velocity_mode,
        }
    partitioner_data["partitioner_meta"] = {
        "type": type(partitioner).__name__,
        "label": partitioner.label,
        "n_cells": partitioner.n_cells,
    }

    species_data = {}

    # ✅ NOUVEAU : Sauvegarde de la matrice complète des cell_ids
    if "matrix" in results:
        species_data["states_matrix"] = results["matrix"]
        print(f"   📦 states_matrix à sauvegarder : {results['matrix'].shape}")

    # Sauvegarde des matrices par espèce
    for species, data in results.items():
        if species == "matrix":
            continue  # Déjà traité ci-dessus
        species_data[f"transitionmatrix_{species}"] = data["P"]
        species_data[f"S_matrix_{species}"] = data["S_matrix"]
        species_data[f"times_{species}"] = data["times"]

    stats_with_species = {
        **stats,
        "species_list": [k for k in results if k != "matrix"],
    }

    save_experiment_to_bucket(
        folder_name=folder_name,
        species_data=species_data,
        stats=stats_with_species,
        config=asdict(config),
        partitioner_data=partitioner_data,
        image_data=image_data,
        particle_diameter=config.particle_diameter,
    )

    bucket_name = (
        "BIG"
        if config.particle_diameter == 0.008
        else "SMALL"
        if config.particle_diameter == 0.004
        else "Experiments"
    )
    print(
        f"   💾 Bucket: {bucket_name}/{folder_name}/ "
        f"({[k for k in results if k != 'matrix']})"
    )


def sample_coordinates(timestep_dict: dict[int, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne coords, velocities, diameters empilés sur tous les timesteps."""
    all_coords, all_velocities, all_diameters = [], [], []

    for idx in sorted(timestep_dict.keys()):  # Recupère toutes les features
        df = timestep_dict[idx]
        all_coords.append(
            np.column_stack(
                [
                    df["coordinates:0"].to_numpy(),
                    df["coordinates:1"].to_numpy(),
                    df["coordinates:2"].to_numpy(),
                ]
            )
        )
        all_velocities.append(
            np.column_stack(
                [
                    df["Velocity:0"].to_numpy(),
                    df["Velocity:1"].to_numpy(),
                    df["Velocity:2"].to_numpy(),
                ]
            )
        )
        all_diameters.append(df["Diameter"].to_numpy())

    print(f"   📏 Diamètres chargés: {sum(len(d) for d in all_diameters)} particules")
    return (
        np.vstack(all_coords),
        np.vstack(all_velocities),
        np.concatenate(all_diameters),
    )


def _detect_species(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Détecte automatiquement les espèces par diamètre et retourne un masque booléen par espèce."""
    diameters = df["Diameter"].to_numpy()
    unique_diams = np.sort(np.unique(diameters))
    if len(unique_diams) == 1:
        print(
            f"   ⚠️  Un seul diamètre trouvé ({unique_diams[0]}) — pas de séparation d'espèces"
        )
        return {"all": np.ones(len(diameters), dtype=bool)}
    if len(unique_diams) == 2:
        labels = ["small", "large"]
    else:
        labels = [f"d{str(d).replace('.', '')}" for d in unique_diams]
        print(f"   ℹ️  {len(unique_diams)} diamètres détectés : {unique_diams}")
    species_masks = {}
    for label, diam in zip(labels, unique_diams):
        mask = diameters == diam
        species_masks[label] = mask
        print(f"   ✅ Espèce '{label}' (d={diam:.4f}) : {mask.sum()} particules")
    return species_masks


def run_experiment(
    config: ExperimentConfig, partitioner: part.BasePartitioner, timestep_dict: dict[int, pd.DataFrame], device: str = "cpu"
) -> tuple[dict, dict]:
    """Construit une matrice de transition P et des matrices d'états S par espèce."""
    n_states = getattr(partitioner, "n_states", partitioner.n_cells)
    tau = config.tau
    step = config.step
    dt = config.dt
    start_base = config.start_index

    print(f"   📐 Configuration: NLT={config.nlt}, step={step}, dt={dt}, tau={tau}")
    print(
        f"   📦 {len(timestep_dict)} timesteps disponibles "
        f"(index {min(timestep_dict)} → {max(timestep_dict)})"
    )

    try:
        df_init = timestep_dict[start_base]
    except KeyError:
        raise KeyError(f"Timestep start_base={start_base} absent du dict")

    species_masks = _detect_species(df_init)

    sorted_indices = sorted(timestep_dict.keys())
    # ✅ On ne garde que les timesteps >= start_base
    sorted_indices = list(sorted_indices)
    n_timesteps = len(sorted_indices)
    n_particles = len(timestep_dict[sorted_indices[0]])
    idx_to_row = {idx: row for row, idx in enumerate(sorted_indices)}

    print(
        f"   🔧 Calcul des états : {n_timesteps} timesteps × {n_particles} particules "
        f"(à partir de t={start_base})..."
    )

    all_x, all_y, all_z = [], [], []
    all_vx, all_vy, all_vz = [], [], []
    is_physics = isinstance(partitioner, part.PhysicsAwarePartitioner)
    is_bicluster = isinstance(partitioner, part.SpectralBiclusteringPartitioner)
    is_gaussian = isinstance(partitioner, part.GaussianMixturePartitioner)
    is_dbscan = isinstance(partitioner, part.DBSCANPartitioner)
    is_spectral = isinstance(partitioner, part.SpectralClusteringPartitioner)

    for idx in sorted_indices:
        df = timestep_dict[idx]
        all_x.append(df["coordinates:0"].to_numpy())
        all_y.append(df["coordinates:1"].to_numpy())
        all_z.append(df["coordinates:2"].to_numpy())
        # all_x.append(df["Velocity:0"].to_numpy())
        # all_y.append(df["Velocity:1"].to_numpy())
        # all_z.append(df["Velocity:2"].to_numpy())

        if is_physics or is_bicluster or is_gaussian or is_dbscan or is_spectral:
            all_vx.append(df["Velocity:0"].to_numpy())
            all_vy.append(df["Velocity:1"].to_numpy())
            all_vz.append(df["Velocity:2"].to_numpy())

    coords_x = np.concatenate(all_x)
    coords_y = np.concatenate(all_y)
    coords_z = np.concatenate(all_z)

    if is_physics or is_bicluster or is_gaussian or is_dbscan or is_spectral:
        vx_all = np.concatenate(all_vx)
        vy_all = np.concatenate(all_vy)
        vz_all = np.concatenate(all_vz)
        if is_physics or is_gaussian or is_bicluster or is_spectral or is_dbscan:
            partitioner.use_velocity = True
            partitioner.dem_velocities = np.column_stack([vx_all, vy_all, vz_all])
        states_flat = partitioner.compute_states(
            coords_x, coords_y, coords_z, vx_all, vy_all, vz_all
        )
    else:
        states_flat = partitioner.compute_states(coords_x, coords_y, coords_z)

    # ✅ shape correcte : (n_timesteps, n_particles) depuis start_base
    states_matrix = states_flat.reshape(n_timesteps, n_particles)
    print(f"   ✅ states_matrix: {states_matrix.shape}")

    S_matrices = {}
    for species, mask in species_masks.items():
        states_species = states_matrix[:, mask]
        S = np.zeros((n_timesteps, n_states), dtype=np.float64)
        for t in range(n_timesteps):
            S[t] = np.bincount(states_species[t], minlength=n_states)
        S_matrices[species] = S
        print(
            f"   ✅ S_matrix '{species}': {S.shape} | "
            f"sum t=0: {S[0].sum():.0f} particules ({mask.sum()} attendues)"
        )

    all_pairs = []
    for nlt_idx in range(config.nlt):
        current_start_base = start_base + nlt_idx * (step + tau)
        max_end_possible = max(timestep_dict.keys())
        max_start_possible = max_end_possible - tau

        if current_start_base > max_start_possible:
            print(
                f"   ⚠️  Bloc {nlt_idx + 1} ignoré "
                f"(start={current_start_base} > max={max_start_possible})"
            )
            break

        if nlt_idx == config.nlt - 1:
            remaining_range = max_start_possible - current_start_base
            n_apprentissages = min((step + tau) // dt, remaining_range // dt) + 1
        else:
            n_apprentissages = (step + tau) // dt

        for i in range(n_apprentissages):
            start_idx = current_start_base + i * dt
            end_idx = start_idx + tau
            if start_idx not in idx_to_row or end_idx not in idx_to_row:
                break
            all_pairs.append((start_idx, end_idx))

    if not all_pairs:
        raise ValueError("Aucune paire possible avec ces paramètres")

    print(
        f"   📊 {len(all_pairs)} paires | "
        f"data_{all_pairs[0][0]}→{all_pairs[0][1]} … "
        f"data_{all_pairs[-1][0]}→{all_pairs[-1][1]}"
    )

    accumulators = {
        species: {
            "prev": np.empty(0, dtype=np.int64),
            "curr": np.empty(0, dtype=np.int64),
        }
        for species in species_masks
    }

    for idx_prev, idx_curr in tqdm(all_pairs, desc="   Paires", leave=False):
        row_prev = idx_to_row[idx_prev]
        row_curr = idx_to_row[idx_curr]

        for species, mask in species_masks.items():
            accumulators[species]["prev"] = np.concatenate(
                (
                    accumulators[species]["prev"],
                    states_matrix[row_prev][mask],
                )
            )
            accumulators[species]["curr"] = np.concatenate(
                (
                    accumulators[species]["curr"],
                    states_matrix[row_curr][mask],
                )
            )

    results = {}
    results["matrix"] = (
        states_matrix  # shape (n_timesteps, n_particles), depuis start_base
    )

    for species in species_masks:
        print(f"\n   📐 Matrice P — espèce '{species}'...")

        P = (
            compute_P_matrix_torch(
                accumulators[species]["prev"],
                accumulators[species]["curr"],
                n_states,
                device,
                species_labels=None,
            )
            .cpu()
            .numpy()
        )

        n_visited = int((P.sum(axis=0) > 0).sum())
        S_mat = S_matrices[species]

        print(
            f"      {n_states} états | {n_visited} visités | "
            f"P(rester)={np.diag(P).mean():.4f} | "
            f"S0 sum={S_mat[0].sum():.0f}"
        )

        results[species] = {
            "P": P,
            "S_matrix": S_mat,
            "times": np.array(sorted_indices),
        }

    n_paires_par_bloc = (step + tau) // dt
    n_blocs_complets = len(all_pairs) // n_paires_par_bloc
    n_paires_dernier_bloc = len(all_pairs) % n_paires_par_bloc

    first_species = next(iter(species_masks))
    P_ref = results[first_species]["P"]
    col_sums = P_ref.sum(axis=0)
    visited = col_sums > 0

    stats = {
        "n_pairs_used": len(all_pairs),
        "n_nlt_requested": config.nlt,
        "n_blocs_complets": n_blocs_complets,
        "n_paires_dernier_bloc": n_paires_dernier_bloc,
        "n_states": n_states,
        "n_states_visited": int(visited.sum()),
        "n_states_empty": int((~visited).sum()),
        "fraction_visited": round(float(visited.sum()) / n_states, 4),
        "diagonal_mean": float(np.diag(P_ref).mean()),
        "diagonal_std": float(np.diag(P_ref).std()),
        "method": config.method,
        "species": list(species_masks.keys()),
        "n_timesteps": n_timesteps,
        "tau": tau,
        "step": step,
        "dt": dt,
        "raffinage_ratio": n_paires_par_bloc,
        "plage_temporelle": int(all_pairs[-1][1] - all_pairs[0][0]),
        "start_index": config.start_index,
        "first_pair": list(all_pairs[0]),
        "last_pair": list(all_pairs[-1]),
        "particle_diameter": config.particle_diameter,
    }

    return results, stats


# =============================================================================
# run_markov_sweep
# =============================================================================
def run_markov_sweep(
    method: str,
    configs: list[ExperimentConfig] | None = None,
    particle_diameter: float | None = None,
    base_dir: str = BASE_OUTPUT_DIR,
) -> list[dict]:

    print("=" * 70)
    print(f"  SWEEP MARKOVIEN — méthode: {method.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    fs = HfFileSystem()
    print("\n📦 Chargement du fichier Parquet...")
    timestep_dict = load_parquet_as_timestep_dict(HF_FOLDER, fs)

    print("\n🔍 Échantillonnage des coordonnées pour le fit...")
    sample_coords, s_velocities, _all_diameters = sample_coordinates(timestep_dict)
    print(f"   {len(sample_coords)} points échantillonnés")

    if method == "all":
        methods = list(REGISTRY.keys())
    else:
        methods = [method]

    if configs is None:
        all_configs = []
        for m in methods:
            all_configs.extend(get_configs(m, particle_diameter=particle_diameter))
    else:
        all_configs = configs

    print(f"\n📋 {len(all_configs)} expériences à lancer:")
    print("-" * 70)

    results_summary = []
    for i, config in enumerate(all_configs):
        # ✅ MODIFICATION 7 : physics et physics_full_vel ont besoin de sample_coords pour le fit

        try:
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            permanent_start = 250 * 1030
            print("   🔧 Fit partitionneur...")

            # ✅ MODIFICATION 3 : Les deux variantes physics ont besoin des vitesses
            if config.method in (
                "physics",
                "physics_full_vel",
                "spectral_biclustering",
                "dbscan",
                #   "spectral",
            ):
                partitioner.use_velocity = True
                partitioner.dem_velocities = s_velocities[permanent_start:, :]
                partitioner.fit(
                    sample_coords[permanent_start:, :]
                )  # utilise la vitesse grâce à l'attribut use_velocities et dem_velocities
                # diag = partitioner.diagnostics(sample_coords[250:,:])
            elif config.method == "cartesian":
                partitioner.fit(sample_coords)
            else:
                # partitioner.fit(sample_coords[250:,:]) # effectue le fit sur les coordonnées sur la phase stationnaire
                # partitioner.fit(s_velocities[permanent_start:,:]) # effectue le fit sur les coordonnées sur la phase stationnaire
                partitioner.fit(
                    sample_coords[permanent_start:, :]
                )  # effectue le fit sur les coordonnées sur la phase stationnaire
                # diag = partitioner.diagnostics(sample_coords)
                # diag = partitioner.diagnostics(s_velocities)

            if config.method in [
                "adaptive",
                "multizone",
                "physics",
                "physics_full_vel",
                "dbscan",
            ]:
                folder_name = config.output_folder(sample_coords=sample_coords)
            else:
                folder_name = config.output_folder()
            print(f"\n[{i + 1}/{len(all_configs)}] {folder_name}")
            # print(
            #     f"   📊 {partitioner.n_cells} cellules | "
            #     f"{diag['n_visited']} visitées | "
            #     f"pop: [{diag['pop_min']}, {diag['pop_max']}] "
            #     f"μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}"
            # )

            results, stats = run_experiment(config, partitioner, timestep_dict, device)

            save_results(
                config=config,
                partitioner=partitioner,
                results=results,
                stats=stats,
                image_data=None,
                folder_name=folder_name,
            )

            results_summary.append(
                {"config": asdict(config), "stats": stats, "success": True}
            )
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} états | "
                f"P(rester)={stats['diagonal_mean']:.4f} | "
                f"espèces={stats['species']} | "
                f"pairs={stats['n_pairs_used']}"
            )
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results_summary.append(
                {
                    "config": asdict(config),
                    "stats": None,
                    "success": False,
                    "error": str(e),
                }
            )

    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    ok = [r for r in results_summary if r["success"]]
    ko = [r for r in results_summary if not r["success"]]
    print(f"\n✅ Réussies: {len(ok)}/{len(results_summary)}")
    if ko:
        print(f"❌ Échouées: {len(ko)}")
        for r in ko:
            print(f"   - {r['config']['method']}: {r.get('error', '?')}")

    summary_data = {
        "method": method,
        "total": len(results_summary),
        "success": len(ok),
        "failed": len(ko),
        "results": results_summary,
    }
    try:
        save_experiment_to_bucket(
            folder_name=f"_summary_{method}",
            species_data={},
            stats=summary_data,
            config={"type": "summary", "method": method},
        )
        print(f"\n💾 Résumé sauvegardé: _summary_{method}/")
    except Exception as e:
        print(f"\n⚠️  Impossible de sauvegarder le résumé: {e}")

    print("✨ Terminé!")
    return results_summary


# =============================================================================
# PIPELINE INHOMOGÈNE (sibling du pipeline homogène ci-dessus)
# =============================================================================


def run_inhomogeneous_experiment(
    config: ExperimentConfig, partitioner: part.BasePartitioner, timestep_dict: dict[int, pd.DataFrame], device: str = "cpu"
) -> tuple[dict, dict]:
    """
    Construit une matrice de transition P_k par NLT (chaîne inhomogène).

    Sibling de run_experiment() — même signature, même logique de préparation,
    mais retourne une liste de matrices (une par NLT) au lieu d'une seule.

    Returns:
        results: dict avec "matrix" + pour chaque espèce: {"P_blocks", "S_matrix", "times"}
        stats:   dict avec métriques étendues (n_blocks, n_pairs_per_block, inhomogeneous=True)
    """
    n_states = partitioner.n_cells
    tau = config.tau
    step = config.step
    dt = config.dt
    start_base = config.start_index

    print(
        f"   📐 Configuration inhomogène: NLT={config.nlt}, step={step}, dt={dt}, tau={tau}"
    )
    print(
        f"   📦 {len(timestep_dict)} timesteps disponibles "
        f"(index {min(timestep_dict)} → {max(timestep_dict)})"
    )

    try:
        df_init = timestep_dict[start_base]
    except KeyError:
        raise KeyError(f"Timestep start_base={start_base} absent du dict")

    species_masks = _detect_species(df_init)

    sorted_indices = sorted(timestep_dict.keys())
    sorted_indices = list(sorted_indices)
    n_timesteps = len(sorted_indices)
    n_particles = len(timestep_dict[sorted_indices[0]])
    idx_to_row = {idx: row for row, idx in enumerate(sorted_indices)}

    print(
        f"   🔧 Calcul des états : {n_timesteps} timesteps × {n_particles} particules "
        f"(à partir de t={start_base})..."
    )

    all_x, all_y, all_z = [], [], []
    all_vx, all_vy, all_vz = [], [], []
    is_physics = isinstance(partitioner, part.PhysicsAwarePartitioner)
    is_bicluster = isinstance(partitioner, part.SpectralBiclusteringPartitioner)
    is_gaussian = isinstance(partitioner, part.GaussianMixturePartitioner)
    is_dbscan = isinstance(partitioner, part.DBSCANPartitioner)
    is_spectral = isinstance(partitioner, part.SpectralClusteringPartitioner)

    for idx in sorted_indices:
        df = timestep_dict[idx]
        all_x.append(df["coordinates:0"].to_numpy())
        all_y.append(df["coordinates:1"].to_numpy())
        all_z.append(df["coordinates:2"].to_numpy())

        if is_physics or is_bicluster or is_gaussian or is_dbscan or is_spectral:
            all_vx.append(df["Velocity:0"].to_numpy())
            all_vy.append(df["Velocity:1"].to_numpy())
            all_vz.append(df["Velocity:2"].to_numpy())

    coords_x = np.concatenate(all_x)
    coords_y = np.concatenate(all_y)
    coords_z = np.concatenate(all_z)

    if is_physics or is_bicluster or is_gaussian or is_dbscan or is_spectral:
        vx_all = np.concatenate(all_vx)
        vy_all = np.concatenate(all_vy)
        vz_all = np.concatenate(all_vz)
        if is_physics or is_gaussian or is_bicluster or is_spectral or is_dbscan:
            partitioner.use_velocity = True
            partitioner.dem_velocities = np.column_stack([vx_all, vy_all, vz_all])
        states_flat = partitioner.compute_states(
            coords_x, coords_y, coords_z, vx_all, vy_all, vz_all
        )
    else:
        states_flat = partitioner.compute_states(coords_x, coords_y, coords_z)

    # shape correcte : (n_timesteps, n_particles)
    states_matrix = states_flat.reshape(n_timesteps, n_particles)
    print(f"   ✅ states_matrix: {states_matrix.shape}")

    S_matrices = {}
    for species, mask in species_masks.items():
        states_species = states_matrix[:, mask]
        S = np.zeros((n_timesteps, n_states), dtype=np.float64)
        for t in range(n_timesteps):
            S[t] = np.bincount(states_species[t], minlength=n_states)
        S_matrices[species] = S
        print(
            f"   ✅ S_matrix '{species}': {S.shape} | "
            f"sum t=0: {S[0].sum():.0f} particules ({mask.sum()} attendues)"
        )

    # ══════════════════════════════════════════════════════════════════════
    # CONSTRUCTION PAR BLOC NLT — chaque bloc produit sa propre P_k
    # ══════════════════════════════════════════════════════════════════════

    blocks = []  # listes de paires (idx_prev, idx_curr), une par NLT
    for nlt_idx in range(config.nlt):
        block_pairs = []
        """step est la distance entre end et start du prochain"""
        current_start_base = start_base + nlt_idx * (step + tau)
        max_end_possible = max(timestep_dict.keys())
        max_start_possible = max_end_possible - tau

        if current_start_base > max_start_possible:
            print(
                f"   ⚠️  Bloc {nlt_idx + 1} ignoré "
                f"(start={current_start_base} > max={max_start_possible})"
            )
            break

        if nlt_idx == config.nlt - 1:
            remaining_range = max_start_possible - current_start_base
            n_apprentissages = min((tau) // dt, remaining_range // dt) + 1
        else:
            n_apprentissages = (tau) // dt

        for i in range(n_apprentissages):
            start_idx = current_start_base + i * dt
            end_idx = start_idx + tau
            if start_idx not in idx_to_row or end_idx not in idx_to_row:
                break
            block_pairs.append((start_idx, end_idx))

        if block_pairs:
            blocks.append(block_pairs)
            print(
                f"   📦 Bloc {len(blocks)}: {len(block_pairs)} paires "
                f"[{block_pairs[0][0]}→{block_pairs[0][1]} … "
                f"{block_pairs[-1][0]}→{block_pairs[-1][1]}]"
            )

    if not blocks:
        raise ValueError("Aucune paire valide — impossible de construire des blocs")

    # Calcul d'une matrice P_k par bloc et par espèce
    P_blocks_by_species = {species: [] for species in species_masks}

    for block_idx, block_pairs in enumerate(blocks):
        accum = {
            species: {
                "prev": np.empty(0, dtype=np.int64),
                "curr": np.empty(0, dtype=np.int64),
            }
            for species in species_masks
        }

        for idx_prev, idx_curr in block_pairs:
            row_prev = idx_to_row[idx_prev]
            row_curr = idx_to_row[idx_curr]
            for species, mask in species_masks.items():
                accum[species]["prev"] = np.concatenate(
                    [
                        accum[species]["prev"],
                        states_matrix[row_prev][mask],
                    ]
                )
                accum[species]["curr"] = np.concatenate(
                    [
                        accum[species]["curr"],
                        states_matrix[row_curr][mask],
                    ]
                )

        for species in species_masks:
            P_k = (
                compute_P_matrix_torch(
                    accum[species]["prev"],
                    accum[species]["curr"],
                    n_states,
                    device,
                    species_labels=None,
                )
                .cpu()
                .numpy()
            )
            P_blocks_by_species[species].append(P_k)

        n_visited_k = int(
            (
                P_blocks_by_species[next(iter(species_masks))][block_idx].sum(axis=0)
                > 0
            ).sum()
        )
        print(f"      P_{block_idx}: {n_states} états, {n_visited_k} visités")

    # Assemblage du dict results
    results = {"matrix": states_matrix}
    for species in species_masks:
        results[species] = {
            "P_blocks": np.array(
                P_blocks_by_species[species]
            ),  # (n_blocks, n_states, n_states)
            "S_matrix": S_matrices[species],
            "times": np.array(sorted_indices),
        }

    # Stats enrichies
    first_species = next(iter(species_masks))
    n_visited_total = int(
        (results[first_species]["P_blocks"].sum(axis=(0, 1)) > 0).sum()
    )

    stats = {
        "n_blocks": len(blocks),
        "n_pairs_per_block": [len(b) for b in blocks],
        "n_nlt_requested": config.nlt,
        "n_states": n_states,
        "n_states_visited": n_visited_total,
        "fraction_visited": round(float(n_visited_total) / n_states, 4),
        "method": config.method,
        "species": list(species_masks.keys()),
        "n_timesteps": n_timesteps,
        "tau": tau,
        "step": step,
        "dt": dt,
        "start_index": config.start_index,
        "particle_diameter": config.particle_diameter,
        "inhomogeneous": True,
    }

    return results, stats


def save_inhomogeneous_results(
    config: ExperimentConfig, partitioner: part.BasePartitioner, results: dict, stats: dict, image_data: dict | None = None, folder_name: str | None = None
) -> None:
    """
    Sauvegarde les résultats inhomogènes (P_blocks par espèce) dans le bucket.

    Sibling de save_results() — même structure mais sauvegarde P_blocks_{species}.npy
    (array 3D) au lieu de transitionmatrix_{species}.npy, et ajoute
    inhomogeneous_metadata.json.
    """
    if folder_name is None:
        folder_name = config.output_folder()

    # Données du partitionneur (identique à save_results)
    partitioner_data = {}
    if hasattr(partitioner, "centroids") and partitioner.centroids is not None:
        partitioner_data["centroids"] = partitioner.centroids
    if hasattr(partitioner, "_r_edges") and partitioner._r_edges is not None:
        partitioner_data["r_edges"] = partitioner._r_edges
    if hasattr(partitioner, "_leaves") and partitioner._leaves:
        partitioner_data["leaves"] = np.array(partitioner._leaves)
    if hasattr(partitioner, "_x_edges") and partitioner._x_edges is not None:
        partitioner_data["x_edges"] = partitioner._x_edges
        partitioner_data["y_edges"] = partitioner._y_edges
        partitioner_data["z_edges"] = partitioner._z_edges
    if isinstance(partitioner, part.PhysicsAwarePartitioner):
        if partitioner._mean is not None:
            partitioner_data["mean"] = partitioner._mean
        if partitioner._std is not None:
            partitioner_data["std"] = partitioner._std
        partitioner_data["physics_params"] = {
            "n_features": partitioner._n_features,
            "velocity_weight": partitioner.velocity_weight,
            "velocity_mode": partitioner.velocity_mode,
        }
    partitioner_data["partitioner_meta"] = {
        "type": type(partitioner).__name__,
        "label": partitioner.label,
        "n_cells": partitioner.n_cells,
    }

    # Species data : P_blocks au lieu de transitionmatrix
    species_data = {}
    if "matrix" in results:
        species_data["states_matrix"] = results["matrix"]
        print(f"   📦 states_matrix à sauvegarder : {results['matrix'].shape}")

    for species, data in results.items():
        if species == "matrix":
            continue
        species_data[f"P_blocks_{species}"] = data["P_blocks"]  # (n_blocks, S, S)
        species_data[f"S_matrix_{species}"] = data["S_matrix"]
        species_data[f"times_{species}"] = data["times"]

    # Metadata inhomogène
    inhomogeneous_metadata = {
        "n_blocks": stats["n_blocks"],
        "n_pairs_per_block": stats["n_pairs_per_block"],
        "species_list": [k for k in results if k != "matrix"],
        "block_start_indices": [
            config.start_index + i * (config.step + config.tau)
            for i in range(stats["n_blocks"])
        ],
    }

    stats_with_species = {
        **stats,
        "species_list": [k for k in results if k != "matrix"],
    }

    save_experiment_to_bucket(
        folder_name=folder_name,
        species_data=species_data,
        stats=stats_with_species,
        config=asdict(config),
        partitioner_data=partitioner_data,
        image_data=image_data,
        particle_diameter=config.particle_diameter,
        inhomogeneous_metadata=inhomogeneous_metadata,
    )

    bucket_name = (
        "BIG"
        if config.particle_diameter == 0.008
        else "SMALL"
        if config.particle_diameter == 0.004
        else "Experiments"
    )
    print(
        f"   💾 Bucket: {bucket_name}/{folder_name}/ "
        f"(inhomogène, {stats['n_blocks']} blocs, "
        f"espèces={[k for k in results if k != 'matrix']})"
    )


def run_inhomogeneous_markov_sweep(
    method: str,
    configs: list[ExperimentConfig] | None = None,
    particle_diameter: float | None = None,
    base_dir: str = BASE_OUTPUT_DIR,
) -> list[dict]:
    """
    Sweep Markovien inhomogène — sibling de run_markov_sweep().

    Même orchestration que run_markov_sweep() mais utilise les fonctions
    inhomogènes pour l'exécution et la sauvegarde.
    """
    print("=" * 70)
    print(f"  SWEEP INHOMOGÈNE — méthode: {method.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    fs = HfFileSystem()
    print("\n📦 Chargement du fichier Parquet...")
    timestep_dict = load_parquet_as_timestep_dict(HF_FOLDER, fs)

    print("\n🔍 Échantillonnage des coordonnées pour le fit...")
    sample_coords, s_velocities, _all_diameters = sample_coordinates(timestep_dict)
    print(f"   {len(sample_coords)} points échantillonnés")

    if method == "all":
        methods = list(REGISTRY.keys())
    else:
        methods = [method]

    if configs is None:
        all_configs = []
        for m in methods:
            for c in get_configs(m, particle_diameter=particle_diameter):
                c.inhomogeneous = True  # Forcer le mode inhomogène
                all_configs.append(c)
    else:
        all_configs = configs

    print(f"\n📋 {len(all_configs)} expériences inhomogènes à lancer:")
    print("-" * 70)

    results_summary = []
    for i, config in enumerate(all_configs):
        try:
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            permanent_start = 250 * 1030
            print("   🔧 Fit partitionneur...")

            if config.method in (
                "physics",
                "physics_full_vel",
                "spectral_biclustering",
                "dbscan",
            ):
                partitioner.use_velocity = True
                partitioner.dem_velocities = s_velocities[permanent_start:, :]
                partitioner.fit(sample_coords[permanent_start:, :])
            else:
                partitioner.fit(sample_coords[permanent_start:, :])

            if config.method in [
                "adaptive",
                "multizone",
                "physics",
                "physics_full_vel",
                "dbscan",
            ]:
                folder_name = config.output_folder(sample_coords=sample_coords)
            else:
                folder_name = config.output_folder()
            print(f"\n[{i + 1}/{len(all_configs)}] {folder_name}")

            results, stats = run_inhomogeneous_experiment(
                config, partitioner, timestep_dict, device
            )

            save_inhomogeneous_results(
                config=config,
                partitioner=partitioner,
                results=results,
                stats=stats,
                image_data=None,
                folder_name=folder_name,
            )

            results_summary.append(
                {"config": asdict(config), "stats": stats, "success": True}
            )
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} états | "
                f"{stats['n_blocks']} blocs | "
                f"espèces={stats['species']}"
            )
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            import traceback

            traceback.print_exc()
            results_summary.append(
                {
                    "config": asdict(config),
                    "stats": None,
                    "success": False,
                    "error": str(e),
                }
            )

    print("\n" + "=" * 70)
    print("RÉSUMÉ (inhomogène)")
    print("=" * 70)
    ok = [r for r in results_summary if r["success"]]
    ko = [r for r in results_summary if not r["success"]]
    print(f"\n✅ Réussies: {len(ok)}/{len(results_summary)}")
    if ko:
        print(f"❌ Échouées: {len(ko)}")
        for r in ko:
            print(f"   - {r['config']['method']}: {r.get('error', '?')}")

    summary_data = {
        "method": method,
        "total": len(results_summary),
        "success": len(ok),
        "failed": len(ko),
        "results": results_summary,
        "inhomogeneous": True,
    }
    try:
        save_experiment_to_bucket(
            folder_name=f"_summary_inhomogeneous_{method}",
            species_data={},
            stats=summary_data,
            config={"type": "summary", "method": method, "inhomogeneous": True},
        )
        print(f"\n💾 Résumé sauvegardé: _summary_inhomogeneous_{method}/")
    except Exception as e:
        print(f"\n⚠️  Impossible de sauvegarder le résumé: {e}")

    print("✨ Terminé!")
    return results_summary


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Markovien multi-partitionnement (homogène ou inhomogène)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cartesian",
        choices=[*list(REGISTRY.keys()), "all"],
        help="Type de partitionnement (default: cartesian)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=BASE_OUTPUT_DIR,
        help=f"Dossier de sortie (default: {BASE_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        choices=[0.004, 0.008],
        default=None,
        help="Filtrer par diamètre de particule: 0.004 (SMALL), 0.008 (BIG), ou None (tous)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les configurations sans lancer les calculs",
    )
    parser.add_argument(
        "--inhomogeneous",
        action="store_true",
        help="Activer le mode inhomogène (une matrice P par NLT)",
    )
    args = parser.parse_args()

    if args.list:
        if args.method == "all":
            for m in REGISTRY:
                configs = get_configs(m, particle_diameter=args.diameter)
                print(f"\n{m.upper()} ({len(configs)} configs):")
                for c in configs:
                    p = create_partitioner(c.method, **c.method_kwargs)
                    print(
                        f"  {p.label} NLT={c.nlt} step={c.step} dt={c.dt} diameter={c.particle_diameter}"
                    )
        else:
            configs = get_configs(args.method, particle_diameter=args.diameter)
            print(f"{args.method.upper()} ({len(configs)} configs):")
            for c in configs:
                p = create_partitioner(c.method, **c.method_kwargs)
                print(
                    f"  {p.label} NLT={c.nlt} step={c.step} dt={c.dt} diameter={c.particle_diameter}"
                )
        return

    if args.inhomogeneous:
        run_inhomogeneous_markov_sweep(
            args.method, particle_diameter=args.diameter, base_dir=args.output
        )
    else:
        run_markov_sweep(
            args.method, particle_diameter=args.diameter, base_dir=args.output
        )


if __name__ == "__main__":
    main()
