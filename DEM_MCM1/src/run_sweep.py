"""
===================================================================================
SWEEP MARKOVIEN — Lance les calculs pour un type de partitionnement donné
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

import os
import json
import argparse
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field, asdict
from huggingface_hub import HfFileSystem
from functools import lru_cache

# from partitioners import create_partitioner, REGISTRY  
# from bucket_io import save_experiment_to_bucket, BUCKET_BASE
# import partitioners as part

# Imports relatifs (notebooks) vs absolus (script direct)
try:
    from .import partitioners as part     # pour le notebook  .ipynb
    from .bucket_io import save_experiment_to_bucket, BUCKET_BASE
    from .partitioners import create_partitioner, REGISTRY            # pour le terminal et fichiers .py
    from .utils import apply_species_mask,load_parquet_as_timestep_dict
except ImportError:
    # Imports absolus quand lancé directement comme script
    import partitioners as part
    from bucket_io import save_experiment_to_bucket, BUCKET_BASE
    from partitioners import create_partitioner, REGISTRY
    from utils import apply_species_mask,load_parquet_as_timestep_dict




# =============================================================================
# CONFIGURATION GÉNÉRALE
# =============================================================================

# BASE_OUTPUT_DIR = "NewResultsMCM"
# BASE_OUTPUT_DIR = "ResultsDtMCM"
BASE_OUTPUT_DIR = "RaffinageTemporel"
# HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
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
    nlt: int = 10
    tau: int = 50  # Écart entre start et end pour chaque paire
    step: int = 10  # Distance entre 2 starts principaux (quand NLT > 1)
    dt: int = None  #type: ignore # Raffinage temporel à l'intérieur de chaque step
    start_index: int = 250
    particle_diameter: Optional[float] = None  # ✅ NEW: Diamètre de particule (0.004, 0.008, ou None)

    def __post_init__(self):
        if self.method_kwargs is None:
            self.method_kwargs = {}
        if self.dt is None:
            # Raffinage par défaut : 5 apprentissages par step
            self.dt = max(1, self.step // 5)
            
    def output_folder(self, base_dir=BASE_OUTPUT_DIR, sample_coords=None):
        part = create_partitioner(self.method, **self.method_kwargs)
        if sample_coords is not None:
            part.fit(sample_coords)
            
        # ✅ Retourner seulement le nom du dossier, pas un chemin
        folder_name = (
            f"{part.label}_NLT{self.nlt}_step{self.step}_"
            f"dt{self.dt}_tau{self.tau}_start{self.start_index}"
        )
        
        # ✅ Ajouter le suffixe particle_diameter si spécifié
        if self.particle_diameter is not None:
            diameter_str = str(self.particle_diameter).replace(".", "")
            folder_name += f"_d{diameter_str}"
    
        return folder_name    # ✅ Retourne juste le nom
# =============================================================================
# CONFIGURATIONS PAR MÉTHODE
# =============================================================================


def get_configs(method, particle_diameter=None):
    """
    Retourne la liste de configs pour une méthode donnée.
    
    Structure :
    1. Configurations spatiales pures (paramètres temporels par défaut)
    2. Configurations temporelles pures (paramètres spatiaux par défaut) 
    3. Pas de dédoublonnage abusif qui supprime des combinaisons légitimes
    
    Args:
        method: str - La méthode de partitionnement
        particle_diameter: Optional[float] - Diamètre de particule (0.004, 0.008, ou None)
    """
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
        # nr variable (axisymétrique pur)
        for nr in [3, 4, 5, 6]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": nr, "ntheta": 1, "nz": 1,
                        "radial_mode": "equal_area",
                    },
                    particle_diameter=particle_diameter,
                )
            )
        # ntheta variable
        for nth in [1, 2, 3, 4]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 1, "ntheta": nth, "nz": 1,
                        "radial_mode": "equal_area",
                    },
                    particle_diameter=particle_diameter,
                )
            )
        # nz variable
        for nz in [1, 2]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 2, "ntheta": 2, "nz": nz,
                        "radial_mode": "equal_area",
                    },
                    particle_diameter=particle_diameter,
                )
            )
        # equal_dr vs equal_area
        for mode in ["equal_dr", "equal_area"]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 2, "ntheta": 2, "nz": 1,
                        "radial_mode": mode,
                    },
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "voronoi":
        # for nc in [8, 10, 12, 14, 16, 18, 20, 24, 27, 30, 64, 100]:
        for nc in [ 20,30]:
            configs.append(
                ExperimentConfig(
                    method="voronoi",
                    method_kwargs={"n_cells": nc},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "quantile":
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            configs.append(
                ExperimentConfig(
                    method="quantile",
                    method_kwargs={"nx": n, "ny": n, "nz": 1},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "octree":
        # max_particles variable
        for mp in [20, 40, 80, 16, 32, 64, 28,50]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": mp, "max_depth": 2},
                    particle_diameter=particle_diameter,
                )
            )
        # max_depth variable
        # for md in [3, 4, 5, 6, 7]:
        for md in [3, 2,1]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": 100, "max_depth": md},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "physics":
        # Sweep n_cells avec velocity_weight par défaut
        for nc in [20,30,40,50]:
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

    elif method == "adaptive":
        # ── Sweep y_split (quantile) ─────────────────────────────────
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
                            "nr": 1, "ntheta": 36, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )

        # ── Sweep finesse zone basse (nr) ────────────────────────────
        for nr in [2, 3,1]:
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
                            "nr": nr, "ntheta": 30, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )

        # ── Sweep finesse zone basse (nz) ────────────────────────────
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
                            "nr": 1, "ntheta": 30, "nz": nz,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )

        # ── Sweep ntheta zone basse ──────────────────────────────────
        for nth in [1, 4, 8, 12, 16,20,30,21,23,22,35,37,39,40,50,60,70,80,90,10,12,23,40]:
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
                            "nr": 1, "ntheta": nth, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )

        # ── Zone haute avec quelques cellules ────────────────────────
        for n_top in [1, 2, 3, 4]:
            top_method = "single" if n_top == 1 else "cylindrical"
            top_kwargs = {} if n_top == 1 else {
                "nr": 1, "ntheta": n_top, "nz": 1,
                "radial_mode": "equal_area",
            }
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
                            "nr": 1, "ntheta": 30, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )

        # ── Voronoï en bas au lieu de cylindrique ────────────────────
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
        # Ajoutez ici les configs multizone...
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

    # ✅ Debug : afficher les configs spatiales
    spatial_count = len(configs)
    print(f"   📊 Configs spatiales pour {method}: {spatial_count}")
    if configs:
        print(f"      Exemple: {configs[0].method_kwargs}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. SWEEP TEMPOREL (avec paramètres spatiaux par défaut)
    # ══════════════════════════════════════════════════════════════════════

    default_spatial_kwargs = _get_default_kwargs(method)
    print(f"   🕒 Ajout des sweeps temporels avec: {default_spatial_kwargs}")

    temporal_configs = []

    # ── Sweep NLT ────────────────────────────────────────────────────────
    for nlt in [1, 2, 3, 5]:  # Réduit pour éviter trop de configs
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                nlt=nlt,
                particle_diameter=particle_diameter,
            )
        )

    # ── Sweep step (distance entre blocs NLT) ──────────
    for step in [20, 30, 40]:  # Réduit
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                step=step,
                particle_diameter=particle_diameter,
            )
        )

    # ── Sweep dt (raffinage temporel) ──────────
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

    # ── Sweep tau (longueur des paires) ──────────
    for tau in [20, 50, 100, 200]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                tau=tau,
                particle_diameter=particle_diameter,
            )
        )

    # ✅ Configurations recommandées
    recommended_configs = [
        ExperimentConfig(
            method=method, method_kwargs=default_spatial_kwargs,
            nlt=3, step=100, dt=1, tau=50,
            particle_diameter=particle_diameter,
        ),
        ExperimentConfig(
            method=method, method_kwargs=default_spatial_kwargs,
            nlt=5, step=20, dt=2, tau=100,
            particle_diameter=particle_diameter,
        ),
        ExperimentConfig(
            method=method, method_kwargs=default_spatial_kwargs,
            nlt=2, step=20, dt=1, tau=100,
            particle_diameter=particle_diameter,
        ),
    ]
    
    temporal_configs.extend(recommended_configs)
    
    print(f"   🕒 Configs temporelles générées: {len(temporal_configs)}")
    
    # ══════════════════════════════════════════════════════════════════════
    # 3. COMBINAISON ET DÉDOUBLONNAGE INTELLIGENT
    # ══════════════════════════════════════════════════════════════════════
    
    all_configs = configs + temporal_configs
    print(f"   🔗 Total avant dédoublonnage: {len(all_configs)} ({spatial_count} spatiales + {len(temporal_configs)} temporelles)")
    
    # ✅ Dédoublonnage intelligent basé sur output_folder() qui inclut TOUS les paramètres
    seen = set()
    unique = []
    duplicates = 0
    
    for c in all_configs:
        if c.method in ["adaptive", "multizone"]:
            # Ces méthodes nécessitent sample_coords pour le fit, on ne peut pas générer la clé ici
            # On utilise une clé approximative
            key = f"{c.method}_{c.method_kwargs}_NLT{c.nlt}_step{c.step}_dt{c.dt}_tau{c.tau}_start{c.start_index}"
        else:
            key = c.output_folder()
            
        if key not in seen:
            seen.add(key)
            unique.append(c)
        else:
            duplicates += 1
            print(f"      🔄 Doublon supprimé: {key}")

    print(f"   🔄 Dédoublonnage: {len(all_configs)} → {len(unique)} ({duplicates} doublons supprimés)")
    return unique

def _get_default_kwargs(method):
    """Paramètres de discrétisation par défaut pour les sweeps temporels."""
    defaults = {
        "cartesian": {"nx": 5, "ny": 5, "nz": 5},
        "cylindrical": {
            "nr": 3, "ntheta": 8, "nz": 1,
            "radial_mode": "equal_area",
        },
        "voronoi": {"n_cells": 40},
        "quantile": {"nx": 5, "ny": 5, "nz": 5},
        "octree": {"max_particles": 100, "max_depth": 1},
        "physics": {"n_cells": 30, "velocity_weight": 0.5},
        "adaptive": {
            "y_split": 0.90,
            "y_split_mode": "quantile",
            "n_cells_top": 1,
            "top_method": "single",
            "top_kwargs": {},
            "bottom_method": "voronoi",
            # "bottom_kwargs": {
            #     "nr": 3, "ntheta": 12, "nz": 1,
            #     "radial_mode": "equal_area",
            # },
            "bottom_kwargs": {
                "n_cells": 100,
            },
        },
        "multizone": {
            "y_mode": "quantile",
            "zones": [
                {
                    "y_min": 0.0, "y_max": 0.75,
                    "method": "cylindrical",
                    "kwargs": {
                        "nr": 2, "ntheta": 2, "nz": 1,
                        "radial_mode": "equal_area",
                    },
                },
                {
                    "y_min": 0.75, "y_max": 1.0,
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



# =============================================================================
# CALCUL MATRICE DE TRANSITION
# =============================================================================




import torch

def compute_P_matrix_torch(states_prev, states_curr, n_states, device="cpu", species_labels=None):
    """
    Calcule P_n pour un timestep - version entièrement vectorisée.
    P[j,i] = probabilité de transition de l'état i vers l'état j
    
    Args:
        states_prev: états des particules au temps t (array ou tensor)
        states_curr: états des particules au temps t+dt (array ou tensor)
        n_states: nombre total d'états
        device: "cpu" ou "cuda"
        species_labels: (DÉPRÉCIÉ) masque booléen optionnel - le filtrage doit être 
                        appliqué AVANT d'appeler cette fonction
    """
    # Conversion en tensor si nécessaire
    if isinstance(states_prev, np.ndarray):
        states_prev = torch.from_numpy(states_prev)
    if isinstance(states_curr, np.ndarray):
        states_curr = torch.from_numpy(states_curr)
    
    s_prev = states_prev.to(device).long()
    s_curr = states_curr.to(device).long()
    
    n = min(len(s_prev), len(s_curr))
    s_prev = s_prev[:n]
    s_curr = s_curr[:n]
    
    # Création des masques one-hot pour chaque particule
    # phi_prev[p, i] = 1 si particule p était dans état i
    # phi_curr[p, j] = 1 si particule p est dans état j
    phi_prev = (s_prev.unsqueeze(1) == torch.arange(n_states, device=device)).float()  # (n, n_states)
    phi_curr = (s_curr.unsqueeze(1) == torch.arange(n_states, device=device)).float()  # (n, n_states)
    
    # Matrice de co-occurrence : transitions[i, j] = nombre de transitions i → j
    # Somme sur toutes les particules de phi_prev[:, i] * phi_curr[:, j]
    transitions = phi_prev.T @ phi_curr  # (n_states, n_states)
    
    # Dénominateur : nombre de particules dans chaque état au temps précédent
    denominator = phi_prev.sum(dim=0)  # (n_states,)
    
    # P[i, j] = transitions[i, j] / denominator[i]
    # P = transitions.T / denominator.unsqueeze(1).clamp(min=1e-10)
    P=transitions.T/denominator

    
    # Mettre à zéro les lignes sans particules
    P[denominator == 0] = 0.0
    
    # Transposition pour avoir P[j, i] = prob(i → j)
    # P = P.T
    
    return P.to(torch.float64)




def save_results(config, partitioner, P, stats, image_data=None, folder_name=None):  # ← Changé image_paths en image_data
    """Sauvegarde les résultats dans le bucket HuggingFace."""
    
    # ✅ folder_name est maintenant juste un nom, pas un chemin
    if folder_name is None:
        folder_name = config.output_folder()
    
    # Préparer les données du partitionneur
    partitioner_data = {}
    if hasattr(partitioner, 'centroids') and partitioner.centroids is not None:
        partitioner_data["centroids"] = partitioner.centroids
    if hasattr(partitioner, '_r_edges') and partitioner._r_edges is not None:
        partitioner_data["r_edges"] = partitioner._r_edges
    if hasattr(partitioner, '_leaves') and partitioner._leaves:
        partitioner_data["leaves"] = np.array(partitioner._leaves)
    if hasattr(partitioner, '_x_edges') and partitioner._x_edges is not None:
        partitioner_data["x_edges"] = partitioner._x_edges
        partitioner_data["y_edges"] = partitioner._y_edges
        partitioner_data["z_edges"] = partitioner._z_edges
    
    # Données spécifiques au partitionneur physics-aware
    if isinstance(partitioner, part.PhysicsAwarePartitioner):
        if partitioner._mean is not None:
            partitioner_data["mean"] = partitioner._mean
        if partitioner._std is not None:
            partitioner_data["std"] = partitioner._std
        partitioner_data["physics_params"] = {
            "n_features": partitioner._n_features,
            "velocity_weight": partitioner.velocity_weight,
        }
    
    # Métadonnées du partitionneur
    partitioner_data["partitioner_meta"] = {
        "type": type(partitioner).__name__,
        "label": partitioner.label,
        "n_cells": partitioner.n_cells,
    }
    
    # ✅ Sauvegarder directement dans le bucket avec particle_diameter
    save_experiment_to_bucket(
        folder_name=folder_name,
        matrix=P,
        stats=stats,
        config=asdict(config),
        partitioner_data=partitioner_data,
        image_data=image_data,
        particle_diameter=config.particle_diameter  # ✅ PASS DIAMETER
    )
    
    # Afficher le bucket utilisé (déterminé par particle_diameter)
    bucket_name = "BIG" if config.particle_diameter == 0.008 else \
                  "SMALL" if config.particle_diameter == 0.004 else \
                  "Experiments"
    print(f"   💾 Bucket: {bucket_name}/{folder_name}/")







# ════════════════════════════════════════════════════════════════════
# sample_coordinates — utilise le dict déjà chargé
# ════════════════════════════════════════════════════════════════════
def sample_coordinates(timestep_dict: dict[int, pd.DataFrame]):
    """
    Retourne coords, velocities, diameters empilés sur tous les timesteps.
    """
    all_coords, all_velocities, all_diameters = [], [], []

    for idx in sorted(timestep_dict.keys()):
        df = timestep_dict[idx]
        all_coords.append(np.column_stack([
            df["coordinates:0"].to_numpy(),
            df["coordinates:1"].to_numpy(),
            df["coordinates:2"].to_numpy(),
        ]))
        all_velocities.append(np.column_stack([
            df["Velocity:0"].to_numpy(),
            df["Velocity:1"].to_numpy(),
            df["Velocity:2"].to_numpy(),
        ]))
        all_diameters.append(df["Diameter"].to_numpy())

    print(f"   📏 Diamètres chargés: {sum(len(d) for d in all_diameters)} particules")
    return (
        np.vstack(all_coords),
        np.vstack(all_velocities),
        np.concatenate(all_diameters),
    )


# ════════════════════════════════════════════════════════════════════
# run_experiment — utilise le dict déjà chargé
# ════════════════════════════════════════════════════════════════════
def run_experiment(config, partitioner, timestep_dict: dict[int, pd.DataFrame], device="cpu"):
    """
    Exécute une expérience complète avec raffinage temporel.
    timestep_dict : {idx: DataFrame} issu de load_parquet_as_timestep_dict()
    """
    n_states   = partitioner.n_cells
    tau        = config.tau
    step       = config.step
    dt         = config.dt
    start_base = config.start_index
    n_timesteps = max(timestep_dict.keys()) + 1   # borne supérieure d'index

    print(f"   📐 Configuration: NLT={config.nlt}, step={step}, dt={dt}, tau={tau}")
    print(f"   📦 {len(timestep_dict)} timesteps disponibles "
          f"(index {min(timestep_dict)} → {max(timestep_dict)})")

    # ── Species labels ───────────────────────────────────────────────
    species_labels = None
    try:
        df_init   = timestep_dict[start_base]
        diameters = df_init["Diameter"].to_numpy() if "Diameter" in df_init.columns else None

        if diameters is not None:
            unique_vals = np.unique(diameters)

            if config.particle_diameter is not None:
                # ✅ Filtre explicite depuis la config
                if config.particle_diameter not in unique_vals:
                    raise ValueError(
                        f"Diamètre {config.particle_diameter} absent des données "
                        f"(valeurs trouvées : {unique_vals})"
                    )
                species_labels = (diameters == config.particle_diameter)
                label_str = "SMALL" if config.particle_diameter == unique_vals[0] else "BIG"
                print(f"   ✅ Filtre diamètre {config.particle_diameter} ({label_str}) : "
                    f"{species_labels.sum()} particules retenues")

            elif len(unique_vals) == 2:
                # Comportement legacy : toutes les espèces, pas de filtre
                print(f"   ℹ️  2 diamètres détectés {unique_vals} — aucun filtre appliqué "
                    f"(passez particle_diameter pour filtrer)")
            else:
                print(f"   ⚠️  {len(unique_vals)} diamètres trouvés — masque non appliqué")
        else:
            print("   ⚠️  Colonne 'Diameter' non trouvée — masque non appliqué")

    except KeyError:
        print(f"   ⚠️  Timestep {start_base} absent du dict — masque non appliqué")
    except Exception as e:
        print(f"   ⚠️  Erreur espèces: {e}")

    # ── Accesseurs rapides ───────────────────────────────────────────
    def get_coords(*_):
        # df = timestep_dict[idx]
        df = pd.concat([timestep_dict[i] for i in timestep_dict.keys()],ignore_index=True) # Recupération de toutes les particules pour tous les pas de temps de la DEM
        return (
            df["coordinates:0"].to_numpy(),
            df["coordinates:1"].to_numpy(),
            df["coordinates:2"].to_numpy(),
        )

    def get_velocities(*_):
        # df = timestep_dict[idx]
        df = pd.concat([timestep_dict[i] for i in timestep_dict.keys()],ignore_index=True) # Recupération de toutes les particules pour tous les pas de temps de la DEM
        vx = df["Velocity:0"].to_numpy()
        vy = df["Velocity:1"].to_numpy()
        vz = df["Velocity:2"].to_numpy()
        partitioner.dem_velocities = np.column_stack([vx, vy, vz])
        return vx, vy, vz

    # ── Construction des paires ─────────────────────────────────────
    """
    le but est non plus de partir lire dans un step de la dem pour calculer les états des particules pour la suite regrouper ces états pourr calculer la matrice de transition 
    mais de calculer les états de toutes les particules en une fois et puis par la suite choisir celle des instants considérés pour calculer la matrice de transition.
    
    une fois les états 
    """
    all_pairs = []
    for nlt_idx in range(config.nlt):
        current_start_base = start_base + nlt_idx * (step+tau)# on évite de se répéter lors de l'apprentissage du modèle 

        if nlt_idx == config.nlt - 1:   # dernier bloc
            max_end_possible   = max(timestep_dict.keys())
            max_start_possible = max_end_possible - tau

            if current_start_base > max_start_possible:
                print(f"   ⚠️  Bloc {nlt_idx+1} ignoré "
                      f"(start={current_start_base} > max={max_start_possible})")
                break

            remaining_range  = max_start_possible - current_start_base
            n_apprentissages = min(step // dt, remaining_range // dt) + 1
        else:
            n_apprentissages = step // dt

        for i in range(n_apprentissages):
            start_idx = current_start_base + i * dt
            end_idx   = start_idx + tau

            if end_idx not in timestep_dict:
                print(f"   ⚠️  Paire ({start_idx},{end_idx}) ignorée (absent du dict)")
                break
            if start_idx not in timestep_dict:
                print(f"   ⚠️  Paire ({start_idx},{end_idx}) ignorée (absent du dict)")
                break

            all_pairs.append((start_idx, end_idx))

    if not all_pairs:
        raise ValueError("Aucune paire possible avec ces paramètres")

    print(f"   📊 {len(all_pairs)} paires générées:")
    print(f"      Premier: data_{all_pairs[0][0]} → data_{all_pairs[0][1]}")
    print(f"      Dernier: data_{all_pairs[-1][0]} → data_{all_pairs[-1][1]}")

    n_paires_par_step    = step // dt
    n_blocs_complets     = len(all_pairs) // n_paires_par_step
    n_paires_dernier_bloc = len(all_pairs) % n_paires_par_step
    print(f"      Structure: {n_blocs_complets} blocs complets de {n_paires_par_step} paires"
          + (f" + 1 bloc partiel de {n_paires_dernier_bloc}" if n_paires_dernier_bloc else ""))

    # ── Traitement des paires ───────────────────────────────────────
    states_prev_acc = np.array([])
    states_curr_acc = np.array([])
    coef_idx=1030
    coords=get_coords()
    velocities=get_velocities()
    if isinstance(partitioner, part.PhysicsAwarePartitioner):
        states= partitioner.compute_states(*coords, *velocities)
        # states_curr = partitioner.compute_states(*coords_curr, *velocities(idx_curr))
    else:
        states = partitioner.compute_states(*coords)
        # states_curr = partitioner.compute_states(*coords_curr)
states=np.reshape(states,(6000,1030))
    for idx_prev, idx_curr in tqdm(all_pairs, desc="   Paires", leave=False):
        states_prev = states[idx_prev]#coef_idx*idx_prev est l'indice de début de la sélection de 1030 particules 
        states_curr = states[idx_curr]
        # Calcule des partitions

        if species_labels is not None:
            states_prev = (states[idx_prev])[species_labels]
            states_curr = (states[idx_curr])[species_labels]

        states_prev_acc = np.concatenate((states_prev_acc, np.asarray(states_prev)))
        states_curr_acc = np.concatenate((states_curr_acc, np.asarray(states_curr)))
    # calcule de la matrice de transition
    P_np = compute_P_matrix_torch(
        states_prev_acc, states_curr_acc, n_states, device, species_labels=None
    ).cpu().numpy()

    # ── Statistiques ────────────────────────────────────────────────
    column_sums = P_np.sum(axis=0)
    visited     = column_sums > 0
    diag        = np.diag(P_np)

    stats = {
        "n_pairs_used":          len(all_pairs),
        "n_nlt_requested":       config.nlt,
        "n_blocs_complets":      n_blocs_complets,
        "n_paires_dernier_bloc": n_paires_dernier_bloc,
        "n_states":              n_states,
        "n_states_visited":      int(visited.sum()),
        "n_states_empty":        int((~visited).sum()),
        "fraction_visited":      round(float(visited.sum()) / n_states, 4),
        "column_sum_min":        float(column_sums[visited].min()) if visited.any() else 0,
        "column_sum_max":        float(column_sums[visited].max()) if visited.any() else 0,
        "column_sum_mean":       float(column_sums[visited].mean()) if visited.any() else 0,
        "diagonal_mean":         float(diag.mean()),
        "diagonal_std":          float(diag.std()),
        "method":                config.method,
        "tau": tau, "step": step, "dt": dt,
        "raffinage_ratio":       step // dt,
        "plage_temporelle":      int(all_pairs[-1][1] - all_pairs[0][0]),
        "start_index":           config.start_index,
        "first_pair":            list(all_pairs[0]),
        "last_pair":             list(all_pairs[-1]),
    }
    return P_np, stats

def run_exp(config,partitioner,timestep_dict: dict[int,pd.DataFrame], device="cpu "):
    df=pd.concat([timestep_dict[i] for i in timestep_dict.keys()],ignore_index=True) # Recupération de toutes les particules pour tous les pas de temps de la DEM
    # datas=pd.DataFrame(timestep_dict)# Du fait que le timestep_dict est un dictionnaire, peut être convertissable en DataFrame facilement
    coords=np.array(
            df["coordinates:0"].to_numpy(),
            df["coordinates:1"].to_numpy(),
            df["coordinates:2"].to_numpy(),
    )
    
    # states=partitioner.fit(datas.drop(columns=[]))

# ════════════════════════════════════════════════════════════════════
# run_markov_sweep
# ════════════════════════════════════════════════════════════════════
def run_markov_sweep(method: str, configs: list[ExperimentConfig] = None,
                     particle_diameter: float = None, base_dir=BASE_OUTPUT_DIR) -> list[dict]:

    print("=" * 70)
    print(f"  SWEEP MARKOVIEN — méthode: {method.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── Chargement unique du parquet ─────────────────────────────────
    fs = HfFileSystem()
    print("\n📦 Chargement du fichier Parquet...")
    timestep_dict = load_parquet_as_timestep_dict(HF_FOLDER, fs)

    # ── Coordonnées pour fit ─────────────────────────────────────────
    print("\n🔍 Échantillonnage des coordonnées pour le fit...")
    sample_coords, s_velocities, all_diameters = sample_coordinates(timestep_dict)
    print(f"   {len(sample_coords)} points échantillonnés")

    # ── Configs ──────────────────────────────────────────────────────
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

    # ── Boucle principale ────────────────────────────────────────────
    results = []
    for i, config in enumerate(all_configs):
        if config.method in ["adaptive", "multizone", "physics"]:
            folder_name = config.output_folder(base_dir=base_dir, sample_coords=sample_coords)
        else:
            folder_name = config.output_folder(base_dir)

        print(f"\n[{i+1}/{len(all_configs)}] {folder_name}")

        try:
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            print("   🔧 Fit partitionneur...")
            if config.method == "physics":
                partitioner.fit(sample_coords, use_velocities=True)
                diag = partitioner.diagnostics(s_velocities)
            else:
                partitioner.fit(sample_coords)
                diag = partitioner.diagnostics(sample_coords)

            print(
                f"   📊 {partitioner.n_cells} cellules | "
                f"{diag['n_visited']} visitées | "
                f"pop: [{diag['pop_min']}, {diag['pop_max']}] "
                f"μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}"
            )

            # ── run_experiment reçoit le dict, plus fs ni parquet_path ──
            P, stats = run_experiment(config, partitioner, timestep_dict, device)

            save_results(
                config=config, partitioner=partitioner, P=P,
                stats=stats, image_data=None, folder_name=folder_name
            )

            results.append({"config": asdict(config), "stats": stats, "success": True})
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} états | "
                f"P(rester)={stats['diagonal_mean']:.4f} | "
                f"pairs={stats['n_pairs_used']} | "
                f"step={config.step} dt={config.dt}"
            )

        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results.append({"config": asdict(config), "stats": None,
                            "success": False, "error": str(e)})

    # ── Résumé ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    ok = [r for r in results if r["success"]]
    ko = [r for r in results if not r["success"]]
    print(f"\n✅ Réussies: {len(ok)}/{len(results)}")
    if ko:
        print(f"❌ Échouées: {len(ko)}")
        for r in ko:
            print(f"   - {r['config']['method']}: {r.get('error', '?')}")

    summary_data = {
        "method": method, "total": len(results),
        "success": len(ok), "failed": len(ko), "results": results,
    }
    try:
        save_experiment_to_bucket(
            folder_name=f"_summary_{method}",
            matrix=np.array([]),
            stats=summary_data,
            config={"type": "summary", "method": method},
        )
        print(f"\n💾 Résumé sauvegardé: _summary_{method}/")
    except Exception as e:
        print(f"\n⚠️  Impossible de sauvegarder le résumé: {e}")

    print("✨ Terminé!")
    return results
# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Markovien multi-partitionnement"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cartesian",
        choices=list(REGISTRY.keys()) + ["all"],  # ← inclut automatiquement adaptive, multizone, single
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
    args = parser.parse_args()

    if args.list:
        if args.method == "all":
            for m in REGISTRY:
                configs = get_configs(m, particle_diameter=args.diameter)
                print(f"\n{m.upper()} ({len(configs)} configs):")
                for c in configs:
                    p = create_partitioner(c.method, **c.method_kwargs)
                    print(f"  {p.label} NLT={c.nlt} step={c.step} dt={c.dt} diameter={c.particle_diameter}")
        else:
            configs = get_configs(args.method, particle_diameter=args.diameter)
            print(f"{args.method.upper()} ({len(configs)} configs):")
            for c in configs:
                p = create_partitioner(c.method, **c.method_kwargs)
                print(f"  {p.label} NLT={c.nlt} step={c.step} dt={c.dt} diameter={c.particle_diameter}")
        return

    run_markov_sweep(args.method, particle_diameter=args.diameter, base_dir=args.output)


if __name__ == "__main__":
    main()