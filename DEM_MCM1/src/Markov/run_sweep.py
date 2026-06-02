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
    nlt: int = 30
    tau: int = 157  # Écart entre start et end pour chaque paire
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
                        "nr": nr, "ntheta": 3, "nz": 3,
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
        for nc in [ 10,15,20,25,30]:
            configs.append(
                ExperimentConfig(
                    method="voronoi",
                    method_kwargs={"n_cells": nc},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "quantile":
        for n in [2, 3, 4, 5, 6,]:
            configs.append(
                ExperimentConfig(
                    method="quantile",
                    method_kwargs={"nx": n, "ny": n, "nz": 1},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "octree":
        # max_particles variable
        for mp in [20, 40, 80, 16, 32, 64, 28,50,60,70,100]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": mp, "max_depth": 2},
                    particle_diameter=particle_diameter,
                )
            )
        # max_depth variable
        # for md in [3, 4, 5, 6, 7]:
        for md in [ 2,1]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": 100, "max_depth": md},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "physics":
        # Sweep n_cells avec velocity_weight par défaut
        for nc in [ 10,15,20,25,30]:
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
        # On réduit un peu la plage car on a 3 features de vitesse au lieu d'1
        for vw in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="physics_full_vel",
                    method_kwargs={"n_cells": 30, "velocity_weight": vw},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "spectral":
        # Sweep n_cells (basé sur la topologie/connectivité)
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={"n_cells": nc, "velocity_weight": 0.5, "n_neighbors": 15},
                    particle_diameter=particle_diameter,
                )
            )
        # Sweep velocity_weight 
        for vw in [0.1, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={"n_cells": 20, "velocity_weight": vw, "n_neighbors": 15},
                    particle_diameter=particle_diameter,
                )
            )
        # Sweep n_neighbors (crucial pour la construction du graphe d'affinité)
        for k in [5, 10, 15, 20, 30]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={"n_cells": 20, "velocity_weight": 0.5, "n_neighbors": k},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "gmm":
        # Sweep n_cells (cellules ellipsoïdales anisotropes)
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="gmm",
                    method_kwargs={"n_cells": nc, "velocity_weight": 0.5},
                    particle_diameter=particle_diameter,
                )
            )
        # Sweep velocity_weight
        for vw in [0.1, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="gmm",
                    method_kwargs={"n_cells": 20, "velocity_weight": vw},
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
        "physics": {"n_cells": 30, "velocity_weight": 0.5},
        "physics_full_vel": {"n_cells": 30, "velocity_weight": 0.5},
        "spectral": {"n_cells": 20, "velocity_weight": 0.5, "n_neighbors": 15, "max_samples": 5000},
        "gmm": {"n_cells": 20, "velocity_weight": 0.5},
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




def save_results(config, partitioner, results: dict, stats: dict, 
                 image_data=None, folder_name=None):
    """
    Sauvegarde les résultats par espèce dans le bucket HuggingFace.
    
    results : {
        "small": {"P": ndarray, "S_matrix": ndarray, "times": ndarray},
        "large": {"P": ndarray, "S_matrix": ndarray, "times": ndarray},
    }
    """
    if folder_name is None:
        folder_name = config.output_folder()

    # ── Données du partitionneur ─────────────────────────────────────
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
    if isinstance(partitioner, part.PhysicsAwarePartitioner):
        if partitioner._mean is not None:
            partitioner_data["mean"] = partitioner._mean
        if partitioner._std is not None:
            partitioner_data["std"] = partitioner._std
        partitioner_data["physics_params"] = {
            "n_features": partitioner._n_features,
            "velocity_weight": partitioner.velocity_weight,
        }
    partitioner_data["partitioner_meta"] = {
        "type":    type(partitioner).__name__,
        "label":   partitioner.label,
        "n_cells": partitioner.n_cells,
    }

    # ── Construire species_data : un dict de arrays numpy par espèce ─
    # Structure dans le bucket :
    #   folder_name/
    #     transitionmatrix_small.npy
    #     S_matrix_small.npy
    #     times_small.npy          ← même pour toutes les espèces mais sauvegardé une fois
    #     transitionmatrix_large.npy
    #     S_matrix_large.npy
    #     species_list.json         ← ["small", "large"]
    #     stats.json
    #     config.json
    #     partitioner/...
    species_data = {}
    for species, data in results.items():
        species_data[f"transitionmatrix_{species}"] = data["P"]
        species_data[f"S_matrix_{species}"]         = data["S_matrix"]
        species_data[f"times_{species}"]            = data["times"]

    # Ajouter la liste des espèces dans stats pour faciliter le rechargement
    stats_with_species = {
        **stats,
        "species_list": list(results.keys()),
    }

    save_experiment_to_bucket(
        folder_name=folder_name,
        species_data=species_data,     # ← nommé
        stats=stats_with_species,
        config=asdict(config),
        partitioner_data=partitioner_data,
        image_data=image_data,
        particle_diameter=config.particle_diameter,
    )

    bucket_name = (
        "BIG"         if config.particle_diameter == 0.008 else
        "SMALL"       if config.particle_diameter == 0.004 else
        "Experiments"
    )
    print(f"   💾 Bucket: {bucket_name}/{folder_name}/ "
          f"({list(results.keys())})")







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
def _detect_species(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Détecte automatiquement les espèces par diamètre et retourne un masque booléen par espèce.
    Fonctionne pour N espèces différentes (pas seulement 2).
    
    Returns:
        {"small": array([True, False, ...]), "large": array([False, True, ...])}
        ou {"d0004": ..., "d0008": ...} si plus de 2 diamètres
    """
    diameters = df["Diameter"].to_numpy()
    unique_diams = np.sort(np.unique(diameters))
    
    if len(unique_diams) == 1:
        print(f"   ⚠️  Un seul diamètre trouvé ({unique_diams[0]}) — pas de séparation d'espèces")
        return {"all": np.ones(len(diameters), dtype=bool)}
    
    if len(unique_diams) == 2:
        labels = ["small", "large"]
    else:
        # Cas générique : nommer par diamètre
        labels = [f"d{str(d).replace('.', '')}" for d in unique_diams]
        print(f"   ℹ️  {len(unique_diams)} diamètres détectés : {unique_diams}")
    
    species_masks = {}
    for label, diam in zip(labels, unique_diams):
        mask = diameters == diam
        species_masks[label] = mask
        print(f"   ✅ Espèce '{label}' (d={diam:.4f}) : {mask.sum()} particules")
    
    return species_masks


def run_experiment(config, partitioner, timestep_dict: dict[int, pd.DataFrame], device="cpu"):
    """
    Construit une matrice de transition P et des matrices d'états S par espèce.
    Il est crutial de comprendre que les S_matrix_espece ici sont pour toute la DEM et qu'une seule partie sera utilisé (à l'instant initial de construction du modèle de Markov pour la prédictoin)

    
    Pour chaque espèce (small, large) :
      - P_espece        : matrice de transition (n_states × n_states)
      - S_matrix_espece : matrice d'états (n_timesteps × n_states)
                          S[t, k] = nombre de particules de cette espèce dans la cellule k au temps t
    
    Returns:
        results : dict {
            "small": {
                "P":        ndarray (n_states, n_states),
                "S_matrix": ndarray (n_timesteps, n_states),  # tous les timesteps DEM
                "times":    ndarray (n_timesteps,),           # indices DEM correspondants
            },
            "large": { ... },
        }
        stats : dict de métriques
    """
    n_states   = partitioner.n_cells    #nombre de cellules du mélangeur
    tau        = config.tau             # pas de temps de markov
    step       = config.step            # écart entre l'indice de fin d'un timestep et l'indice de début d'un autre
    dt         = config.dt              # temps de raffinage temporel: pas de temps de décalage de l'indice de début sur les (step+tau) pour reapprendre sur ces intervalles
    start_base = config.start_index     # indice de début

    print(f"   📐 Configuration: NLT={config.nlt}, step={step}, dt={dt}, tau={tau}")
    print(f"   📦 {len(timestep_dict)} timesteps disponibles "
          f"(index {min(timestep_dict)} → {max(timestep_dict)})")

    # ── Détection des espèces ────────────────────────────────────────
    try:
        df_init = timestep_dict[start_base] # données de simulation DEM de l'instant initial ()
    except KeyError:
        raise KeyError(f"Timestep start_base={start_base} absent du dict")
    
    species_masks = _detect_species(df_init)        # detection des tailles des particules. est dictionnaire contenant un vecteur de booléen de taille (1,1030) par espèce

    # ── Calcul des états sur tous les timesteps (une seule fois) ─────
    sorted_indices = sorted(timestep_dict.keys())
    n_timesteps    = len(sorted_indices)
    n_particles    = len(timestep_dict[sorted_indices[0]])
    idx_to_row     = {idx: row for row, idx in enumerate(sorted_indices)}

    print(f"   🔧 Calcul des états : {n_timesteps} timesteps × {n_particles} particules...")

    all_x, all_y, all_z = [], [], []
    all_vx, all_vy, all_vz = [], [], []
    is_physics = isinstance(partitioner, part.PhysicsAwarePartitioner)

    for idx in sorted_indices:
        df = timestep_dict[idx]
        all_x.append(df["coordinates:0"].to_numpy())
        all_y.append(df["coordinates:1"].to_numpy())
        all_z.append(df["coordinates:2"].to_numpy())
        if is_physics:
            all_vx.append(df["Velocity:0"].to_numpy())
            all_vy.append(df["Velocity:1"].to_numpy())
            all_vz.append(df["Velocity:2"].to_numpy())
    # la fonction concatenate s'applique sur des listes pour les concatener en un seul np.array
    coords_x = np.concatenate(all_x)
    coords_y = np.concatenate(all_y)
    coords_z = np.concatenate(all_z)

    if is_physics:
        vx_all = np.concatenate(all_vx)
        vy_all = np.concatenate(all_vy)
        vz_all = np.concatenate(all_vz)
        partitioner.dem_velocities = np.column_stack([vx_all, vy_all, vz_all])
        states_flat = partitioner.compute_states(
            coords_x, coords_y, coords_z, vx_all, vy_all, vz_all
        )
    else:
        states_flat = partitioner.compute_states(coords_x, coords_y, coords_z) # états des particules sur toute la simulation dem vecteur de taille (1,6000*1030)

    # (n_timesteps, n_particles) — ligne t = snapshot DEM à l'instant sorted_indices[t]
    states_matrix = states_flat.reshape(n_timesteps, n_particles) # états des particules sur toute la simulation dem mais, matrice de taille (6000,1030)
    print(f"   ✅ states_matrix brute: {states_matrix.shape}")

    # ── S_matrix par espèce : (n_timesteps, n_states) ───────────────
    # S_matrix[t, k] = nombre de particules de l'espèce dans la cellule k au temps t
    S_matrices = {}
    for species, mask in species_masks.items():
        # states_matrix[:, mask] → (n_timesteps, n_particles_species)
        # Autrement dit pour des particules bidisperses comme dans notre cas, les states_species sont de (6000,350) pour les petites particules et
        # (6000,680) pour les grandes particules
        states_species = states_matrix[:, mask]  # seulement les colonnes de cette espèce

        # bincount vectorisé sur chaque ligne
        S = np.zeros((n_timesteps, n_states), dtype=np.float64)
        for t in range(n_timesteps):
            S[t] = np.bincount(states_species[t], minlength=n_states) # compte le nombre de particules dans chacune des partitions au cours du temps pour chacune des espèces

        S_matrices[species] = S # S_matrix est de taille (6000,n_states), dont chaque colonne contient le nombre de particules pour un instant pour une espèce
        print(f"   ✅ S_matrix '{species}': {S.shape} | "
              f"sum t=0: {S[0].sum():.0f} particules ({mask.sum()} attendues)")

    # ── Construction des paires ──────────────────────────────────────
    all_pairs = [] # les pairs sont tout simplement les couples (idx_prev,idx_curr)
    for nlt_idx in range(config.nlt):
        current_start_base = start_base + nlt_idx * (step + tau) # indices de départ pour faire l'apprentissage

        max_end_possible   = max(timestep_dict.keys())
        max_start_possible = max_end_possible - tau

        if current_start_base > max_start_possible:
            print(f"   ⚠️  Bloc {nlt_idx+1} ignoré "
                  f"(start={current_start_base} > max={max_start_possible})")
            break

        if nlt_idx == config.nlt - 1:
            remaining_range  = max_start_possible - current_start_base
            n_apprentissages = min((step + tau) // dt, remaining_range // dt) + 1
        else:
            n_apprentissages = (step + tau) // dt

        for i in range(n_apprentissages):
            start_idx = current_start_base + i * dt
            end_idx   = start_idx + tau
            if start_idx not in idx_to_row or end_idx not in idx_to_row:
                break
            all_pairs.append((start_idx, end_idx))

    if not all_pairs:
        raise ValueError("Aucune paire possible avec ces paramètres")

    print(f"   📊 {len(all_pairs)} paires | "
          f"data_{all_pairs[0][0]}→{all_pairs[0][1]} … "
          f"data_{all_pairs[-1][0]}→{all_pairs[-1][1]}")

    # ── Accumulation des transitions PAR ESPÈCE ─────────────────────
    # Tout comme all_pairs contient les indices (idx_prev,idx_curr), accumulators contient les vecteurs d'états (prev, curr) par espèce
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
            accumulators[species]["prev"] = np.concatenate(( # je ne savais pas qu'on pouvait concatener un np.empty() avec un np.array() sans aucun problème
                accumulators[species]["prev"],
                states_matrix[row_prev][mask],
            ))
            accumulators[species]["curr"] = np.concatenate((
                accumulators[species]["curr"],
                states_matrix[row_curr][mask],
            ))

    # ── Calcul de P par espèce + assemblage final ────────────────────
    results = {}
    for species in species_masks:
        print(f"\n   📐 Matrice P — espèce '{species}'...")

        P = compute_P_matrix_torch(
            accumulators[species]["prev"],
            accumulators[species]["curr"],
            n_states,
            device,
            species_labels=None,
        ).cpu().numpy()

        n_visited = int((P.sum(axis=0) > 0).sum())
        S_mat     = S_matrices[species] # je rappelle que S_matrices est un dictionnaire de l'évolution temporelle du nombre de particules au cours du temps pour une espèce donnée. Il est taille (6000,n_states) par espèce

        print(f"      {n_states} états | {n_visited} visités | "
              f"P(rester)={np.diag(P).mean():.4f} | "
              f"S0 sum={S_mat[0].sum():.0f}")

        results[species] = {
            "P":        P,
            "S_matrix": S_mat,                          # (n_timesteps, n_states)  ← DEM réel
            "times":    np.array(sorted_indices),       # indices DEM (250, 260, ..., 5999)
        }

    # ── Statistiques ────────────────────────────────────────────────
    n_paires_par_bloc     = (step + tau) // dt
    n_blocs_complets      = len(all_pairs) // n_paires_par_bloc
    n_paires_dernier_bloc = len(all_pairs) % n_paires_par_bloc

    first_species = next(iter(results))
    P_ref         = results[first_species]["P"]
    col_sums      = P_ref.sum(axis=0)
    visited       = col_sums > 0

    stats = {
        "n_pairs_used":          len(all_pairs),
        "n_nlt_requested":       config.nlt,
        "n_blocs_complets":      n_blocs_complets,
        "n_paires_dernier_bloc": n_paires_dernier_bloc,
        "n_states":              n_states,
        "n_states_visited":      int(visited.sum()),
        "n_states_empty":        int((~visited).sum()),
        "fraction_visited":      round(float(visited.sum()) / n_states, 4),
        "diagonal_mean":         float(np.diag(P_ref).mean()),
        "diagonal_std":          float(np.diag(P_ref).std()),
        "method":                config.method,
        "species":               list(species_masks.keys()),
        "n_timesteps":           n_timesteps,
        "tau": tau, "step": step, "dt": dt,
        "raffinage_ratio":       n_paires_par_bloc,
        "plage_temporelle":      int(all_pairs[-1][1] - all_pairs[0][0]),
        "start_index":           config.start_index,
        "first_pair":            list(all_pairs[0]),
        "last_pair":             list(all_pairs[-1]),
        "particle_diameter":     config.particle_diameter,
    }

    return results, stats
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
    results_summary = []
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
            results, stats = run_experiment(config, partitioner, timestep_dict, device)

            save_results(
                config=config,
                partitioner=partitioner,
                results=results,        # dict par espèce
                stats=stats,
                image_data=None,
                folder_name=folder_name,
            )

            results_summary.append({"config": asdict(config), "stats": stats, "success": True})
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} états | "
                f"P(rester)={stats['diagonal_mean']:.4f} | "
                f"espèces={stats['species']} | "
                f"pairs={stats['n_pairs_used']}"
            )
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results_summary.append({"config": asdict(config), "stats": None,
                            "success": False, "error": str(e)})

    # ── Résumé ───────────────────────────────────────────────────────
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
        "method": method, "total": len(results_summary),
        "success": len(ok), "failed": len(ko), "results": results_summary,
    }
    try:
        save_experiment_to_bucket(
            folder_name=f"_summary_{method}",
            species_data={},           # pas de matrices pour le résumé
            stats=summary_data,
            config={"type": "summary", "method": method},
        )
        print(f"\n💾 Résumé sauvegardé: _summary_{method}/")
    except Exception as e:
        print(f"\n⚠️  Impossible de sauvegarder le résumé: {e}")

    print("✨ Terminé!")
    return results_summary
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