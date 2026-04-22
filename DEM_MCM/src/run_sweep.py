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
import torch
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from huggingface_hub import HfFileSystem

from partitioners import create_partitioner, REGISTRY  
import partitioners as part     # pour le notebook  .ipynb
from bucket_io import save_experiment_to_bucket, BUCKET_BASE
# from .bucket_io import save_experiment_to_bucket, BUCKET_BASE
# from .partitioners import create_partitioner, REGISTRY            # pour le terminal et fichiers .py
# from .bucket_io import save_experiment_to_bucket, BUCKET_BASE



# =============================================================================
# CONFIGURATION GÉNÉRALE
# =============================================================================

# BASE_OUTPUT_DIR = "NewResultsMCM"
# BASE_OUTPUT_DIR = "ResultsDtMCM"
BASE_OUTPUT_DIR = "RaffinageTemporel"
HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
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
    dt: int = None  # Raffinage temporel à l'intérieur de chaque step
    start_index: int = 250

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
    
        return folder_name    # ✅ Retourne juste le nom
# =============================================================================
# CONFIGURATIONS PAR MÉTHODE
# =============================================================================


def get_configs(method):
    """
    Retourne la liste de configs pour une méthode donnée.
    
    Structure :
    1. Configurations spatiales pures (paramètres temporels par défaut)
    2. Configurations temporelles pures (paramètres spatiaux par défaut) 
    3. Pas de dédoublonnage abusif qui supprime des combinaisons légitimes
    """
    configs = []
    
    print(f"   🔍 Génération des configs pour {method}...")

    # ══════════════════════════════════════════════════════════════════════
    # 1. SWEEP DE DISCRÉTISATION SPATIALE (avec paramètres temporels par défaut)
    # ══════════════════════════════════════════════════════════════════════

    if method == "cartesian":
        for n in [2, 3, 4, 5]:
            configs.append(
                ExperimentConfig(
                    method="cartesian",
                    method_kwargs={"nx": n, "ny": n, "nz": n},
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
                )
            )

    elif method == "voronoi":
        # for nc in [8, 10, 12, 14, 16, 18, 20, 24, 27, 30, 64, 100]:
        for nc in [ 200,300]:
            configs.append(
                ExperimentConfig(
                    method="voronoi",
                    method_kwargs={"n_cells": nc},
                )
            )

    elif method == "quantile":
        for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            configs.append(
                ExperimentConfig(
                    method="quantile",
                    method_kwargs={"nx": n, "ny": n, "nz": 1},
                )
            )

    elif method == "octree":
        # max_particles variable
        for mp in [20, 40, 80, 16, 32, 64, 100,200]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": mp, "max_depth": 2},
                )
            )
        # max_depth variable
        # for md in [3, 4, 5, 6, 7]:
        for md in [3, 2,1,4]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": 100, "max_depth": md},
                )
            )

    elif method == "physics":
        # for nc in [2, 4, 8, 16, 32, 64, 100]:
        for nc in [200,300,400,500]:
            configs.append(
                ExperimentConfig(
                    method="physics",
                    method_kwargs={"n_cells": nc},
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
                )
            )

        # ── Sweep ntheta zone basse ──────────────────────────────────
        for nth in [1, 4, 8, 12, 16,20,30,21,23,22,35,37,39,40,50,60,70,80,90,100,120,130,140]:
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
                )
            )

        # ── Voronoï en bas au lieu de cylindrique ────────────────────
        for nc in [10, 20, 30, 50, 64, 100, 125, 250, 500]:
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
            )
        )

    # ── Sweep step (distance entre blocs NLT) ──────────
    for step in [20, 30, 40]:  # Réduit
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                step=step,
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
            )
        )

    # ── Sweep tau (longueur des paires) ──────────
    for tau in [20, 50, 100, 200]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                tau=tau,
            )
        )

    # ✅ Configurations recommandées
    recommended_configs = [
        ExperimentConfig(
            method=method, method_kwargs=default_spatial_kwargs,
            nlt=3, step=100, dt=1, tau=50
        ),
        ExperimentConfig(
            method=method, method_kwargs=default_spatial_kwargs,
            nlt=5, step=20, dt=2, tau=100
        ),
        ExperimentConfig(
            method=method, method_kwargs=default_spatial_kwargs,
            nlt=2, step=20, dt=1, tau=100
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
        "voronoi": {"n_cells": 400},
        "quantile": {"nx": 5, "ny": 5, "nz": 5},
        "octree": {"max_particles": 100, "max_depth": 2},
        "physics": {"n_cells": 125},
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


def sample_coordinates(files, fs, sample_rate=SAMPLE_RATE):
    """
    Échantillonne des coordonnées pour le fit des partitionneurs.

    Returns:
        np.ndarray shape (N, 3)
    """
    all_coords = []
    for f in tqdm(files[5::sample_rate], desc="   Échantillonnage", leave=False):
        with fs.open(f, "rb") as fh:
            df = pl.read_csv(fh)
        coords = np.column_stack(
            [
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ]
        )
        all_coords.append(coords)
    return np.vstack(all_coords)

def sample_velocities(files, fs, sample_rate=SAMPLE_RATE):
    """
    Échantillonne des coordonnées pour le fit des partitionneurs.

    Returns:
        np.ndarray shape (N, 3)
    """
    all_coords = []
    for f in tqdm(files[5::sample_rate], desc="   Échantillonnage", leave=False):
        with fs.open(f, "rb") as fh:
            df = pl.read_csv(fh)
        coords = np.column_stack(
            [
                df["Velocity:0"].to_numpy(),
                df["Velocity:1"].to_numpy(),
                df["Velocity:2"].to_numpy(),
            ]
        )
        all_coords.append(coords)
    return np.vstack(all_coords)


# =============================================================================
# CALCUL MATRICE DE TRANSITION
# =============================================================================




# def phi_particule(state: int, partition: int) -> bool:
#     """Vérifie si une particule est bien dans une partition"""
#     return 1 if state == partition else 0

# def phi_sum_partition(states, partition: int) -> int:
#     """Somme les particules qui sont dans une partition"""
#     phi_s = 0
#     for i in range(len(states)):
#         phi_s += phi_particule(states[i], partition=partition)
#     return phi_s

# def compute_P_matrix_torch(states_prev, states_curr, n_states, device="cpu"):
#     """
#     Calcule P_n pour un timestep en utilisant phi_particule et phi_sum_partition.
#     Normalisation par colonnes (somme des colonnes = 1).
#     """
#     # Conversion en tensor si nécessaire
#     if isinstance(states_curr, np.ndarray):
#         states_curr = torch.from_numpy(states_curr)
#     if isinstance(states_prev, np.ndarray):
#         states_prev = torch.from_numpy(states_prev)
    
#     s_prev = states_prev.to(device).long()
#     s_curr = states_curr.to(device).long()
    
#     # Initialisation de la matrice de transition
#     P = torch.zeros((n_states, n_states), device=device, dtype=torch.float64)
    
#     # Calcul des transitions P[i,j] = probabilité d'aller de i à j
#     for i in range(n_states):
#         for j in range(n_states):
#             # Compte les transitions de i vers j
#             inter = 0
#             n = min(len(s_prev), len(s_curr))
#             for p in range(n):
#                 inter += phi_particule(state=s_prev[p].item(), partition=i) * phi_particule(state=s_curr[p].item(), partition=j)
            
#             # Normalisation par le nombre de particules dans l'état i au temps précédent
#             denominator = phi_sum_partition(s_prev.cpu().numpy(), i)
#             P[i, j] = inter / denominator if denominator > 0 else 0.0
    
#     # Transposition pour avoir les états courants en lignes, précédents en colonnes
#     P = P.T
    
#     # # Normalisation par colonnes (somme des colonnes = 1) avec torch.sum(dim=0)
#     # col_sums = torch.sum(P, dim=0)
    
#     # P = torch.where(col_sums > 0, P / col_sums, torch.zeros_like(P))
    
#     return P

import torch

def compute_P_matrix_torch(states_prev, states_curr, n_states, device="cpu"):
    """
    Calcule P_n pour un timestep - version entièrement vectorisée.
    P[j,i] = probabilité de transition de l'état i vers l'état j
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





# =============================================================================
# EXPÉRIENCE
# =============================================================================
def run_experiment(config, partitioner, files, fs, device):
    """
    Exécute une expérience complète avec raffinage temporel.

    Logique temporelle :
    - NLT blocs principaux séparés de `step`
    - Dans chaque bloc : int(step/dt) apprentissages avec décalage `dt`
    - Chaque apprentissage : paire (start, start + tau)
    - En fin de fichier : int((endfile-end)/dt) au lieu de int(step/dt)

    Exemple: NLT=2, step=100, dt=20, tau=50, start=0

    Bloc 1 (base=0):
        (0,50), (20,70), (40,90), (60,110), (80,130)    # 5 apprentissages
    Bloc 2 (base=100):  
        (100,150), (120,170), (140,190), (160,210), (180,230)
    """
    n_states = partitioner.n_cells
    tau = config.tau
    step = config.step
    dt = config.dt
    start_base = config.start_index
    idx_prev = start_base  # Initialize properly
    idx_curr = start_base+tau

    print(f"   📐 Configuration: NLT={config.nlt}, step={step}, dt={dt}, tau={tau}")
    def load_coords(file_path):
        with fs.open(file_path, "rb") as fh:
            df = pl.read_csv(fh)
        # ✅ Indexation directe plus rapide que select(), et conversion immédiate
        return (
            df["coordinates:0"].to_numpy(),
            df["coordinates:1"].to_numpy(),
            df["coordinates:2"].to_numpy()
        )
    def load_velocities(file_path):
        with fs.open(file_path, "rb") as fh:
            df = pl.read_csv(fh)
        # ✅ Indexation directe plus rapide que select(), et conversion immédiate
        return (
            df["Velocity:0"].to_numpy(),
            df["Velocity:1"].to_numpy(),
            df["Velocity:2"].to_numpy()
        )


    # ── Construire toutes les paires ──
    all_pairs = []
    
    for nlt_idx in range(config.nlt):
        # Start de base pour ce bloc NLT
        current_start_base = start_base + nlt_idx * step
        
        # Calculer combien d'apprentissages dans ce bloc
        if nlt_idx == config.nlt - 1:  # Dernier bloc
            # Vérifier combien on peut faire avant la fin des fichiers
            max_end_possible = len(files) - 1
            max_start_possible = max_end_possible - tau
            
            if current_start_base > max_start_possible:
                # Ce bloc ne peut pas commencer
                print(f"   ⚠️  Bloc {nlt_idx+1} ignoré (start={current_start_base} > max={max_start_possible})")
                break
                
            # Nombre d'apprentissages possibles dans ce dernier bloc
            remaining_range = max_start_possible - current_start_base
            n_apprentissages = min(step // dt, remaining_range // dt) + 1 # important de noter que nous choisissons le minimum entre le nombre cycles d'apprentissages
                    # entre ceux initialement prévus et ceux effectivement disponibles
            
        else:
            # Bloc normal : int(step/dt) apprentissages
            n_apprentissages = step // dt
            
        # Générer les paires pour ce bloc
        for i in range(n_apprentissages):
            start_idx = current_start_base + i * dt
            end_idx = start_idx + tau
            
            if end_idx >= len(files):
                print(f"   ⚠️  Paire ({start_idx},{end_idx}) ignorée (dépasse les fichiers)")
                break
                
            all_pairs.append((start_idx, end_idx))
    
    if not all_pairs:
        raise ValueError("Aucune paire possible avec ces paramètres")
    
    print(f"   📊 {len(all_pairs)} paires générées:")
    print(f"      Premier: files[{all_pairs[0][0]}] → files[{all_pairs[0][1]}]")
    print(f"      Dernier: files[{all_pairs[-1][0]}] → files[{all_pairs[-1][1]}]")
    
    # Analyser la structure
    n_paires_par_step = step // dt
    n_blocs_complets = len(all_pairs) // n_paires_par_step
    n_paires_dernier_bloc = len(all_pairs) % n_paires_par_step
    
    print(f"      Structure: {n_blocs_complets} blocs complets de {n_paires_par_step} paires")
    if n_paires_dernier_bloc > 0:
        print(f"                 1 bloc partiel de {n_paires_dernier_bloc} paires")

    # ── Accumulateur ──
    P_acc = torch.zeros(
        (n_states, n_states), dtype=torch.float64, device=device
    )
    states_prev_acc=np.array([])
    states_curr_acc=np.array([])

    # ── Traitement des paires ──
    for i, (idx_prev, idx_curr) in enumerate(tqdm(all_pairs, desc="   Paires", leave=False)):
        # ✅ Charger les coordonnées pour cette paire
        coords_prev = load_coords(files[idx_prev])
        coords_curr = load_coords(files[idx_curr])
        
        # Lecture des fichiers
        # with fs.open(files[idx_prev], "rb") as f:
        #     df_prev = pl.read_csv(f)
        # with fs.open(files[idx_curr], "rb") as f:
        #     df_curr = pl.read_csv(f)

        # ✅ Conversion immédiate en numpy (un seul appel à to_numpy())
        if isinstance(partitioner, part.PhysicsAwarePartitioner):
            # vel_prev = load_velocities(files[idx_prev])
            # vel_curr = load_velocities(files[idx_curr])

            # states_prev = partitioner.compute_states_with_physics(*coords_prev, *vel_prev)
            # states_curr = partitioner.compute_states_with_physics(*coords_curr, *vel_curr)
            states_prev = partitioner.compute_states(*coords_prev)
            states_curr = partitioner.compute_states(*coords_curr)

            states_prev_acc = np.concatenate((states_prev_acc, np.asarray(states_prev)))
            states_curr_acc = np.concatenate((states_curr_acc, np.asarray(states_curr)))
        else:
            states_prev = partitioner.compute_states(*coords_prev)
            states_curr = partitioner.compute_states(*coords_curr)

            states_prev_acc = np.concatenate((states_prev_acc, np.asarray(states_prev)))
            states_curr_acc = np.concatenate((states_curr_acc, np.asarray(states_curr)))

        # Calcul de la matrice de transition
    P_acc = compute_P_matrix_torch(states_prev_acc, states_curr_acc, n_states, device)

    # ── Moyenne ──
    P = P_acc 
    P_np = P.cpu().numpy()

    # ── Statistiques ──
    column_sums = P_np.sum(axis=0)
    visited = column_sums > 0
    diag = np.diag(P_np)

    stats = {
        "n_pairs_used": len(all_pairs),
        "n_nlt_requested": config.nlt,
        "n_blocs_complets": n_blocs_complets,
        "n_paires_dernier_bloc": n_paires_dernier_bloc,
        "n_states": n_states,
        "n_states_visited": int(visited.sum()),
        "n_states_empty": int((~visited).sum()),
        "fraction_visited": round(float(visited.sum()) / n_states, 4),
        "column_sum_min": float(column_sums[visited].min()) if visited.any() else 0,
        "column_sum_max": float(column_sums[visited].max()) if visited.any() else 0,
        "column_sum_mean": float(column_sums[visited].mean()) if visited.any() else 0,
        "diagonal_mean": float(diag.mean()),
        "diagonal_std": float(diag.std()),
        "method": config.method,
        "tau": tau,
        "step": step,
        "dt": dt,
        "raffinage_ratio": step // dt,
        "plage_temporelle": int(all_pairs[-1][1] - all_pairs[0][0]),
        "start_index": config.start_index,
        "first_pair": list(all_pairs[0]),
        "last_pair": list(all_pairs[-1]),
    }

    return P_np, stats
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
    
    # Métadonnées du partitionneur
    partitioner_data["partitioner_meta"] = {
        "type": type(partitioner).__name__,
        "label": partitioner.label,
        "n_cells": partitioner.n_cells,
    }
    
    # ✅ Sauvegarder directement dans le bucket
    save_experiment_to_bucket(
        folder_name=folder_name,
        matrix=P,
        stats=stats,
        config=asdict(config),
        partitioner_data=partitioner_data,
        image_data=image_data  # ← Changé
    )
    
    print(f"   💾 Bucket: {BUCKET_BASE}/{folder_name}/")
# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def run_markov_sweep(method: str, configs: list[ExperimentConfig] = None, base_dir=BASE_OUTPUT_DIR) -> list[dict]:
    """
    Lance le sweep Markovien pour une méthode de partitionnement.
    """

    print("=" * 70)
    print(f"  SWEEP MARKOVIEN — méthode: {method.upper()}")
    print("=" * 70)
    
    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── Fichiers ──
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))
    print(f"📁 Fichiers disponibles: {len(files)}")

    # ── Coordonnées pour fit ──
    print("\n🔍 Échantillonnage des coordonnées pour le fit...")
    sample_coords = sample_coordinates(files, fs)
    s_velocities=sample_velocities(files,fs)
    print(f"   {len(sample_coords)} points échantillonnés")

    # ── Configs ──
    if method == "all":
        methods = list(REGISTRY.keys())
    else:
        methods = [method]

    if configs is None:
        all_configs = []
        for m in methods:
            all_configs.extend(get_configs(m))
    else:
        all_configs = configs

    print(f"\n📋 {len(all_configs)} expériences à lancer:")
    print("-" * 70)

    # ── Boucle principale (SANS CACHE) ──
    results = []
    for i, config in enumerate(all_configs):
        # ✅ Générer le nom du dossier (pas un chemin)
        if config.method in ["adaptive", "multizone"]:
            folder_name = config.output_folder(base_dir=base_dir, sample_coords=sample_coords)
        else:
            folder_name = config.output_folder(base_dir)
        
        print(f"\n[{i + 1}/{len(all_configs)}] {folder_name}")

        try:
            # ✅ Créer et fitter le partitionneur (pas de cache)
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            print(f"   🔧 Fit partitionneur...")
            if method=='physics':
                # partitioner.fit_with_physics(sample_coords,s_velocities)
                partitioner.fit(sample_coords)
            else:
                partitioner.fit(sample_coords)

            # ✅ Diagnostics (maintenant disponible)
            diag = partitioner.diagnostics(sample_coords)
            print(
                f"   📊 {partitioner.n_cells} cellules | "
                f"{diag['n_visited']} visitées | "
                f"pop: [{diag['pop_min']}, {diag['pop_max']}] "
                f"μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}"
            )
            
            # ✅ Générer les visualisations
            image_data = None
            if hasattr(partitioner, 'visualize'):
                try:
                    x, y, z ,vx,vy,vz= sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2],s_velocities[:,0],s_velocities[:,1],s_velocities[:,2]
                    # Créer un nom de fichier sûr
                    safe_label = partitioner.label.replace('=', '_').replace(' ', '_').replace('/', '_')
                    if method=='physics':
                        # image_data=partitioner.visualize(x,y,z,vx,vy,vz,save_prefix=f"partition_vis_{safe_label}")
                        image_data = partitioner.visualize(x, y, z, save_prefix=f"partition_vis_{safe_label}")
                        print(f"   🎨 {len(image_data)} images générées")
                    else:
                        image_data = partitioner.visualize(x, y, z, save_prefix=f"partition_vis_{safe_label}")
                        print(f"   🎨 {len(image_data)} images générées")
                except Exception as e:
                    print(f"   ⚠️  Visualisation échouée: {e}")

            # Lancer l'expérience
            P, stats = run_experiment(config, partitioner, files, fs, device)

            # Sauvegarder
            save_results(
                config=config, 
                partitioner=partitioner, 
                P=P, 
                stats=stats,
                image_data=image_data,
                folder_name=folder_name
            )

            results.append(
                {"config": asdict(config), "stats": stats, "success": True}
            )
            
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} états | "
                f"P(rester)={stats['diagonal_mean']:.4f} | "
                f"pairs={stats['n_pairs_used']} | "
                f"step={config.step} dt={config.dt}"
            )

        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results.append(
                {
                    "config": asdict(config),
                    "stats": None,
                    "success": False,
                    "error": str(e),
                }
            )

    # ── Résumé ──
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

    # ✅ Sauvegarder le résumé dans le bucket
    summary_data = {
        "method": method,
        "total": len(results),
        "success": len(ok),
        "failed": len(ko),
        "results": results
    }
    
    try:
        save_experiment_to_bucket(
            folder_name=f"_summary_{method}",
            matrix=np.array([]),  # Pas de matrice pour un résumé
            stats=summary_data,
            config={"type": "summary", "method": method}
        )
        print(f"\n💾 Résumé sauvegardé dans le bucket: _summary_{method}/")
    except Exception as e:
        print(f"\n⚠️  Impossible de sauvegarder le résumé: {e}")
    
    print("✨ Terminé!")

    return results

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
        "--list",
        action="store_true",
        help="Lister les configurations sans lancer les calculs",
    )
    args = parser.parse_args()

    if args.list:
        if args.method == "all":
            for m in REGISTRY:
                configs = get_configs(m)
                print(f"\n{m.upper()} ({len(configs)} configs):")
                for c in configs:
                    p = create_partitioner(c.method, **c.method_kwargs)
                    print(f"  {p.label} NLT={c.nlt} step={c.step} dt={c.dt}")
        else:
            configs = get_configs(args.method)
            print(f"{args.method.upper()} ({len(configs)} configs):")
            for c in configs:
                p = create_partitioner(c.method, **c.method_kwargs)
                print(f"  {p.label} NLT={c.nlt} step={c.step} dt={c.dt}")
        return

    run_markov_sweep(args.method, base_dir=args.output)


if __name__ == "__main__":
    main()