"""
===================================================================================
MODÉLISATION MARKOVIENNE GPU - VERSION FACTORISÉE
===================================================================================

Permet de tester automatiquement plusieurs configurations d'hyperparamètres:
- N_LT: nombre de pas de temps pour l'apprentissage
- Discrétisation spatiale: nx, ny, nz
- Pas de temps: step_size (1 = tous les fichiers, 2 = 1 sur 2, etc.)
- Position de départ: start_index (après régime transitoire)

Les résultats sont sauvegardés dans: outputs/NLT_{nlt}_nx{nx}_ny{ny}_nz{nz}_step{step}_start{start}/
===================================================================================
"""

import os
import polars as pl
from huggingface_hub import HfFileSystem
import torch
import numpy as np
from tqdm import tqdm
from itertools import product
from dataclasses import dataclass, asdict
import json

# ===================================================================================
# CONFIGURATION
# ===================================================================================

# Dossier de base pour les résultats
BASE_OUTPUT_DIR = "markov_sweep_results"


@dataclass
class Params:
    """Hyperparamètres pour une expérience."""

    nlt: int = 100  # Nombre de pas de temps pour l'apprentissage
    nx: int = 5  # Discrétisation X
    ny: int = 5  # Discrétisation Y
    nz: int = 5  # Discrétisation Z
    step_size: int = 1  # Pas de temps (1 = tous, 2 = 1 sur 2, etc.)
    start_index: int = 0  # Index de départ (après régime transitoire)

    def output_folder(self, base_dir=BASE_OUTPUT_DIR):
        """Génère le chemin du dossier de sortie."""
        folder_name = f"NLT_{self.nlt}_nx{self.nx}_ny{self.ny}_nz{self.nz}_step{self.step_size}_start{self.start_index}"
        return os.path.join(base_dir, folder_name)


# ===================================================================================
# CONFIGURATIONS À TESTER
# ===================================================================================

# Grille d'hyperparamètres
CONFIGURATIONS = [
    # Format: Params(nlt, nx, ny, nz, step_size, start_index)
    # Discrétisation variable
    Params(nlt=100, nx=2, ny=2, nz=2, step_size=1, start_index=0),
    Params(nlt=100, nx=3, ny=3, nz=3, step_size=1, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=100, nx=6, ny=6, nz=6, step_size=1, start_index=0),
    Params(nlt=100, nx=7, ny=7, nz=7, step_size=1, start_index=0),
    Params(nlt=100, nx=8, ny=8, nz=8, step_size=1, start_index=0),
    Params(nlt=100, nx=9, ny=9, nz=9, step_size=1, start_index=0),
    Params(nlt=100, nx=10, ny=10, nz=10, step_size=1, start_index=0),
    Params(nlt=100, nx=11, ny=11, nz=11, step_size=1, start_index=0),
    Params(nlt=100, nx=12, ny=12, nz=12, step_size=1, start_index=0),
    Params(nlt=100, nx=13, ny=13, nz=13, step_size=1, start_index=0),
    Params(nlt=100, nx=14, ny=14, nz=14, step_size=1, start_index=0),
    Params(nlt=100, nx=15, ny=15, nz=15, step_size=1, start_index=0),
    Params(nlt=100, nx=16, ny=16, nz=16, step_size=1, start_index=0),
    Params(nlt=100, nx=17, ny=17, nz=17, step_size=1, start_index=0),
    Params(nlt=100, nx=18, ny=18, nz=18, step_size=1, start_index=0),
    Params(nlt=100, nx=13, ny=13, nz=13, step_size=1, start_index=0),
    Params(nlt=100, nx=20, ny=20, nz=20, step_size=1, start_index=0),
    # N_LT variable
    Params(nlt=10, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=20, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=30, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=40, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=50, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=60, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=70, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=80, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=90, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=110, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=120, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=130, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=140, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=150, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=160, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=170, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=180, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=190, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=200, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=200, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=400, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=500, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    # Pas de temps variable
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=.1, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=.2, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=.3, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=.7, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=.6, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=2, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=3, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=4, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=5, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=6, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=7, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=8, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=9, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=10, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=11, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=12, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=13, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=14, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=15, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=16, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=17, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=18, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=19, start_index=0),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=20, start_index=0),
    
    # Départ après régime transitoire
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=1000),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=1500),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=2000),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=2500),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=3000),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=3500),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=400),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=4500),
    Params(nlt=100, nx=5, ny=5, nz=5, step_size=1, start_index=5000),
]

# ===================================================================================
# FONCTIONS DE CALCUL
# ===================================================================================


def compute_spatial_limits(files, sample_rate=50):
    """Calcule les limites spatiales du domaine."""
    x_vals, y_vals, z_vals = [], [], []

    for f in files[::sample_rate]:
        with fs.open(f, "rb") as file:
            df = pl.read_csv(file)
        x_vals.extend(df["coordinates:0"].to_list())
        y_vals.extend(df["coordinates:1"].to_list())
        z_vals.extend(df["coordinates:2"].to_list())

    return (
        min(x_vals) - 0.001,
        max(x_vals) + 0.001,
        min(y_vals) - 0.001,
        max(y_vals) + 0.001,
        min(z_vals) - 0.001,
        max(z_vals) + 0.001,
    )


def compute_states(coordinates, xmin, xmax, ymin, ymax, zmin, zmax, nx, ny, nz):
    """Convertit les coordonnées en indices d'état."""
    x, y, z = coordinates

    # Convertir Polars Series en numpy arrays
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    dz = (zmax - zmin) / nz

    ix = np.clip(((x - xmin) / dx).astype(np.int64), 0, nx - 1)
    iy = np.clip(((y - ymin) / dy).astype(np.int64), 0, ny - 1)
    iz = np.clip(((z - zmin) / dz).astype(np.int64), 0, nz - 1)

    return ix + iy * nx + iz * nx * ny


def compute_P_matrix_torch(states_prev, states_curr, n_states, device):
    """
    Calcule la matrice de transition P_n pour un timestep - TOUT GPU.

    Formule: P_ij(n) = T_ij(n) / phi(i, t_{n-1})
    """
    # Move to GPU
    s_prev = states_prev.to(device)
    s_curr = states_curr.to(device)

    # phi(i, t_{n-1}) = nombre de particules par état source
    phi = torch.bincount(s_prev, minlength=n_states).float()

    # Construire T_ij(n) avec scatter_add
    n = min(len(s_prev), len(s_curr))
    indices = s_prev[:n] * n_states + s_curr[:n]
    counts = torch.ones(n, device=device, dtype=torch.float64)

    T = torch.zeros(n_states * n_states, device=device, dtype=torch.float64)
    T.scatter_add_(0, indices, counts)
    T = T.view(n_states, n_states)

    # Normalisation vectorisée
    phi_expanded = phi.unsqueeze(1).expand(n_states, n_states)
    P_n = torch.where(phi_expanded > 0, T / phi_expanded, torch.zeros_like(T))

    return P_n


def run_experiment(params, files, xmin, xmax, ymin, ymax, zmin, zmax, device):
    """
    Execute an experiment to compute a transition matrix from sequential data files.

    This function processes a series of timesteps, reads coordinate data from consecutive
    files, discretizes the coordinates into spatial states, and accumulates transition
    counts to compute an average transition probability matrix.

    Args:
        params: Configuration object containing:
            - nx, ny, nz (int): Grid dimensions for state discretization
            - start_index (int): Starting index for file selection
            - step_size (int): Stride for selecting files
            - nlt (int): Number of timesteps to process
        files (list): List of file paths containing coordinate data
        xmin, xmax (float): X-axis bounds for discretization
        ymin, ymax (float): Y-axis bounds for discretization
        zmin, zmax (float): Z-axis bounds for discretization
        device (torch.device): Computation device (CPU or GPU)

        tuple: A tuple containing:
            - P_np (np.ndarray): Shape (n_states, n_states). Average transition probability matrix
              where each row is normalized by the number of timesteps used
            - stats (dict): Dictionary containing statistics:
                - n_timesteps_used (int): Actual number of timesteps processed
                - n_states (int): Total number of discretized states
                - row_sum_min (float): Minimum row sum among visited states
                - row_sum_max (float): Maximum row sum among visited states
                - diagonal_mean (float): Mean of diagonal elements
                - diagonal_std (float): Standard deviation of diagonal elements
   
    Exécute une expérience avec les paramètres dados.

    Returns:
        dict avec la matrice P et les statistiques
    """
    n_states = params.nx * params.ny * params.nz

    # Sélection des fichiers selon step_size et start_index
    all_indices = list(range(params.start_index, len(files) - 1, params.step_size))
    selected_indices = all_indices[: params.nlt]

    if len(selected_indices) < params.nlt:
        print(
            f"   ⚠️  Seulement {len(selected_indices)} timesteps disponibles (demandé: {params.nlt})"
        )

    # Accumulateur
    P_accumulator = torch.zeros(
        (n_states, n_states), dtype=torch.float64, device=device
    )

    # Boucle principale
    for idx in tqdm(selected_indices, desc=f"   Timesteps", leave=False):
        i = idx
        j = idx + 1

        # Lecture
        with fs.open(files[i], "rb") as f:
            df_prev = pl.read_csv(f)
        with fs.open(files[j], "rb") as f:
            df_curr = pl.read_csv(f)

        # Conversion vers indices
        states_prev_np = compute_states(
            (
                df_prev["coordinates:0"],
                df_prev["coordinates:1"],
                df_prev["coordinates:2"],
            ),
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            params.nx,
            params.ny,
            params.nz,
        )
        states_curr_np = compute_states(
            (
                df_curr["coordinates:0"],
                df_curr["coordinates:1"],
                df_curr["coordinates:2"],
            ),
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
            params.nx,
            params.ny,
            params.nz,
        )

        # Conversion GPU
        states_prev = torch.from_numpy(states_prev_np).to(device)
        states_curr = torch.from_numpy(states_curr_np).to(device)

        # Calcul P_n
        P_n = compute_P_matrix_torch(states_prev, states_curr, n_states, device)
        P_accumulator += P_n

    # Moyenne
    actual_nlt = len(selected_indices)
    P = P_accumulator / actual_nlt

    # Statistiques
    P_np = P.cpu().numpy()
    row_sums = P_np.sum(axis=1)
    visited = row_sums > 0

    stats = {
        "n_timesteps_used": actual_nlt,
        "n_states": n_states,
        "row_sum_min": float(row_sums[visited].min()) if visited.any() else 0,
        "row_sum_max": float(row_sums[visited].max()) if visited.any() else 0,
        "diagonal_mean": float(np.diag(P_np).mean()),
        "diagonal_std": float(np.diag(P_np).std()),
    }

    return P_np, stats


def save_results(params, P, stats, output_dir):
    """Sauvegarde les résultats dans le dossier de sortie."""
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder la matrice
    np.save(os.path.join(output_dir, "transition_matrix.npy"), P)

    # Sauvegarder les statistiques
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Sauvegarder les paramètres
    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(asdict(params), f, indent=2)

    print(f"   💾 Sauvegardé: {output_dir}/")


# ===================================================================================
# MAIN
# ===================================================================================


def main():
    """Point d'entrée principal."""

    print("=" * 70)
    print("MODÉLISATION MARKOVIENNE GPU - MULTI-CONFIGURATIONS")
    print("=" * 70)

    # Créer le dossier de sortie
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"\n📂 Dossier de sortie: {BASE_OUTPUT_DIR}/")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Connexion HuggingFace
    global fs
    fs = HfFileSystem()
    folder_path = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
    files = sorted(fs.glob(f"{folder_path}/*.csv"))
    print(f"📁 Fichiers disponibles: {len(files)}")

    # Calcul des limites (une seule fois)
    print("\n🔍 Calcul des limites spatiales...")
    xmin, xmax, ymin, ymax, zmin, zmax = compute_spatial_limits(files)
    print(f"   X=[{xmin:.4f}, {xmax:.4f}]")
    print(f"   Y=[{ymin:.4f}, {ymax:.4f}]")
    print(f"   Z=[{zmin:.4f}, {zmax:.4f}]")

    # Résumé des expériences
    print(f"\n📋 {len(CONFIGURATIONS)} configurations à tester:")
    print("-" * 70)
    for i, p in enumerate(CONFIGURATIONS):
        print(
            f"   {i + 1:2d}. N_LT={p.nlt:4d}, nx={p.nx}, ny={p.ny}, nz={p.nz}, "
            f"step={p.step_size}, start={p.start_index}"
        )
    print("-" * 70)

    # Exécution des expériences
    print("\n🚀 Lancement des expériences...")
    results = []

    for i, params in enumerate(CONFIGURATIONS):
        print(f"\n[{i + 1}/{len(CONFIGURATIONS)}] {params.output_folder()}")

        try:
            P, stats = run_experiment(
                params, files, xmin, xmax, ymin, ymax, zmin, zmax, device
            )

            save_results(params, P, stats, params.output_folder())

            results.append({"params": asdict(params), "stats": stats, "success": True})

            print(
                f"   ✅ Terminé | timesteps={stats['n_timesteps_used']} | "
                f"P(rester)={stats['diagonal_mean']:.4f}"
            )

        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results.append(
                {
                    "params": asdict(params),
                    "stats": None,
                    "success": False,
                    "error": str(e),
                }
            )

    # Résumé final
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    successful = [r for r in results if r["success"]]
    print(f"\n✅ Réussies: {len(successful)}/{len(results)}")

    if successful:
        print("\nConfigurations testées:")
        for r in successful:
            p = r["params"]
            s = r["stats"]
            print(
                f"   N_LT={p['nlt']:4d}, nx={p['nx']}, "
                f"P(rester)={s['diagonal_mean']:.4f}"
            )

    # Sauvegarder le résumé
    summary_path = os.path.join(BASE_OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Résumé: {summary_path}")

    print("\n✨ Terminé!")


if __name__ == "__main__":
    main()
