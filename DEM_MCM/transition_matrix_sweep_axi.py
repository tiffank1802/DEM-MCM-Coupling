"""
===================================================================================
MODÉLISATION MARKOVIENNE GPU - PARTITIONNEMENT CYLINDRIQUE
===================================================================================
Discrétisation en coordonnées cylindriques (r, θ, z) pour mélangeur axisymétrique.

Deux modes de discrétisation radiale:
  - "equal_dr":   Δr constant  → cellules extérieures plus grandes (en volume)
  - "equal_area": aire de section constante → meilleure équirépartition des particules
===================================================================================
"""

import os
import polars as pl
from huggingface_hub import HfFileSystem
import torch
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, asdict, field
import json


BASE_OUTPUT_DIR = "markov_sweep_results_cylindrical"


# ===================================================================================
# CONFIGURATION
# ===================================================================================

@dataclass
class CylindricalParams:
    """Hyperparamètres pour discrétisation cylindrique."""

    nlt: int = 100
    nr: int = 5          # Nombre de couronnes radiales
    ntheta: int = 8      # Nombre de secteurs angulaires (1 = axisymétrique pur)
    nz: int = 5          # Nombre de tranches axiales
    step_size: int = 1
    start_index: int = 0
    radial_mode: str = "equal_area"  # "equal_dr" ou "equal_area"

    @property
    def n_states(self):
        return self.nr * self.ntheta * self.nz

    def output_folder(self, base_dir=BASE_OUTPUT_DIR):
        folder_name = (
            f"NLT_{self.nlt}_nr{self.nr}_nth{self.ntheta}_nz{self.nz}"
            f"_step{self.step_size}_start{self.start_index}_{self.radial_mode}"
        )
        return os.path.join(base_dir, folder_name)


# ===================================================================================
# CONFIGURATIONS À TESTER
# ===================================================================================

CONFIGURATIONS = [
    # --- Axisymétrique pur (ntheta=1): pas de dépendance angulaire ---
    CylindricalParams(nlt=100, nr=3,  ntheta=1,  nz=5,  radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=5,  ntheta=1,  nz=5,  radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=10, ntheta=1,  nz=10, radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=15, ntheta=1,  nz=15, radial_mode="equal_area"),

    # --- Cylindrique complet (r, θ, z) ---
    CylindricalParams(nlt=100, nr=5,  ntheta=4,  nz=5,  radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=5,  ntheta=8,  nz=5,  radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=5,  ntheta=12, nz=5,  radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=5,  ntheta=16, nz=5,  radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=10, ntheta=8,  nz=10, radial_mode="equal_area"),

    # --- Comparaison equal_dr vs equal_area ---
    CylindricalParams(nlt=100, nr=5,  ntheta=8, nz=5, radial_mode="equal_dr"),
    CylindricalParams(nlt=100, nr=5,  ntheta=8, nz=5, radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=10, ntheta=8, nz=10, radial_mode="equal_dr"),
    CylindricalParams(nlt=100, nr=10, ntheta=8, nz=10, radial_mode="equal_area"),

    # --- Raffinement radial ---
    CylindricalParams(nlt=100, nr=3,  ntheta=8, nz=5, radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=5,  ntheta=8, nz=5, radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=8,  ntheta=8, nz=5, radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=10, ntheta=8, nz=5, radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=15, ntheta=8, nz=5, radial_mode="equal_area"),
    CylindricalParams(nlt=100, nr=20, ntheta=8, nz=5, radial_mode="equal_area"),
]


# ===================================================================================
# FONCTIONS GÉOMÉTRIQUES
# ===================================================================================

def compute_cylindrical_limits(files, fs, sample_rate=50):
    """
    Calcule les limites du domaine + centre du cylindre.
    
    Returns:
        dict avec x_center, y_center, r_max, z_min, z_max
    """
    x_vals, y_vals, z_vals = [], [], []

    for f in files[::sample_rate]:
        with fs.open(f, "rb") as file:
            df = pl.read_csv(file)
        x_vals.extend(df["coordinates:0"].to_list())
        y_vals.extend(df["coordinates:1"].to_list())
        z_vals.extend(df["coordinates:2"].to_list())

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)
    z_arr = np.array(z_vals)

    # Centre = barycentre des positions (ou milieu du domaine)
    x_center = (x_arr.min() + x_arr.max()) / 2
    y_center = (y_arr.min() + y_arr.max()) / 2

    # Rayon max depuis le centre
    r_all = np.sqrt((x_arr - x_center)**2 + (y_arr - y_center)**2)
    r_max = r_all.max() + 0.001

    z_min = z_arr.min() - 0.001
    z_max = z_arr.max() + 0.001

    return {
        "x_center": float(x_center),
        "y_center": float(y_center),
        "r_max": float(r_max),
        "z_min": float(z_min),
        "z_max": float(z_max),
    }


def compute_radial_edges(r_max, nr, mode="equal_area"):
    """
    Calcule les bords des couronnes radiales.

    Args:
        r_max: rayon maximum
        nr: nombre de couronnes
        mode: "equal_dr" ou "equal_area"

    Returns:
        np.array de taille (nr+1,) avec les bords [0, r1, r2, ..., r_max]
    """
    if mode == "equal_dr":
        # Δr constant
        return np.linspace(0, r_max, nr + 1)

    elif mode == "equal_area":
        # Aire de chaque couronne = π(r_{i+1}² - r_i²) = constante
        # → r_i = r_max * sqrt(i / nr)
        return r_max * np.sqrt(np.linspace(0, 1, nr + 1))

    else:
        raise ValueError(f"Mode radial inconnu: {mode}")


def compute_states_cylindrical(coordinates, limits, params):
    """
    Convertit les coordonnées (x,y,z) en indices d'état cylindriques.

    State index = ir + itheta * nr + iz * nr * ntheta

    Args:
        coordinates: tuple (x, y, z) - Series Polars ou arrays
        limits: dict avec x_center, y_center, r_max, z_min, z_max
        params: CylindricalParams

    Returns:
        np.array d'indices d'état
    """
    x = np.array(coordinates[0])
    y = np.array(coordinates[1])
    z = np.array(coordinates[2])

    # ── Conversion en coordonnées cylindriques ──
    dx = x - limits["x_center"]
    dy = y - limits["y_center"]
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)          # [-π, π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)  # [0, 2π]

    # ── Discrétisation radiale ──
    r_edges = compute_radial_edges(limits["r_max"], params.nr, params.radial_mode)
    ir = np.searchsorted(r_edges, r, side="right") - 1
    ir = np.clip(ir, 0, params.nr - 1)

    # ── Discrétisation angulaire ──
    dtheta = 2 * np.pi / params.ntheta
    itheta = np.clip((theta / dtheta).astype(np.int64), 0, params.ntheta - 1)

    # ── Discrétisation axiale ──
    dz = (limits["z_max"] - limits["z_min"]) / params.nz
    iz = np.clip(((z - limits["z_min"]) / dz).astype(np.int64), 0, params.nz - 1)

    # ── Index global ──
    states = ir + itheta * params.nr + iz * params.nr * params.ntheta

    return states


# ===================================================================================
# FONCTIONS DE CALCUL (identiques au cartésien)
# ===================================================================================

def compute_P_matrix_torch(states_prev, states_curr, n_states, device):
    """Calcule la matrice de transition pour un timestep."""
    s_prev = states_prev.to(device)
    s_curr = states_curr.to(device)

    phi = torch.bincount(s_prev, minlength=n_states).float()

    n = min(len(s_prev), len(s_curr))
    indices = s_prev[:n] * n_states + s_curr[:n]
    counts = torch.ones(n, device=device, dtype=torch.float64)

    T = torch.zeros(n_states * n_states, device=device, dtype=torch.float64)
    T.scatter_add_(0, indices, counts)
    T = T.view(n_states, n_states)

    phi_expanded = phi.unsqueeze(1).expand(n_states, n_states)
    P_n = torch.where(phi_expanded > 0, T / phi_expanded, torch.zeros_like(T))

    return P_n


def run_experiment(params, files, fs, limits, device):
    """Exécute une expérience cylindrique."""
    n_states = params.n_states

    all_indices = list(range(params.start_index, len(files) - 1, params.step_size))
    selected_indices = all_indices[:params.nlt]

    if len(selected_indices) < params.nlt:
        print(f"   ⚠️  {len(selected_indices)} timesteps disponibles (demandé: {params.nlt})")

    P_accumulator = torch.zeros((n_states, n_states), dtype=torch.float64, device=device)

    for idx in tqdm(selected_indices, desc="   Timesteps", leave=False):
        i, j = idx, idx + 1

        with fs.open(files[i], "rb") as f:
            df_prev = pl.read_csv(f)
        with fs.open(files[j], "rb") as f:
            df_curr = pl.read_csv(f)

        coords_prev = (
            df_prev["coordinates:0"],
            df_prev["coordinates:1"],
            df_prev["coordinates:2"],
        )
        coords_curr = (
            df_curr["coordinates:0"],
            df_curr["coordinates:1"],
            df_curr["coordinates:2"],
        )

        # ← ICI la seule différence: compute_states_cylindrical
        states_prev_np = compute_states_cylindrical(coords_prev, limits, params)
        states_curr_np = compute_states_cylindrical(coords_curr, limits, params)

        states_prev = torch.from_numpy(states_prev_np).to(device)
        states_curr = torch.from_numpy(states_curr_np).to(device)

        P_n = compute_P_matrix_torch(states_prev, states_curr, n_states, device)
        P_accumulator += P_n

    actual_nlt = len(selected_indices)
    P = P_accumulator / actual_nlt

    P_np = P.cpu().numpy()
    row_sums = P_np.sum(axis=1)
    visited = row_sums > 0

    # Compter les états visités
    n_visited = int(visited.sum())

    stats = {
        "n_timesteps_used": actual_nlt,
        "n_states": n_states,
        "n_states_visited": n_visited,
        "n_states_empty": n_states - n_visited,
        "fraction_visited": round(n_visited / n_states, 4),
        "row_sum_min": float(row_sums[visited].min()) if visited.any() else 0,
        "row_sum_max": float(row_sums[visited].max()) if visited.any() else 0,
        "diagonal_mean": float(np.diag(P_np).mean()),
        "diagonal_std": float(np.diag(P_np).std()),
        "radial_mode": params.radial_mode,
    }

    return P_np, stats


def save_results(params, P, stats, limits, output_dir):
    """Sauvegarde les résultats."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "transition_matrix.npy"), P)

    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(output_dir, "params.json"), "w") as f:
        json.dump(asdict(params), f, indent=2)

    # Sauvegarder aussi les limites géométriques
    with open(os.path.join(output_dir, "limits.json"), "w") as f:
        json.dump(limits, f, indent=2)

    # Sauvegarder les bords radiaux pour post-traitement
    r_edges = compute_radial_edges(limits["r_max"], params.nr, params.radial_mode)
    np.save(os.path.join(output_dir, "radial_edges.npy"), r_edges)

    print(f"   💾 Sauvegardé: {output_dir}/")


# ===================================================================================
# MAIN
# ===================================================================================

def main():
    print("=" * 70)
    print("MODÉLISATION MARKOVIENNE GPU - PARTITIONNEMENT CYLINDRIQUE")
    print("=" * 70)

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    fs = HfFileSystem()
    folder_path = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
    files = sorted(fs.glob(f"{folder_path}/*.csv"))
    print(f"📁 Fichiers: {len(files)}")

    # Limites cylindriques (une seule fois)
    print("\n🔍 Calcul des limites cylindriques...")
    limits = compute_cylindrical_limits(files, fs)
    print(f"   Centre:  ({limits['x_center']:.4f}, {limits['y_center']:.4f})")
    print(f"   R_max:   {limits['r_max']:.4f}")
    print(f"   Z:       [{limits['z_min']:.4f}, {limits['z_max']:.4f}]")

    # Résumé
    print(f"\n📋 {len(CONFIGURATIONS)} configurations:")
    print("-" * 70)
    for i, p in enumerate(CONFIGURATIONS):
        print(
            f"   {i+1:2d}. nr={p.nr:2d}, nθ={p.ntheta:2d}, nz={p.nz:2d} "
            f"→ {p.n_states:5d} états  [{p.radial_mode}]"
        )
    print("-" * 70)

    # Exécution
    results = []
    for i, params in enumerate(CONFIGURATIONS):
        print(f"\n[{i+1}/{len(CONFIGURATIONS)}] {params.output_folder()}")

        try:
            P, stats = run_experiment(params, files, fs, limits, device)
            save_results(params, P, stats, limits, params.output_folder())
            results.append({"params": asdict(params), "stats": stats, "success": True})
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} états visités | "
                f"P(rester)={stats['diagonal_mean']:.4f}"
            )
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results.append({"params": asdict(params), "error": str(e), "success": False})

    # Résumé
    with open(os.path.join(BASE_OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✨ Terminé!")


if __name__ == "__main__":
    main()