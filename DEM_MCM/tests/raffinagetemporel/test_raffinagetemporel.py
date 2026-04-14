import pytest
import numpy as np
import torch 
import polars as pl
from DEM_MCM.src.bucket_io import (
    HfFileSystem,
)
from DEM_MCM.src.partitioners import (
    create_partitioner,
)
from DEM_MCM.src.run_sweep import (
    ExperimentConfig,
    compute_P_matrix_torch,
    sample_coordinates,
    HF_FOLDER,
)


def test_simple_config():
    """Test avec une configuration très simple"""
    config = ExperimentConfig(
        method="single",  # Une seule cellule = plus simple
        method_kwargs={},
        nlt=2,           # Seulement 2 blocs
        step=100,
        dt=50,           # 2 apprentissages par bloc  
        tau=10,          # Paires courtes
        start_index=100
    )
    
    print("=== TEST SIMPLE ===")
    print(f"Config: NLT={config.nlt}, step={config.step}, dt={config.dt}, tau={config.tau}")
    
    # Calculer manuellement les paires attendues
    expected_pairs = [
        # Bloc 0 (start_base=100)
        (100, 110),      # 100 + 0*50, 100 + 0*50 + 10
        (150, 160),      # 100 + 1*50, 100 + 1*50 + 10
        # Bloc 1 (start_base=200)  
        (200, 210),      # 200 + 0*50, 200 + 0*50 + 10
        (250, 260),      # 200 + 1*50, 200 + 1*50 + 10
    ]
    
    print(f"Paires attendues: {expected_pairs}")
    return config

def verify_transition_matrix(config, max_pairs=5):
    """Vérifier quelques matrices manuellement"""
    
    # Créer un partitionneur simple
    partitioner = create_partitioner(config.method, **config.method_kwargs)
    
    # Charger les fichiers
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))
    
    # Échantillonner pour le fit
    sample_coords = sample_coordinates(files, fs, sample_rate=100)
    partitioner.fit(sample_coords)
    
    print(f"\n=== VÉRIFICATION MANUELLE ===")
    print(f"Partitionneur: {partitioner.n_cells} cellules")
    
    # Tester quelques paires manuellement
    test_pairs = [(100, 110), (150, 160), (200, 210)]
    
    for i, (idx_prev, idx_curr) in enumerate(test_pairs[:max_pairs]):
        if idx_curr >= len(files):
            continue
            
        print(f"\n--- Paire {i+1}: files[{idx_prev}] → files[{idx_curr}] ---")
        
        # Charger les données
        with fs.open(files[idx_prev], "rb") as f:
            df_prev = pl.read_csv(f)
        with fs.open(files[idx_curr], "rb") as f:
            df_curr = pl.read_csv(f)
            
        print(f"Particules: {len(df_prev)} → {len(df_curr)}")
        
        # États
        states_prev = partitioner.compute_states(
            df_prev["coordinates:0"], df_prev["coordinates:1"], df_prev["coordinates:2"]
        )
        states_curr = partitioner.compute_states(
            df_curr["coordinates:0"], df_curr["coordinates:1"], df_curr["coordinates:2"]
        )
        
        print(f"États prev: min={states_prev.min()}, max={states_prev.max()}")
        print(f"États curr: min={states_curr.min()}, max={states_curr.max()}")
        
        # Distribution des états
        for state in range(partitioner.n_cells):
            n_prev = (states_prev == state).sum()
            n_curr = (states_curr == state).sum()
            print(f"  État {state}: {n_prev} → {n_curr}")
        
        # Matrice de transition
        P = compute_P_matrix_torch(states_prev, states_curr, partitioner.n_cells, "cpu")
        print(f"Matrice P:\n{P.numpy()}")
        
        # Vérifications
        col_sums = P.sum(dim=0)
        print(f"Sommes colonnes: {col_sums}")
        print(f"Toutes ≈ 1? {torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)}")


def verify_matrix_properties(P_np, tolerance=1e-6):
    """Vérifier les propriétés d'une matrice de transition"""
    
    print(f"\n=== PROPRIÉTÉS MATRICE ===")
    print(f"Shape: {P_np.shape}")
    
    # 1. Sommes des colonnes = 1 (conservation)
    col_sums = P_np.sum(axis=0)
    print(f"Sommes colonnes: min={col_sums.min():.6f}, max={col_sums.max():.6f}")
    
    valid_cols = col_sums > tolerance
    if valid_cols.any():
        col_sums_valid = col_sums[valid_cols]
        close_to_one = np.abs(col_sums_valid - 1.0) < tolerance
        print(f"Colonnes ≈ 1: {close_to_one.sum()}/{len(col_sums_valid)} ✅" if close_to_one.all() 
              else f"❌ Colonnes non-normalisées!")
    
    # 2. Valeurs entre 0 et 1
    print(f"Valeurs: min={P_np.min():.6f}, max={P_np.max():.6f}")
    valid_range = (P_np.min() >= -tolerance) and (P_np.max() <= 1 + tolerance)
    print(f"Dans [0,1]: {'✅' if valid_range else '❌'}")
    
    # 3. États visités
    visited = col_sums > tolerance
    print(f"États visités: {visited.sum()}/{len(visited)}")
    
    # 4. Diagonale (probabilités de rester)
    diag = np.diag(P_np)[visited]
    if len(diag) > 0:
        print(f"P(rester): μ={diag.mean():.4f}, σ={diag.std():.4f}")
    
    return col_sums, visited, diag

def verify_particle_tracking(idx1, idx2, max_particles=100):
    """Vérifier que les mêmes particules sont bien suivies"""
    
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))
    
    print(f"\n=== SUIVI PARTICULES ===")
    print(f"Fichiers: {idx1} → {idx2}")
    
    # Charger les données
    with fs.open(files[idx1], "rb") as f:
        df1 = pl.read_csv(f)
    with fs.open(files[idx2], "rb") as f:
        df2 = pl.read_csv(f)
    
    # Vérifier les IDs (si disponibles)
    if "id" in df1.columns:
        ids1 = set(df1["id"][:max_particles])
        ids2 = set(df2["id"][:max_particles])
        
        common = ids1.intersection(ids2)
        print(f"Particules communes: {len(common)}/{min(len(ids1), len(ids2))}")
        
        if len(common) != min(len(ids1), len(ids2)):
            print("❌ Particules différentes entre timesteps!")
            return False
    
    # Vérifier le nombre total
    print(f"Nombre particules: {len(df1)} → {len(df2)}")
    if len(df1) != len(df2):
        print("❌ Nombre de particules différent!")
        return False
        
    print("✅ Cohérence particules OK")
    return True

def test_regression():
    """Test sur des données où on connaît le résultat attendu"""

    # Créer des données artificielles
    n_particles = 1000
    np.random.seed(42)

    # Particules qui ne bougent jamais (diagonal = 1)
    coords = np.random.random((n_particles, 3)) * 100

    # Partitionneur simple
    partitioner = create_partitioner("single", config={})
    partitioner.fit(coords)
    
    states = partitioner.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])
    
    # Matrice avec mêmes états → diagonale pure
    P = compute_P_matrix_torch(states, states, 1, "cpu")
    
    print(f"\n=== TEST RÉGRESSION ===")
    print(f"Matrice (particules immobiles): {P}")
    
    # Doit être [[1.0]]
    expected = torch.tensor([[1.0]])
    success = torch.allclose(P, expected, atol=1e-6)
    print(f"Diagonal = 1? {'✅' if success else '❌'}")
    
    return success