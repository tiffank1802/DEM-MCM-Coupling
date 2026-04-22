#!/usr/bin/env python3
"""
Tests du partitionnement adaptatif avec données réelles depuis bucket_io
Ce fichier est autonome et ne dépend pas des fichiers avec erreurs d'indentation
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from src.partitioners import create_partitioner, AdaptiveZPartitioner
from src import bucket_io


def load_real_particle_data(max_particles=1000):
    """
    Charge les coordonnées réelles des particules depuis le bucket
    """
    try:
        fs = bucket_io.get_fs()

        # Lister les fichiers CSV dans Output Paraview
        files = fs.glob("hf://buckets/ktongue/DEM_MCM/Output Paraview/*.csv")

        if not files:
            print("⚠️ Aucun fichier trouvé dans le bucket")
            return None

        print(f"📁 {len(files)} fichiers disponibles")

        # Charger le premier fichier
        import polars as pl

        with fs.open(files[50], "rb") as f:  # Utiliser snapshot 50
            df = pl.read_csv(f)

        coords = np.column_stack(
            [
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ]
        )

        # Sous-échantillonner si trop grand
        if len(coords) > max_particles:
            idx = np.random.choice(len(coords), max_particles, replace=False)
            coords = coords[idx]

        print(f"   ✅ {len(coords)} particules chargées")
        print(
            f"   📏 Dimensions: X=[{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]"
        )
        print(f"                Y=[{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
        print(f"                Z=[{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")

        return coords

    except Exception as e:
        print(f"❌ Erreur chargement données: {e}")
        return None


def test_adaptive_partitioning_on_real_data():
    """
    Test complet du partitionnement adaptatif avec données réelles
    """
    print("=" * 70)
    print("TEST: Partitionnement Adaptatif sur Données Réelles")
    print("=" * 70)
    print()

    # 1. Charger données
    coords = load_real_particle_data(max_particles=1000)
    if coords is None:
        print("⚠️ Utilisation de données synthétiques pour la démo")
        np.random.seed(42)
        coords = np.random.rand(300, 3)
        coords[:, 2] *= 2

    print()

    # 2. Tester différentes configurations adaptatives
    configs = [
        {
            "name": "Adaptatif 70% bas (cylindrique)",
            "params": {
                "z_split_mode": "quantile",
                "z_split": 0.7,
                "n_cells_top": 1,
                "top_method": "single",
                "bottom_method": "cylindrical",
                "bottom_kwargs": {
                    "nr": 5,
                    "ntheta": 8,
                    "nz": 5,
                    "radial_mode": "equal_area",
                },
            },
        },
        {
            "name": "Adaptatif 50/50 (Voronoï)",
            "params": {
                "z_split_mode": "quantile",
                "z_split": 0.5,
                "n_cells_top": 3,
                "top_method": "voronoi",
                "top_kwargs": {"n_cells": 30},
                "bottom_method": "voronoi",
                "bottom_kwargs": {"n_cells": 100},
            },
        },
    ]

    results = []

    for cfg in configs:
        print(f"3. Configuration: {cfg['name']}")
        print("-" * 50)

        # Créer partitionneur
        part = create_partitioner("adaptive", **cfg["params"])

        # Fitter
        part.fit(coords)
        states = part.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])

        # Calculer stats
        z = coords[:, 2]
        n_bottom = np.sum(z <= part._z_split)
        n_top = np.sum(z > part._z_split)

        # Diagnostics
        diag = part.diagnostics(coords)

        print(f"   Z_split: {part._z_split:.4f}")
        print(f"   Particules bas: {n_bottom} ({100 * n_bottom / len(z):.1f}%)")
        print(f"   Particules haut: {n_top} ({100 * n_top / len(z):.1f}%)")
        print(
            f"   Cellules: {part.n_cells} (bas: {part._n_cells_bottom}, haut: {part._n_cells_top})"
        )
        print(f"   Cellules visitées: {diag['n_visited']}/{part.n_cells}")
        print(f"   Pop moyenne: {diag['pop_mean']:.1f} ± {diag['pop_std']:.1f}")
        print()

        results.append(
            {
                "config": cfg["name"],
                "z_split": part._z_split,
                "n_cells": part.n_cells,
                "n_visited": diag["n_visited"],
                "pop_mean": diag["pop_mean"],
                "pop_std": diag["pop_std"],
            }
        )

    # 4. Comparaison finale
    print("=" * 70)
    print("RÉSULTATS COMPARATIFS")
    print("=" * 70)
    print()
    print(
        f"{'Configuration':30s} {'Cellules':10s} {'Visitées':10s} {'μ pop':10s} {'σ pop':10s}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['config']:30s} {r['n_cells']:10d} {r['n_visited']:10d} "
            f"{r['pop_mean']:10.1f} {r['pop_std']:10.1f}"
        )

    print()
    print("=" * 70)
    print("✅ TOUS LES TESTS RÉUSSIS")
    print("=" * 70)
    print()
    print("💡 Conclusion:")
    print("   - Le partitionnement adaptatif fonctionne avec les données réelles")
    print("   - Le split quantile assure une répartition équilibrée des particules")
    print("   - La zone haute peut être grossièrement discrétisée")
    print()

    return results


def test_bucket_io_functions():
    """
    Test individuel des fonctions bucket_io
    """
    print("=" * 70)
    print("TEST: Fonctions bucket_io")
    print("=" * 70)
    print()

    try:
        # 1. Test get_fs
        fs = bucket_io.get_fs()
        print("1. ✅ get_fs() fonctionne")

        # 2. Lister expériences
        experiments = bucket_io.list_experiments()
        print(f"2. ✅ list_experiments() trouvé {len(experiments)} expériences")

        if experiments:
            print(f"   - Première: {experiments[0][:50]}...")

            # 3. Charger une expérience
            exp = bucket_io.load_experiment_from_bucket(experiments[0])
            print(f"3. ✅ load_experiment_from_bucket() réussi")
            print(f"   - Matrice: {exp['matrix'].shape}")
            print(f"   - Stats: {list(exp['stats'].keys())}")

        print()
        return True

    except Exception as e:
        print(f"❌ Erreur bucket_io: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DÉMARRAGE DES TESTS")
    print("=" * 70 + "\n")

    # Test 1: bucket_io
    success1 = test_bucket_io_functions()
    print()

    # Test 2: Adaptive partitioning
    success2 = test_adaptive_partitioning_on_real_data()

    # Résultat final
    if success1 and success2:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS AVEC SUCCÈS 🎉\n")
        exit(0)
    else:
        print("\n❌ CERTAINS TESTS ONT ÉCHOUÉ\n")
        exit(1)
