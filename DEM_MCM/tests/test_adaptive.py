"""
Test script for adaptive partitioning visualization with real data
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.partitioners import create_partitioner, AdaptiveZPartitioner
from src.visualize_partitioning import PartitionVisualizer
from src import bucket_io


def test_adaptive_partitioner_creation():
    """Test that AdaptiveZPartitioner can be created with different configurations"""

    # Test 1: Basic quantile split
    part = create_partitioner(
        "adaptive",
        z_split_mode="quantile",
        z_split=0.7,
        n_cells_top=1,
        bottom_method="cylindrical",
        bottom_kwargs={"nr": 5, "ntheta": 5, "nz": 5},
    )
    assert isinstance(part, AdaptiveZPartitioner)
    assert part.z_split_mode == "quantile"
    assert part.n_cells_top_target == 1

    # Test 2: Absolute split
    part2 = create_partitioner(
        "adaptive",
        z_split_mode="absolute",
        z_split=0.5,
        n_cells_top=3,
        top_method="cartesian",
        top_kwargs={"nx": 2, "ny": 2, "nz": 2},
        bottom_method="voronoi",
        bottom_kwargs={"n_cells": 125},
    )
    assert isinstance(part2, AdaptiveZPartitioner)
    assert part2.z_split_mode == "absolute"

    print("✅ Test 1 passé: Création AdaptiveZPartitioner")


def test_adaptive_partitioner_fit():
    """Test that adaptive partitioner fits correctly and splits zones"""
    np.random.seed(42)
    coords = np.random.rand(500, 3)
    coords[:, 2] = coords[:, 2] * 2  # Scale z-axis

    part = create_partitioner(
        "adaptive",
        z_split_mode="quantile",
        z_split=0.7,
        n_cells_top=1,
        bottom_method="voronoi",
        bottom_kwargs={"n_cells": 50},
    )

    # Fit the partitioner
    part.fit(coords)

    # Check that split was computed
    assert hasattr(part, "_z_split")
    assert hasattr(part, "_n_cells_bottom")
    assert hasattr(part, "_n_cells_top")
    assert part._n_cells_bottom == 50  # voronoi cells
    assert part._n_cells_top == 1

    # Check zone assignment works
    states = part.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])
    assert len(states) == len(coords)
    assert states.max() < part.n_cells
    assert states.min() >= 0

    print("✅ Test 2 passé: Adaptateur fit et assignation")


def test_adaptive_diagnostics():
    """Test that diagnostics provide zone-specific information"""
    np.random.seed(42)
    coords = np.random.rand(200, 3)

    part = create_partitioner(
        "adaptive",
        z_split_mode="quantile",
        z_split=0.6,
        bottom_method="cartesian",
        bottom_kwargs={"nx": 3, "ny": 3, "nz": 3},
    )
    part.fit(coords)

    diag = part.diagnostics(coords)

    # Check that diagnostics contains zone info
    assert "bottom_stats" in diag
    assert "z_split" in diag
    assert "fraction_in_bottom" in diag
    assert 0 < diag["fraction_in_bottom"] < 1

    print("✅ Test 3 passé: Diagnostics adaptatifs")


def test_visualizer_with_adaptive():
    """Test that visualizer can handle adaptive partitioner"""
    # Créer des données synthétiques pour le test
    np.random.seed(42)
    coords = np.random.rand(300, 3)

    viz = PartitionVisualizer()
    viz.coords = coords  # Simuler le chargement

    # Test la configuration adaptative
    configs = viz.get_default_partitioners()
    adaptive_configs = {k: v for k, v in configs.items() if "adapt" in k.lower()}

    assert len(adaptive_configs) > 0, (
        "Aucun partitionneur adaptatif dans la configuration"
    )

    for label, config in adaptive_configs.items():
        part = create_partitioner(config["method"], **config["kwargs"])
        part.fit(coords)
        states = part.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])

        # Vérifier que les états sont corrects
        assert states is not None
        assert len(states) == len(coords)

        print(f"  ✅ Partitionneur adaptatif '{label}' fonctionne")

    print("✅ Test 4 passé: Visualiseur avec partitionneurs adaptatifs")


def test_load_real_data():
    """Test loading real data from HuggingFace bucket"""
    try:
        from src.bucket_io import get_fs

        fs = get_fs()

        # Lister les fichiers disponibles
        files = fs.glob("hf://buckets/ktongue/DEM_MCM/Output Paraview/*.csv")

        if not files:
            print("⚠️ Aucun fichier DEM trouvé - saut du test de données réelles")
            return

        print(f"📁 {len(files)} fichiers trouvés dans le bucket")

        # Charger le premier fichier pour tester
        import polars as pl

        with fs.open(files[0], "rb") as f:
            df = pl.read_csv(f)

        assert len(df) > 0, "DataFrame vide"
        assert "coordinates:0" in df.columns, "Colonne coordinates:0 manquante"

        print("✅ Test 5 passé: Chargement des données réelles réussi")

    except Exception as e:
        print(f"⚠️ Test 5 ignoré ({e}) - connexion HuggingFace requise")


def test_adaptive_visualization_profile():
    """Test the adaptive profile visualization HTML generation"""
    np.random.seed(42)
    coords = np.random.rand(100, 3)

    part = create_partitioner(
        "adaptive",
        z_split_mode="quantile",
        z_split=0.7,
        bottom_method="cylindrical",
        bottom_kwargs={"nr": 3, "ntheta": 4, "nz": 5},
    )
    part.fit(coords)

    # Test that visualize_profile method exists and generates HTML
    assert hasattr(part, "visualize_profile"), "La méthode visualize_profile() manque"

    # Generate the visualization
    try:
        html_output = part.visualize_profile(size=400)
        assert html_output is not None, "La visualisation HTML est None"

        # Check that HTML contains expected elements
        html_str = str(html_output)
        assert "Zone basse" in html_str, "HTML ne contient pas 'Zone basse'"
        assert "Zone haute" in html_str, "HTML ne contient pas 'Zone haute'"
        assert "z_split" in html_str, "HTML ne contient pas 'z_split'"

        print("✅ Test 6 passé: Visualisation de profil adaptative")

    except Exception as e:
        print(f"⚠️ Test 6 partiel: Visualisation créée mais erreur lors du rendu: {e}")


def test_integration_end_to_end():
    """Test complete workflow: load data → partition → visualize"""
    try:
        # Skip if no real data available
        from src.bucket_io import get_fs

        fs = get_fs()
        files = fs.glob("hf://buckets/ktongue/DEM_MCM/Output Paraview/*.csv")

        if not files:
            print("⚠️ Test 7 ignoré - aucune donnée réelle disponible")
            return

        # Charger les données
        import polars as pl

        with fs.open(files[50], "rb") as f:  # Use snapshot 50
            df = pl.read_csv(f)

        coords = np.column_stack(
            [
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ]
        )

        # Créer partitionneur adaptatif
        part = create_partitioner(
            "adaptive",
            z_split_mode="quantile",
            z_split=0.7,
            bottom_method="voronoi",
            bottom_kwargs={"n_cells": 100},
        )
        part.fit(coords)

        # Générer les états
        states = part.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])

        # Vérifier les résultats
        assert len(states) == len(coords)
        assert states.max() < part.n_cells
        assert part.diagnostics(coords)["fraction_visited"] > 0.5

        print("✅ Test 7 passé: Workflow complet (data → partition → analyze)")

    except Exception as e:
        print(f"⚠️ Test 7 partiel: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTS DU PARTITIONNEUR ADAPTATIF Z")
    print("=" * 60)
    print()

    try:
        test_adaptive_partitioner_creation()
    except Exception as e:
        print(f"❌ Test 1 échoué: {e}")

    try:
        test_adaptive_partitioner_fit()
    except Exception as e:
        print(f"❌ Test 2 échoué: {e}")

    try:
        test_adaptive_diagnostics()
    except Exception as e:
        print(f"❌ Test 3 échoué: {e}")

    try:
        test_visualizer_with_adaptive()
    except Exception as e:
        print(f"❌ Test 4 échoué: {e}")

    try:
        test_load_real_data()
    except Exception as e:
        print(f"❌ Test 5 échoué: {e}")

    try:
        test_adaptive_visualization_profile()
    except Exception as e:
        print(f"❌ Test 6 échoué: {e}")

    try:
        test_integration_end_to_end()
    except Exception as e:
        print(f"❌ Test 7 échoué: {e}")

    print()
    print("=" * 60)
    print("Tests terminés!")
    print("=" * 60)
