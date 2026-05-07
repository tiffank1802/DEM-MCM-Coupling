# -*- coding: utf-8 -*-
"""
Sweep comparatif — toutes les méthodes de coupe oblique pour OctreePartitioner

Lance les 6 variantes (axis, pca, kmeans2, 2medians, random, svm) avec
des paramètres identiques, génère les images et matrices de transition,
et sauvegarde le tout dans le bucket HuggingFace.

Usage:
    python runs/run_oblique_comparison.py
    python runs/run_oblique_comparison.py --diameter 0.008
    python runs/run_oblique_comparison.py --dry-run
"""
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import HfFileSystem

try:
    from run_sweep import (
        ExperimentConfig, run_experiment, sample_coordinates,
        sample_velocities, save_results, HF_FOLDER
    )
    from partitioners import OctreePartitioner
    from bucket_io import save_experiment_to_bucket, list_experiments
except ImportError:
    from src.run_sweep import (
        ExperimentConfig, run_experiment, sample_coordinates,
        sample_velocities, save_results, HF_FOLDER
    )
    from src.partitioners import OctreePartitioner
    from src.bucket_io import save_experiment_to_bucket, list_experiments

IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images', 'oblique_studies')
os.makedirs(IMAGES_DIR, exist_ok=True)

OBLIQUE_METHODS = ["axis", "pca", "kmeans2", "2medians", "random", "svm"]

OCTREE_KWARGS = {"max_particles": 100, "max_depth": 3}

EXPERIMENT_PARAMS = {
    "nlt": 10,
    "tau": 50,
    "step": 10,
    "dt": 2,
    "start_index": 250,
}


def run_oblique_sweep(particle_diameter=0.004, dry_run=False):
    print("=" * 70)
    print("  SWEEP OBLIQUE OCTREE — Comparaison des méthodes de coupe")
    print("=" * 70)
    print(f"  Diamètre: {particle_diameter}")
    print(f"  Méthodes: {OBLIQUE_METHODS}")
    print(f"  Dry-run: {dry_run}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}\n")

    fs = HfFileSystem()
    files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))
    print(f"  Fichiers DEM: {len(files)}\n")

    print("  Échantillonnage coordonnées pour fit...")
    sample_coords = sample_coordinates(files, fs)
    s_velocities = sample_velocities(files, fs)
    print(f"    {len(sample_coords)} points\n")

    sample_diameters = None
    try:
        with fs.open(files[5], "rb") as fh:
            import polars as pl
            df = pl.read_csv(fh)
            if "Diameter" in df.columns:
                sample_diameters = df["Diameter"].to_numpy()
    except Exception:
        pass

    all_results = {}

    for om in OBLIQUE_METHODS:
        safe_om = om.replace(" ", "_")
        folder_name = (
            f"octree_mp{OCTREE_KWARGS['max_particles']}"
            f"_md{OCTREE_KWARGS['max_depth']}_{safe_om}"
            f"_NLT{EXPERIMENT_PARAMS['nlt']}"
            f"_step{EXPERIMENT_PARAMS['step']}"
            f"_dt{EXPERIMENT_PARAMS['dt']}"
            f"_tau{EXPERIMENT_PARAMS['tau']}"
            f"_start{EXPERIMENT_PARAMS['start_index']}"
            f"_d{str(particle_diameter).replace('.', '')}"
        )

        print(f"\n{'─' * 60}")
        print(f"  [{om}] {folder_name}")

        if dry_run:
            all_results[om] = {"folder": folder_name}
            continue

        try:
            partitioner = OctreePartitioner(
                max_particles=OCTREE_KWARGS["max_particles"],
                max_depth=OCTREE_KWARGS["max_depth"],
                oblique_method=om,
            )

            print(f"    Fit partitionneur...")
            partitioner.fit(sample_coords)

            diag = partitioner.diagnostics(sample_coords)
            print(f"    {partitioner.n_cells} cellules | "
                  f"{diag['n_visited']} visitées | "
                  f"pop [{diag['pop_min']}, {diag['pop_max']}] "
                  f"μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}")

            image_data = None
            if hasattr(partitioner, 'visualize'):
                try:
                    x, y, z = sample_coords[:, 0], sample_coords[:, 1], sample_coords[:, 2]
                    safe_label = partitioner.label.replace('=', '_').replace(' ', '_').replace('/', '_')
                    vis_kwargs = {
                        "x": x, "y": y, "z": z,
                        "save_prefix": f"oblique_{safe_label}"
                    }
                    if sample_diameters is not None and len(sample_diameters) == len(x):
                        vis_kwargs["particle_diameters"] = sample_diameters
                    image_data = partitioner.visualize(**vis_kwargs)
                    print(f"    {len(image_data)} images générées")
                except Exception as e:
                    print(f"    ⚠️  Visualisation: {e}")

            config = ExperimentConfig(
                method="octree",
                method_kwargs={
                    "max_particles": OCTREE_KWARGS["max_particles"],
                    "max_depth": OCTREE_KWARGS["max_depth"],
                    "oblique_method": om,
                },
                nlt=EXPERIMENT_PARAMS["nlt"],
                tau=EXPERIMENT_PARAMS["tau"],
                step=EXPERIMENT_PARAMS["step"],
                dt=EXPERIMENT_PARAMS["dt"],
                start_index=EXPERIMENT_PARAMS["start_index"],
                particle_diameter=particle_diameter,
            )

            print(f"    Calcul matrice de transition...")
            P, stats = run_experiment(config, partitioner, files, fs, device)

            save_results(
                config=config,
                partitioner=partitioner,
                P=P,
                stats=stats,
                image_data=image_data,
                folder_name=folder_name,
            )

            print(f"    ✅ {stats['n_states_visited']}/{stats['n_states']} états | "
                  f"P(rester)={stats['diagonal_mean']:.4f} | "
                  f"pairs={stats['n_pairs_used']}")

            all_results[om] = {
                "folder": folder_name,
                "n_cells": partitioner.n_cells,
                "stats": stats,
                "success": True,
            }

        except Exception as e:
            print(f"    ❌ Erreur: {e}")
            all_results[om] = {"folder": folder_name, "success": False, "error": str(e)}

    print(f"\n{'=' * 70}")
    print("  RÉSUMÉ")
    print(f"{'=' * 70}")
    for om, res in all_results.items():
        status = "✅" if res.get("success") else "❌"
        nc = res.get("n_cells", "?")
        print(f"  {status} {om}: {nc} cellules → {res['folder']}")

    summary_data = {
        "method": "octree_oblique",
        "diameter": particle_diameter,
        "oblique_methods": OBLIQUE_METHODS,
        "results": {
            om: {"folder": r["folder"], "success": r.get(False, False)}
            for om, r in all_results.items()
        },
        "params": {**OCTREE_KWARGS, **EXPERIMENT_PARAMS},
    }

    try:
        save_experiment_to_bucket(
            folder_name=f"_summary_octree_oblique_d{str(particle_diameter).replace('.', '')}",
            matrix=np.array([]),
            stats=summary_data,
            config={"type": "summary", "method": "octree_oblique"},
        )
        print(f"\n  Résumé sauvegardé dans le bucket")
    except Exception as e:
        print(f"\n  ⚠️  Résumé non sauvegardé: {e}")

    print("\n  Terminé!")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sweep oblique Octree")
    parser.add_argument("--diameter", type=float, default=0.004, choices=[0.004, 0.008])
    parser.add_argument("--dry-run", action="store_true", help="Simuler sans lancer")
    parser.add_argument("--visualize", action="store_true",
                        help="Générer les visualisations depuis le bucket après le sweep")
    args = parser.parse_args()

    results = run_oblique_sweep(particle_diameter=args.diameter, dry_run=args.dry_run)

    if args.visualize and not args.dry_run:
        print("\n" + "=" * 70)
        print("  GÉNÉRATION DES VISUALISATIONS DEPUIS LE BUCKET")
        print("=" * 70)
        try:
            from src.partitioners import OctreePartitioner
            for method in OBLIQUE_METHODS:
                print(f"\n  Méthode: {method}")
                imgs = OctreePartitioner.visualize_from_bucket(
                    method=method,
                    particle_diameter=args.diameter,
                    save_prefix=f"oblique_{method}_d{str(args.diameter).replace('.', '')}",
                    plot_types=["matrix", "rsd"],
                )
                if imgs:
                    img_dir = os.path.join(PROJECT_ROOT, 'images', 'oblique_studies')
                    os.makedirs(img_dir, exist_ok=True)
                    for name, data in imgs.items():
                        path = os.path.join(img_dir, name)
                        with open(path, 'wb') as f:
                            f.write(data)
                        print(f"    Image: {path}")

            # Comparaison RSD toutes méthodes
            print(f"\n  Comparaison globale:")
            imgs = OctreePartitioner.visualize_from_bucket(
                method="pca",
                particle_diameter=args.diameter,
                save_prefix=f"oblique_comparison_d{str(args.diameter).replace('.', '')}",
                plot_types=["comparison"],
            )
            for name, data in imgs.items():
                path = os.path.join(PROJECT_ROOT, 'images', 'oblique_studies', name)
                with open(path, 'wb') as f:
                    f.write(data)
                print(f"    Image: {path}")

        except Exception as e:
            print(f"  ⚠️  Visualisation depuis bucket: {e}")
