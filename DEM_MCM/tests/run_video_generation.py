"""
Lance la génération de vidéos 3D pour toutes les méthodes de partitionnement.

Usage:
    python tests/run_video_generation.py --method all
    python tests/run_video_generation.py --method cartesian
    python tests/run_video_generation.py --method cylindrical voronoi
"""

import sys
import os
import argparse
import numpy as np

# Ajouter src au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from partitioners import REGISTRY, create_partitioner


def generate_video_for_method(method_name, particle_diameter=0.004, 
                              duration=12, fps=30, n_particles=2000):
    """
    Génère une vidéo 3D pour une méthode de partitionnement.
    
    Args:
        method_name: Nom de la méthode (cartesian, cylindrical, etc.)
        particle_diameter: Diamètre des particules (0.004 ou 0.008)
        duration: Durée de la vidéo en secondes
        fps: Images par seconde
        n_particles: Nombre de particules à utiliser (échantillonnage)
    """
    print(f"\n{'='*60}")
    print(f"Génération vidéo 3D pour: {method_name.upper()}")
    print(f"{'='*60}")
    
    # Créer le partitionneur avec des paramètres par défaut
    if method_name == "cartesian":
        partitioner = create_partitioner(method_name, nx=5, ny=5, nz=3)
    elif method_name == "cylindrical":
        partitioner = create_partitioner(method_name, nr=3, ntheta=8, nz=3, radial_mode="equal_area")
    elif method_name == "voronoi":
        partitioner = create_partitioner(method_name, n_cells=125)
    elif method_name == "quantile":
        partitioner = create_partitioner(method_name, nx=5, ny=5, nz=3)
    elif method_name == "octree":
        partitioner = create_partitioner(method_name, max_particles=100, max_depth=3)
    elif method_name == "physics":
        partitioner = create_partitioner(method_name, n_cells=125, velocity_weight=0.5)
    elif method_name == "adaptive":
        partitioner = create_partitioner(
            method_name,
            y_split=0.90, y_split_mode="quantile",
            n_cells_top=1, top_method="single", top_kwargs={},
            bottom_method="voronoi", bottom_kwargs={"n_cells": 100}
        )
    else:
        print(f"❌ Méthode non supportée: {method_name}")
        return None
    
    # Générer des données de test (coordonnées simulées)
    np.random.seed(42)
    
    if method_name == "cylindrical":
        # Pour le cylindrique, utiliser des coordonnées polaires
        r = np.random.rand(n_particles) * 0.5
        theta = np.random.rand(n_particles) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.rand(n_particles) * 1.0
    else:
        # Pour les autres, coordonnées cartésiennes
        x = np.random.randn(n_particles) * 0.3
        y = np.random.randn(n_particles) * 0.3
        z = np.random.rand(n_particles) * 1.0
    
    # Diamètres bidisperses
    diameters = np.random.choice([0.004, 0.008], n_particles)
    
    # Chemin de sortie
    output_dir = os.path.join("images", "3d_rotation")
    os.makedirs(output_dir, exist_ok=True)
    safe_label = partitioner.label.replace('=', '_').replace(' ', '_').replace('/', '_')
    output_path = os.path.join(output_dir, f"{safe_label}_d{str(particle_diameter).replace('.', '')}_rotation.mp4")
    
    print(f"   📊 Nombre de particules: {n_particles}")
    print(f"   🎥 Durée: {duration}s à {fps} fps ({duration*fps} frames)")
    print(f"   📏 Diamètres: 0.004m et 0.008m (bidisperses)")
    print(f"   💾 Sortie: {output_path}")
    
    try:
        result = partitioner.visualize_3d_rotation(
            x, y, z, particle_diameters=diameters,
            output_path=output_path,
            duration=duration, fps=fps
        )
        print(f"   ✅ Vidéo générée: {result}")
        print(f"   📦 Taille: {os.path.getsize(result) / 1024 / 1024:.2f} MB")
        return result
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Génère des vidéos 3D de rotation pour les méthodes de partitionnement"
    )
    parser.add_argument(
        '--method', type=str, nargs='+', default=['all'],
        choices=list(REGISTRY.keys()) + ['all'],
        help="Méthode(s) à traiter (défaut: all)"
    )
    parser.add_argument(
        '--duration', type=int, default=10,
        help="Durée de la vidéo en secondes (défaut: 10)"
    )
    parser.add_argument(
        '--fps', type=int, default=60,
        help="Images par seconde (défaut: 60)"
    )
    parser.add_argument(
        '--particles', type=int, default=2000,
        help="Nombre de particules à utiliser (défaut: 2000)"
    )
    parser.add_argument(
        '--diameter', type=float, default=0.004,
        help="Diamètre des particules (défaut: 0.004)"
    )
    
    args = parser.parse_args()
    
    # Déterminer les méthodes à traiter
    if 'all' in args.method:
        methods = list(REGISTRY.keys())
    else:
        methods = args.method
    
    print(f"\n🎥 Génération de vidéos 3D pour {len(methods)} méthode(s)")
    print(f"   Durée: {args.duration}s, FPS: {args.fps}, Particules: {args.particles}")
    
    results = {}
    for method in methods:
        result = generate_video_for_method(
            method_name=method,
            particle_diameter=args.diameter,
            duration=args.duration,
            fps=args.fps,
            n_particles=args.particles
        )
        results[method] = result
    
    # Résumé
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print(f"{'='*60}")
    success = sum(1 for v in results.values() if v is not None)
    print(f"✅ Réussies: {success}/{len(methods)}")
    
    if success < len(methods):
        print(f"❌ Échouées: {len(methods) - success}")
        for method, result in results.items():
            if result is None:
                print(f"   - {method}")
    
    print(f"\n💾 Vidéos sauvegardées dans: images/3d_rotation/")
    print("✨ Terminé!\n")


if __name__ == "__main__":
    main()
