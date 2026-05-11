# -*- coding: utf-8 -*-
"""
CONFIG: Sweep des méthodes de coupe oblique pour OctreePartitioner

Lance les 6 méthodes (axis, pca, kmeans2, 2medians, random, svm)
pour les deux diamètres de particules.

Paramètres fixes: max_particles=100, max_depth=3, nlt=10, step=10, dt=2, tau=50, start=250
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    from src import run_sweep as r_s
except ImportError:
    import run_sweep as r_s

OBLIQUE_METHODS = [
    # "axis",
    #   "pca",
    #     "kmeans2",
    #     "2medians", 
        "random",
        #   "svm"
          ]
BASE_KWARGS = {"max_particles": 90, "max_depth": 3}
BASE_PARAMS = {"nlt": 10, "step": 10, "dt": 2, "tau": 50, "start_index": 250}

configs = []

# SMALL (d=0.004)
for om in OBLIQUE_METHODS:
    kw = {**BASE_KWARGS, "oblique_method": om}
    configs.append(
        r_s.ExperimentConfig(
            method="octree", method_kwargs=kw,
            **BASE_PARAMS, particle_diameter=0.004
        )
    )

# BIG (d=0.008)
# for om in OBLIQUE_METHODS:
#     kw = {**BASE_KWARGS, "oblique_method": om}
#     configs.append(
#         r_s.ExperimentConfig(
#             method="octree", method_kwargs=kw,
#             **BASE_PARAMS, particle_diameter=0.008
#         )
#     )

print(f"\n📋 {len(configs)} configurations octree oblique:")
print(f"{'─' * 70}")
for cfg in configs:
    d = cfg.particle_diameter
    label = "SMALL" if d == 0.004 else "BIG"
    om = cfg.method_kwargs.get("oblique_method", "axis")
    print(f"   [{label}] oblique={om:>10s} → {cfg.output_folder()}")
print(f"{'─' * 70}\n")

r_s.run_markov_sweep("octree", configs=configs, particle_diameter=None)
