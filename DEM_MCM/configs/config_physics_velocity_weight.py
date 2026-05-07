# -*- coding: utf-8 -*-
"""
CONFIG: Étude de l'influence du poids de la vitesse — Méthode PHYSICS

Étudie l'effet de velocity_weight sur la qualité de prédiction du modèle
Markov pour les deux diamètres de particules.

Paramètres fixes: n_cells=30, nlt=20, step=50, dt=2, start=250, tau=50
velocity_weight varie: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from src import run_sweep as r_s

VELOCITY_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]

configs = []

# ══════════════════════════════════════════════════════════════════════
# PHYSICS — VELOCITY WEIGHT STUDY — SMALL (d=0.004)
# ══════════════════════════════════════════════════════════════════════

for vw in VELOCITY_WEIGHTS:
    configs.append(
        r_s.ExperimentConfig(
            method="physics",
            method_kwargs={"n_cells": 30, "velocity_weight": vw},
            nlt=20, tau=50, step=50, dt=2, start_index=250,
            particle_diameter=0.004
        )
    )

# ══════════════════════════════════════════════════════════════════════
# PHYSICS — VELOCITY WEIGHT STUDY — BIG (d=0.008)
# ══════════════════════════════════════════════════════════════════════

for vw in VELOCITY_WEIGHTS:
    configs.append(
        r_s.ExperimentConfig(
            method="physics",
            method_kwargs={"n_cells": 30, "velocity_weight": vw},
            nlt=20, tau=50, step=50, dt=2, start_index=250,
            particle_diameter=0.008
        )
    )

print(f"\n📋 {len(configs)} configurations velocity_weight (physics):")
print(f"    Poids testés: {VELOCITY_WEIGHTS}")
print()
for cfg in configs:
    vw = cfg.method_kwargs.get("velocity_weight", 0)
    d = cfg.particle_diameter
    label = "SMALL" if d == 0.004 else ("BIG" if d == 0.008 else "ALL")
    print(f"   [{label:5s}] vw={vw:.1f} → {cfg.output_folder()}")
print()

r_s.run_markov_sweep("physics", configs=configs, particle_diameter=None)
