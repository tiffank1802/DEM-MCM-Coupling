# -*- coding: utf-8 -*-
"""
CONFIG: Étude de l'influence de tau — Méthode PHYSICS (avec vitesses)

Étudie l'effet du pas de temps Markov (tau) sur la cinétique de mélange
pour les deux diamètres de particules avec intégration des vitesses.

Paramètres fixes: n_cells=30, velocity_weight=0.5, nlt=20, step=50, dt=2, start=250
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from src import run_sweep as r_s

# ══════════════════════════════════════════════════════════════════════
# PHYSICS — TAU STUDY — SMALL (d=0.004)
# ══════════════════════════════════════════════════════════════════════
# velocity_weight=0.5, n_cells=30, nlt=20, step=50, dt=2, start=250
# ══════════════════════════════════════════════════════════════════════

configs = [
    # SMALL (d=0.004) — tau sweep
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=10,  step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=25,  step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=50,  step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=100, step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=150, step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=200, step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=300, step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=500, step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=750, step=50, dt=2, start_index=250, particle_diameter=0.004),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=1000, step=50, dt=2, start_index=250, particle_diameter=0.004),

    # BIG (d=0.008) — tau sweep
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=10,  step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=25,  step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=50,  step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=100, step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=150, step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=200, step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=300, step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=500, step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=750, step=50, dt=2, start_index=250, particle_diameter=0.008),
    r_s.ExperimentConfig(method="physics", method_kwargs={"n_cells": 30, "velocity_weight": 0.5}, nlt=20, tau=1000, step=50, dt=2, start_index=250, particle_diameter=0.008),
]

print(f"\n📋 {len(configs)} configurations tau (physics):")
for cfg in configs:
    d = cfg.particle_diameter
    label = "SMALL" if d == 0.004 else ("BIG" if d == 0.008 else "ALL")
    print(f"   [{label:5s}] tau={cfg.tau:5d} → {cfg.output_folder()}")
print()

r_s.run_markov_sweep("physics", configs=configs, particle_diameter=None)
