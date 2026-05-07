from src import run_sweep as r_s


configs = [
    # ══════════════════════════════════════════════════════════════════════
    # PHYSICS — SMALL (d=0.004) — Étude tau avec intégration des vitesses
    # velocity_weight=0.5, nlt=20, step=50, dt=2, start=250
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    # PHYSICS — BIG (d=0.008) — Étude tau avec intégration des vitesses
    # velocity_weight=0.5, nlt=20, step=50, dt=2, start=250
    # ══════════════════════════════════════════════════════════════════════

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


r_s.run_markov_sweep("physics", configs=configs, particle_diameter=None)
