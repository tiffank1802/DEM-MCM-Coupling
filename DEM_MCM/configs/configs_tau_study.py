import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    from src import run_sweep as r_s
except ImportError:
    import run_sweep as r_s

# ══════════════════════════════════════════════════════════════════════
# ÉTUDE DE L'INFLUENCE DU PAS DE TEMPS (tau) SUR LA CINÉTIQUE DE MÉLANGE
# ══════════════════════════════════════════════════════════════════════
# Partitionneur: cylindrical ~20 cellules (nr=4, ntheta=5, nz=1)
# tau: différents pas de temps Markov
# step=50, dt=2, start=250, nlt=20 (constants)
# ══════════════════════════════════════════════════════════════════════

configs_tau = [
    # ~20 cellules: nr=4 * ntheta=5 * nz=1 = 20 cellules
    r_s.ExperimentConfig(
        method="cylindrical",
        method_kwargs={"nr": 2, "ntheta": 4, "nz": 4, "radial_mode": "equal_area"},
        nlt=20, step=50, dt=2, tau=50, start_index=250,
        particle_diameter=0.004
    ),
    r_s.ExperimentConfig(
        method="cylindrical",
        method_kwargs={"nr": 2, "ntheta": 4, "nz": 4, "radial_mode": "equal_area"},
        nlt=20, step=50, dt=2, tau=100, start_index=250,
        particle_diameter=0.004
    ),
    r_s.ExperimentConfig(
        method="cylindrical",
        method_kwargs={"nr": 2, "ntheta": 4, "nz": 4, "radial_mode": "equal_area"},
        nlt=20, step=50, dt=2, tau=200, start_index=250,
        particle_diameter=0.004
    ),
    r_s.ExperimentConfig(
        method="cylindrical",
        method_kwargs={"nr": 2, "ntheta": 4, "nz": 4, "radial_mode": "equal_area"},
        nlt=20, step=50, dt=2, tau=500, start_index=250,
        particle_diameter=0.004
    ),
    r_s.ExperimentConfig(
        method="cylindrical",
        method_kwargs={"nr": 2, "ntheta": 4, "nz": 4, "radial_mode": "equal_area"},
        nlt=20, step=50, dt=2, tau=1000, start_index=250,
        particle_diameter=0.004
    ),
]

print("🎯 Étude tau - 20 cellules avec différents pas de temps Markov:")
for cfg in configs_tau:
    print(f"   tau={cfg.tau} → {cfg.output_folder()}")

# Exécuter le sweep
r_s.run_markov_sweep("cylindrical", configs=configs_tau, particle_diameter=0.004)
