# -*- coding: utf-8 -*-
"""
Étude comparative: Influence de tau sur la cinétique de mélange — Méthode PHYSICS (avec vitesses)

Charge les modèles physics (velocity_weight=0.5) avec différents tau et compare
les courbes RSD. Fonctionne pour les deux diamètres (SMALL=0.004, BIG=0.008).

Usage:
    python run_tau_comparison_physics.py --diameter 0.004
    python run_tau_comparison_physics.py --diameter 0.008
    python run_tau_comparison_physics.py --diameter 0.004 --max-time 60
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

try:
    from analyze_results import MarkovAnalyzer
    from partitioners import PhysicsAwarePartitioner
except ImportError:
    from src.analyze_results import MarkovAnalyzer
    from src.partitioners import PhysicsAwarePartitioner

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

TAU_LIST = [10, 25, 50, 100, 150, 200, 300, 500, 750, 1000]
N_CELLS = 30
VELOCITY_WEIGHT = 0.5
NLT = 20
STEP = 50
DT = 2
START = 250


def build_folder_name(tau, diameter):
    """Construit le nom du folder pour une config donnée."""
    diameter_str = str(diameter).replace(".", "")
    vw = VELOCITY_WEIGHT
    suffix = "withvel" if vw != 0 else "pos"
    return (
        f"physics_{N_CELLS}cells_{suffix}_vw{vw}_NLT{NLT}_step{STEP}_"
        f"dt{DT}_tau{tau}_start{START}_d{diameter_str}"
    )


# ══════════════════════════════════════════════════════════════════════
# 1. Parse arguments
# ══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Étude tau — méthode Physics avec vitesses")
parser.add_argument("--diameter", type=float, required=True, help="Diamètre: 0.004 (SMALL) ou 0.008 (BIG)")
parser.add_argument("--max-time", type=int, default=60, help="Temps max en secondes (défaut: 60)")
args = parser.parse_args()

diameter = args.diameter
max_time_seconds = args.max_time

# ══════════════════════════════════════════════════════════════════════
# 2. Initialisation
# ══════════════════════════════════════════════════════════════════════

analyzer = MarkovAnalyzer()
analyzer.load_method("physics")

diam_label = "SMALL" if diameter == 0.004 else "BIG"
print(f"\n{'='*70}")
print(f"ÉTUDE TAU — PHYSICS (velocity_weight={VELOCITY_WEIGHT}) — {diam_label} (d={diameter})")
print(f"{'='*70}")

# ══════════════════════════════════════════════════════════════════════
# 3. Vérifier les modèles tau disponibles
# ══════════════════════════════════════════════════════════════════════

available_tau = []
for tau in TAU_LIST:
    folder = build_folder_name(tau, diameter)
    if folder in analyzer.results:
        M = analyzer.results[folder]["matrix"]
        print(f"   ✅ tau={tau:5d}: {folder} (shape={M.shape})")
        available_tau.append(tau)
    else:
        print(f"   ❌ tau={tau:5d}: {folder} NON TROUVÉ")

if not available_tau:
    print("\n❌ Aucun modèle tau trouvé! Vérifiez le bucket.")
    sys.exit(1)

print(f"\n📊 {len(available_tau)} modèles disponibles sur {len(TAU_LIST)} configurés")

# ══════════════════════════════════════════════════════════════════════
# 4. Partitionneur (pour le calcul RSD)
# ══════════════════════════════════════════════════════════════════════

partitioner = PhysicsAwarePartitioner(n_cells=N_CELLS, velocity_weight=VELOCITY_WEIGHT)

# ══════════════════════════════════════════════════════════════════════
# 5. Tracé comparatif RSD vs tau
# ══════════════════════════════════════════════════════════════════════

save_name = f"/kaggle/working/rsd_tau_physics_{diam_label.lower()}_comparison.png"

analyzer.plot_rsd_vs_tau_comparison(
    partitioner=partitioner,
    method="physics",
    folder_name_template=build_folder_name("{tau}", diameter),
    tau_list=available_tau,
    max_time_seconds=max_time_seconds,
    figsize=(14, 8),
    save_name=save_name
)

print(f"\n✅ Étude comparative terminée!")
print(f"   Graphique sauvegardé: {save_name}")
