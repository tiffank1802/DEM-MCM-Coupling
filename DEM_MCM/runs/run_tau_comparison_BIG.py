# -*- coding: utf-8 -*-
"""
Étude comparative: Influence de tau sur la cinétique de mélange (diamètre 0.008m)
Utilise les modèles déjà lancés sur le bucket BIG.
"""
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from analyze_results import MarkovAnalyzer
    from partitioners import CylindricalPartitioner
except ImportError:
    from src.analyze_results import MarkovAnalyzer
    from src.partitioners import CylindricalPartitioner

IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images', 'tau_studies')
os.makedirs(IMAGES_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# 1. Initialisation
# ══════════════════════════════════════════════════════════════════════

analyzer = MarkovAnalyzer()

# Charger uniquement les modèles cylindriques
analyzer.load_method("cylindrical")

# ══════════════════════════════════════════════════════════════════════
# 2. Vérifier les modèles tau disponibles (BIG diameter 0.008)
# ══════════════════════════════════════════════════════════════════════

tau_list = [50, 100, 200, 500, 1000]
folder_template = "cylindrical_nr2_nth4_nz4_equal_area_NLT20_step50_dt2_tau{tau}_start250_d0008"

print("\n Vérification des modèles tau (BIG d=0.008) disponibles:")
available_tau = []
for tau in tau_list:
    folder = folder_template.format(tau=tau)
    if folder in analyzer.results:
        M = analyzer.results[folder]["matrix"]
        print(f"   ✅ tau={tau}: {folder} (shape={M.shape})")
        available_tau.append(tau)
    else:
        print(f"   ❌ tau={tau}: {folder} NON TROUVÉ")

if not available_tau:
    print("\n❌ Aucun modèle tau trouvé! Vérifiez le bucket.")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════
# 3. Lancer l'étude comparative
# ══════════════════════════════════════════════════════════════════════

partitioner = CylindricalPartitioner(nr=2, ntheta=4, nz=4, radial_mode="equal_area")

# ✅ Pour d=0.008, on étudie les GRANDES particules (criterion="large")
save_name = os.path.join(IMAGES_DIR, "rsd_tau_comparison_BIG.png")

analyzer.plot_rsd_vs_tau_comparison(
    partitioner=partitioner,
    method="cylindrical",
    folder_name_template=folder_template,
    tau_list=available_tau,
    max_time_seconds=60,
    figsize=(14, 8),
    save_name=save_name,
    species_criterion="large"
)

print("\n✅ Étude comparative BIG (d=0.008) terminée!")
print(f"   Graphique sauvegardé: {save_name}")
