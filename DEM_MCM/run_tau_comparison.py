# -*- coding: utf-8 -*-
"""
Étude comparative: Influence de tau sur la cinétique de mélange
Utilise les modèles déjà lancés sur le bucket SMALL.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import matplotlib.pyplot as plt

from analyze_results import MarkovAnalyzer
from partitioners import CylindricalPartitioner

# ══════════════════════════════════════════════════════════════════════
# 1. Initialisation
# ══════════════════════════════════════════════════════════════════════

analyzer = MarkovAnalyzer()

# Charger uniquement les modèles cylindriques (plus rapide)
analyzer.load_method("cylindrical")

# ══════════════════════════════════════════════════════════════════════
# 2. Vérifier les modèles tau disponibles
# ══════════════════════════════════════════════════════════════════════

tau_list = [50, 100, 200, 500, 1000]
folder_template = "cylindrical_nr2_nth4_nz4_equal_area_NLT20_step50_dt2_tau{tau}_start250_d0004"

print("\n🔍 Vérification des modèles tau disponibles:")
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

analyzer.plot_rsd_vs_tau_comparison(
    partitioner=partitioner,
    method="cylindrical",
    folder_name_template=folder_template,
    tau_list=available_tau,
    max_time_seconds=60,
    figsize=(14, 8),
    save_name="/kaggle/working/rsd_tau_comparison_study.png"
)

print("\n✅ Étude comparative terminée!")
print(f"   Graphique sauvegardé: /kaggle/working/rsd_tau_comparison_study.png")
