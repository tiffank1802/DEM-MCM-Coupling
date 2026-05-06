# -*- coding: utf-8 -*-
"""
Étude comparative: Influence de tau sur la cinétique de mélange (Voronoi, 30 cellules, d=0.008)
Utilise les modèles déjà lancés sur le bucket BIG.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import matplotlib.pyplot as plt

from analyze_results import MarkovAnalyzer
from partitioners import VoronoiPartitioner

# ══════════════════════════════════════════════════════════════════════
# 1. Initialisation
# ══════════════════════════════════════════════════════════════════════

analyzer = MarkovAnalyzer()

# Charger uniquement les modèles Voronoi
analyzer.load_method("voronoi")

# ══════════════════════════════════════════════════════════════════════
# 2. Vérifier les modèles tau disponibles (Voronoi 30 cellules)
# ═════════════════════════════════════════════════════════════════════

tau_list = [50, 100, 200, 500, 1000]
folder_template = "voronoi_30cells_NLT20_step50_dt2_tau{tau}_start250_d0008"

print("\n🔍 Vérification des modèles tau (Voronoi 30 cellules, BIG d=0.008) disponibles:")
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

partitioner = VoronoiPartitioner(n_cells=30, random_state=42)

analyzer.plot_rsd_vs_tau_comparison(
    partitioner=partitioner,
    method="voronoi",
    folder_name_template=folder_template,
    tau_list=available_tau,
    max_time_seconds=60,
    figsize=(14, 8),
    save_name="/kaggle/working/rsd_tau_comparison_voronoi_30cells_BIG.png",
    species_criterion="large"
)

print("\n✅ Étude comparative Voronoi (30 cellules, d=0.008 BIG) terminée!")
print(f"   Graphique sauvegardé: /kaggle/working/rsd_tau_comparison_voronoi_30cells_BIG.png")
