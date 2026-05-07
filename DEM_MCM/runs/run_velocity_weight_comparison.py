# -*- coding: utf-8 -*-
"""
RUN: Étude comparative du poids de la vitesse — Méthode PHYSICS

Charge les modèles physics avec différents velocity_weight et compare
les courbes RSD et les métriques de mélange.

Usage:
    python runs/run_velocity_weight_comparison.py --diameter 0.004
    python runs/run_velocity_weight_comparison.py --diameter 0.008
    python runs/run_velocity_weight_comparison.py --diameter 0.004 --max-time 60
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
import argparse
import re

try:
    from analyze_results import MarkovAnalyzer
    from partitioners import PhysicsAwarePartitioner
except ImportError:
    from src.analyze_results import MarkovAnalyzer
    from src.partitioners import PhysicsAwarePartitioner

IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images', 'velocity_weight')
os.makedirs(IMAGES_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

VELOCITY_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
N_CELLS = 30
NLT = 20
STEP = 50
DT = 2
TAU = 50
START = 250

# Couleurs pour les courbes velocity_weight
VW_COLORS = plt.cm.viridis(np.linspace(0.1, 0.9, len(VELOCITY_WEIGHTS)))


def build_folder_name(vw, diameter):
    vw_str = f"{vw:.1f}".replace(".", "p")
    diameter_str = str(diameter).replace(".", "")
    return (
        f"physics_{N_CELLS}cells_withvel_NLT{NLT}_step{STEP}_"
        f"dt{DT}_tau{TAU}_start{START}_d{diameter_str}"
    )


def extract_velocity_weight(folder_name):
    vw_match = re.search(r'physics_\d+cells_withvel', folder_name)
    if not vw_match:
        return None
    for vw in VELOCITY_WEIGHTS:
        vw_str = f"{vw:.1f}".replace(".", "p")
        if f"vw{vw_str}" in folder_name or f"vel{vw_str}" in folder_name:
            return vw
    return None


def find_matching_folders(analyzer, diameter):
    diameter_str = str(diameter).replace(".", "")
    matching = []
    for vw in VELOCITY_WEIGHTS:
        for folder_name in analyzer.results:
            if f"_d{diameter_str}" not in folder_name:
                continue
            if "physics" not in folder_name:
                continue
            if "withvel" not in folder_name:
                continue
            folder_vw = analyzer.results[folder_name].get("params", {}).get("method_kwargs", {}).get("velocity_weight")
            if folder_vw is not None and abs(folder_vw - vw) < 0.01:
                matching.append((vw, folder_name))
                break
    return matching


# ══════════════════════════════════════════════════════════════════════
# 1. Parse arguments
# ══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Étude velocity_weight — méthode Physics")
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
print(f"ÉTUDE VELOCITY_WEIGHT — PHYSICS — {diam_label} (d={diameter})")
print(f"{'='*70}")

# ══════════════════════════════════════════════════════════════════════
# 3. Trouver les modèles disponibles
# ══════════════════════════════════════════════════════════════════════

available = find_matching_folders(analyzer, diameter)

if not available:
    print("\n❌ Aucun modèle trouvé! Vérifiez le bucket.")
    sys.exit(1)

print(f"\n📊 {len(available)} modèles trouvés:")
for vw, folder in available:
    M = analyzer.results[folder]["matrix"]
    print(f"   ✅ vw={vw:.1f}: {folder} (shape={M.shape})")

# ══════════════════════════════════════════════════════════════════════
# 4. Tracé comparatif RSD
# ══════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 8))

partitioner_base = PhysicsAwarePartitioner(n_cells=N_CELLS, velocity_weight=0.0)

for i, (vw, folder) in enumerate(sorted(available, key=lambda x: x[0])):
    vw_color = VW_COLORS[i]
    result = analyzer.results[folder]
    matrix = result["matrix"]
    params = result.get("params", {})
    stats = result.get("stats", {})

    step_match = re.search(r'step(\d+)', folder)
    markov_step = int(step_match.group(1)) if step_match else STEP

    start_match = re.search(r'start(\d+)', folder)
    initial_time = int(start_match.group(1)) if start_match else START

    dt_markov_seconds = markov_step * 0.01
    n_steps = int(max_time_seconds / dt_markov_seconds)
    time_seconds = np.arange(n_steps) * dt_markov_seconds

    n_states = matrix.shape[0]
    C0 = np.ones(n_states) / n_states
    C = C0.copy()
    rsd_values = []
    rsd_initial = np.sqrt(np.mean((C - 1.0/n_states)**2)) / np.mean(C) if np.mean(C) > 0 else 0

    for k in range(n_steps):
        C = matrix @ C
        total = C.sum()
        if total > 0:
            C_norm = C / total
            rsd = np.sqrt(np.mean((C_norm - 1.0/n_states)**2)) / np.mean(C_norm) if np.mean(C_norm) > 0 else 0
            rsd_values.append(rsd * 100)

    rsd_initial_pct = analyzer.rsd_percent(C0) if hasattr(analyzer, 'rsd_percent') else (rsd_initial * 100)
    if rsd_values:
        rsd_values = [rsd_values[0] * (r / rsd_values[0]) if rsd_values[0] > 0 else r for r in [rsd_initial_pct] + rsd_values[1:]]

    label = f"v_w={vw:.1f}"
    ax.plot(time_seconds[:len(rsd_values)], rsd_values,
           color=vw_color, linewidth=2.5, alpha=0.85, label=label)

ax.set_xlabel("Temps (s)", fontsize=13, fontweight='bold')
ax.set_ylabel("RSD (%)", fontsize=13, fontweight='bold')
ax.set_title(
    f"Influence du poids de la vitesse sur la cinétique de mélange\n"
    f"PHYSICS | {N_CELLS} cellules | {diam_label} (d={diameter}) | {len(available)} configurations",
    fontsize=14, fontweight='bold', pad=15
)
ax.legend(fontsize=11, loc='best', framealpha=0.95, edgecolor='black', title="velocity_weight")
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
ax.set_xlim(0, max_time_seconds)
ax.set_ylim(bottom=0)
ax.minorticks_on()
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)

plt.tight_layout()

save_name = os.path.join(IMAGES_DIR, f"rsd_velocity_weight_{diam_label.lower()}_comparison.png")
plt.savefig(save_name, dpi=200, bbox_inches='tight', facecolor='white')
print(f"\n✅ Figure sauvegardée: {save_name}")
plt.close()

# ══════════════════════════════════════════════════════════════════════
# 5. Tableau récapitulatif des métriques
# ══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print("RÉSUMÉ DES MÉTRIQUES")
print(f"{'='*70}")
print(f"{'velocity_weight':>15s} | {'P(stay)':>8s} | {'λ₂':>8s} | {'Spectral gap':>12s} | {'RSD final (%)':>13s}")
print("-" * 70)

for vw, folder in sorted(available, key=lambda x: x[0]):
    stats = analyzer.results[folder].get("stats", {})
    p_stay = stats.get("p_stay", "N/A")
    lambda2 = stats.get("lambda_2", "N/A")
    spectral_gap = stats.get("spectral_gap", "N/A")
    rsd_final = rsd_values[-1] if rsd_values else "N/A"

    if isinstance(p_stay, (int, float)):
        p_stay = f"{p_stay:.4f}"
    if isinstance(lambda2, (int, float)):
        lambda2 = f"{lambda2:.4f}"
    if isinstance(spectral_gap, (int, float)):
        spectral_gap = f"{spectral_gap:.4f}"
    if isinstance(rsd_final, (int, float)):
        rsd_final = f"{rsd_final:.2f}"

    print(f"{vw:>15.1f} | {p_stay:>8s} | {lambda2:>8s} | {spectral_gap:>12s} | {rsd_final:>13s}")

print(f"\n✅ Étude velocity_weight terminée!")
