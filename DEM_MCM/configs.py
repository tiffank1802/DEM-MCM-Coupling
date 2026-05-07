# -*- coding: utf-8 -*-
"""
Point d'entrée principal pour les configurations d'expériences.

Structure du projet:
    configs/   — Fichiers de configuration d'expériences
    runs/      — Scripts de lancement des études comparatives
    images/    — Sorties graphiques des runs

Usage:
    # Lancer une config spécifique:
    python configs/config_physics_tau.py
    python configs/config_physics_velocity_weight.py
    python configs/configs_tau_study.py

    # Lancer une étude comparative:
    python runs/run_tau_comparison_physics.py --diameter 0.004
    python runs/run_velocity_weight_comparison.py --diameter 0.004
"""
print("📂 Structure du projet:")
print("   configs/     — Configurations d'expériences")
print("   runs/        — Scripts d'études comparatives")
print("   images/      — Sorties graphiques")
print()
print("📋 Configurations disponibles:")
print("   python configs/config_physics_tau.py              — Étude tau (physics)")
print("   python configs/config_physics_velocity_weight.py  — Étude velocity_weight (physics)")
print("   python configs/configs_tau_study.py               — Étude tau (cylindrical)")
print("   python configs/config_octree_oblique.py           — Sweep méthodes obliques octree")
print()
print("📊 Runs comparatifs:")
print("   python runs/run_tau_comparison_physics.py --diameter 0.004")
print("   python runs/run_tau_comparison.py")
print("   python runs/run_velocity_weight_comparison.py --diameter 0.004")
