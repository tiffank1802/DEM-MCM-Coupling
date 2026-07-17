"""
postprocess.py — Post-traitement automatisé des expériences DEM/Markov.

Usage :
  # --- Style argparse classique ---
  python postprocess.py --folder voronoi_20cells_NLT30_step10_dt2_tau157_start250
  python postprocess.py --keywords voronoi_20cells NLT30
  python postprocess.py --category voronoi_simulations
  python postprocess.py --folder mon_exp --top-states 5 --bucket-prefix _Good/Experiment

  # --- Style sous-commandes ---
  python postprocess.py single voronoi_20cells_NLT30_step10_dt2_tau157_start250
  python postprocess.py keywords voronoi_20cells NLT30
  python postprocess.py category voronoi_simulations
"""

import argparse
import asyncio
import io
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
from scipy.spatial import ConvexHull

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from matplotlib.lines import Line2D

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
pv.OFF_SCREEN = True

# ── Imports projet ────────────────────────────────────────────────────────────
from huggingface_hub import HfFileSystem

from DEM_MCM1.src.bucket_io import (
    ALL_CATEGORIES,
    BUCKET_ID,
    CATEGORY_MAP,
    PostprocessingBucketUploader,
    get_simulation_category,
)
from DEM_MCM1.src.utils import load_parquet_as_timestep_dict

fs = HfFileSystem()

# ═════════════════════════════════════════════════════════════════════════════
# STYLE MATPLOTLIB GLOBAL
# ═════════════════════════════════════════════════════════════════════════════
STYLE = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "lines.linewidth": 1.6,
}
plt.rcParams.update(STYLE)

# Palette cohérente : DEM = bleu/orange selon espèce, Markov = rouge/vert
SPECIES_COLORS = {
    "small": {"dem": "#2196F3", "markov": "#E53935"},
    "large": {"dem": "#FF9800", "markov": "#43A047"},
}
DEFAULT_COLOR = {"dem": "#607D8B", "markov": "#9C27B0"}


# ═════════════════════════════════════════════════════════════════════════════
# RECHERCHE DE DOSSIER DANS LE BUCKET
# ═════════════════════════════════════════════════════════════════════════════


def find_experiment_paths(
    bucket_base_hf: str,
    folder_name: str | None = None,
    keywords: list[str] | None = None,
) -> list[tuple[str, str]]:
    """
    Retourne une liste de (hf_path, short_name) pour TOUS les dossiers d'expérience trouvés.
    Cherche dans tous les sous-dossiers de catégorie, puis à la racine (fallback).

    Priority : folder_name exact > keywords
    """

    def _search_in(base: str) -> list[str]:
        """Retourne TOUS les chemins correspondants dans base."""
        results = []
        try:
            items = fs.ls(base)
        except FileNotFoundError:
            return results

        for item in items:
            if item["type"] != "directory":
                continue
            name = item["name"].split("/")[-1]

            # Vérifier si ça correspond aux critères
            matches = False
            if (folder_name and name == folder_name) or (
                keywords and all(k in name for k in keywords)
            ):
                matches = True

            if matches:
                results.append(item["name"])

        return results

    all_found = []

    # 1. Chercher dans chaque sous-dossier de catégorie
    for cat in ALL_CATEGORIES:
        found = _search_in(f"{bucket_base_hf}/{cat}")
        for path in found:
            short = path.split("/")[-1]
            all_found.append((f"hf://{path}", short))

    # 2. Fallback racine (avant migration) - seulement si rien trouvé dans les catégories
    if not all_found:
        found = _search_in(bucket_base_hf.replace("hf://", ""))
        for path in found:
            short = path.split("/")[-1]
            all_found.append((f"hf://{path}", short))

    if not all_found:
        raise FileNotFoundError(
            f"Aucun dossier trouvé pour "
            f"{'folder=' + folder_name if folder_name else 'keywords=' + str(keywords)}"
        )

    return all_found


def list_category_paths(bucket_base_hf: str, category: str) -> list[tuple[str, str]]:
    """Retourne [(hf_path, short_name), ...] pour tous les dossiers d'une catégorie."""
    results = []
    cat_base = f"{bucket_base_hf}/{category}"
    try:
        # bucket_base_hf commence par hf://, fs.ls veut le chemin sans hf://
        items = fs.ls(cat_base.replace("hf://", ""))
    except FileNotFoundError:
        return results
    for item in items:
        if item["type"] == "directory":
            short = item["name"].split("/")[-1]
            results.append((f"hf://{item['name']}", short))
    return sorted(results, key=lambda x: x[1])


# ═════════════════════════════════════════════════════════════════════════════
# CHARGEMENT
# ═════════════════════════════════════════════════════════════════════════════


def _load_npy(path_hf: str, filename: str) -> np.ndarray:
    with fs.open(f"{path_hf}/{filename}", "rb") as f:
        return np.load(io.BytesIO(f.read()))


def _load_json(path_hf: str, filename: str) -> dict:
    with fs.open(f"{path_hf}/{filename}", "r") as f:
        return json.load(f)


def load_experiment(path_hf: str) -> dict:
    """
    Charge config, stats et données par espèce depuis le bucket.
    Détecte automatiquement le format homogène vs inhomogène.
    Retourne un dict complet avec les matrices et méta-données.
    """
    config = _load_json(path_hf, "config.json")
    stats = _load_json(path_hf, "stats.json")
    species_list = stats.get("species_list", ["small", "large"])

    # Détection du format inhomogène
    inhomogeneous = False
    inhomogeneous_metadata = None
    try:
        inhomogeneous_metadata = _load_json(path_hf, "inhomogeneous_metadata.json")
        inhomogeneous = True
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    species_data = {}
    for sp in species_list:
        if inhomogeneous:
            P_blocks = _load_npy(path_hf, f"P_blocks_{sp}.npy")  # (n_blocks, S, S)
            species_data[sp] = {
                "P_raw": P_blocks[0],  # 1ère matrice pour compatibilité
                "P_blocks": P_blocks,
                "S_matrix": _load_npy(path_hf, f"S_matrix_{sp}.npy"),
                "times": _load_npy(path_hf, f"times_{sp}.npy"),
            }
        else:
            species_data[sp] = {
                "P_raw": _load_npy(path_hf, f"transitionmatrix_{sp}.npy"),
                "S_matrix": _load_npy(path_hf, f"S_matrix_{sp}.npy"),
                "times": _load_npy(path_hf, f"times_{sp}.npy"),
            }

    # Charger la matrice des états par particule (pour fig_mesh)
    matrix = None
    try:
        matrix = _load_npy(path_hf, "states_matrix.npy")
    except FileNotFoundError:
        print(f"   ⚠️  states_matrix.npy introuvable dans {path_hf}")

    return {
        "config": config,
        "stats": stats,
        "species": species_data,
        "matrix": matrix,
        "inhomogeneous": inhomogeneous,
        "inhomogeneous_metadata": inhomogeneous_metadata,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PRÉPARATION MATHÉMATIQUE
# ═════════════════════════════════════════════════════════════════════════════


def clean_transition_matrix(P: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    P_clean = P.copy()
    col_sums = P_clean.sum(axis=0)
    activated = col_sums >= threshold
    P_clean[:, ~activated] = 0.0
    safe = col_sums.copy()
    safe[~activated] = 1.0
    P_clean = P_clean / safe
    return P_clean, activated


def propagate_markov(S0: np.ndarray, P: np.ndarray, times: np.ndarray, start_idx: int, tau: int, activated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_start = np.searchsorted(times, start_idx)
    times_full = times[row_start:]
    markov_idx = np.arange(0, len(times_full), tau)
    times_markov = times_full[markov_idx]
    S = S0.copy().astype(float)
    S[~activated] = 0.0
    traj = [S.copy()]
    for _ in range(1, len(markov_idx)):
        S = P @ S
        traj.append(S.copy())
    return np.array(traj), times_markov


def propagate_markov_inhomogeneous(S0: np.ndarray, P_blocks: np.ndarray, times: np.ndarray, start_idx: int, tau: int, activated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagation markovienne avec matrices de transition variables dans le temps.

    P_blocks : ndarray (n_blocks, n_states, n_states) — une matrice P_k par NLT.
    À chaque pas de temps, la matrice utilisée dépend du bloc en cours :
        block_idx = min(t // block_size, n_blocks - 1)

    Returns:
        traj_markov : ndarray (n_steps+1, n_states)
        times_markov : ndarray (n_steps+1,)
    """
    row_start = np.searchsorted(times, start_idx)
    times_full = times[row_start:]
    markov_idx = np.arange(0, len(times_full), tau)
    times_markov = times_full[markov_idx]
    n_steps = len(markov_idx) - 1  # nombre de propagations
    n_blocks = len(P_blocks)

    # Taille d'un bloc : nombre de pas de markov par bloc
    block_size = max(1, n_steps // n_blocks) if n_blocks > 0 else 1

    S = S0.copy().astype(float)
    S[~activated] = 0.0
    traj = [S.copy()]
    for t in range(1, len(markov_idx)):
        block_idx = min((t - 1) // block_size, n_blocks - 1)
        S = P_blocks[block_idx] @ S
        traj.append(S.copy())
    return np.array(traj), times_markov


def prepare_species(exp: dict) -> dict:
    """
    Nettoie P, calcule les cellules activées, propage Markov, extrait DEM tronqué.
    Retourne un dict enrichi par espèce.
    """
    config = exp["config"]
    # start  = config.get("start_index", 250)
    start = 0
    tau = config.get("tau", 50)
    out = {}

    for sp, data in exp["species"].items():
        P_clean, activated = clean_transition_matrix(data["P_raw"])
        row_start = np.searchsorted(data["times"], start)
        S0 = data["S_matrix"][row_start].astype(float)
        traj, times_markov = propagate_markov(
            S0, P_clean, data["times"], start, tau, activated
        )
        out[sp] = {
            "P": P_clean,
            "P_raw": data["P_raw"],
            "S_matrix": data["S_matrix"],
            "times": data["times"],
            "S_dem": data["S_matrix"][row_start:],
            "times_dem": data["times"][row_start:],
            "traj_markov": traj,
            "times_markov": times_markov,
            "activated": activated,
        }
    return out


def prepare_species_inhomogeneous(exp: dict) -> dict:
    """
    Version inhomogène de prepare_species.

    Utilise P_blocks (une matrice par NLT) pour la propagation au lieu
    d'une matrice unique. Le format de sortie est identique à prepare_species
    pour que les figures existantes fonctionnent sans modification.

    - "P" : première matrice P_blocks[0] (proprement nettoyée)
    - "P_blocks" : toutes les matrices (spécifique inhomogène)
    - "traj_markov" : propagation avec matrices variables
    """
    config = exp["config"]
    start = 0
    tau = config.get("tau", 50)
    out = {}

    for sp, data in exp["species"].items():
        P_blocks = data.get("P_blocks")
        if P_blocks is None:
            raise KeyError(
                f"Données inhomogènes attendues pour '{sp}' "
                f"mais 'P_blocks' est absent. Utiliser prepare_species() "
                f"pour le format homogène."
            )

        # Nettoyer la première matrice comme référence
        P_clean, activated = clean_transition_matrix(P_blocks[0])
        row_start = np.searchsorted(data["times"], start)
        S0 = data["S_matrix"][row_start].astype(float)
        traj, times_markov = propagate_markov_inhomogeneous(
            S0, P_blocks, data["times"], start, tau, activated
        )
        out[sp] = {
            "P": P_clean,  # 1ère matrice (compatibilité figures)
            "P_raw": P_blocks[0],  # 1ère matrice brute
            "P_blocks": P_blocks,  # toutes les matrices
            "S_matrix": data["S_matrix"],
            "times": data["times"],
            "S_dem": data["S_matrix"][row_start:],
            "times_dem": data["times"][row_start:],
            "traj_markov": traj,  # propagé avec matrices variables
            "times_markov": times_markov,
            "activated": activated,
        }
    return out


def _short_label(name: str, all_names: list[str]) -> str:
    """
    Extrait uniquement les parties du nom qui varient entre les expériences.
    Fallback : extrait les paramètres clés connus (tau, start, step, dt, NLT).
    """
    # Découper chaque nom en tokens (séparateur "_")
    parts_list = [n.split("_") for n in all_names]

    # Trouver les positions qui varient
    min_len = min(len(p) for p in parts_list)
    varying = []
    for i in range(min_len):
        values = {p[i] for p in parts_list}
        if len(values) > 1:
            varying.append(i)

    parts = name.split("_")

    if varying:
        # Garder uniquement les tokens qui varient
        return "_".join(parts[i] for i in varying if i < len(parts))

    # Fallback : extraire les paramètres clés avec regex
    tokens = []
    for key in ["tau", "start", "step", "dt", "NLT"]:
        m = re.search(rf"{key}(\d+)", name)
        if m:
            tokens.append(f"{key}{m.group(1)}")
    return "_".join(tokens) if tokens else name[:20]


def _common_prefix(all_names: list[str]) -> str:
    """Retourne la partie du nom identique entre toutes les expériences."""
    parts_list = [n.split("_") for n in all_names]
    min_len = min(len(p) for p in parts_list)
    common = []
    for i in range(min_len):
        values = {p[i] for p in parts_list}
        if len(values) == 1:
            common.append(parts_list[0][i])
        else:
            break  # dès qu'un token varie, on s'arrête
    return "_".join(common)


def _extract_model_type(names: list[str]) -> str:
    """Extrait le type de modèle (voronoi, cartesian, gmm, ...) depuis les noms."""
    known = list(CATEGORY_MAP.keys())
    for name in names:
        for model in known:
            if model in name.lower():
                return model.replace("_", " ").capitalize()
    return "unknown"


def find_all_experiments_by_keywords(
    bucket_base_hf: str,
    keywords: list[str],
) -> list[tuple[str, str]]:
    """
    Retourne [(hf_path, short_name), ...] pour TOUS les dossiers
    dont le nom contient tous les mots-clés, dans toutes les catégories.
    """
    results = []

    def _search_in(base: str) -> None:
        try:
            items = fs.ls(base)
        except FileNotFoundError:
            return
        for item in items:
            if item["type"] != "directory":
                continue
            name = item["name"].split("/")[-1]
            if all(k in name for k in keywords):
                results.append((f"hf://{item['name']}", name))

    for cat in ALL_CATEGORIES:
        _search_in(f"{bucket_base_hf}/{cat}")

    # Fallback racine
    _search_in(bucket_base_hf.replace("hf://", ""))

    return sorted(results, key=lambda x: x[1])


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES DE COMPARAISON
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# RÉFÉRENCE DEM POUR LA COMPARAISON
# ═════════════════════════════════════════════════════════════════════════════


def get_dem_reference(
    all_species_data: dict[str, dict],
    dem_ref: str,
) -> tuple[str, dict]:
    names = list(all_species_data.keys())

    # ── Nom exact ────────────────────────────────────────────────────────────
    if dem_ref in all_species_data:
        short_part = _short_label(dem_ref, names)
        common = _common_prefix(names)
        label = f"{common}_{short_part}"
        print(f"   📌 DEM référence : {label}")
        return label, all_species_data[dem_ref]

    # ── Moyenne ──────────────────────────────────────────────────────────────
    if dem_ref == "mean":
        all_species = sorted({sp for sd in all_species_data.values() for sp in sd})
        mean_data: dict[str, dict] = {}
        for sp in all_species:
            sds = [sd[sp] for sd in all_species_data.values() if sp in sd]
            n = min(len(d["S_dem"]) for d in sds)
            S_mean = np.mean([d["S_dem"][:n] for d in sds], axis=0)
            mean_data[sp] = {
                **sds[0],
                "S_dem": S_mean,
                "times_dem": sds[0]["times_dem"][:n],
            }
        label = f"{_common_prefix(names)}_mean"
        print(f"   📌 DEM référence : {label}")
        return label, mean_data

    # ── Fallback : première expérience ───────────────────────────────────────
    if dem_ref != "first":
        print(f"   ⚠️  dem-ref '{dem_ref}' introuvable → fallback sur 'first'")
    name = names[0]
    short_part = _short_label(name, names)
    common = _common_prefix(names)
    label = f"{common}_{short_part}"
    print(f"   📌 DEM référence : {label}")
    names = list(all_species_data.keys())
    return label, all_species_data[name]


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES DE COMPARAISON
# ═════════════════════════════════════════════════════════════════════════════


def fig_compare_rsd(
    all_species_data: dict[str, dict],
    out_dir: Path,
    dem_ref_label: str = "",
    dem_ref_data: dict | None = None,
    model_type: str = "",
) -> None:
    """
    RSD de concentration (cross-espèce) — compare la variation du ratio
    small/(small+large) entre les cellules.
    - Une courbe DEM de référence (grise).
    - Une courbe Markov par expérience (couleurs viridis).
    """
    all_species = sorted({sp for sd in all_species_data.values() for sp in sd})
    if len(all_species) < 2:
        print("   ⚠️  Moins de 2 espèces — fig_compare_rsd ignorée.")
        return

    sp_a, sp_b = all_species[1], all_species[0]
    # sp_a, sp_b = all_species[0], all_species[1]
    names = list(all_species_data.keys())

    # ── Détection du paramètre NLT ─────────────────────────────────────────
    import re

    nlt_values = []
    for name in names:
        m = re.search(r"NLT(\d+)", name)
        nlt_values.append(int(m.group(1)) if m else None)

    is_nlt_comparison = (
        all(v is not None for v in nlt_values) and len(set(nlt_values)) > 1
    )

    # ── Calcul de l'erreur relative entre matrices P (NLT seulement) ───────
    error_text_parts = []
    if is_nlt_comparison:
        # Trier les expériences par NLT croissant
        sorted_indices = sorted(range(len(names)), key=lambda i: nlt_values[i])
        [names[i] for i in sorted_indices]
        sorted_values = [nlt_values[i] for i in sorted_indices]

        for sp in [sp_a, sp_b]:
            parts = []
            for j in range(1, len(sorted_indices)):
                idx_prev = sorted_indices[j - 1]
                idx_curr = sorted_indices[j]
                names[idx_prev]
                names[idx_curr]

                # Récupérer les matrices P des deux expériences pour cette espèce
                P_prev = list(all_species_data.values())[idx_prev][sp]["P"]
                P_curr = list(all_species_data.values())[idx_curr][sp]["P"]

                norm_diff = np.sum(np.abs(P_curr - P_prev))
                norm_prev = np.sum(np.abs(P_prev))
                rel_err = norm_diff / norm_prev if norm_prev > 0 else 0.0

                parts.append(
                    f"{sorted_values[j - 1]}→{sorted_values[j]}: {rel_err:.4f}"
                )

            error_text_parts.append(f"{sp}: {' | '.join(parts)}")

    # ── Construction de la figure ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle(f"Comparaison RSD de concentration — {model_type}", fontweight="bold")

    # Palette viridis (triée par NLT si c'est une comparaison NLT)
    if is_nlt_comparison:
        ordered_names = sorted(names, key=lambda n: nlt_values[names.index(n)])
    else:
        ordered_names = names

    cmap = plt.cm.viridis
    colors = {
        n: cmap(i / max(len(ordered_names) - 1, 1)) for i, n in enumerate(ordered_names)
    }

    # ── DEM référence (concentration) ────────────────────────────────────
    rsd_d = None
    if dem_ref_data and sp_a in dem_ref_data and sp_b in dem_ref_data:
        d_ref_a = dem_ref_data[sp_a]
        d_ref_b = dem_ref_data[sp_b]
        n_d = min(len(d_ref_a["times_dem"]), len(d_ref_b["times_dem"]))
        rsd_d = rsd_concentration(
            d_ref_a["S_dem"][:n_d],
            d_ref_b["S_dem"][:n_d],
            d_ref_a["activated"],
            d_ref_b["activated"],
        )
        # ax.plot(  # commenté intentionnellement
        #     d_ref_a["times_dem"][:n_d], rsd_d,
        #     "-", color="#AAAAAA", lw=1.5, alpha=0.4,
        #     label=f"DEM — {dem_ref_label}",
        #     zorder=5,
        # )

    # ── Markov : une courbe par expérience ───────────────────────────────
    for name in ordered_names:
        species_data = all_species_data[name]
        if sp_a not in species_data or sp_b not in species_data:
            continue
        d_a = species_data[sp_a]
        d_b = species_data[sp_b]
        n_m = min(len(d_a["times_markov"]), len(d_b["times_markov"]))
        rsd_m = rsd_concentration(
            d_a["traj_markov"][:n_m],
            d_b["traj_markov"][:n_m],
            d_a["activated"],
            d_b["activated"],
        )

        # RMSE entre la courbe Markov et la référence DEM
        label = _short_label(name, names)
        if rsd_d is not None:
            n_min = min(len(rsd_m), len(rsd_d))
            if n_min > 0:
                rmse = np.sqrt(np.mean((rsd_m[:n_min] - rsd_d[:n_min]) ** 2))
                label += f"  (RMSE={rmse:.4f})"

        ax.plot(
            d_a["times_markov"][:n_m],
            rsd_m,
            "-P",
            color=colors[name],
            lw=1.6,
            alpha=0.8,
            label=label,
            zorder=3,
        )

    ax.set_title(f"RSD de concentration C({sp_a})")
    ax.set_xlabel("Temps (centièmes de seconde)")
    ax.set_ylabel("RSD de concentration")
    ax.legend(
        fontsize=7,
        ncol=1,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.5,
        handletextpad=0.4,
    )
    ax.set_ylim(bottom=0)

    # ── Affichage de l'erreur relative P (NLT seulement) ───────────────────
    if is_nlt_comparison and error_text_parts:
        error_text = "Erreur relative P (norme 1):\n" + "\n".join(error_text_parts)
        ax.text(
            0.02,
            0.02,
            error_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox={
                "boxstyle": "round,pad=0.4",
                "fc": "#f8f9fa",
                "ec": "#b0bec5",
                "alpha": 0.9,
            },
        )

    fig.tight_layout()
    fig.savefig(out_dir / "compare_rsd.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 compare_rsd.png  (RSD de concentration)")


def fig_compare_states(
    all_species_data: dict[str, dict],
    out_dir: Path,
    k: int = 3,
    dem_ref_label: str = "",
    dem_ref_data: dict | None = None,
    model_type: str = "",
) -> None:
    """
    États des k cellules les plus peuplées.
    - Une courbe DEM de référence (noire, épaisse).
    - Une courbe Markov par expérience (couleurs tab10).
    """
    all_species = sorted({sp for sd in all_species_data.values() for sp in sd})
    names = list(all_species_data.keys())
    cmap = plt.cm.tab10
    colors = {n: cmap(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}

    for sp in all_species:
        # Union des top-k cellules de toutes les expériences
        top_cells_set: set[int] = set()
        for sd in all_species_data.values():
            if sp not in sd:
                continue
            mean_occ = sd[sp]["S_dem"].mean(axis=0)
            top_cells_set.update(np.argsort(mean_occ)[::-1][:k].tolist())
        cells = sorted(top_cells_set)

        n = len(cells)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.5 * ncols, 4 * nrows),
            squeeze=False,
        )
        fig.suptitle(
            f"Comparaison états — {model_type} — espèce '{sp}'",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        for idx, cell in enumerate(cells):
            ax = axes[idx // ncols][idx % ncols]

            # ── DEM référence ────────────────────────────────────────────
            if dem_ref_data and sp in dem_ref_data:
                d_ref = dem_ref_data[sp]
                if cell < d_ref["S_dem"].shape[1]:
                    ax.plot(
                        d_ref["times_dem"],
                        d_ref["S_dem"][:, cell],
                        "-",
                        color="#AAAAAA",
                        lw=1.5,
                        alpha=0.4,
                        label=f"DEM — {dem_ref_label}",
                        zorder=5,
                    )

            # ── Markov : une courbe par expérience ───────────────────────
            for name, sd in all_species_data.items():
                if sp not in sd:
                    continue
                d = sd[sp]
                if cell < d["traj_markov"].shape[1]:
                    ax.plot(
                        d["times_markov"],
                        d["traj_markov"][:, cell],
                        "-",
                        color=colors[name],
                        lw=1.6,
                        alpha=0.8,
                        label=_short_label(name, names),
                        zorder=3,
                    )

            ax.set_title(f"Cellule {cell}")
            ax.set_xlabel("Temps (centièmes de seconde)")
            ax.set_ylabel("Nb particules")
            ax.legend(
                fontsize=6,
                ncol=1,
                borderpad=0.4,
                labelspacing=0.3,
                handlelength=1.5,
                handletextpad=0.4,
            )
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout()
        fname = f"compare_etats_{sp}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"   💾 {fname}")


def fig_compare_n_particles(
    all_species_data: dict[str, dict],
    out_dir: Path,
    dem_ref_label: str = "",
    dem_ref_data: dict | None = None,
    model_type: str = "",
) -> None:
    """
    Nombre total de particules par espèce.
    - Une courbe DEM de référence (noire, épaisse).
    - Une courbe Markov par expérience (couleurs tab10).
    """
    all_species = sorted({sp for sd in all_species_data.values() for sp in sd})
    names = list(all_species_data.keys())
    cmap = plt.cm.tab10
    colors = {n: cmap(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}

    fig, axes = plt.subplots(
        1,
        len(all_species),
        figsize=(8.5 * len(all_species), 5),
        squeeze=False,
    )
    fig.suptitle(f"Nombre total de particules — {model_type}", fontweight="bold")

    for col, sp in enumerate(all_species):
        ax = axes[0][col]

        # ── DEM référence ────────────────────────────────────────────────
        if dem_ref_data and sp in dem_ref_data:
            d_ref = dem_ref_data[sp]
            n_ref = d_ref["S_dem"][:, d_ref["activated"]].sum(axis=1)
            ax.plot(
                d_ref["times_dem"],
                n_ref,
                "-",
                color="#AAAAAA",
                lw=1.5,
                alpha=0.4,
                label=f"DEM — {dem_ref_label}",
                zorder=5,
            )

        # ── Markov : une courbe par expérience ───────────────────────────
        for name, sd in all_species_data.items():
            if sp not in sd:
                continue
            d = sd[sp]
            n_markov = d["traj_markov"][:, d["activated"]].sum(axis=1)
            ax.plot(
                d["times_markov"],
                n_markov,
                "-",
                color=colors[name],
                lw=1.6,
                alpha=0.8,
                label=_short_label(name, names),
                zorder=3,
            )

        ax.set_title(f"N particules — espèce '{sp}'")
        ax.set_xlabel("Temps (centièmes de seconde)")
        ax.set_ylabel("N particules")
        ax.legend(
            fontsize=6,
            ncol=1,
            borderpad=0.4,
            labelspacing=0.3,
            handlelength=1.5,
            handletextpad=0.4,
        )
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "compare_n_particules.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 compare_n_particules.png")


def fig_compare_teneur(
    all_species_data: dict[str, dict],
    out_dir: Path,
    dem_ref_label: str = "",
    dem_ref_data: dict | None = None,
    model_type: str = "",
    max_cells: int = 9,
) -> None:
    """
    Comparaison de la teneur (fraction de petites particules) par cellule.
    - DEM référence : traits gris fins.
    - Une courbe Markov par expérience (couleurs viridis).
    """
    all_species = sorted({sp for sd in all_species_data.values() for sp in sd})
    if len(all_species) < 2:
        print("   ⚠️  Moins de 2 espèces — fig_compare_teneur ignorée.")
        return

    sp_a, sp_b = all_species[1], all_species[0]
    # sp_a, sp_b = all_species[0], all_species[1]
    names = list(all_species_data.keys())

    # ── Cellules activées (top max_cells par occupation DEM) ──────────────
    if dem_ref_data and sp_a in dem_ref_data and sp_b in dem_ref_data:
        ref_a, ref_b = dem_ref_data[sp_a], dem_ref_data[sp_b]
        activated = ref_a["activated"]
    else:
        first_key = next(iter(all_species_data.keys()))
        ref_a = all_species_data[first_key][sp_a]
        ref_b = all_species_data[first_key][sp_b]
        activated = ref_a["activated"]

    active_indices = np.where(activated)[0]
    if len(active_indices) == 0:
        print("   ⚠️  Aucune cellule activée — fig_compare_teneur ignorée.")
        return

    # Trier par occupation totale moyenne (small + large) dans la DEM référence
    occ_total = np.asarray(ref_a["S_dem"]).squeeze()[:, active_indices].mean(
        axis=0
    ) + np.asarray(ref_b["S_dem"]).squeeze()[:, active_indices].mean(axis=0)
    sorted_cells = active_indices[np.argsort(occ_total)[::-1]]
    cells_to_plot = sorted_cells[:max_cells]
    n_cells = len(cells_to_plot)
    ncols = min(3, n_cells)
    nrows = (n_cells + ncols - 1) // ncols

    cmap_exp = plt.cm.viridis
    exp_colors = {n: cmap_exp(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"Comparaison de la teneur — {model_type}",
        fontweight="bold",
        fontsize=14,
    )

    for idx, cell in enumerate(cells_to_plot):
        ax = axes[idx // ncols][idx % ncols]

        # ── DEM référence ────────────────────────────────────────────────
        if dem_ref_data and sp_a in dem_ref_data and sp_b in dem_ref_data:
            d_ref_a = dem_ref_data[sp_a]
            d_ref_b = dem_ref_data[sp_b]
            t_d = np.asarray(d_ref_a["times_dem"]).ravel()
            S_d_a = np.asarray(d_ref_a["S_dem"]).squeeze()
            S_d_b = np.asarray(d_ref_b["S_dem"]).squeeze()
            n_d = min(len(t_d), len(np.asarray(d_ref_b["times_dem"]).ravel()))

            tot_d = S_d_a[:n_d, cell] + S_d_b[:n_d, cell]
            teneur_d = np.divide(
                S_d_a[:n_d, cell], tot_d, out=np.zeros(n_d), where=tot_d != 0
            )

            ax.plot(
                t_d[:n_d],
                teneur_d,
                "-",
                color="#AAAAAA",
                lw=1.5,
                alpha=0.4,
                label=f"DEM — {dem_ref_label}",
                zorder=5,
            )

        # ── Markov : une courbe par expérience ───────────────────────────
        for name, species_data in all_species_data.items():
            if sp_a not in species_data or sp_b not in species_data:
                continue
            d_a = species_data[sp_a]
            d_b = species_data[sp_b]
            t_m = np.asarray(d_a["times_markov"]).ravel()
            S_m_a = np.asarray(d_a["traj_markov"]).squeeze()
            S_m_b = np.asarray(d_b["traj_markov"]).squeeze()
            n_m = min(len(t_m), len(np.asarray(d_b["times_markov"]).ravel()))

            tot_m = S_m_a[:n_m, cell] + S_m_b[:n_m, cell]
            teneur_m = np.divide(
                S_m_a[:n_m, cell], tot_m, out=np.zeros(n_m), where=tot_m != 0
            )

            label = _short_label(name, names)
            ax.plot(
                t_m[:n_m],
                teneur_m,
                "-",
                color=exp_colors[name],
                lw=1.6,
                alpha=0.8,
                label=label,
                zorder=3,
            )

        ax.set_title(f"Cellule {cell}")
        ax.set_xlabel("Temps (centièmes de seconde)")
        ax.set_ylabel(f"Teneur en {sp_a}")
        ax.set_ylim(0, 1)
        ax.legend(
            fontsize=6,
            ncol=1,
            borderpad=0.3,
            labelspacing=0.3,
            handlelength=1.5,
            handletextpad=0.4,
        )

    for idx in range(n_cells, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "compare_teneur.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("   💾 compare_teneur.png")


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPARAISON  (remplace l'ancienne run_comparison)
# ═════════════════════════════════════════════════════════════════════════════


def run_comparison(
    experiments: list[tuple[str, str]],
    keywords: list[str],
    bucket_prefix: str = "_Good/Experiment",
    top_states: int = 3,
    dem_ref: str = "first",
) -> None:
    keyword_slug = "_".join(keywords)
    print(f"\n{'═' * 60}")
    print(f"🔀 Comparaison : {keyword_slug}")
    print(f"   {len(experiments)} expériences  |  dem-ref = '{dem_ref}'")
    print(f"{'═' * 60}")

    # ── Chargement ───────────────────────────────────────────────────────────
    all_species_data: dict[str, dict] = {}
    for path_hf, short in experiments:
        print(f"\n   📥 Chargement : {short}")
        try:
            exp = load_experiment(path_hf)
            species_data = prepare_species(exp)
            all_species_data[short] = species_data
        except Exception as e:
            print(f"   ⚠️  {short} ignoré : {e}")
    model_type = _extract_model_type(list(all_species_data.keys()))

    if not all_species_data:
        print("❌ Aucune expérience chargée.")
        return

    # ── Référence DEM ─────────────────────────────────────────────────────────
    dem_ref_label, dem_ref_data = get_dem_reference(all_species_data, dem_ref)

    bucket_subfolder = f"comparaisons/{keyword_slug}"

    with PostprocessingBucketUploader(bucket_subfolder=bucket_subfolder) as tmp:
        out_dir = tmp / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Figures individuelles par expérience ──────────────────────────
        # print("\n📈 Figures individuelles par expérience...")
        # for short, species_data in all_species_data.items():
        #     # États par cellule (par espèce)
        #     for sp, sd in species_data.items():
        #         try:
        #             fig_states_by_species(sp, sd, short, out_dir)
        #         except Exception as e:
        #             print(f"   ⚠️  {short}/{sp} ignoré : {e}")
        #     # RSD individuel (DEM vs Markov)
        #     try:
        #         fig_rsd(species_data, short, out_dir)
        #     except Exception as e:
        #         print(f"   ⚠️  RSD {short} ignoré : {e}")
        #     # Concentration individuelle (DEM vs Markov)
        #     try:
        #         fig_concentration(species_data, short, out_dir)
        #     except Exception as e:
        #         print(f"   ⚠️  Concentration {short} ignorée : {e}")

        # ── Figures de comparaison ────────────────────────────────────────
        print("\n📊 Figures de comparaison...")
        fig_compare_rsd(
            all_species_data,
            out_dir,
            dem_ref_label=dem_ref_label,
            dem_ref_data=dem_ref_data,
            model_type=model_type,
        )
        fig_compare_states(
            all_species_data,
            out_dir,
            k=top_states,
            dem_ref_label=dem_ref_label,
            dem_ref_data=dem_ref_data,
            model_type=model_type,
        )
        # fig_compare_n_particles(
        #     all_species_data, out_dir,
        #     dem_ref_label=dem_ref_label, dem_ref_data=dem_ref_data,
        #     model_type=model_type,
        # )
        fig_compare_teneur(
            all_species_data,
            out_dir,
            dem_ref_label=dem_ref_label,
            dem_ref_data=dem_ref_data,
            model_type=model_type,
            max_cells=top_states,
        )

    print(f"\n✅ Comparaison '{keyword_slug}' — terminée.\n")


# ═════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ═════════════════════════════════════════════════════════════════════════════


def rsd_from_S(S: np.ndarray, activated: np.ndarray) -> np.ndarray:
    S_a = S[:, activated]
    mean = S_a.mean(axis=1)
    std = S_a.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)


def rsd_concentration(S_small: np.ndarray, S_large: np.ndarray, act_s: np.ndarray, act_l: np.ndarray) -> np.ndarray:
    act = act_s & act_l
    total = S_small[:, act] + S_large[:, act]
    C = np.where(total > 0, S_small[:, act] / total, 0.0)
    mean = C.mean(axis=1)
    std = C.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)


def entropy_from_S(S: np.ndarray, activated: np.ndarray) -> np.ndarray:
    S_a = S[:, activated]
    N = S_a.sum(axis=1, keepdims=True)
    N = np.where(N > 0, N, 1.0)
    p = S_a / N
    H = np.zeros(len(S_a))
    for t in range(len(S_a)):
        pt = p[t]
        m = pt > 0
        if m.any():
            H[t] = -np.sum(pt[m] * np.log(pt[m]))
    return H


def entropy_concentration(S_small: np.ndarray, S_large: np.ndarray, act_s: np.ndarray, act_l: np.ndarray) -> np.ndarray:
    act = act_s & act_l
    total = S_small[:, act] + S_large[:, act]
    total = np.where(total > 0, total, 1.0)
    C = S_small[:, act] / total
    H = np.zeros(len(C))
    for t in range(len(C)):
        Ct = C[t]
        v = (Ct > 0) & (Ct < 1)
        if v.any():
            Cv = Ct[v]
            H[t] = -np.sum(Cv * np.log(Cv) + (1 - Cv) * np.log(1 - Cv))
    return H


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════


def _dem_label(sp: str) -> str:
    return f"DEM — {sp}"


def _markov_label(sp: str) -> str:
    return f"Markov — {sp}"


def _colors(sp: str) -> str:
    return SPECIES_COLORS.get(sp, DEFAULT_COLOR)


# ── États : sous-figure par cellule ──────────────────────────────────────────


def _plot_states_grid(
    cell_indices: list[int],
    sp: str,
    sp_data: dict,
    short_name: str,
    title_suffix: str,
    out_dir: Path,
    filename: str,
) -> None:
    """
    Trace une grille de sous-figures (max 3 colonnes).
    Chaque sous-figure = une cellule.
    DEM tracé EN PREMIER (au premier plan via zorder=3), Markov derrière (zorder=2).
    """
    n = len(cell_indices)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    colors = _colors(sp)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"États par partition — espèce '{sp}' — {title_suffix}\n{short_name}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    for idx, cell in enumerate(cell_indices):
        ax = axes[idx // ncols][idx % ncols]
        t_d = sp_data["times_dem"]
        S_d = sp_data["S_dem"]
        t_m = sp_data["times_markov"]
        S_m = sp_data["traj_markov"]

        # ── [FIX] DEM tracé en premier — au premier plan (zorder=3) ──
        ax.plot(
            t_d,
            S_d[:, cell],
            "-",
            color=colors["dem"],
            linewidth=2.0,
            alpha=0.9,
            label=_dem_label(sp),
            zorder=3,
        )
        # ── Markov tracé en second — derrière (zorder=2) ──
        ax.plot(
            t_m,
            S_m[:, cell],
            "o--",
            color=colors["markov"],
            markersize=4,
            linewidth=1.4,
            alpha=0.85,
            label=_markov_label(sp),
            zorder=2,
        )

        mean_occ = S_d[:, cell].mean()
        ax.set_title(f"Cellule {cell}  (occupation moy. DEM : {mean_occ:.1f})", pad=6)
        ax.set_xlabel("Temps (centièmes de seconde)")
        ax.set_ylabel("Nb particules")
        ax.legend(loc="upper right")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    # Masquer les axes vides
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {filename}")


def fig_states_top3_index(
    sp: str, sp_data: dict, short_name: str, out_dir: Path
) -> None:
    """Cellules 0, 1, 2 (par index)."""
    n_states = sp_data["traj_markov"].shape[1]
    cells = list(range(min(3, n_states)))
    _plot_states_grid(
        cells,
        sp,
        sp_data,
        short_name,
        title_suffix="3 premiers états (index 0-1-2)",
        out_dir=out_dir,
        filename=f"etats_top3_index_{sp}.png",
    )


def fig_states_top_populated(
    sp: str, sp_data: dict, short_name: str, out_dir: Path, k: int = 6
) -> None:
    """Top-k cellules par occupation moyenne DEM."""
    S_dem = sp_data["S_dem"]
    mean_occ = S_dem.mean(axis=0)
    cells = np.argsort(mean_occ)[::-1][:k].tolist()
    _plot_states_grid(
        cells,
        sp,
        sp_data,
        short_name,
        title_suffix=f"Top {k} cellules les plus peuplées (DEM)",
        out_dir=out_dir,
        filename=f"etats_top{k}_peuplees_{sp}.png",
    )


# ── Matrice de transition ─────────────────────────────────────────────────────


def fig_transition_matrix(
    sp: str, sp_data: dict, short_name: str, out_dir: Path
) -> None:
    P = sp_data["P"]
    n = P.shape[0]
    vmax = np.percentile(P[P > 0], 98) if (P > 0).any() else 1.0

    fig, ax = plt.subplots(figsize=(n * 0.7 + 2, n * 0.7 + 2))
    im = ax.imshow(
        P, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest"
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probabilité de transition", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    ax.set_title(
        f"Matrice de transition P nettoyée — {sp}\n{short_name}",
        fontweight="bold",
    )
    ax.set_xlabel("Cellule source (colonne j)")
    ax.set_ylabel("Cellule destination (ligne i)")

    # Annotations numériques sur toutes les cellules > 0.01
    for i in range(n):
        for j in range(n):
            val = P[i, j]
            if val > 0.01:
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val > vmax * 0.5 else "black",
                )

    # Ticks sans grille minor (suppression des traits parasites)
    ax.set_xticks(range(n))
    ax.set_xticklabels(range(n), fontsize=7, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(range(n), fontsize=7)
    ax.tick_params(which="both", bottom=False, left=False)
    ax.grid(False)

    fig.tight_layout()
    fname = f"matrice_transition_{sp}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname}")


# ── Diagnostic spectral ───────────────────────────────────────────────────────


def fig_spectral_diagnostic(
    sp: str, sp_data: dict, short_name: str, out_dir: Path
) -> None:
    P = sp_data["P"]
    activated = sp_data["activated"]

    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    pi = np.abs(eigvecs[:, 0])
    pi /= pi.sum()
    pi_act = pi[activated]
    rsd_pi = pi_act.std() / pi_act.mean() if pi_act.mean() > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Diagnostic spectral de P — espèce '{sp}'\n{short_name}",
        fontweight="bold",
    )

    # Spectre dans le disque unité
    theta = np.linspace(0, 2 * np.pi, 300)
    axes[0].plot(np.cos(theta), np.sin(theta), "r--", lw=0.8, label="Cercle unité")
    axes[0].scatter(
        eigvals.real,
        eigvals.imag,
        s=25,
        alpha=0.75,
        color="#2196F3",
        edgecolors="white",
        linewidth=0.4,
        zorder=3,
    )
    axes[0].axhline(0, color="grey", lw=0.5)
    axes[0].axvline(0, color="grey", lw=0.5)
    axes[0].set_title("Spectre de P (valeurs propres)")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Re(λ)")
    axes[0].set_ylabel("Im(λ)")

    # Distribution stationnaire π
    axes[1].bar(range(len(pi)), pi, color="#43A047", alpha=0.8, edgecolor="white")
    axes[1].set_title(f"Distribution stationnaire π\n(RSD actif = {rsd_pi:.3f})")
    axes[1].set_xlabel("Cellule")
    axes[1].set_ylabel("π(i)")
    axes[1].axhline(
        1 / activated.sum(),
        color="red",
        lw=1.2,
        linestyle="--",
        label="Uniforme (actif)",
    )
    axes[1].legend()

    # Matrice P (vue compacte)
    im = axes[2].imshow(
        P, aspect="auto", cmap="viridis", vmin=0, interpolation="nearest"
    )
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Matrice P (nettoyée)")
    axes[2].set_xlabel("Cellule source")
    axes[2].set_ylabel("Cellule dest.")

    fig.tight_layout()
    fname = f"diagnostic_spectral_{sp}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname}")


# ── RSD par espèce ────────────────────────────────────────────────────────────


def fig_rsd(species_data: dict, short_name: str, out_dir: Path) -> None:
    species_list = list(species_data.keys())
    n = len(species_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    fig.suptitle(f"RSD global par espèce\n{short_name}", fontweight="bold")

    for i, sp in enumerate(species_list):
        d = species_data[sp]
        colors = _colors(sp)
        ax = axes[0][i]

        rsd_d = rsd_from_S(d["S_dem"], d["activated"])
        rsd_m = rsd_from_S(d["traj_markov"], d["activated"])

        ax.plot(
            d["times_dem"],
            rsd_d,
            "-",
            color=colors["dem"],
            lw=2.0,
            alpha=0.9,
            label=_dem_label(sp),
            zorder=3,
        )
        ax.plot(
            d["times_markov"],
            rsd_m,
            "o--",
            color=colors["markov"],
            markersize=4,
            lw=1.4,
            alpha=0.85,
            label=_markov_label(sp),
            zorder=2,
        )

        ax.set_title(f"RSD — espèce '{sp}'")
        ax.set_xlabel("Temps (centièmes de seconde)")
        ax.set_ylabel("RSD")
        ax.legend()
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "rsd_par_espece.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 rsd_par_espece.png")


# ── Concentration ─────────────────────────────────────────────────────────────


def fig_concentration(species_data: dict, short_name: str, out_dir: Path) -> None:
    sps = list(species_data.keys())
    if len(sps) < 2:
        print("   ⚠️  Moins de 2 espèces — figure concentration ignorée.")
        return

    sp_a, sp_b = sps[0], sps[1]
    da, db = species_data[sp_a], species_data[sp_b]
    n_m = min(len(da["times_markov"]), len(db["times_markov"]))
    n_d = min(len(da["times_dem"]), len(db["times_dem"]))

    rsd_c_d = rsd_concentration(
        da["S_dem"][:n_d],
        db["S_dem"][:n_d],
        da["activated"],
        db["activated"],
    )
    rsd_c_m = rsd_concentration(
        da["traj_markov"][:n_m],
        db["traj_markov"][:n_m],
        da["activated"],
        db["activated"],
    )
    ent_c_d = entropy_concentration(
        da["S_dem"][:n_d],
        db["S_dem"][:n_d],
        da["activated"],
        db["activated"],
    )
    ent_c_m = entropy_concentration(
        da["traj_markov"][:n_m],
        db["traj_markov"][:n_m],
        da["activated"],
        db["activated"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Concentration C({sp_a}) — RSD & Entropie\n{short_name}",
        fontweight="bold",
    )

    for ax, (yd, ym, ylabel, title) in zip(
        axes,
        [
            (rsd_c_d, rsd_c_m, "RSD de C", f"RSD de concentration C({sp_a})"),
            (ent_c_d, ent_c_m, "Entropie de C", f"Entropie de concentration C({sp_a})"),
        ],
    ):
        ax.plot(
            da["times_dem"][:n_d],
            yd,
            "-",
            color="#2196F3",
            lw=2.0,
            alpha=0.9,
            label="DEM",
            zorder=3,
        )
        ax.plot(
            da["times_markov"][:n_m],
            ym,
            "o--",
            color="#E53935",
            markersize=4,
            lw=1.4,
            alpha=0.85,
            label="Markov",
            zorder=2,
        )
        ax.set_title(title)
        ax.set_xlabel("Temps (centièmes de seconde)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "concentration_rsd_entropie.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 concentration_rsd_entropie.png")


def fig_states_by_species(
    sp: str,
    sp_data: dict,
    short_name: str,
    out_dir: Path,
) -> None:
    S_d = np.asarray(sp_data["S_dem"]).squeeze()
    S_m = np.asarray(sp_data["traj_markov"]).squeeze()
    t_d = np.asarray(sp_data["times_dem"]).ravel()
    t_m = np.asarray(sp_data["times_markov"]).ravel()
    activated = sp_data["activated"]

    # Debug temporaire — à supprimer une fois confirmé
    print(f"      S_d={S_d.shape}  S_m={S_m.shape}  t_d={t_d.shape}  t_m={t_m.shape}")

    n_cells = len(activated)
    cmap = plt.cm.tab20
    colors = [cmap(i / max(n_cells - 1, 1)) for i in range(n_cells)]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Par (utiliser les indices entiers, pas le booléen) :
    active_indices = np.where(activated)[0]
    for idx, cell in enumerate(active_indices):
        color = colors[idx]
        ax.plot(
            t_d,
            S_d[:, cell],
            "-",
            color=color,
            linewidth=1.8,
            alpha=0.85,
            label=f"DEM — Cellule {cell}",
            zorder=3,
        )
        ax.plot(
            t_m,
            S_m[:, cell],
            "o--",
            color=color,
            markersize=3,
            linewidth=1.2,
            alpha=0.7,
            label=f"Markov — Cellule {cell}",
            zorder=2,
        )

    ax.set_title(f"États par cellule — espèce '{sp}'\n{short_name}", fontweight="bold")
    ax.set_xlabel("Temps (centièmes de seconde)")
    ax.set_ylabel("Nombre de particules")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fname = f"etats_espece_{sp}_{short_name}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname}")


# ── Entropie totale ───────────────────────────────────────────────────────────


def fig_entropy_total(species_data: dict, short_name: str, out_dir: Path) -> None:
    sps = list(species_data.keys())
    if len(sps) < 2:
        return
    sp_a, sp_b = sps[0], sps[1]
    da, db = species_data[sp_a], species_data[sp_b]
    n_m = min(len(da["times_markov"]), len(db["times_markov"]))
    n_d = min(len(da["times_dem"]), len(db["times_dem"]))

    ent_d = entropy_from_S(da["S_dem"][:n_d], da["activated"]) + entropy_from_S(
        db["S_dem"][:n_d], db["activated"]
    )
    ent_m = entropy_from_S(da["traj_markov"][:n_m], da["activated"]) + entropy_from_S(
        db["traj_markov"][:n_m], db["activated"]
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        da["times_dem"][:n_d],
        ent_d,
        "-",
        color="#607D8B",
        lw=2.0,
        alpha=0.9,
        label="DEM (total)",
        zorder=3,
    )
    ax.plot(
        da["times_markov"][:n_m],
        ent_m,
        "o--",
        color="#9C27B0",
        markersize=4,
        lw=1.4,
        alpha=0.85,
        label="Markov (total)",
        zorder=2,
    )
    ax.set_title(f"Entropie totale ({sp_a} + {sp_b})\n{short_name}", fontweight="bold")
    ax.set_xlabel("Temps (centièmes de seconde)")
    ax.set_ylabel("H (nats)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "entropie_totale.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 entropie_totale.png")


def fig_teneur(
    species_data: dict,
    short_name: str,
    out_dir: Path,
) -> None:
    sps = list(species_data.keys())
    if len(sps) < 2:
        print("   ⚠️ fig_teneur annulée : Moins de 2 espèces trouvées.")
        return

    sp_a, sp_b = sps[0], sps[1]
    da, db = species_data[sp_a], species_data[sp_b]

    # squeeze() comme dans fig_states_by_species pour éviter les dims fantômes
    S_d_a = np.asarray(da["S_dem"]).squeeze()
    S_d_b = np.asarray(db["S_dem"]).squeeze()
    S_m_a = np.asarray(da["traj_markov"]).squeeze()
    S_m_b = np.asarray(db["traj_markov"]).squeeze()

    t_d_a = np.asarray(da["times_dem"]).ravel()
    t_d_b = np.asarray(db["times_dem"]).ravel()
    t_m_a = np.asarray(da["times_markov"]).ravel()
    t_m_b = np.asarray(db["times_markov"]).ravel()

    n_d = min(len(t_d_a), len(t_d_b))
    n_m = min(len(t_m_a), len(t_m_b))
    t_d, t_m = t_d_a[:n_d], t_m_a[:n_m]

    # activated est un masque booléen -> on récupère les vrais indices de cellules
    active_indices = np.where(da["activated"])[0]
    len(active_indices)

    tot_dem = S_d_a[:n_d] + S_d_b[:n_d]
    teneur_dem = np.zeros(tot_dem.shape, dtype=float)
    np.divide(S_d_a[:n_d], tot_dem, out=teneur_dem, where=tot_dem != 0)

    tot_m = S_m_a[:n_m] + S_m_b[:n_m]
    teneur_markov = np.zeros(tot_m.shape, dtype=float)
    np.divide(S_m_a[:n_m], tot_m, out=teneur_markov, where=tot_m != 0)

    # ── Construction d'un DataFrame long-format pour Seaborn ──────────────
    rows = []
    for cell in active_indices:
        for t, val in zip(t_d, teneur_dem[:, cell].squeeze()):
            rows.append(
                {
                    "Temps": t,
                    "Teneur": val,
                    "Cellule": f"Cellule {cell}",
                    "Source": "DEM",
                }
            )
        for t, val in zip(t_m, teneur_markov[:, cell].squeeze()):
            rows.append(
                {
                    "Temps": t,
                    "Teneur": val,
                    "Cellule": f"Cellule {cell}",
                    "Source": "Markov",
                }
            )
    df = pd.DataFrame(rows)

    # ── Style Seaborn ───────────────────────────────────────────────────
    # sns.set_theme(style="whitegrid", context="talk")
    # palette = sns.color_palette("viridis", n_colors=n_cells)

    fig, ax = plt.subplots(figsize=(16, 8))

    tab10 = plt.get_cmap("tab10").colors  # 10 couleurs, cycle si >10 cellules
    color_map = {cell: tab10[cell % 10] for cell in active_indices}
    palette = [color_map[cell] for cell in active_indices]

    fig, ax = plt.subplots(figsize=(16, 8))

    # DEM : traits pleins épais
    sns.lineplot(
        data=df[df["Source"] == "DEM"],
        x="Temps",
        y="Teneur",
        hue="Cellule",
        palette=palette,
        linewidth=2.5,
        alpha=0.5,
        hue_order=[f"Cellule {c}" for c in active_indices],
        ax=ax,
        legend=True,
        zorder=3,
    )

    # Markov : pointillés avec marqueurs, même palette, légende désactivée
    sns.lineplot(
        data=df[df["Source"] == "Markov"],
        x="Temps",
        y="Teneur",
        hue="Cellule",
        palette=palette,
        linewidth=1.6,
        alpha=0.9,
        hue_order=[f"Cellule {c}" for c in active_indices],
        style="Source",
        markers=["o"],
        dashes=[(2, 2)],
        markersize=6,
        ax=ax,
        legend=False,
        zorder=2,
    )

    ax.set_title(
        f"Teneur en {sp_a} par cellule ({sp_a} + {sp_b})\n{short_name}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Temps (centièmes de seconde)", fontsize=14)
    ax.set_ylabel(f"Teneur en {sp_a} (fraction)", fontsize=14)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=12)

    # Légende : cellules (couleur) + rappel du style de trait DEM/Markov
    handles, labels = ax.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", label="DEM"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.6,
            linestyle="--",
            marker="o",
            markersize=6,
            label="Markov",
        ),
    ]
    ax.legend(
        handles=handles + style_handles,
        labels=[*labels, "DEM", "Markov"],
        title="Cellule / Source",
        fontsize=11,
        title_fontsize=12,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )

    sns.despine(fig)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "teneur.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("   💾 teneur.png")


def fig_states_totale(
    species_data: dict,
    short_name: str,
    out_dir: Path,
) -> None:
    """
    Trace l'évolution du nombre total de particules (big + small) dans chaque cellule.
    - DEM : traits pleins
    - Markov : pointillés avec marqueurs.
    """
    sps = list(species_data.keys())
    if len(sps) < 2:
        print("   ⚠️ fig_states_totale annulée : Moins de 2 espèces trouvées.")
        return

    sp_a, sp_b = sps[0], sps[1]
    da, db = species_data[sp_a], species_data[sp_b]

    # squeeze() pour éviter les dims fantômes
    S_d_a = np.asarray(da["S_dem"]).squeeze()
    S_d_b = np.asarray(db["S_dem"]).squeeze()
    S_m_a = np.asarray(da["traj_markov"]).squeeze()
    S_m_b = np.asarray(db["traj_markov"]).squeeze()

    t_d_a = np.asarray(da["times_dem"]).ravel()
    t_d_b = np.asarray(db["times_dem"]).ravel()
    t_m_a = np.asarray(da["times_markov"]).ravel()
    t_m_b = np.asarray(db["times_markov"]).ravel()

    n_d = min(len(t_d_a), len(t_d_b))
    n_m = min(len(t_m_a), len(t_m_b))
    t_d, t_m = t_d_a[:n_d], t_m_a[:n_m]

    # activated est un masque booléen -> on récupère les vrais indices de cellules
    active_indices = np.where(da["activated"])[0]
    len(active_indices)

    # Total particles = S_big + S_small
    tot_dem = S_d_a[:n_d] + S_d_b[:n_d]
    tot_markov = S_m_a[:n_m] + S_m_b[:n_m]

    # ── Construction d'un DataFrame long-format pour Seaborn ──────────────
    rows = []
    for cell in active_indices:
        for t, val in zip(t_d, tot_dem[:, cell].squeeze()):
            rows.append(
                {
                    "Temps": t,
                    "Total": val,
                    "Cellule": f"Cellule {cell}",
                    "Source": "DEM",
                }
            )
        for t, val in zip(t_m, tot_markov[:, cell].squeeze()):
            rows.append(
                {
                    "Temps": t,
                    "Total": val,
                    "Cellule": f"Cellule {cell}",
                    "Source": "Markov",
                }
            )
    df = pd.DataFrame(rows)

    # ── Style ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 8))

    tab10 = plt.get_cmap("tab10").colors
    color_map = {cell: tab10[cell % 10] for cell in active_indices}
    palette = [color_map[cell] for cell in active_indices]

    # DEM : traits pleins épais
    sns.lineplot(
        data=df[df["Source"] == "DEM"],
        x="Temps",
        y="Total",
        hue="Cellule",
        palette=palette,
        linewidth=2.5,
        alpha=0.5,
        hue_order=[f"Cellule {c}" for c in active_indices],
        ax=ax,
        legend=True,
        zorder=3,
    )

    # Markov : pointillés avec marqueurs
    sns.lineplot(
        data=df[df["Source"] == "Markov"],
        x="Temps",
        y="Total",
        hue="Cellule",
        palette=palette,
        linewidth=1.6,
        alpha=0.9,
        hue_order=[f"Cellule {c}" for c in active_indices],
        style="Source",
        markers=["o"],
        dashes=[(2, 2)],
        markersize=6,
        ax=ax,
        legend=False,
        zorder=2,
    )

    ax.set_title(
        f"Nombre total de particules par cellule ({sp_a} + {sp_b})\n{short_name}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Temps (centièmes de seconde)", fontsize=14)
    ax.set_ylabel("Nombre total de particules", fontsize=14)
    ax.tick_params(labelsize=12)

    # Légende
    handles, labels = ax.get_legend_handles_labels()
    style_handles = [
        Line2D([0], [0], color="black", lw=2.5, linestyle="-", label="DEM"),
        Line2D(
            [0],
            [0],
            color="black",
            lw=1.6,
            linestyle="--",
            marker="o",
            markersize=6,
            label="Markov",
        ),
    ]
    ax.legend(
        handles=handles + style_handles,
        labels=[*labels, "DEM", "Markov"],
        title="Cellule / Source",
        fontsize=11,
        title_fontsize=12,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
    )

    sns.despine(fig)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "states_totale.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("   💾 states_totale.png")


def fig_compaction_population(
    volumes_dict: dict, short_name: str, out_dir: Path
) -> None:
    """
    Génère des graphiques explicites séparés pour comparer le volume des partitions,
    le volume des particules et la compacité résultante.
    """
    valid_clusters = {
        int(k): v
        for k, v in volumes_dict.items()
        if "error" not in v and v.get("volume_enveloppe", 0) > 0
    }

    if not valid_clusters:
        print("      ⚠️ Aucun cluster valide pour générer les graphiques de volumes.")
        return

    cluster_ids = sorted(valid_clusters.keys())
    v_enveloppe = [valid_clusters[cid]["volume_enveloppe"] for cid in cluster_ids]
    v_particules = [
        valid_clusters[cid]["volume_particules_total"] for cid in cluster_ids
    ]
    compacite = [valid_clusters[cid]["compaction_locale"] for cid in cluster_ids]

    x = np.arange(len(cluster_ids))
    width = 0.35

    # ══════════════════════════════════════════════════════════════════════
    # IMAGE 1 : COMPARAISON DES VOLUMES (Partition vs Matière)
    # ══════════════════════════════════════════════════════════════════════
    fig1, ax_vol = plt.subplots(figsize=(11, 5))

    # Échelle logarithmique recommandée car le volume enveloppe est souvent beaucoup
    # plus grand que le volume net des particules isolées
    ax_vol.bar(
        x - width / 2,
        v_enveloppe,
        width,
        label="Volume de la Partition ($V_{enveloppe}$)",
        color="#78909c",
    )
    ax_vol.bar(
        x + width / 2,
        v_particules,
        width,
        label="Volume Réel des Particules ($V_{particules}$)",
        color="#1e88e5",
    )

    ax_vol.set_yscale("log")
    ax_vol.set_ylabel("Volume ($m^3$) - Échelle Log", fontweight="bold")
    ax_vol.set_xlabel("Identifiant du Cluster (Cell ID)", fontweight="bold")
    ax_vol.set_title(
        f"Comparaison des Volumes : Domaines vs Particules\n{short_name}",
        fontweight="bold",
    )
    ax_vol.set_xticks(x)
    ax_vol.set_xticklabels([str(cid) for cid in cluster_ids])
    ax_vol.legend(loc="upper right")
    ax_vol.grid(True, which="both", linestyle="--", alpha=0.5)

    # Ajout du texte de la formule sur l'image
    formule_vol = r"$V_{particules} = N_{0.008} \cdot V_{sph}(8mm) + N_{0.004} \cdot V_{sph}(4mm)$"
    ax_vol.text(
        0.02,
        0.05,
        formule_vol,
        transform=ax_vol.transAxes,
        fontsize=10,
        bbox={
            "boxstyle": "round,pad=0.3",
            "fc": "#f8f9fa",
            "ec": "#b0bec5",
            "alpha": 0.9,
        },
    )

    fig1.tight_layout()
    fname_vol = f"comparaison_volumes_details_{short_name}.png"
    fig1.savefig(out_dir / fname_vol, bbox_inches="tight", dpi=150)
    plt.close(fig1)
    print(f"   💾 {fname_vol}")

    # ══════════════════════════════════════════════════════════════════════
    # IMAGE 2 : COMPACITÉ LOCALE DÉDUITE
    # ══════════════════════════════════════════════════════════════════════
    fig2, ax_comp = plt.subplots(figsize=(11, 4.5))

    ax_comp.plot(
        x,
        compacite,
        marker="o",
        lw=2.5,
        color="#43a047",
        label=r"Compacité calculée $\phi_{locale}$",
    )
    ax_comp.axhline(
        y=0.64,
        color="#e53935",
        linestyle=":",
        lw=1.8,
        label="Limite théorique RCP (~0.64)",
    )

    ax_comp.set_ylabel(r"Compacité $\phi$ (-)", fontweight="bold")
    ax_comp.set_xlabel("Identifiant du Cluster (Cell ID)", fontweight="bold")
    ax_comp.set_title(
        f"Analyse de la Compacité par Partition\n{short_name}", fontweight="bold"
    )
    ax_comp.set_xticks(x)
    ax_comp.set_xticklabels([str(cid) for cid in cluster_ids])
    ax_comp.set_ylim(0, 0.8)
    ax_comp.legend(loc="upper right")
    ax_comp.grid(True, linestyle="--", alpha=0.5)

    # Ajout du texte de la formule de compacité
    formule_comp = r"$\phi_{locale} = \frac{V_{particules}}{V_{enveloppe}}$"
    ax_comp.text(
        0.02,
        0.05,
        formule_comp,
        transform=ax_comp.transAxes,
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.4",
            "fc": "#f8f9fa",
            "ec": "#b0bec5",
            "alpha": 0.9,
        },
    )

    fig2.tight_layout()
    fname_comp = f"comparaison_compacite_details_{short_name}.png"
    fig2.savefig(out_dir / fname_comp, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"   💾 {fname_comp}")


def fig_mesh(
    exp: dict,
    df_start: pd.DataFrame,
    short_name: str,
    out_dir_img: Path,
    out_dir_files: Path,
    timestep_dict: dict | None = None,
    frame_stride: int = 157,
    series_theta_resolution: int = 8,
    series_phi_resolution: int = 8,
) -> None:
    """
    timestep_dict : dict {t_value -> DataFrame} donnant les positions réelles
                    des particules à chaque pas de temps (ex: sortie de
                    load_parquet_as_timestep_dict). Les clés doivent être dans
                    la même unité que exp["species"][sp]["times"].
    frame_stride  : ne garder qu'une frame sur N dans la série .vtp exportée
                    (le label restant identique à toutes les frames, ce
                    paramètre ne sert qu'à limiter le volume de fichiers et
                    le temps d'exécution — pas de perte d'information sur le
                    label).
    """
    config = exp["config"]
    start = config.get("start_index", 250)
    frame_stride = config.get("tau", 157)

    # 1. Vérifications initiales des données
    states_matrix = exp.get("matrix")  # Format attendu : (n_timesteps, n_particles)
    if states_matrix is None:
        print("   ⚠️  states_matrix.npy introuvable — maillage ignoré")
        return

    coord_cols = ["coordinates:0", "coordinates:1", "coordinates:2"]
    missing_cols = [c for c in coord_cols if c not in df_start.columns]
    if missing_cols:
        print(
            f"   ⚠️  Colonnes manquantes dans df_start: {missing_cols} — maillage ignoré"
        )
        return

    # Détection de la colonne de diamètre
    diam_col = (
        "Diameter"
        if "Diameter" in df_start.columns
        else ("diameter" if "diameter" in df_start.columns else None)
    )
    if diam_col is None:
        print(
            "   ⚠️  Colonne de diamètre introuvable dans df_start — Calcul précis de compacité impossible."
        )
        return

    coords = df_start[coord_cols].to_numpy()
    diameters = df_start[diam_col].to_numpy()

    # Identifiant stable de particule (indispensable pour réaligner les
    # positions de chaque frame sur le même ordre que cell_ids/diameters)
    id_candidates = ["id", "particle_id", "Particle_ID", "Particle_Id", "ID"]
    id_col = next((c for c in id_candidates if c in df_start.columns), None)
    if id_col is not None:
        particle_ids = df_start[id_col].to_numpy()
    else:
        particle_ids = np.arange(len(df_start))
        print(
            "      ⚠️  Aucune colonne d'identifiant stable ('id'/'particle_id') dans df_start — "
            "on suppose que l'ordre des lignes est identique dans toutes les frames de timestep_dict."
        )

    first_species = next(iter(exp["species"]))
    times = exp["species"][first_species]["times"]

    if states_matrix.shape[1] != len(df_start):
        print(
            "      ⚠️  Mismatch du nombre de particules (matrix vs df_start) — alignement requis"
        )

    # =========================================================================
    # 🔄 EXPORT DE LA MATRICE TEMPORELLE COMPLÈTE
    # =========================================================================
    print("      📊 Extraction de la matrice temporelle complète...")
    states_matrix_clean = np.squeeze(states_matrix)
    n_timesteps, n_particles = states_matrix_clean.shape

    npy_matrix_path = out_dir_files / f"matrix_cell_ids_evolution_{short_name}.npy"
    np.save(npy_matrix_path, states_matrix_clean)

    df_matrix = pd.DataFrame(
        data=states_matrix_clean,
        index=times[:n_timesteps],
        columns=[f"Particule_{i}" for i in range(n_particles)],
    )
    df_matrix.index.name = "Temps"
    csv_matrix_path = out_dir_files / f"matrix_cell_ids_evolution_{short_name}.csv"
    df_matrix.to_csv(csv_matrix_path)
    print(
        f"   💾 Matrice temporelle sauvegardée ({n_timesteps} pas x {n_particles} particules) [NPY & CSV]"
    )

    # =========================================================================
    # 🏷️ LABEL DE PARTITION FIGÉ À L'INSTANT DE RÉFÉRENCE (START)
    # =========================================================================
    row_start = np.searchsorted(times, start)
    if row_start >= len(times) or times[row_start] != start:
        print(
            f"      ⚠️  Pas de timestep exact={start} dans 'times', utilisation de l'index le plus proche."
        )
        row_start = min(row_start, len(times) - 1)

    cell_ids = states_matrix_clean[row_start].astype(int)
    print(
        f"      Focus instantané t={times[row_start]} : cell_ids={len(cell_ids)}  coords={len(coords)}  ✅"
    )
    print(
        "      🔒 Label de partition figé à cet instant — appliqué identique à toutes les frames "
        "de la série temporelle (maillage fixe, particules mobiles)."
    )

    # =========================================================================
    # 🎞️ EXPORT SÉRIE TEMPORELLE VTP (GLYPHES SPHÉRIQUES) + PVD (COMPRESSÉE EN .ZIP)
    # =========================================================================
    print(
        "      🎞️  Génération de la série temporelle .vtp (glyphes sphériques) + .pvd..."
    )

    if timestep_dict is None:
        print(
            "      ⚠️  Aucun timestep_dict fourni : les particules resteront figées aux coordonnées "
            "de df_start dans la série .vtp (label constant + position constante = fichier statique répété)."
        )
        dict_keys_sorted = None
    else:
        dict_keys_sorted = np.array(sorted(timestep_dict.keys()))

    def _lookup_frame_df(t_value: float) -> tuple[pd.DataFrame | None, float | None]:
        """Retourne le DataFrame le plus proche de t_value dans timestep_dict, ou None."""
        if dict_keys_sorted is None or len(dict_keys_sorted) == 0:
            return None, None
        idx = np.searchsorted(dict_keys_sorted, t_value)
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(dict_keys_sorted)]
        best_idx = min(candidates, key=lambda i: abs(dict_keys_sorted[i] - t_value))
        key = dict_keys_sorted[best_idx]
        return timestep_dict.get(key), key

    tmp_series_dir = out_dir_files / f"_tmp_vtp_series_{short_name}"
    tmp_series_dir.mkdir(parents=True, exist_ok=True)

    # Géométrie du glyphe pour la série temporelle (résolution réduite par défaut
    # pour limiter le poids des fichiers — la série statique mesh_3d_*.vtp plus
    # bas garde une résolution plus fine)
    sphere_geom_series = pv.Sphere(
        radius=0.5,
        theta_resolution=series_theta_resolution,
        phi_resolution=series_phi_resolution,
    )

    frame_indices = range(0, n_timesteps, max(1, frame_stride))
    n_frames_written = 0
    n_frames_missing = 0
    pvd_entries = []
    log_every = max(1, len(frame_indices) // 10)

    for count, t_idx in enumerate(frame_indices):
        t_value = float(times[t_idx])

        if timestep_dict is not None:
            df_t, matched_key = _lookup_frame_df(t_value)
        else:
            df_t, matched_key = None, None

        if df_t is None:
            frame_coords = coords  # repli sur la position de référence
            n_frames_missing += 1
        elif id_col is not None and id_col in df_t.columns:
            df_t_aligned = df_t.set_index(id_col).reindex(particle_ids)
            if df_t_aligned[coord_cols].isna().any().any():
                print(
                    f"         ⚠️  Certaines particules absentes au pas t={t_value} (clé matchée={matched_key}) "
                    f"— positions manquantes remplacées par la référence de df_start."
                )
                df_t_aligned[coord_cols] = df_t_aligned[coord_cols].fillna(
                    pd.DataFrame(coords, columns=coord_cols, index=df_t_aligned.index)
                )
            frame_coords = df_t_aligned[coord_cols].to_numpy()
        else:
            # Pas d'identifiant fiable : on suppose l'ordre des lignes identique à df_start
            frame_coords = df_t[coord_cols].to_numpy()

        frame_points = pv.PolyData(frame_coords)
        frame_points.point_data["partition_label"] = (
            cell_ids  # figé, identique pour toutes les frames
        )
        frame_points.point_data["particle_id"] = particle_ids
        frame_points.point_data["Diameter"] = diameters

        frame_glyph = frame_points.glyph(
            geom=sphere_geom_series, orient=False, factor=1.0, scale="Diameter"
        )

        frame_name = f"frame_{t_idx:04d}.vtp"
        frame_glyph.save(str(tmp_series_dir / frame_name))
        pvd_entries.append((t_value, frame_name))
        n_frames_written += 1

        if count % log_every == 0:
            print(
                f"         ...frame {count + 1}/{len(frame_indices)} écrite (t={t_value})"
            )

    if n_frames_missing:
        print(
            f"      ⚠️  {n_frames_missing} frame(s) sans correspondance dans timestep_dict "
            f"(position de repli utilisée)."
        )

    # Génération du fichier .pvd (collection ParaView)
    pvd_lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1">',
        "  <Collection>",
    ]
    for t_val, fname in pvd_entries:
        pvd_lines.append(f'    <DataSet timestep="{t_val}" file="{fname}"/>')
    pvd_lines.append("  </Collection>")
    pvd_lines.append("</VTKFile>")

    pvd_path = tmp_series_dir / f"series_{short_name}.pvd"
    pvd_path.write_text("\n".join(pvd_lines), encoding="utf-8")
    print(
        f"      💾 {pvd_path.name} généré ({n_frames_written} frames référencées, stride={frame_stride})"
    )

    # Compression en .zip puis suppression du dossier non compressé
    zip_base_path = (
        out_dir_files / f"vtp_series_{short_name}"
    )  # sans extension, make_archive l'ajoute
    shutil.make_archive(
        base_name=str(zip_base_path),
        format="zip",
        root_dir=str(tmp_series_dir.parent),
        base_dir=tmp_series_dir.name,
    )
    shutil.rmtree(tmp_series_dir)
    print(
        f"   💾 {zip_base_path.name}.zip (série .vtp + .pvd compressée, dossier temporaire supprimé)"
    )

    # =========================================================================
    # CALCUL GÉOMÉTRIQUE POUR L'INSTANT DE RÉFÉRENCE (START)
    # =========================================================================
    v_individuel_particules = (4.0 / 3.0) * np.pi * (diameters / 2.0) ** 3

    unique_cells = np.unique(cell_ids)
    volumes_dict = {}
    hulls_list = []

    print(
        f"      📊 Analyse géométrique des {len(unique_cells)} domaines à l'instant t={start}..."
    )

    for c_id in unique_cells:
        cluster_mask = cell_ids == c_id
        cluster_coords = coords[cluster_mask]
        cluster_diams = diameters[cluster_mask]
        cluster_v_reels = v_individuel_particules[cluster_mask]

        n_total = len(cluster_coords)
        n_gros = int(np.sum(np.isclose(cluster_diams, 0.008, atol=1e-4)))
        n_petits = int(np.sum(np.isclose(cluster_diams, 0.004, atol=1e-4)))
        v_total_spheres = float(np.sum(cluster_v_reels))
        fraction_gros = n_gros / n_total if n_total > 0 else 0.0

        if n_total >= 4:
            try:
                hull = ConvexHull(cluster_coords)
                v_mesh_hull = float(hull.volume)
                compaction = v_total_spheres / v_mesh_hull if v_mesh_hull > 0 else 0.0

                volumes_dict[int(c_id)] = {
                    "volume_enveloppe": v_mesh_hull,
                    "volume_particules_total": v_total_spheres,
                    "compaction_locale": compaction,
                    "n_total": n_total,
                    "n_espece_008": n_gros,
                    "n_espece_004": n_petits,
                    "fraction_numerique_008": fraction_gros,
                }

                n_faces = len(hull.simplices)
                faces = np.column_stack((np.full(n_faces, 3), hull.simplices)).ravel()

                hull_poly = pv.PolyData(hull.points, faces)
                hull_poly.cell_data["cell_id"] = np.full(n_faces, c_id)
                hull_poly.cell_data["cluster_volume"] = np.full(n_faces, v_mesh_hull)
                hull_poly.cell_data["cluster_compaction"] = np.full(n_faces, compaction)
                hull_poly.cell_data["fraction_008"] = np.full(n_faces, fraction_gros)

                hulls_list.append(hull_poly)

            except Exception as e:
                volumes_dict[int(c_id)] = {
                    "volume_enveloppe": 0.0,
                    "n_total": n_total,
                    "error": str(e),
                }
        else:
            volumes_dict[int(c_id)] = {
                "volume_enveloppe": 0.0,
                "n_total": n_total,
                "error": "Pas assez de points",
            }

    # Sauvegarde du rapport JSON pour l'instant T
    json_vol_path = out_dir_files / f"volumes_clusters_{short_name}_t{start}.json"
    with open(json_vol_path, "w") as f:
        json.dump(volumes_dict, f, indent=4)

    # 3. Assemblage propre des enveloppes 3D
    if hulls_list:
        boundaries_poly = pv.PolyData()
        for h in hulls_list:
            boundaries_poly += h

        if boundaries_poly.n_points > 0:
            hulls_vtp_path = out_dir_files / f"boundaries_3d_{short_name}_t{start}.vtp"
            boundaries_poly.save(str(hulls_vtp_path))
            print(f"   💾 {hulls_vtp_path.name} (Surfaces ConvexHull enregistrées)")

    # =========================================================================
    # INJECTION DES SCALAIRES SUR LE MAILLAGE DES PARTICULES
    # =========================================================================
    vol_env_par_particule = np.zeros(len(cell_ids))
    compaction_par_particule = np.zeros(len(cell_ids))
    fraction_008_par_particule = np.zeros(len(cell_ids))

    for c_id, data in volumes_dict.items():
        if "error" not in data and data["volume_enveloppe"] > 0:
            mask = cell_ids == c_id
            vol_env_par_particule[mask] = data["volume_enveloppe"]
            compaction_par_particule[mask] = data["compaction_locale"]
            fraction_008_par_particule[mask] = data["fraction_numerique_008"]

    mesh = pv.PolyData(coords)
    mesh.point_data["cell_id"] = cell_ids
    mesh.point_data["Diameter"] = diameters
    mesh.point_data["cluster_volume_enveloppe"] = vol_env_par_particule
    mesh.point_data["cluster_compaction"] = compaction_par_particule
    mesh.point_data["cluster_fraction_grosses_particules"] = fraction_008_par_particule

    # Application des tailles réelles sur les sphères 3D
    sphere = pv.Sphere(radius=0.5, theta_resolution=12, phi_resolution=12)
    glyph = mesh.glyph(geom=sphere, orient=False, factor=1.0, scale="Diameter")
    vtp_path = out_dir_files / f"mesh_3d_{short_name}_t{start}.vtp"
    glyph.save(str(vtp_path))
    print(f"   💾 {vtp_path.name} (Particules enrichies enregistrées)")

    # =========================================================================
    # VISUALISATIONS GRAPHIQUES ET HEATMAP D'ÉVOLUTION COMPLÈTE
    # =========================================================================
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        glyph,
        scalars="cluster_compaction",
        cmap="viridis",
        show_scalar_bar=True,
        scalar_bar_args={"title": f"Compacite Locale (-) t={start}"},
    )
    plotter.add_title(
        f"Analyse de Compacite Multi-Especes — {short_name}", font_size=10
    )
    fname_3d = f"mesh_3d_volume_{short_name}_t{start}.png"
    plotter.screenshot(str(out_dir_img / fname_3d))
    plotter.close()
    print(f"   💾 {fname_3d}")

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=cell_ids,
        cmap="hsv",
        s=12,
        alpha=0.85,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Cell ID")
    ax.set_title(
        f"Maillage 2D — Projection XY (t={start})\n{short_name}", fontweight="bold"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fname_2d = f"mesh_2d_{short_name}_t{start}.png"
    fig.tight_layout()
    fig.savefig(out_dir_img / fname_2d, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname_2d}")

    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(
        states_matrix_clean,
        aspect="auto",
        cmap="hsv",
        origin="lower",
        extent=[0, n_particles, times[0], times[n_timesteps - 1]],
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Cell ID assignée")
    ax.set_title(
        f"Évolution de l'assignation des cellules par particule\n{short_name}",
        fontweight="bold",
    )
    ax.set_xlabel("Index de la particule")
    ax.set_ylabel("Temps (pas)")
    fname_heatmap = f"mesh_evolution_heatmap_{short_name}.png"
    fig.tight_layout()
    fig.savefig(out_dir_img / fname_heatmap, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname_heatmap}")

    try:
        fig_compaction_population(volumes_dict, short_name, out_dir_img)
    except NameError:
        pass  # Si la fonction n'est pas définie globalement


def fig_population_par_cellule(
    sp: str,
    sp_data: dict,
    short_name: str,
    out_dir: Path,
) -> None:
    """
    Visualise l'évolution du nombre de particules dans chaque cellule au cours du temps.
    Génère :
    1. Un graphique temporel (courbes superposées)
    2. Une heatmap de la distribution moyenne.
    """
    S_dem = np.asarray(sp_data["S_dem"]).squeeze()
    times_dem = np.asarray(sp_data["times_dem"]).ravel()
    activated = sp_data["activated"]

    # Filtrer uniquement les cellules activées
    active_indices = np.where(activated)[0]
    S_active = S_dem[:, active_indices]

    n_cells = len(active_indices)

    # ══════════════════════════════════════════════════════════════════════
    # 1. Graphique temporel : évolution du nombre de particules par cellule
    # ══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(14, 7))

    # Colormap pour distinguer les cellules
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_cells - 1, 1)) for i in range(n_cells)]

    # Tracer chaque cellule
    for idx, cell in enumerate(active_indices):
        ax.plot(
            times_dem,
            S_active[:, idx],
            "-",
            color=colors[idx],
            linewidth=1.2,
            alpha=0.7,
            label=f"Cellule {cell}",
        )

    ax.set_title(
        f"Évolution du nombre de particules par cellule — espèce '{sp}'\n{short_name}",
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xlabel("Temps (centièmes de seconde)", fontsize=11)
    ax.set_ylabel("Nombre de particules", fontsize=11)
    ax.legend(
        fontsize=7,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        title="Cellules activées",
    )
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fname_time = f"population_par_cellule_temps_{sp}.png"
    fig.savefig(out_dir / fname_time, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   💾 {fname_time}")

    # ══════════════════════════════════════════════════════════════════════
    # 2. Heatmap : distribution moyenne des particules par cellule
    # ══════════════════════════════════════════════════════════════════════
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Calculer la moyenne temporelle pour chaque cellule
    mean_population = S_active.mean(axis=0)
    std_population = S_active.std(axis=0)

    # Créer une barre horizontale pour chaque cellule
    y_pos = np.arange(n_cells)

    # Barres avec erreur standard
    bars = ax2.barh(
        y_pos,
        mean_population,
        xerr=std_population,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Cellule {cell}" for cell in active_indices], fontsize=9)
    ax2.set_xlabel("Nombre moyen de particules (± écart-type)", fontsize=11)
    ax2.set_title(
        f"Distribution moyenne des particules par cellule — espèce '{sp}'\n{short_name}",
        fontweight="bold",
        fontsize=13,
    )
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.invert_yaxis()  # Cellule 0 en haut

    # Ajouter les valeurs numériques sur les barres
    for idx, (bar, mean_val) in enumerate(zip(bars, mean_population)):
        ax2.text(
            bar.get_width() + std_population[idx] + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{mean_val:.1f}",
            va="center",
            ha="left",
            fontsize=8,
            color="darkblue",
        )

    fig2.tight_layout()
    fname_heatmap = f"population_par_cellule_moyenne_{sp}.png"
    fig2.savefig(out_dir / fname_heatmap, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"   💾 {fname_heatmap}")

    # ══════════════════════════════════════════════════════════════════════
    # 3. Graphique empilé (stacked area) : vue globale de la distribution
    # ══════════════════════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(14, 7))

    # Limiter à un sous-ensemble de cellules si trop nombreuses (pour lisibilité)
    max_cells_to_plot = min(n_cells, 20)
    if n_cells > max_cells_to_plot:
        # Prendre les cellules les plus peuplées
        top_indices = np.argsort(mean_population)[::-1][:max_cells_to_plot]
        S_plot = S_active[:, top_indices]
        cells_plot = active_indices[top_indices]
        colors_plot = [colors[i] for i in top_indices]
        title_suffix = f" (top {max_cells_to_plot} cellules)"
    else:
        S_plot = S_active
        cells_plot = active_indices
        colors_plot = colors
        title_suffix = ""

    ax3.stackplot(
        times_dem,
        S_plot.T,
        labels=[f"Cellule {cell}" for cell in cells_plot],
        colors=colors_plot,
        alpha=0.7,
    )

    ax3.set_title(
        f"Répartition temporelle des particules par cellule{title_suffix} — espèce '{sp}'\n{short_name}",
        fontweight="bold",
        fontsize=13,
    )
    ax3.set_xlabel("Temps (centièmes de seconde)", fontsize=11)
    ax3.set_ylabel("Nombre de particules", fontsize=11)
    ax3.legend(
        fontsize=7,
        ncol=3,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        title="Cellules",
    )
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig3.tight_layout()
    fname_stacked = f"population_par_cellule_stacked_{sp}.png"
    fig3.savefig(out_dir / fname_stacked, bbox_inches="tight", dpi=150)
    plt.close(fig3)
    print(f"   💾 {fname_stacked}")

    # ══════════════════════════════════════════════════════════════════════
    # 4. Export des données en CSV (tableau numérique)
    # ══════════════════════════════════════════════════════════════════════
    # Créer un DataFrame avec les données
    df_population = pd.DataFrame(
        S_active,
        columns=[f"cellule_{cell}" for cell in active_indices],
        index=times_dem,
    )
    df_population.index.name = "temps"

    csv_path = out_dir / f"population_par_cellule_{sp}.csv"
    df_population.to_csv(csv_path)
    print(f"   💾 population_par_cellule_{sp}.csv")


def fig_matrice_population_heatmap(
    sp: str,
    sp_data: dict,
    short_name: str,
    out_dir: Path,
) -> None:
    """Crée une heatmap 2D (temps × cellules) montrant l'évolution de la population."""
    S_dem = np.asarray(sp_data["S_dem"]).squeeze()
    times_dem = np.asarray(sp_data["times_dem"]).ravel()
    activated = sp_data["activated"]

    # Filtrer les cellules activées
    active_indices = np.where(activated)[0]
    S_active = S_dem[:, active_indices]

    # Sous-échantillonner le temps si trop de points (pour performance)
    max_time_points = 500
    if len(times_dem) > max_time_points:
        step = len(times_dem) // max_time_points
        S_plot = S_active[::step, :]
        times_plot = times_dem[::step]
    else:
        S_plot = S_active
        times_plot = times_dem

    # Créer la heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        S_plot.T,  # Transposer pour avoir temps en x, cellules en y
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
        extent=[times_plot[0], times_plot[-1], -0.5, len(active_indices) - 0.5],
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Nombre de particules", fontsize=10)

    ax.set_title(
        f"Heatmap : Population par cellule au cours du temps — espèce '{sp}'\n{short_name}",
        fontweight="bold",
        fontsize=13,
    )
    ax.set_xlabel("Temps (centièmes de seconde)", fontsize=11)
    ax.set_ylabel("Cellule", fontsize=11)

    # Labels des cellules (seulement quelques-uns si trop nombreux)
    n_cells = len(active_indices)
    if n_cells <= 30:
        ax.set_yticks(range(n_cells))
        ax.set_yticklabels([str(cell) for cell in active_indices], fontsize=8)
    else:
        # Afficher seulement quelques labels
        n_labels = min(20, n_cells)
        label_indices = np.linspace(0, n_cells - 1, n_labels, dtype=int)
        ax.set_yticks(label_indices)
        ax.set_yticklabels([str(active_indices[i]) for i in label_indices], fontsize=8)

    fig.tight_layout()
    fname = f"heatmap_population_{sp}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"   💾 {fname}")


# ── Export matrices de transition ─────────────────────────────────────────────


def export_transition_matrices(
    species_data: dict, short_name: str, out_dir: Path
) -> None:
    for sp, d in species_data.items():
        np.save(out_dir / f"P_{sp}_{short_name}.npy", d["P"])
        np.savetxt(out_dir / f"P_{sp}_{short_name}.txt", d["P"], fmt="%.6f")
        print(f"   💾 P_{sp}_{short_name}.npy / .txt")


def run_postprocess(
    path_hf: str,
    short_name: str,
    bucket_prefix: str = "_Good/Experiment",
    top_states: int = 6,
    particle_diameter: float | None = None,
    df_start: pd.DataFrame | None = None,
    timestep_dict: dict | None = None,
) -> None:
    """Post-traite une expérience et upload les résultats dans le bucket."""
    category = get_simulation_category(short_name)
    print(f"\n{'═' * 60}")
    print(f"🔬 Post-traitement : {short_name}")
    print(f"   Catégorie : {category}")
    print(f"   Chemin    : {path_hf}")
    print(f"{'═' * 60}")

    # Chargement
    exp = load_experiment(path_hf)
    species_data = prepare_species(exp)
    bucket_subfolder = f"postraitement/{category}/{short_name}"

    with PostprocessingBucketUploader(
        bucket_subfolder=bucket_subfolder,
        particle_diameter=particle_diameter,
    ) as tmp:
        # ── Arborescence locale ──────────────────────────────────────────
        img_etats = tmp / "images" / "etats"
        img_rsd = tmp / "images" / "rsd"
        img_matrices = tmp / "images" / "matrices"
        img_mesh = tmp / "images" / "mesh"
        f_mesh = tmp / "fichiers" / "mesh"
        f_trans = tmp / "fichiers" / "transitions"
        # img_teneur  = tmp / "images"  / "teneurs"
        for d in [img_etats, img_rsd, img_matrices, img_mesh, f_mesh, f_trans]:
            d.mkdir(parents=True, exist_ok=True)

        # ── 1. Figures d'états par espèce ────────────────────────────────
        print("\n📊 Figures d'états...")
        for sp, sd in species_data.items():
            fig_states_top3_index(sp, sd, short_name, img_etats)
            fig_states_top_populated(sp, sd, short_name, img_etats, k=top_states)

        # ── 2. Matrices de transition ────────────────────────────────────
        print("\n🔲 Matrices de transition...")
        for sp, sd in species_data.items():
            fig_transition_matrix(sp, sd, short_name, img_matrices)
            fig_spectral_diagnostic(sp, sd, short_name, img_matrices)

        # ── Figure états par espèce (nouvelle) ──────────────────────────────
        print("\n📈 États par espèce...")
        for sp, sd in species_data.items():
            try:
                fig_states_by_species(sp, sd, short_name, img_etats)
            except Exception as e:
                print(f"   ⚠️  {sp} ignoré : {e}")

        # ── 3. RSD ───────────────────────────────────────────────────────
        print("\n📈 RSD & métriques...")
        fig_rsd(species_data, short_name, img_rsd)
        fig_concentration(species_data, short_name, img_rsd)
        fig_entropy_total(species_data, short_name, img_rsd)
        print("Teneur !!!")
        try:
            fig_teneur(species_data, short_name, img_etats)
        except Exception as e:
            print(f" Teneur non construite \n{e}")

        print("États totaux !!!")
        try:
            fig_states_totale(species_data, short_name, img_etats)
        except Exception as e:
            print(f" États totaux non construits \n{e}")

        # ── 4. Maillage ──────────────────────────────────────────────────
        print("\n🗺️  Maillage...")
        try:
            fig_mesh(
                exp, df_start, short_name, img_mesh, f_mesh, timestep_dict=timestep_dict
            )
        except Exception as e:
            print(f"   ⚠️  Maillage ignoré : {e}")

        # ── 5. Export matrices brutes ─────────────────────────────────────
        print("\n📁 Export matrices brutes...")
        export_transition_matrices(species_data, short_name, f_trans)

        # ── 2.5 Population par cellule (NOUVEAU) ──────────────────────────────
        print("\n📊 Population par cellule...")
        for sp, sd in species_data.items():
            try:
                fig_population_par_cellule(sp, sd, short_name, img_etats)
                fig_matrice_population_heatmap(sp, sd, short_name, img_etats)
            except Exception as e:
                print(f"   ⚠️  {sp} ignoré : {e}")

    print(f"\n✅ {short_name} — terminé.\n")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-traitement DEM/Markov → bucket HuggingFace",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemples :
  # Argparse classique
  python postprocess.py --folder voronoi_20cells_NLT30_step10_dt2_tau157_start250
  python postprocess.py --keywords voronoi_20cells NLT30
  python postprocess.py --category voronoi_simulations
  python postprocess.py --folder mon_exp --top-states 8 --dry-run

  # Sous-commandes
  python postprocess.py single voronoi_20cells_NLT30_step10_dt2_tau157_start250
  python postprocess.py keywords voronoi_20cells NLT30
  python postprocess.py category voronoi_simulations --top-states 4
""",
    )

    # Options communes
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--bucket-prefix",
        default="_Good/Experiment",
        help="Préfixe du bucket (défaut : _Good/Experiment)",
    )
    common.add_argument(
        "--top-states",
        type=int,
        default=6,
        help="Nombre de cellules les plus peuplées à tracer (défaut : 6)",
    )
    common.add_argument(
        "--dry-run",
        action="store_true",
        help="Liste les expériences sans les post-traiter",
    )
    common.add_argument(
        "--dem-ref",
        default="first",
        help="Référence DEM : 'first' (défaut), 'mean', ou nom exact d'une expérience",
    )
    p.add_argument("--dem-ref", default="first", dest="dem_ref")
    # --- Style classique (pas de sous-commande) ---
    p.add_argument("--folder", help="Nom exact du dossier d'expérience")
    p.add_argument("--keywords", nargs="+", help="Mots-clés pour trouver le dossier")
    p.add_argument("--category", help="Catégorie complète à traiter")
    p.add_argument(
        "--bucket-prefix",
        default="_Good/Experiment",
        dest="bucket_prefix",
        help="Préfixe du bucket (défaut : _Good/Experiment)",
    )
    p.add_argument("--top-states", type=int, default=6, dest="top_states")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")

    # --- Sous-commandes ---
    sub = p.add_subparsers(dest="subcommand", metavar="SOUS-COMMANDE")

    s_single = sub.add_parser(
        "single", parents=[common], help="Post-traite un dossier par son nom exact"
    )
    s_single.add_argument("folder", help="Nom exact du dossier")

    s_kw = sub.add_parser(
        "keywords", parents=[common], help="Trouve un dossier par mots-clés"
    )
    s_kw.add_argument("keywords", nargs="+", help="Mots-clés")

    s_cat = sub.add_parser(
        "category", parents=[common], help="Post-traite toute une catégorie"
    )
    s_cat.add_argument("category", help="Nom de la catégorie (ex: voronoi_simulations)")
    s_cmp = sub.add_parser(
        "compare",
        parents=[common],
        help="Compare toutes les expériences matchant les mots-clés",
    )
    s_cmp.add_argument("keywords", nargs="+", help="Mots-clés communs aux expériences")

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    bucket_prefix = getattr(args, "bucket_prefix", "_Good/Experiment")
    top_states = getattr(args, "top_states", 6)
    dry_run = getattr(args, "dry_run", False)
    dem_ref = getattr(args, "dem_ref", "first")
    bucket_hf = f"hf://buckets/{BUCKET_ID}/{bucket_prefix}"

    # ── Résolution du mode et de la liste d'expériences ─────────────────────
    mode = None
    keywords = None
    experiments = []

    if args.subcommand == "single":
        mode = "single"
        results = find_experiment_paths(bucket_hf, folder_name=args.folder)
        if not results:
            print(f"❌ Aucune expérience trouvée pour folder='{args.folder}'")
            sys.exit(1)
        experiments = results

    elif args.subcommand == "keywords":
        mode = "keywords"
        keywords = args.keywords
        results = find_experiment_paths(bucket_hf, keywords=keywords)
        if not results:
            print(f"❌ Aucune expérience trouvée pour keywords={keywords}")
            sys.exit(1)
        experiments = results

    elif args.subcommand == "category":
        mode = "category"
        experiments = list_category_paths(bucket_hf, args.category)
        if not experiments:
            print(f"❌ Aucune expérience trouvée dans '{args.category}'")
            sys.exit(1)

    elif args.subcommand == "compare":
        mode = "compare"
        keywords = args.keywords
        experiments = find_all_experiments_by_keywords(bucket_hf, keywords)
        if not experiments:
            print(f"❌ Aucune expérience trouvée pour {keywords}")
            sys.exit(1)

    # ── Style classique (pas de sous-commande) ───────────────────────────────
    elif getattr(args, "folder", None):
        mode = "single"
        results = find_experiment_paths(bucket_hf, folder_name=args.folder)
        if not results:
            print(f"❌ Aucune expérience trouvée pour folder='{args.folder}'")
            sys.exit(1)
        experiments = results

    elif getattr(args, "keywords", None):
        mode = "keywords"
        keywords = args.keywords
        results = find_experiment_paths(bucket_hf, keywords=keywords)
        if not results:
            print(f"❌ Aucune expérience trouvée pour keywords={keywords}")
            sys.exit(1)
        experiments = results

    elif getattr(args, "category", None):
        mode = "category"
        experiments = list_category_paths(bucket_hf, args.category)
        if not experiments:
            print(f"❌ Aucune expérience trouvée dans '{args.category}'")
            sys.exit(1)

    else:
        print("❌ Aucun argument fourni. Utilisez --help pour voir les options.")
        sys.exit(1)

    # ── Dry-run ──────────────────────────────────────────────────────────────
    if dry_run:
        print(f"[DRY] mode={mode} — {len(experiments)} expérience(s) :")
        for _, s in experiments:
            print(f"  - {s}")
        return

    print(f"\n📋 {len(experiments)} expérience(s) à traiter")
    for i, (_, s) in enumerate(experiments, 1):
        print(f"  [{i}/{len(experiments)}] {s}")

    # ── Chargement unique du parquet (sauf compare qui n'en a pas besoin) ────
    df_start = None
    timestep_dict = None
    if mode != "compare":
        print("\n🔄 Chargement du parquet (une seule fois)...")
        first_exp = load_experiment(experiments[0][0])
        start = first_exp["config"].get("start_index", 250)
        timestep_dict = load_parquet_as_timestep_dict(
            f"hf://buckets/{BUCKET_ID}/simulation_complete.parquet", fs
        )
        df_start = timestep_dict[start]
        print(f"   ✅ df_start chargé (timestep {start}, {len(df_start)} particules)")
        print(
            f"   ✅ timestep_dict chargé ({len(timestep_dict)} pas de temps disponibles)\n"
        )

    # ── Exécution ────────────────────────────────────────────────────────────
    if mode == "compare":
        run_comparison(
            experiments, keywords, bucket_prefix, top_states, dem_ref=dem_ref
        )

    else:
        for path_hf, short in experiments:
            try:
                # Extraire particle_diameter depuis config si disponible
                exp_config = load_experiment(path_hf)["config"]
                particle_diameter = exp_config.get("particle_diameter")

                run_postprocess(
                    path_hf,
                    short,
                    bucket_prefix,
                    top_states,
                    particle_diameter=particle_diameter,
                    df_start=df_start,
                    timestep_dict=timestep_dict,
                )
            except Exception as e:
                print(f"⚠️  {short} — erreur : {e}")


if __name__ == "__main__":
    main()
