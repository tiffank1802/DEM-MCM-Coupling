"""
postprocess.py — Post-traitement automatisé des expériences DEM/Markov

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
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pyvista as pv

asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
pv.OFF_SCREEN = True

# ── Imports projet ────────────────────────────────────────────────────────────
from huggingface_hub import HfFileSystem
from DEM_MCM1.src.partitioners import create_partitioner
from DEM_MCM1.src.utils import load_parquet_as_timestep_dict
from DEM_MCM1.src.bucket_io import (
    BUCKET_ID,
    ALL_CATEGORIES,
    get_simulation_category,
    PostprocessingBucketUploader,
    _get_bucket_prefix_from_particle_diameter,
)

fs = HfFileSystem()

# ═════════════════════════════════════════════════════════════════════════════
# STYLE MATPLOTLIB GLOBAL
# ═════════════════════════════════════════════════════════════════════════════
STYLE = {
    "figure.dpi":          150,
    "figure.facecolor":    "white",
    "axes.facecolor":      "#f8f9fa",
    "axes.grid":           True,
    "grid.color":          "white",
    "grid.linewidth":      1.0,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.labelsize":      11,
    "axes.titlesize":      12,
    "axes.titleweight":    "bold",
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "legend.fontsize":     9,
    "legend.framealpha":   0.9,
    "lines.linewidth":     1.6,
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

def find_experiment_path(
    bucket_base_hf: str,
    folder_name: str = None,
    keywords: list[str] = None,
) -> tuple[str, str]:
    """
    Retourne (hf_path, short_name) du dossier d'expérience.
    Cherche dans tous les sous-dossiers de catégorie, puis à la racine (fallback).

    Priority : folder_name exact > keywords
    """
    def _search_in(base: str) -> str | None:
        try:
            items = fs.ls(base)
        except FileNotFoundError:
            return None
        for item in items:
            if item["type"] != "directory":
                continue
            name = item["name"].split("/")[-1]
            if folder_name and name == folder_name:
                return item["name"]
            if keywords and all(k in name for k in keywords):
                return item["name"]
        return None

    # 1. Chercher dans chaque sous-dossier de catégorie
    for cat in ALL_CATEGORIES:
        found = _search_in(f"{bucket_base_hf}/{cat}")
        if found:
            short = found.split("/")[-1]
            return f"hf://{found}", short

    # 2. Fallback racine (avant migration)
    found = _search_in(bucket_base_hf.replace("hf://", ""))
    if found:
        short = found.split("/")[-1]
        return f"hf://{found}", short

    raise FileNotFoundError(
        f"Aucun dossier trouvé pour "
        f"{'folder=' + folder_name if folder_name else 'keywords=' + str(keywords)}"
    )


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
    Retourne un dict complet avec les matrices et méta-données.
    """
    config       = _load_json(path_hf, "config.json")
    stats        = _load_json(path_hf, "stats.json")
    species_list = stats.get("species_list", ["small", "large"])

    species_data = {}
    for sp in species_list:
        P        = _load_npy(path_hf, f"transitionmatrix_{sp}.npy")
        S_matrix = _load_npy(path_hf, f"S_matrix_{sp}.npy")
        times    = _load_npy(path_hf, f"times_{sp}.npy")
        species_data[sp] = {"P_raw": P, "S_matrix": S_matrix, "times": times}

    return {"config": config, "stats": stats, "species": species_data}


# ═════════════════════════════════════════════════════════════════════════════
# PRÉPARATION MATHÉMATIQUE
# ═════════════════════════════════════════════════════════════════════════════

def clean_transition_matrix(P: np.ndarray, threshold: float = 0.5):
    P_clean   = P.copy()
    col_sums  = P_clean.sum(axis=0)
    activated = col_sums >= threshold
    P_clean[:, ~activated] = 0.0
    safe = col_sums.copy(); safe[~activated] = 1.0
    P_clean = P_clean / safe
    return P_clean, activated


def propagate_markov(S0, P, times, start_idx, tau, activated):
    row_start    = np.searchsorted(times, start_idx)
    times_full   = times[row_start:]
    markov_idx   = np.arange(0, len(times_full), tau)
    times_markov = times_full[markov_idx]
    S = S0.copy().astype(float); S[~activated] = 0.0
    traj = [S.copy()]
    for _ in range(1, len(markov_idx)):
        S = P @ S
        traj.append(S.copy())
    return np.array(traj), times_markov


def prepare_species(exp: dict) -> dict:
    """
    Nettoie P, calcule les cellules activées, propage Markov, extrait DEM tronqué.
    Retourne un dict enrichi par espèce.
    """
    config = exp["config"]
    start  = config.get("start_index", 250)
    tau    = config.get("tau", 50)
    out    = {}

    for sp, data in exp["species"].items():
        P_clean, activated = clean_transition_matrix(data["P_raw"])
        row_start = np.searchsorted(data["times"], start)
        S0        = data["S_matrix"][row_start].astype(float)
        traj, times_markov = propagate_markov(
            S0, P_clean, data["times"], start, tau, activated
        )
        out[sp] = {
            "P":            P_clean,
            "P_raw":        data["P_raw"],
            "S_matrix":     data["S_matrix"],
            "times":        data["times"],
            "S_dem":        data["S_matrix"][row_start:],
            "times_dem":    data["times"][row_start:],
            "traj_markov":  traj,
            "times_markov": times_markov,
            "activated":    activated,
        }
    return out

def find_all_experiments_by_keywords(
    bucket_base_hf: str,
    keywords: list[str],
) -> list[tuple[str, str]]:
    """
    Retourne [(hf_path, short_name), ...] pour TOUS les dossiers
    dont le nom contient tous les mots-clés, dans toutes les catégories.
    """
    results = []

    def _search_in(base: str):
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

def fig_compare_rsd(all_species_data: dict[str, dict], out_dir: Path):
    """
    RSD par espèce — toutes les expériences superposées.
    all_species_data : { short_name: { sp: sp_data } }
    """
    # Collecter toutes les espèces présentes
    all_species = sorted({sp for sd in all_species_data.values() for sp in sd})

    fig, axes = plt.subplots(1, len(all_species),
                             figsize=(8 * len(all_species), 5), squeeze=False)
    fig.suptitle("Comparaison RSD par espèce", fontweight="bold")

    cmap   = plt.cm.tab10
    names  = list(all_species_data.keys())
    colors = {n: cmap(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}

    for col, sp in enumerate(all_species):
        ax = axes[0][col]
        for name, species_data in all_species_data.items():
            if sp not in species_data:
                continue
            d     = species_data[sp]
            color = colors[name]
            rsd_d = rsd_from_S(d["S_dem"],       d["activated"])
            rsd_m = rsd_from_S(d["traj_markov"], d["activated"])
            label = name[:40]  # tronquer pour la lisibilité
            ax.plot(d["times_dem"],    rsd_d, "-",   color=color, lw=1.8,
                    alpha=0.85, label=f"DEM — {label}", zorder=3)
            ax.plot(d["times_markov"], rsd_m, "o--", color=color, markersize=3,
                    lw=1.2, alpha=0.7, label=f"Markov — {label}", zorder=2)

        ax.set_title(f"RSD — espèce '{sp}'")
        ax.set_xlabel("Temps (pas)"); ax.set_ylabel("RSD")
        ax.legend(fontsize=7, ncol=2); ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "compare_rsd.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 compare_rsd.png")


def fig_compare_states(all_species_data: dict[str, dict], out_dir: Path, k: int = 3):
    """
    États des k cellules les plus peuplées (union des top-k de chaque exp)
    — toutes les expériences superposées, une sous-figure par (espèce, cellule).
    """
    all_species = sorted({sp for sd in all_species_data.values() for sp in sd})
    names       = list(all_species_data.keys())
    cmap        = plt.cm.tab10
    colors      = {n: cmap(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}

    for sp in all_species:
        # Union des top-k cellules de toutes les expériences
        top_cells_set = set()
        for sd in all_species_data.values():
            if sp not in sd:
                continue
            mean_occ = sd[sp]["S_dem"].mean(axis=0)
            top_cells_set.update(np.argsort(mean_occ)[::-1][:k].tolist())
        cells = sorted(top_cells_set)

        n     = len(cells)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
        fig.suptitle(f"Comparaison états — espèce '{sp}'",
                     fontsize=13, fontweight="bold", y=1.01)

        for idx, cell in enumerate(cells):
            ax = axes[idx // ncols][idx % ncols]
            for name, sd in all_species_data.items():
                if sp not in sd:
                    continue
                d     = sd[sp]
                color = colors[name]
                label = name[:35]
                if cell < d["S_dem"].shape[1]:
                    ax.plot(d["times_dem"], d["S_dem"][:, cell],
                            "-", color=color, lw=1.8, alpha=0.85,
                            label=f"DEM — {label}", zorder=3)
                    ax.plot(d["times_markov"], d["traj_markov"][:, cell],
                            "o--", color=color, markersize=3, lw=1.2, alpha=0.7,
                            label=f"Markov — {label}", zorder=2)

            ax.set_title(f"Cellule {cell}")
            ax.set_xlabel("Temps (pas)"); ax.set_ylabel("Nb particules")
            ax.legend(fontsize=6, ncol=2)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: f"{int(x)}")
            )

        for idx in range(n, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout()
        fname = f"compare_etats_{sp}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"   💾 {fname}")


def fig_compare_n_particles(all_species_data: dict[str, dict], out_dir: Path):
    """
    Nombre total de particules par espèce au cours du temps
    (somme sur toutes les cellules actives) — DEM vs Markov.
    """
    all_species = sorted({sp for sd in all_species_data.values() for sp in sd})
    names       = list(all_species_data.keys())
    cmap        = plt.cm.tab10
    colors      = {n: cmap(i / max(len(names) - 1, 1)) for i, n in enumerate(names)}

    fig, axes = plt.subplots(1, len(all_species),
                             figsize=(8 * len(all_species), 5), squeeze=False)
    fig.suptitle("Nombre total de particules par espèce", fontweight="bold")

    for col, sp in enumerate(all_species):
        ax = axes[0][col]
        for name, sd in all_species_data.items():
            if sp not in sd:
                continue
            d     = sd[sp]
            color = colors[name]
            label = name[:40]
            n_dem    = d["S_dem"][:, d["activated"]].sum(axis=1)
            n_markov = d["traj_markov"][:, d["activated"]].sum(axis=1)
            ax.plot(d["times_dem"],    n_dem,    "-",   color=color, lw=1.8,
                    alpha=0.85, label=f"DEM — {label}", zorder=3)
            ax.plot(d["times_markov"], n_markov, "o--", color=color, markersize=3,
                    lw=1.2, alpha=0.7, label=f"Markov — {label}", zorder=2)

        ax.set_title(f"N particules — espèce '{sp}'")
        ax.set_xlabel("Temps (pas)"); ax.set_ylabel("N particules")
        ax.legend(fontsize=7, ncol=2); ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "compare_n_particules.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 compare_n_particules.png")


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE COMPARAISON
# ═════════════════════════════════════════════════════════════════════════════

def run_comparison(
    experiments: list[tuple[str, str]],
    keywords: list[str],
    bucket_prefix: str = "_Good/Experiment",
    top_states: int = 3,
):
    keyword_slug = "_".join(keywords)
    print(f"\n{'═'*60}")
    print(f"🔀 Comparaison : {keyword_slug}")
    print(f"   {len(experiments)} expériences")
    print(f"{'═'*60}")

    # Chargement de toutes les expériences
    all_species_data: dict[str, dict] = {}
    for path_hf, short in experiments:
        print(f"\n   📥 Chargement : {short}")
        try:
            exp          = load_experiment(path_hf)
            species_data = prepare_species(exp)
            all_species_data[short] = species_data
        except Exception as e:
            print(f"   ⚠️  {short} ignoré : {e}")

    if not all_species_data:
        print("❌ Aucune expérience chargée."); return

    bucket_subfolder = f"comparaisons/{keyword_slug}"

    with PostprocessingBucketUploader(bucket_subfolder=bucket_subfolder) as tmp:
        out_dir = tmp / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── États par espèce pour chaque expérience ──────────────────────
        print("\n📈 États par espèce (individuels)...")
        for short, species_data in all_species_data.items():
            for sp, sd in species_data.items():
                try:
                    fig_states_by_species(sp, sd, short, out_dir)
                except Exception as e:
                    print(f"   ⚠️  {short} / {sp} ignoré : {e}")

        # ── Figures de comparaison ────────────────────────────────────────
        print("\n📊 Figures de comparaison...")
        fig_compare_rsd(all_species_data, out_dir)
        fig_compare_states(all_species_data, out_dir, k=top_states)
        fig_compare_n_particles(all_species_data, out_dir)

    print(f"\n✅ Comparaison '{keyword_slug}' — terminée.\n")
# ═════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ═════════════════════════════════════════════════════════════════════════════

def rsd_from_S(S, activated):
    S_a  = S[:, activated]
    mean = S_a.mean(axis=1); std = S_a.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)


def rsd_concentration(S_small, S_large, act_s, act_l):
    act   = act_s & act_l
    total = S_small[:, act] + S_large[:, act]
    C     = np.where(total > 0, S_small[:, act] / total, 0.0)
    mean  = C.mean(axis=1); std = C.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)


def entropy_from_S(S, activated):
    S_a   = S[:, activated]
    N     = S_a.sum(axis=1, keepdims=True); N = np.where(N > 0, N, 1.0)
    p     = S_a / N
    H     = np.zeros(len(S_a))
    for t in range(len(S_a)):
        pt = p[t]; m = pt > 0
        if m.any(): H[t] = -np.sum(pt[m] * np.log(pt[m]))
    return H


def entropy_concentration(S_small, S_large, act_s, act_l):
    act   = act_s & act_l
    total = S_small[:, act] + S_large[:, act]; total = np.where(total > 0, total, 1.0)
    C     = S_small[:, act] / total
    H     = np.zeros(len(C))
    for t in range(len(C)):
        Ct = C[t]; v = (Ct > 0) & (Ct < 1)
        if v.any():
            Cv = Ct[v]
            H[t] = -np.sum(Cv * np.log(Cv) + (1 - Cv) * np.log(1 - Cv))
    return H


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════

def _dem_label(sp):    return f"DEM — {sp}"
def _markov_label(sp): return f"Markov — {sp}"

def _colors(sp):
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
):
    """
    Trace une grille de sous-figures (max 3 colonnes).
    Chaque sous-figure = une cellule.
    DEM tracé EN PREMIER (au premier plan via zorder=3), Markov derrière (zorder=2).
    """
    n      = len(cell_indices)
    ncols  = min(3, n)
    nrows  = (n + ncols - 1) // ncols
    colors = _colors(sp)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"États par partition — espèce '{sp}' — {title_suffix}\n{short_name}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    for idx, cell in enumerate(cell_indices):
        ax   = axes[idx // ncols][idx % ncols]
        t_d  = sp_data["times_dem"]
        S_d  = sp_data["S_dem"]
        t_m  = sp_data["times_markov"]
        S_m  = sp_data["traj_markov"]

        # ── [FIX] DEM tracé en premier — au premier plan (zorder=3) ──
        ax.plot(
            t_d, S_d[:, cell],
            "-", color=colors["dem"], linewidth=2.0, alpha=0.9,
            label=_dem_label(sp),
            zorder=3,
        )
        # ── Markov tracé en second — derrière (zorder=2) ──
        ax.plot(
            t_m, S_m[:, cell],
            "o--", color=colors["markov"], markersize=4,
            linewidth=1.4, alpha=0.85,
            label=_markov_label(sp),
            zorder=2,
        )

        mean_occ = S_d[:, cell].mean()
        ax.set_title(f"Cellule {cell}  (occupation moy. DEM : {mean_occ:.1f})", pad=6)
        ax.set_xlabel("Temps (pas)")
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


def fig_states_top3_index(sp: str, sp_data: dict, short_name: str, out_dir: Path):
    """Cellules 0, 1, 2 (par index)."""
    n_states = sp_data["traj_markov"].shape[1]
    cells    = list(range(min(3, n_states)))
    _plot_states_grid(
        cells, sp, sp_data, short_name,
        title_suffix="3 premiers états (index 0-1-2)",
        out_dir=out_dir,
        filename=f"etats_top3_index_{sp}.png",
    )


def fig_states_top_populated(sp: str, sp_data: dict, short_name: str, out_dir: Path, k: int = 6):
    """Top-k cellules par occupation moyenne DEM."""
    S_dem    = sp_data["S_dem"]
    mean_occ = S_dem.mean(axis=0)
    cells    = np.argsort(mean_occ)[::-1][:k].tolist()
    _plot_states_grid(
        cells, sp, sp_data, short_name,
        title_suffix=f"Top {k} cellules les plus peuplées (DEM)",
        out_dir=out_dir,
        filename=f"etats_top{k}_peuplees_{sp}.png",
    )


# ── Matrice de transition ─────────────────────────────────────────────────────

def fig_transition_matrix(sp: str, sp_data: dict, short_name: str, out_dir: Path):
    P    = sp_data["P"]
    n    = P.shape[0]
    vmax = np.percentile(P[P > 0], 98) if (P > 0).any() else 1.0

    fig, ax = plt.subplots(figsize=(n * 0.7 + 2, n * 0.7 + 2))
    im = ax.imshow(P, aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                   interpolation="nearest")
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
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if val > vmax * 0.5 else "black")

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

def fig_spectral_diagnostic(sp: str, sp_data: dict, short_name: str, out_dir: Path):
    P         = sp_data["P"]
    activated = sp_data["activated"]

    eigvals, eigvecs = np.linalg.eig(P.T)
    idx    = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]; eigvecs = eigvecs[:, idx]
    pi      = np.abs(eigvecs[:, 0]); pi /= pi.sum()
    pi_act  = pi[activated]
    rsd_pi  = pi_act.std() / pi_act.mean() if pi_act.mean() > 0 else 0.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Diagnostic spectral de P — espèce '{sp}'\n{short_name}",
        fontweight="bold",
    )

    # Spectre dans le disque unité
    theta = np.linspace(0, 2 * np.pi, 300)
    axes[0].plot(np.cos(theta), np.sin(theta), "r--", lw=0.8, label="Cercle unité")
    axes[0].scatter(eigvals.real, eigvals.imag, s=25, alpha=0.75,
                    color="#2196F3", edgecolors="white", linewidth=0.4, zorder=3)
    axes[0].axhline(0, color="grey", lw=0.5); axes[0].axvline(0, color="grey", lw=0.5)
    axes[0].set_title("Spectre de P (valeurs propres)")
    axes[0].set_aspect("equal"); axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Re(λ)"); axes[0].set_ylabel("Im(λ)")

    # Distribution stationnaire π
    axes[1].bar(range(len(pi)), pi, color="#43A047", alpha=0.8, edgecolor="white")
    axes[1].set_title(f"Distribution stationnaire π\n(RSD actif = {rsd_pi:.3f})")
    axes[1].set_xlabel("Cellule"); axes[1].set_ylabel("π(i)")
    axes[1].axhline(1 / activated.sum(), color="red", lw=1.2, linestyle="--",
                    label="Uniforme (actif)")
    axes[1].legend()

    # Matrice P (vue compacte)
    im = axes[2].imshow(P, aspect="auto", cmap="viridis", vmin=0,
                        interpolation="nearest")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Matrice P (nettoyée)")
    axes[2].set_xlabel("Cellule source"); axes[2].set_ylabel("Cellule dest.")

    fig.tight_layout()
    fname = f"diagnostic_spectral_{sp}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname}")


# ── RSD par espèce ────────────────────────────────────────────────────────────

def fig_rsd(species_data: dict, short_name: str, out_dir: Path):
    species_list = list(species_data.keys())
    n    = len(species_list)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    fig.suptitle(f"RSD global par espèce\n{short_name}", fontweight="bold")

    for i, sp in enumerate(species_list):
        d      = species_data[sp]
        colors = _colors(sp)
        ax     = axes[0][i]

        rsd_d = rsd_from_S(d["S_dem"],       d["activated"])
        rsd_m = rsd_from_S(d["traj_markov"], d["activated"])

        ax.plot(d["times_dem"],    rsd_d, "-",  color=colors["dem"],    lw=2.0,
                alpha=0.9, label=_dem_label(sp), zorder=3)
        ax.plot(d["times_markov"], rsd_m, "o--", color=colors["markov"], markersize=4,
                lw=1.4, alpha=0.85, label=_markov_label(sp), zorder=2)

        ax.set_title(f"RSD — espèce '{sp}'")
        ax.set_xlabel("Temps (pas)"); ax.set_ylabel("RSD")
        ax.legend(); ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "rsd_par_espece.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 rsd_par_espece.png")


# ── Concentration ─────────────────────────────────────────────────────────────

def fig_concentration(species_data: dict, short_name: str, out_dir: Path):
    sps = list(species_data.keys())
    if len(sps) < 2:
        print("   ⚠️  Moins de 2 espèces — figure concentration ignorée.")
        return

    sp_a, sp_b = sps[0], sps[1]
    da, db     = species_data[sp_a], species_data[sp_b]
    n_m   = min(len(da["times_markov"]), len(db["times_markov"]))
    n_d   = min(len(da["times_dem"]),    len(db["times_dem"]))

    rsd_c_d = rsd_concentration(
        da["S_dem"][:n_d],       db["S_dem"][:n_d],
        da["activated"],         db["activated"],
    )
    rsd_c_m = rsd_concentration(
        da["traj_markov"][:n_m], db["traj_markov"][:n_m],
        da["activated"],         db["activated"],
    )
    ent_c_d = entropy_concentration(
        da["S_dem"][:n_d],       db["S_dem"][:n_d],
        da["activated"],         db["activated"],
    )
    ent_c_m = entropy_concentration(
        da["traj_markov"][:n_m], db["traj_markov"][:n_m],
        da["activated"],         db["activated"],
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Concentration C({sp_a}) — RSD & Entropie\n{short_name}",
        fontweight="bold",
    )

    for ax, (yd, ym, ylabel, title) in zip(axes, [
        (rsd_c_d, rsd_c_m, "RSD de C",       f"RSD de concentration C({sp_a})"),
        (ent_c_d, ent_c_m, "Entropie de C",  f"Entropie de concentration C({sp_a})"),
    ]):
        ax.plot(da["times_dem"][:n_d],    yd, "-",  color="#2196F3", lw=2.0,
                alpha=0.9, label="DEM", zorder=3)
        ax.plot(da["times_markov"][:n_m], ym, "o--", color="#E53935", markersize=4,
                lw=1.4, alpha=0.85, label="Markov", zorder=2)
        ax.set_title(title); ax.set_xlabel("Temps (pas)"); ax.set_ylabel(ylabel)
        ax.legend(); ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / "concentration_rsd_entropie.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 concentration_rsd_entropie.png")
def fig_states_by_species(
    sp: str,
    sp_data: dict,
    short_name: str,
    out_dir: Path,
):
    S_d       = np.asarray(sp_data["S_dem"])
    S_m       = np.asarray(sp_data["traj_markov"])
    t_d       = np.asarray(sp_data["times_dem"])
    t_m       = np.asarray(sp_data["times_markov"])
    activated = sp_data["activated"]

    # Correction de shape : (T, 1, N) ou (1, T, N) → (T, N)
    if S_d.ndim == 3:
        S_d = S_d.reshape(-1, S_d.shape[-1])
    if S_m.ndim == 3:
        S_m = S_m.reshape(-1, S_m.shape[-1])
    if t_d.ndim > 1:
        t_d = t_d.ravel()
    if t_m.ndim > 1:
        t_m = t_m.ravel()

    # Debug temporaire — à supprimer une fois confirmé
    print(f"      S_d={S_d.shape}  S_m={S_m.shape}  t_d={t_d.shape}  t_m={t_m.shape}")

    n_cells = len(activated)
    cmap    = plt.cm.tab20
    colors  = [cmap(i / max(n_cells - 1, 1)) for i in range(n_cells)]

    fig, ax = plt.subplots(figsize=(12, 5))

    for idx, cell in enumerate(activated):
        color = colors[idx]
        ax.plot(t_d, S_d[:, cell], "-",
                color=color, linewidth=1.8, alpha=0.85,
                label=f"DEM — Cellule {cell}", zorder=3)
        ax.plot(t_m, S_m[:, cell], "o--",
                color=color, markersize=3, linewidth=1.2, alpha=0.7,
                label=f"Markov — Cellule {cell}", zorder=2)

    ax.set_title(f"États par cellule — espèce '{sp}'\n{short_name}", fontweight="bold")
    ax.set_xlabel("Temps (pas DEM)")
    ax.set_ylabel("Nombre de particules")
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fname = f"etats_espece_{sp}_{short_name}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname}")
# ── Entropie totale ───────────────────────────────────────────────────────────

def fig_entropy_total(species_data: dict, short_name: str, out_dir: Path):
    sps = list(species_data.keys())
    if len(sps) < 2:
        return
    sp_a, sp_b = sps[0], sps[1]
    da, db     = species_data[sp_a], species_data[sp_b]
    n_m = min(len(da["times_markov"]), len(db["times_markov"]))
    n_d = min(len(da["times_dem"]),    len(db["times_dem"]))

    ent_d = (
        entropy_from_S(da["S_dem"][:n_d],       da["activated"]) +
        entropy_from_S(db["S_dem"][:n_d],       db["activated"])
    )
    ent_m = (
        entropy_from_S(da["traj_markov"][:n_m], da["activated"]) +
        entropy_from_S(db["traj_markov"][:n_m], db["activated"])
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(da["times_dem"][:n_d],    ent_d, "-",  color="#607D8B", lw=2.0,
            alpha=0.9, label="DEM (total)", zorder=3)
    ax.plot(da["times_markov"][:n_m], ent_m, "o--", color="#9C27B0", markersize=4,
            lw=1.4, alpha=0.85, label="Markov (total)", zorder=2)
    ax.set_title(f"Entropie totale ({sp_a} + {sp_b})\n{short_name}", fontweight="bold")
    ax.set_xlabel("Temps (pas)"); ax.set_ylabel("H (nats)"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "entropie_totale.png", bbox_inches="tight")
    plt.close(fig)
    print("   💾 entropie_totale.png")


# ── Maillage 3D ───────────────────────────────────────────────────────────────

def fig_mesh(
    exp: dict,
    df_start,           # ← pd.DataFrame déjà chargé, plus de path_hf
    short_name: str,
    out_dir_img: Path,
    out_dir_files: Path,
):
    config  = exp["config"]
    start   = config.get("start_index", 250)
    method  = config.get("method", "voronoi")
    kwargs  = config.get("method_kwargs", {})

    coords = df_start[["coordinates:0", "coordinates:1", "coordinates:2"]].to_numpy()

    first_sp  = next(iter(exp["species"]))
    times     = exp["species"][first_sp]["times"]
    S_mat     = exp["species"][first_sp]["S_matrix"]
    row_start = np.searchsorted(times, start)
    S_start   = S_mat[row_start]

    cell_ids = np.repeat(np.arange(len(S_start)), S_start.astype(int))

    assert len(cell_ids) == len(coords), (
        f"Désaccord cell_ids ({len(cell_ids)}) vs coords ({len(coords)})"
    )

    mesh = pv.PolyData(coords)
    mesh.point_data["cell_id"] = cell_ids

    if "Diameter" in df_start.columns:
        mesh.point_data["Diameter"] = df_start["Diameter"].to_numpy()
        sphere = pv.Sphere(radius=0.5, theta_resolution=12, phi_resolution=12)
        glyph  = mesh.glyph(geom=sphere, orient=False, factor=1.0, scale="Diameter")
        vtp_path = out_dir_files / f"mesh_3d_{short_name}.vtp"
        glyph.save(str(vtp_path))
        mesh_to_plot = glyph
    else:
        vtp_path = out_dir_files / f"mesh_3d_{short_name}.vtp"
        mesh.save(str(vtp_path))
        mesh_to_plot = mesh

    print(f"   💾 {vtp_path.name}")

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh_to_plot, scalars="cell_id", cmap="hsv",
                     show_scalar_bar=True, scalar_bar_args={"title": "Cell ID"})
    plotter.add_title(f"Maillage 3D — {short_name}", font_size=10)
    fname_3d = f"mesh_3d_{short_name}.png"
    plotter.screenshot(str(out_dir_img / fname_3d))
    plotter.close()
    print(f"   💾 {fname_3d}")

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=cell_ids,
                    cmap="hsv", s=12, alpha=0.85, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="Cell ID")
    ax.set_title(f"Maillage 2D — Projection XY\n{short_name}", fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal")
    fname_2d = f"mesh_2d_{short_name}.png"
    fig.tight_layout()
    fig.savefig(out_dir_img / fname_2d, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname_2d}")


# ── Export matrices de transition ─────────────────────────────────────────────

def export_transition_matrices(species_data: dict, short_name: str, out_dir: Path):
    for sp, d in species_data.items():
        np.save(out_dir / f"P_{sp}_{short_name}.npy", d["P"])
        np.savetxt(out_dir / f"P_{sp}_{short_name}.txt", d["P"], fmt="%.6f")
        print(f"   💾 P_{sp}_{short_name}.npy / .txt")


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════

def run_postprocess(
    path_hf: str,
    short_name: str,
    bucket_prefix: str = "_Good/Experiment",
    top_states: int = 6,
    particle_diameter=None,
    df_start=None, 
):
    """Post-traite une expérience et upload les résultats dans le bucket."""
    category = get_simulation_category(short_name)
    print(f"\n{'═'*60}")
    print(f"🔬 Post-traitement : {short_name}")
    print(f"   Catégorie : {category}")
    print(f"   Chemin    : {path_hf}")
    print(f"{'═'*60}")

    # Chargement
    exp          = load_experiment(path_hf)
    species_data = prepare_species(exp)

    bucket_subfolder = f"postraitement/{category}/{short_name}"

    with PostprocessingBucketUploader(
        bucket_subfolder=bucket_subfolder,
        particle_diameter=particle_diameter,
    ) as tmp:

        # ── Arborescence locale ──────────────────────────────────────────
        img_etats   = tmp / "images" / "etats"
        img_rsd     = tmp / "images" / "rsd"
        img_matrices= tmp / "images" / "matrices"
        img_mesh    = tmp / "images" / "mesh"
        f_mesh      = tmp / "fichiers" / "mesh"
        f_trans     = tmp / "fichiers" / "transitions"
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
                fig_states_by_species(sp, sd, short_name, img_states)
            except Exception as e:
                print(f"   ⚠️  {sp} ignoré : {e}")
        # ── 3. RSD ───────────────────────────────────────────────────────
        print("\n📈 RSD & métriques...")
        fig_rsd(species_data, short_name, img_rsd)
        fig_concentration(species_data, short_name, img_rsd)
        fig_entropy_total(species_data, short_name, img_rsd)

        # ── 4. Maillage ──────────────────────────────────────────────────
        print("\n🗺️  Maillage...")
        try:
            fig_mesh(exp, df_start, short_name, img_mesh, f_mesh)
        except Exception as e:
            print(f"   ⚠️  Maillage ignoré : {e}")

        # ── 5. Export matrices brutes ─────────────────────────────────────
        print("\n📁 Export matrices brutes...")
        export_transition_matrices(species_data, short_name, f_trans)

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
        "--bucket-prefix", default="_Good/Experiment",
        help="Préfixe du bucket (défaut : _Good/Experiment)",
    )
    common.add_argument(
        "--top-states", type=int, default=6,
        help="Nombre de cellules les plus peuplées à tracer (défaut : 6)",
    )
    common.add_argument(
        "--dry-run", action="store_true",
        help="Liste les expériences sans les post-traiter",
    )

    # --- Style classique (pas de sous-commande) ---
    p.add_argument("--folder",   help="Nom exact du dossier d'expérience")
    p.add_argument("--keywords", nargs="+", help="Mots-clés pour trouver le dossier")
    p.add_argument("--category", help="Catégorie complète à traiter")
    p.add_argument("--bucket-prefix", default="_Good/Experiment",
                   dest="bucket_prefix",
                   help="Préfixe du bucket (défaut : _Good/Experiment)")
    p.add_argument("--top-states", type=int, default=6, dest="top_states")
    p.add_argument("--dry-run", action="store_true", dest="dry_run")

    # --- Sous-commandes ---
    sub = p.add_subparsers(dest="subcommand", metavar="SOUS-COMMANDE")

    s_single = sub.add_parser("single", parents=[common],
                               help="Post-traite un dossier par son nom exact")
    s_single.add_argument("folder", help="Nom exact du dossier")

    s_kw = sub.add_parser("keywords", parents=[common],
                           help="Trouve un dossier par mots-clés")
    s_kw.add_argument("keywords", nargs="+", help="Mots-clés")

    s_cat = sub.add_parser("category", parents=[common],
                            help="Post-traite toute une catégorie")
    s_cat.add_argument("category", help="Nom de la catégorie (ex: voronoi_simulations)")
    s_cmp = sub.add_parser("compare", parents=[common],
                        help="Compare toutes les expériences matchant les mots-clés")
    s_cmp.add_argument("keywords", nargs="+", help="Mots-clés communs aux expériences")


    return p


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    bucket_prefix = getattr(args, "bucket_prefix", "_Good/Experiment")
    top_states    = getattr(args, "top_states", 6)
    dry_run       = getattr(args, "dry_run", False)
    bucket_hf     = f"hf://buckets/{BUCKET_ID}/{bucket_prefix}"

    # ── Résolution du mode et de la liste d'expériences ─────────────────────
    mode     = None
    keywords = None

    if args.subcommand == "single":
        mode        = "single"
        path_hf, short = find_experiment_path(bucket_hf, folder_name=args.folder)
        experiments = [(path_hf, short)]

    elif args.subcommand == "keywords":
        mode        = "keywords"
        keywords    = args.keywords
        path_hf, short = find_experiment_path(bucket_hf, keywords=keywords)
        experiments = [(path_hf, short)]

    elif args.subcommand == "category":
        mode        = "category"
        experiments = list_category_paths(bucket_hf, args.category)
        if not experiments:
            print(f"❌ Aucune expérience trouvée dans '{args.category}'")
            sys.exit(1)

    elif args.subcommand == "compare":
        mode        = "compare"
        keywords    = args.keywords
        experiments = find_all_experiments_by_keywords(bucket_hf, keywords)
        if not experiments:
            print(f"❌ Aucune expérience trouvée pour {keywords}")
            sys.exit(1)

    # ── Style classique (pas de sous-commande) ───────────────────────────────
    elif getattr(args, "folder", None):
        mode        = "single"
        path_hf, short = find_experiment_path(bucket_hf, folder_name=args.folder)
        experiments = [(path_hf, short)]

    elif getattr(args, "keywords", None):
        mode        = "keywords"
        keywords    = args.keywords
        path_hf, short = find_experiment_path(bucket_hf, keywords=keywords)
        experiments = [(path_hf, short)]

    elif getattr(args, "category", None):
        mode        = "category"
        experiments = list_category_paths(bucket_hf, args.category)
        if not experiments:
            print(f"❌ Aucune expérience trouvée dans '{args.category}'")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(0)

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
    if mode != "compare":
        print("\n🔄 Chargement du parquet (une seule fois)...")
        first_exp = load_experiment(experiments[0][0])
        start     = first_exp["config"].get("start_index", 250)
        timestep_dict = load_parquet_as_timestep_dict(
            f"hf://buckets/{BUCKET_ID}/simulation_complete.parquet", fs
        )
        df_start = timestep_dict[start]
        print(f"   ✅ df_start chargé (timestep {start}, {len(df_start)} particules)\n")

    # ── Exécution ────────────────────────────────────────────────────────────
    if mode == "compare":
        run_comparison(experiments, keywords, bucket_prefix, top_states)
    else:
        for path_hf, short in experiments:
            try:
                run_postprocess(path_hf, short, bucket_prefix, top_states,
                                df_start=df_start)
            except Exception as e:
                print(f"⚠️  {short} — erreur : {e}")


if __name__ == "__main__":
    main()

