"""
postprocess_inhomogeneous.py.
=============================
Post-traitement automatisé des CHAÎNES INHOMOGÈNES (P_blocks) — DEM/Markov

Usage :
  python postprocess_inhomogeneous.py --folder inhomogeneous_voronoi_125cells_NLT3_step100_dt1_tau50_start157
  python postprocess_inhomogeneous.py --keywords inhomogeneous voronoi
  python postprocess_inhomogeneous.py --keywords inhomogeneous cylindrical NLT3
  python postprocess_inhomogeneous.py list                        # lister les expériences inhomogènes disponibles

Dépend de postprocess.py pour les fonctions de tracé.
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import seaborn as sns
import matplotlib as mpl
import re


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DEM_MCM1.src.bucket_io import (
    PostprocessingBucketUploader,
    get_simulation_category,
)
from DEM_MCM1.src.utils import load_parquet_as_timestep_dict

# ── Import des fonctions de tracé et utilitaires depuis postprocess.py ────────
from postprocess import (
    # Utilitaires
    BUCKET_ID,
    _load_json,
    # Export
    export_transition_matrices,
    fig_concentration,
    fig_discrepancy_analysis,
    fig_entropy_total,
    fig_matrice_population_heatmap,
    fig_mesh,
    fig_population_par_cellule,
    fig_rsd,
    fig_spectral_diagnostic,
    fig_states_by_species,
    # Fonctions de tracé (homogènes, réutilisées)
    fig_states_top3_index,
    fig_states_top_populated,
    fig_states_totale,
    fig_teneur,
    fig_transition_matrix,
    find_all_experiments_by_keywords,
    find_experiment_paths,
    fs,
    load_experiment,
    prepare_species_inhomogeneous,
    prepare_species,
    calculate_abs_error_over_time,
    fig_compare_hom_vs_inhom,
)

HF_BASE = f"hf://buckets/{BUCKET_ID}/_Good/Experiment"


# =============================================================================
# FIGURES SPÉCIFIQUES AUX CHAÎNES INHOMOGÈNES (P_blocks)
# =============================================================================


def fig_matrices_blocks_grid(
    sp: str, sp_data: dict, short_name: str, out_dir: Path
) -> None:
    """
    Grille de heatmaps : une sous-figure par P_k (un bloc NLT).
    Permet de visualiser l'évolution de la matrice de transition
    au fil des blocs temporels.
    """
    P_blocks = sp_data.get("P_blocks")
    if P_blocks is None:
        print(f"      ⚠️  Pas de P_blocks pour '{sp}' — figure ignorée")
        return
    n_blocks, n_states, _ = P_blocks.shape
    all_nonzero = P_blocks[P_blocks > 0]
    vmax = np.percentile(all_nonzero, 98) if len(all_nonzero) > 0 else 1.0
    ncols = min(3, n_blocks)
    nrows = (n_blocks + ncols - 1) // ncols

    # Repart des valeurs par défaut de matplotlib (ignore tout sns.set_theme
    # / sns.set_style / plt.style.use appelé ailleurs dans le pipeline),
    # puis applique explicitement "pas de grille" par-dessus.
    base_rc = dict(mpl.rcParamsDefault)
    base_rc.update(
        {
            "axes.grid": False,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.alpha": 0,
            "grid.linewidth": 0,
        }
    )

    with mpl.rc_context(base_rc):
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), squeeze=False
        )
        fig.suptitle(
            f"Matrices P_k par bloc NLT — espèce '{sp}'\n{short_name}",
            fontsize=13,
            fontweight="bold",
        )
        for k in range(n_blocks):
            ax = axes[k // ncols][k % ncols]
            P_k = P_blocks[k]
            im = ax.imshow(
                P_k,
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(f"Bloc {k + 1}/{n_blocks}", fontsize=10)
            ax.set_xlabel("Source (j)")
            ax.set_ylabel("Dest. (i)")
            ax.tick_params(which="both", bottom=False, left=False, labelsize=7)
            for i in range(n_states):
                for j in range(n_states):
                    val = P_k[i, j]
                    if val > 0.05:
                        color = "white" if val > vmax * 0.5 else "black"
                        ax.text(
                            j,
                            i,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color=color,
                        )
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=6)

            # Nettoyage final, forcé, indépendant des rcParams :
            # supprime toute ligne de grille déjà attachée à l'axe,
            # puis désactive explicitement le grid.
            ax.set_facecolor("white")
            ax.grid(False, which="both")
            for line in ax.get_xgridlines() + ax.get_ygridlines():
                line.set_visible(False)

        for k in range(n_blocks, nrows * ncols):
            axes[k // ncols][k % ncols].axis("off")
        fig.tight_layout()
        fname = f"P_blocks_grid_{sp}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
    print(f"   💾 {fname}")

def fig_matrix_components_evolution(
    sp: str,
    sp_data: dict,
    short_name: str,
    out_dir: Path,
    block_times: np.ndarray | None = None,
) -> None:
    """
    Évolution temporelle des composantes p_ij des matrices P_k.

    Pour chaque paire (i, j) où p_ij est significative (> seuil),
    on trace la valeur de p_ij en fonction du temps de simulation.

    Paramètres
    ----------
    block_times : array, optionnel
        Temps de simulation (centièmes de seconde) pour chaque bloc.
        Si None, utilise les indices de blocs 0, 1, …, n_blocks-1.
    """
    P_blocks = sp_data.get("P_blocks")
    if P_blocks is None:
        print(f"      ⚠️  Pas de P_blocks pour '{sp}' — figure ignorée")
        return

    n_blocks, _n_states, _ = P_blocks.shape
    if block_times is None:
        x_values = np.arange(n_blocks)
        xlabel = "Bloc NLT"
    else:
        x_values = np.asarray(block_times, dtype=float)
        xlabel = "Temps (centièmes de seconde)"

    # Seuil de significativité (on ignore les transitions quasi-nulles)
    threshold = 0.01
    # Union des p_ij significatives sur tous les blocs
    significant = np.any(P_blocks > threshold, axis=0)  # (S, S) booléen
    sig_pairs = np.where(significant)
    n_sig = len(sig_pairs[0])

    if n_sig == 0:
        print(f"      ⚠️  Aucune transition > {threshold} pour '{sp}' — figure ignorée")
        return

    print(f"      📈 {n_sig} transitions significatives à tracer pour '{sp}'")

    # Limiter le nombre de sous-figures par page
    max_cells_per_fig = 16
    n_figs = (n_sig + max_cells_per_fig - 1) // max_cells_per_fig

    for fig_idx in range(n_figs):
        start_idx = fig_idx * max_cells_per_fig
        end_idx = min(start_idx + max_cells_per_fig, n_sig)
        pairs_in_fig = end_idx - start_idx

        ncols = min(4, pairs_in_fig)
        nrows = (pairs_in_fig + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False
        )
        fig.suptitle(
            f"Évolution p_ij au cours du temps — '{sp}' (page {fig_idx + 1}/{n_figs})\n"
            f"{short_name}",
            fontsize=11,
            fontweight="bold",
        )

        for idx_in_fig in range(pairs_in_fig):
            global_idx = start_idx + idx_in_fig
            i = sig_pairs[0][global_idx]  # destination
            j = sig_pairs[1][global_idx]  # source
            values = P_blocks[:, i, j]

            ax = axes[idx_in_fig // ncols][idx_in_fig % ncols]
            ax.plot(
                x_values,
                values,
                "-o",
                color="#2196F3",
                markersize=6,
                linewidth=1.8,
                alpha=0.85,
                zorder=3,
            )
            ax.axhline(0, color="grey", lw=0.5, ls="--", alpha=0.5)
            ax.set_title(f"$p_{{{i}{j}}}$", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel("P", fontsize=8)
            ax.set_ylim(bottom=0)
            ax.tick_params(labelsize=7)

            # Annoter les valeurs
            for k_idx, val in enumerate(values):
                if val > threshold:
                    ax.annotate(
                        f"{val:.2f}",
                        (x_values[k_idx], val),
                        textcoords="offset points",
                        xytext=(0, 6),
                        fontsize=6,
                        ha="center",
                        color="#1565C0",
                    )

        # Cacher les axes vides
        for idx_in_fig in range(pairs_in_fig, nrows * ncols):
            axes[idx_in_fig // ncols][idx_in_fig % ncols].axis("off")

        fig.tight_layout()
        fname = f"P_components_evolution_{sp}_p{fig_idx + 1}.png"
        fig.savefig(out_dir / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"   💾 {fname}")


def fig_matrix_differences(
    sp: str, sp_data: dict, short_name: str, out_dir: Path
) -> None:
    """
    Différence entre matrices P_k consécutives : Δ_k = P_{k+1} - P_k.

    Montre comment la matrice de transition change entre deux blocs NLT.
    """
    P_blocks = sp_data.get("P_blocks")
    if P_blocks is None:
        print(f"      ⚠️  Pas de P_blocks pour '{sp}' — figure ignorée")
        return

    n_blocks, _n_states, _ = P_blocks.shape
    if n_blocks < 2:
        print(f"      ⚠️  Moins de 2 blocs pour '{sp}' — figure ignorée")
        return

    # Calcul des différences
    diffs = np.diff(P_blocks, axis=0)  # (n_blocks-1, S, S)
    n_diffs = len(diffs)

    # Échelle de couleur symétrique
    vmax = max(abs(diffs.min()), abs(diffs.max()))
    vmax = max(vmax, 0.01)  # éviter vmax=0

    ncols = min(3, n_diffs)
    nrows = (n_diffs + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), squeeze=False
    )
    fig.suptitle(
        f"Différence entre matrices P_k consécutives — espèce '{sp}'\n{short_name}",
        fontsize=13,
        fontweight="bold",
    )

    for d_idx in range(n_diffs):
        ax = axes[d_idx // ncols][d_idx % ncols]
        D = diffs[d_idx]
        im = ax.imshow(
            D,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(
            f"Δ_{{ {d_idx + 1} }} = P_{{ {d_idx + 1} }} − P_{{ {d_idx} }}",
            fontsize=10,
        )
        ax.set_xlabel("Source (j)")
        ax.set_ylabel("Dest. (i)")
        ax.tick_params(which="both", bottom=False, left=False, labelsize=7)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Δ P", fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    # Cacher les axes vides
    for d_idx in range(n_diffs, nrows * ncols):
        axes[d_idx // ncols][d_idx % ncols].axis("off")

    fig.tight_layout()
    fname = f"P_blocks_differences_{sp}.png"
    fig.savefig(out_dir / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"   💾 {fname}")


def list_inhomogeneous_experiments() -> None:
    """Liste toutes les expériences inhomogènes disponibles dans le bucket."""
    print("🔍 Recherche des expériences inhomogènes...")
    results = find_all_experiments_by_keywords(HF_BASE, keywords=["inhomogeneous"])
    if not results:
        print("   Aucune expérience inhomogène trouvée.")
        return
    print(f"\n📋 {len(results)} expérience(s) inhomogène(s) trouvée(s) :\n")
    for hf_path, short in sorted(results, key=lambda x: x[1]):
        try:
            config = _load_json(hf_path, "config.json")
            stats = _load_json(hf_path, "stats.json")
            method = config.get("method", "?")
            nlt = config.get("nlt", "?")
            n_blocks = stats.get("n_blocks", "?")
            n_states = stats.get("n_states", "?")
            print(f"   📁 {short}")
            print(
                f"      Méthode : {method} | NLT={nlt} | {n_blocks} blocs | {n_states} états"
            )
        except Exception as e:
            print(f"   📁 {short}  (⚠️  infos non chargées : {e})")


def run_inhomogeneous_postprocess(
    path_hf: str,
    short_name: str,
    bucket_prefix: str = "_Good/Experiment",
    top_states: int = 6,
    particle_diameter: float | None = None,
    df_start: pd.DataFrame | None = None,
    timestep_dict: dict | None = None,
) -> None:
    """
    Post-traite une expérience inhomogène (P_blocks) et upload vers le bucket.

    Similaire à run_postprocess() de postprocess.py mais utilise
    prepare_species_inhomogeneous() pour exploiter les matrices P_k multiples.
    """
    category = get_simulation_category(short_name)
    print(f"\n{'═' * 60}")
    print(f"🔬 Post-traitement INHOMOGÈNE : {short_name}")
    print(f"   Catégorie : {category}")
    print(f"   Chemin    : {path_hf}")
    print(f"{'═' * 60}")

    # ── Chargement (détection auto inhomogène via inhomogeneous_metadata.json) ──
    exp = load_experiment(path_hf)
    if not exp.get("inhomogeneous", False):
        print(f"   ⚠️  ATTENTION : {short_name} n'a pas été détectée comme inhomogène.")
        print("   Le fichier inhomogeneous_metadata.json est peut-être manquant.\n")

    species_data = prepare_species_inhomogeneous(exp)
    n_blocks = len(next(iter(species_data.values())).get("P_blocks", []))
    print(f"   ✅ {len(species_data)} espèce(s) chargée(s) — {n_blocks} blocs P_k")

    # ── Calcul des temps de simulation pour chaque bloc ─────────────────────
    c = exp["config"]
    start_base = c.get("start_index", 157)
    step = c.get("step", 157)
    tau = c.get("tau", 157)
    # Chaque bloc k commence à start_base + k * (step + tau)
    block_times = [start_base + k * (step + tau) for k in range(n_blocks)]
    print(f"   ⏱️  Temps (centièmes de seconde) : {block_times}")

    bucket_subfolder = f"postraitement/{category}/{short_name}"

    with PostprocessingBucketUploader(
        bucket_subfolder=bucket_subfolder,
        particle_diameter=particle_diameter,
    ) as tmp:
        # ── Arborescence locale ──────────────────────────────────────────────
        img_etats = tmp / "images" / "etats"
        img_rsd = tmp / "images" / "rsd"
        img_matrices = tmp / "images" / "matrices"
        img_mesh = tmp / "images" / "mesh"
        f_mesh = tmp / "fichiers" / "mesh"
        f_trans = tmp / "fichiers" / "transitions"
        for d in [img_etats, img_rsd, img_matrices, img_mesh, f_mesh, f_trans]:
            d.mkdir(parents=True, exist_ok=True)

        # ── 1. Figures d'états par espèce ────────────────────────────────────
        print("\n📊 Figures d'états...")
        for sp, sd in species_data.items():
            fig_states_top3_index(sp, sd, short_name, img_etats)
            fig_states_top_populated(sp, sd, short_name, img_etats, k=top_states)

        # ── 2. Matrices de transition — exploitation des P_blocks ────────────
        print("\n🔲 Matrices de transition — P_blocks (une par NLT)...")
        for sp, sd in species_data.items():
            # Matrice unique (P_blocks[0]) pour compatibilité
            fig_transition_matrix(sp, sd, short_name, img_matrices)
            fig_spectral_diagnostic(sp, sd, short_name, img_matrices)

            # INHOMOGÈNE : grille de toutes les matrices P_k
            fig_matrices_blocks_grid(sp, sd, short_name, img_matrices)

            # INHOMOGÈNE : évolution temporelle des composantes p_ij
            fig_matrix_components_evolution(
                sp, sd, short_name, img_matrices, block_times=block_times
            )

            # INHOMOGÈNE : différences entre matrices consécutives
            fig_matrix_differences(sp, sd, short_name, img_matrices)

        # ── Figure états par espèce ───────────────────────────────────────────
        print("\n📈 États par espèce...")
        for sp, sd in species_data.items():
            try:
                fig_states_by_species(sp, sd, short_name, img_etats)
            except Exception as e:
                print(f"   ⚠️  {sp} ignoré : {e}")

        # ── 3. RSD & métriques ───────────────────────────────────────────────
        print("\n📈 RSD & métriques...")
        fig_rsd(species_data, short_name, img_rsd)
        fig_concentration(species_data, short_name, img_rsd)
        fig_entropy_total(species_data, short_name, img_rsd)

        print("Teneur !!!")
        try:
            fig_teneur(species_data, short_name, img_etats)
        except Exception as e:
            print(f"   Teneur non construite : {e}")

        print("États totaux !!!")
        try:
            fig_states_totale(species_data, short_name, img_etats)
        except Exception as e:
            print(f"   États totaux non construits : {e}")

        # ── Analyse de l'écart Markov vs DEM ──────────────────────────────
        print("\n📊 Analyse de l'écart Markov vs DEM (inhomogène)...")
        for sp, sd in species_data.items():
            try:
                fig_discrepancy_analysis(sp, sd, short_name, img_etats)
            except Exception as e:
                print(f"   ⚠️  Discrepancy {sp} ignorée : {e}")

        # ── Comparaison avec la version homogène si disponible ───────────
        try:
            # enlever préfixe 'inhomogeneous_' pour retrouver l'expérience homogène
            if short_name.startswith("inhomogeneous_"):
                hom_short = short_name.replace("inhomogeneous_", "", 1)
            else:
                hom_short = short_name

            try:
                hom_paths = find_experiment_paths(HF_BASE, folder_name=hom_short)
            except FileNotFoundError:
                # Heuristique de secours : chercher par mots-clés extraits
                tokens = [t for t in re.split(r"[_\-]", hom_short) if t and not t.startswith("NLT")]
                print(f"   ℹ️  Recherche alternative homogène par mots-clés: {tokens}")
                hom_paths = find_all_experiments_by_keywords(HF_BASE, tokens)

            if hom_paths:
                hom_path_hf, hom_shortname = hom_paths[0]
                print(f"   ℹ️  Chargement version homogène : {hom_shortname}")
                exp_hom = load_experiment(hom_path_hf)
                hom_species_data = prepare_species(exp_hom)
                # tracer la comparaison pour chaque espèce commune
                for sp in species_data:
                    if sp in hom_species_data:
                        try:
                            fig_compare_hom_vs_inhom(
                                sp,
                                hom_species_data[sp],
                                species_data[sp],
                                short_name_hom=hom_shortname,
                                short_name_inhom=short_name,
                                out_dir=img_etats,
                            )
                        except Exception as e:
                            print(f"   ⚠️  Comparaison {sp} ignorée : {e}")
            else:
                print("   ℹ️  Aucune expérience homogène correspondante trouvée pour comparaison.")
        except Exception as e:
            print(f"   ⚠️  Erreur lors de la tentative de comparaison homogène/inhomogène: {e}")

        # ── 4. Maillage ──────────────────────────────────────────────────────
        print("\n🗺️  Maillage...")
        try:
            fig_mesh(
                exp, df_start, short_name, img_mesh, f_mesh, timestep_dict=timestep_dict
            )
        except Exception as e:
            print(f"   ⚠️  Maillage ignoré : {e}")

        # ── 5. Export matrices brutes ─────────────────────────────────────────
        print("\n📁 Export matrices brutes...")
        export_transition_matrices(species_data, short_name, f_trans)

        # ── 6. Population par cellule ─────────────────────────────────────────
        print("\n📊 Population par cellule...")
        for sp, sd in species_data.items():
            try:
                fig_population_par_cellule(sp, sd, short_name, img_etats)
                fig_matrice_population_heatmap(sp, sd, short_name, img_etats)
            except Exception as e:
                print(f"   ⚠️  {sp} ignoré : {e}")

    print(f"\n✅ {short_name} — post-traitement inhomogène terminé.\n")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Point d'entrée CLI : résout les expériences et lance le post-traitement."""
    parser = argparse.ArgumentParser(
        description="Post-traitement des chaînes Markoviennes INHOMOGÈNES (P_blocks)",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Exemples :
  # Post-traiter une expérience inhomogène par son nom exact
  python postprocess_inhomogeneous.py --folder inhomogeneous_voronoi_125cells_NLT3_...

  # Post-traiter par mots-clés
  python postprocess_inhomogeneous.py --keywords inhomogeneous voronoi NLT3

  # Lister toutes les expériences inhomogènes disponibles
  python postprocess_inhomogeneous.py list

  # Avec options
  python postprocess_inhomogeneous.py --folder mon_exp --top-states 8 --dry-run
""",
    )

    parser.add_argument("--folder", help="Nom exact du dossier d'expérience")
    parser.add_argument(
        "--keywords", nargs="+", help="Mots-clés pour trouver le dossier"
    )
    parser.add_argument(
        "--top-states",
        type=int,
        default=6,
        help="Nombre de cellules les plus peuplées à tracer (défaut : 6)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Liste les expériences sans les post-traiter",
    )
    parser.add_argument(
        "--bucket-prefix",
        default="_Good/Experiment",
        help="Préfixe du bucket (défaut : _Good/Experiment)",
    )

    args = parser.parse_args()

    # ── Sous-commande 'list' ─────────────────────────────────────────────────
    if getattr(args, "folder", None) == "list":
        list_inhomogeneous_experiments()
        return

    # ── Résolution de la liste d'expériences ─────────────────────────────────
    experiments = []

    if args.folder:
        results = find_experiment_paths(
            f"hf://buckets/{BUCKET_ID}/{args.bucket_prefix}",
            folder_name=args.folder,
        )
        if not results:
            print(f"❌ Aucune expérience trouvée pour folder='{args.folder}'")
            print("   Utilisez 'list' pour voir les expériences disponibles.")
            sys.exit(1)
        experiments = results
    elif args.keywords:
        results = find_experiment_paths(
            f"hf://buckets/{BUCKET_ID}/{args.bucket_prefix}",
            keywords=args.keywords,
        )
        if not results:
            print(f"❌ Aucune expérience trouvée pour keywords={args.keywords}")
            print("   Utilisez 'list' pour voir les expériences disponibles.")
            sys.exit(1)
        experiments = results
    else:
        print(
            "❌ Fournissez --folder ou --keywords (ou 'list' pour voir les disponibles)."
        )
        sys.exit(1)

    # ── Dry-run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"[DRY] {len(experiments)} expérience(s) inhomogène(s) :")
        for _, s in experiments:
            print(f"  - {s}")
        return

    print(f"\n📋 {len(experiments)} expérience(s) inhomogène(s) à traiter")
    for i, (_, s) in enumerate(experiments, 1):
        print(f"  [{i}/{len(experiments)}] {s}")

    # ── Chargement unique du parquet ─────────────────────────────────────────
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
    for path_hf, short in experiments:
        try:
            exp_config = load_experiment(path_hf)["config"]
            particle_diameter = exp_config.get("particle_diameter")

            run_inhomogeneous_postprocess(
                path_hf,
                short,
                bucket_prefix=args.bucket_prefix,
                top_states=args.top_states,
                particle_diameter=particle_diameter,
                df_start=df_start,
                timestep_dict=timestep_dict,
            )
        except Exception as e:
            print(f"⚠️  {short} — erreur : {e}")
            import traceback

            traceback.print_exc()

    print("\n✨ Terminé !")


if __name__ == "__main__":
    main()
