"""Études du rapport calculées avec la librairie ``dem_mcm_coupling``.

Contrairement à l'ancienne version (``etudes_rapport.py``, supprimée), ce
script ne réimplémente AUCUN calcul : les données DEM (chunks locaux) sont
couplées à la librairie via :class:`InMemoryDataSource`, et chaque étape
numérique appelle le code déjà implémenté :

* partitionneurs      : :func:`dem_mcm_coupling.partitioners.create_partitioner`
  + :func:`run_sweep._fit_partitioner_for_sweep` (fit sur le régime permanent) ;
* états / comptages   : :func:`run_sweep._compute_state_matrices`
  + :func:`run_sweep._build_state_matrices` ;
* paires de transition: :func:`run_sweep._build_pairs` (structure
  start/tau/step/dt/NLT de la librairie) ;
* matrice de transition (ligne-stochastique, ``S_next = S @ P``) :
  :func:`run_sweep.compute_P_matrix_torch` ;
* nettoyage / propagation / RSD : ``postprocessing.metrics``
  (:func:`clean_transition_matrix`, :func:`propagate_markov`,
  :func:`propagate_markov_inhomogeneous`, :func:`rsd_concentration`) ;
* espèces             : :func:`run_sweep._detect_species` (masques small/large).

Les matrices d'états étant indépendantes des paramètres temporels, elles
sont calculées une seule fois par méthode (exactement comme dans
``run_experiment``) puis réutilisées pour chaque configuration — les
résultats sont identiques à des appels répétés de ``run_experiment``.

Figures produites (mêmes noms que dans le rapport) :
``comparaison_methodes_rsd.png``, ``etude_start_rsd.png``,
``etude_tau_rsd.png``, ``etude_nlt_erreur.png``, ``etude_step_erreur.png``,
``etude_especes_rsd.png``, ``matrice_cylindrique_especes.png``,
``rsd_cylindrique.png``, ``espace_caracteristiques_physique.png``.

Usage::

    python postprocessing/etudes_librairie.py
"""

from __future__ import annotations

import gzip
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dem_mcm_coupling.partitioners import create_partitioner  # noqa: E402
from dem_mcm_coupling.run_sweep import (  # noqa: E402
    ExperimentConfig,
    _build_pairs,
    _build_state_matrices,
    _compute_state_matrices,
    _detect_species,
    _fit_partitioner_for_sweep,
    compute_P_matrix_torch,
    sample_coordinates,
    PERMANENT_START,
    N_PARTICLES_PER_TIMESTEP,
)
from postprocessing.metrics import (  # noqa: E402
    clean_transition_matrix,
    propagate_markov,
    propagate_markov_inhomogeneous,
    rsd_concentration,
)

FIGDIR = ROOT / "template-rapport-stage" / "figures"
FIGDIR.mkdir(exist_ok=True)

TAU = 157
START = 157
N_T = 6000

#: dégradé jaune -> rouge pour distinguer les quatre méthodes
COLORS = {
    "Cartésien": "#fec44f",
    "Cylindrique": "#fe9929",
    "Voronoï": "#ec7014",
    "Physique": "#cc4c02",
}

#: découpages géométriques : labellisation dans le repère du lit
GEO_METHODS = {"cartesien", "cylindrique"}


def make_frame(sample_coords, permanent_rows):
    """Repère du lit : origine au barycentre des positions du régime
    permanent, axes du plan transverse alignés sur les directions
    principales du lit (la surface libre inclinée devient un axe du repère).

    Ce changement de repère, appliqué avant la labellisation des découpages
    géométriques, répartit les particules dans les cellules de manière plus
    équilibrée ; les découpages par k-moyennes y sont insensibles
    (invariance par translation et rotation).
    """
    pts = np.asarray(sample_coords[permanent_rows:], dtype=np.float64)
    c = pts.mean(axis=0)
    xy = pts[:, :2] - c[:2]
    cov = xy.T @ xy / len(xy)
    _, evecs = np.linalg.eigh(cov)
    R2 = evecs[:, ::-1]  # colonnes : direction principale, puis normale
    if np.linalg.det(R2) < 0:
        R2[:, 1] = -R2[:, 1]
    return c, R2


def transform_coords(coords, c, R2):
    out = np.asarray(coords, dtype=np.float64).copy()
    out[:, :2] = (out[:, :2] - c[:2]) @ R2
    out[:, 2] -= c[2]
    return out


def transform_dict(timestep_dict, c, R2):
    """Copie du dictionnaire de pas de temps avec coordonnées transformées
    (colonnes de coordonnées uniquement : suffisant pour la labellisation
    des découpages géométriques)."""
    out = {}
    for idx, df in timestep_dict.items():
        xy = (df[["coordinates:0", "coordinates:1"]].to_numpy() - c[:2]) @ R2
        out[idx] = pd.DataFrame({
            "coordinates:0": xy[:, 0],
            "coordinates:1": xy[:, 1],
            "coordinates:2": df["coordinates:2"].to_numpy() - c[2],
        })
    return out


#: méthode -> (nom d'affichage, identifiant registry, kwargs) — 10 cellules
METHODES = {
    "cartesien": ("Cartésien", "cartesian", dict(nx=10, ny=1, nz=1)),
    "cylindrique": ("Cylindrique", "cylindrical",
                    dict(nr=1, ntheta=10, nz=1, radial_mode="equal_area")),
    "voronoi": ("Voronoï", "voronoi", dict(n_cells=10)),
    "physique": ("Physique", "physics",
                 dict(n_cells=10, velocity_weight=0.5)),
}


# ---------------------------------------------------------------------------
# Couplage des données locales dans la librairie
# ---------------------------------------------------------------------------


def load_timestep_dict() -> dict[int, pd.DataFrame]:
    """Charge les chunks locaux au format attendu par la librairie."""
    cols = ["Fichier_Source", "Diameter",
            "coordinates:0", "coordinates:1", "coordinates:2",
            "Velocity:0", "Velocity:1", "Velocity:2"]
    frames = []
    for path in sorted((ROOT / "data" / "chunks").glob("*.parquet.gz")):
        with gzip.open(path, "rb") as fh:
            frames.append(pd.read_parquet(io.BytesIO(fh.read()), columns=cols))
    df_full = pd.concat(frames, ignore_index=True)
    del frames

    timestep_dict: dict[int, pd.DataFrame] = {}
    for source, group in df_full.groupby("Fichier_Source", sort=False):
        idx = int(str(source).replace("data_", "").replace(".csv", ""))
        timestep_dict[idx] = group.drop(columns="Fichier_Source").reset_index(
            drop=True
        )
    print(f"📦 {len(timestep_dict)} timesteps couplés "
          f"(index {min(timestep_dict)} → {max(timestep_dict)})")
    return dict(sorted(timestep_dict.items()))


# ---------------------------------------------------------------------------
# Préparation par méthode : fit librairie + matrices d'états (une fois)
# ---------------------------------------------------------------------------


class EtudeMethode:
    """États et matrices S d'une méthode, préparés avec la librairie.

    Reproduit la préparation de :func:`run_experiment` (fit du partitionneur
    sur le régime permanent, labélisation de tous les instants, comptages
    par espèce), puis délègue chaque construction de matrice de transition
    à :func:`_build_pairs` + :func:`compute_P_matrix_torch`.
    """

    def __init__(self, key, timestep_dict, sample_coords, s_velocities,
                 frame=None):
        self.key = key
        self.nom, method, kwargs = METHODES[key]
        self.method = method
        self.timestep_dict = timestep_dict
        self.partitioner = create_partitioner(method, **kwargs)
        self.n_states = self.partitioner.n_cells

        cfg_fit = ExperimentConfig(method=method, method_kwargs=kwargs)
        permanent_rows = PERMANENT_START * N_PARTICLES_PER_TIMESTEP

        # Découpages géométriques : labellisation dans le repère du lit
        # (translation au barycentre + rotation aux axes principaux).
        label_dict = timestep_dict
        fit_coords = sample_coords
        if frame is not None and key in GEO_METHODS:
            c, R2 = frame
            fit_coords = transform_coords(sample_coords, c, R2)
            label_dict = transform_dict(timestep_dict, c, R2)
            # Dans le repère du lit, le fit se fait sur le régime permanent
            # pour les deux découpages géométriques : les bornes épousent le
            # lit stationnaire, ce qui équilibre l'occupation des cellules.
            self.partitioner.fit(fit_coords[permanent_rows:])
        else:
            _fit_partitioner_for_sweep(
                self.partitioner, cfg_fit, fit_coords, s_velocities,
                permanent_rows,
            )

        self.species_masks = _detect_species(
            timestep_dict[min(timestep_dict)]
        )
        self.states_matrix, self.sorted_indices = _compute_state_matrices(
            label_dict, self.partitioner, START
        )
        self.idx_to_row = {int(i): r for r, i in
                           enumerate(self.sorted_indices)}
        self.S_matrices = _build_state_matrices(
            self.states_matrix, self.species_masks, self.n_states
        )
        self.times = np.asarray(self.sorted_indices)

    # -- matrices de transition (code librairie) ---------------------------

    def _accumulate(self, pairs, mask):
        prev = np.concatenate(
            [self.states_matrix[self.idx_to_row[a]][mask] for a, b in pairs]
        )
        curr = np.concatenate(
            [self.states_matrix[self.idx_to_row[b]][mask] for a, b in pairs]
        )
        return prev, curr

    def build_P(self, config, species):
        """Matrice homogène d'une espèce (accumulation sur tous les blocs)."""
        pairs = _build_pairs(config, self.timestep_dict, self.idx_to_row)
        if not pairs:
            raise ValueError("Aucune paire valide")
        mask = (np.ones(N_PARTICLES_PER_TIMESTEP, bool) if species == "all"
                else self.species_masks[species])
        prev, curr = self._accumulate(pairs, mask)
        return compute_P_matrix_torch(prev, curr, self.n_states,
                                      "cpu").cpu().numpy()

    def build_P_blocks(self, config, species):
        """Une matrice par bloc NLT (chaîne inhomogène)."""
        mask = (np.ones(N_PARTICLES_PER_TIMESTEP, bool) if species == "all"
                else self.species_masks[species])
        blocks = []
        for k in range(config.nlt):
            cfg_k = ExperimentConfig(
                method=config.method, nlt=1, tau=config.tau,
                step=config.step, dt=config.dt,
                start_index=config.start_index + k * (config.step + config.tau),
            )
            pairs = _build_pairs(cfg_k, self.timestep_dict, self.idx_to_row)
            if not pairs:
                break
            prev, curr = self._accumulate(pairs, mask)
            blocks.append(compute_P_matrix_torch(
                prev, curr, self.n_states, "cpu").cpu().numpy())
        return np.array(blocks)

    # -- propagation + RSD (code postprocessing) ---------------------------

    def markov_rsd(self, config, start_pred=None):
        """RSD de concentration prédit (matrices distinctes par espèce)."""
        start_pred = config.start_index if start_pred is None else start_pred
        trajs, acts = {}, {}
        for sp in ("small", "large"):
            P_raw = self.build_P(config, sp)
            # Les lignes NaN (cellules jamais sources) sont désactivées.
            P_clean, activated = clean_transition_matrix(
                np.nan_to_num(P_raw, nan=0.0))
            row0 = np.searchsorted(self.times, start_pred)
            S0 = self.S_matrices[sp][row0].astype(float)
            traj, t_mk = propagate_markov(
                S0, P_clean, self.times, start_pred, config.tau, activated)
            trajs[sp], acts[sp] = traj, activated
        n = min(len(trajs["small"]), len(trajs["large"]))
        rsd = rsd_concentration(trajs["small"][:n], trajs["large"][:n],
                                acts["small"], acts["large"])
        return rsd, t_mk[:n], acts

    def markov_rsd_single_P(self, config, start_pred=None):
        """RSD prédit avec UNE matrice unique (sans distinction d'espèce)."""
        start_pred = config.start_index if start_pred is None else start_pred
        P_clean, activated = clean_transition_matrix(
            np.nan_to_num(self.build_P(config, "all"), nan=0.0))
        row0 = np.searchsorted(self.times, start_pred)
        trajs = {}
        for sp in ("small", "large"):
            S0 = self.S_matrices[sp][row0].astype(float)
            traj, t_mk = propagate_markov(
                S0, P_clean, self.times, start_pred, config.tau, activated)
            trajs[sp] = traj
        rsd = rsd_concentration(trajs["small"], trajs["large"],
                                activated, activated)
        return rsd, t_mk

    def markov_rsd_inhomogeneous(self, config):
        trajs, acts = {}, {}
        for sp in ("small", "large"):
            P_blocks = np.nan_to_num(self.build_P_blocks(config, sp),
                                      nan=0.0)
            P0_clean, activated = clean_transition_matrix(P_blocks[0])
            row0 = np.searchsorted(self.times, config.start_index)
            S0 = self.S_matrices[sp][row0].astype(float)
            traj, t_mk = propagate_markov_inhomogeneous(
                S0, P_blocks, self.times, config.start_index, config.tau,
                activated, step=config.step, nlt=len(P_blocks))
            trajs[sp], acts[sp] = traj, activated
        n = min(len(trajs["small"]), len(trajs["large"]))
        rsd = rsd_concentration(trajs["small"][:n], trajs["large"][:n],
                                acts["small"], acts["large"])
        return rsd, t_mk[:n]

    def markov_traj(self, config, start_pred=None):
        """Trajectoires prédites par espèce (comptages par cellule)."""
        start_pred = config.start_index if start_pred is None else start_pred
        trajs, acts = {}, {}
        for sp in ("small", "large"):
            P_raw = self.build_P(config, sp)
            # Les lignes NaN (cellules jamais sources) sont désactivées.
            P_clean, activated = clean_transition_matrix(
                np.nan_to_num(P_raw, nan=0.0))
            row0 = np.searchsorted(self.times, start_pred)
            S0 = self.S_matrices[sp][row0].astype(float)
            traj, t_mk = propagate_markov(
                S0, P_clean, self.times, start_pred, config.tau, activated)
            trajs[sp], acts[sp] = traj, activated
        return trajs, t_mk, acts

    def dem_rsd(self, times_probe, acts=None):
        rows = np.searchsorted(self.times, times_probe)
        S_s = self.S_matrices["small"][rows]
        S_l = self.S_matrices["large"][rows]
        if acts is None:
            a = np.ones(self.n_states, bool)
            return rsd_concentration(S_s, S_l, a, a)
        return rsd_concentration(S_s, S_l, acts["small"], acts["large"])


def config_for(method, **kw):
    base = dict(nlt=2, tau=TAU, step=TAU, dt=8, start_index=START)
    base.update(kw)
    return ExperimentConfig(method=method, **base)


# ---------------------------------------------------------------------------
# Études (figures du rapport)
# ---------------------------------------------------------------------------


def comparaison_methodes(etudes):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    resume = {}
    for ax, (key, et) in zip(axes.flat, etudes.items()):
        cfg = config_for(et.method)
        rsd_mk, t_mk, acts = et.markov_rsd(cfg)
        t_dem = np.arange(START, N_T, 20)
        rsd_dem = et.dem_rsd(t_dem, acts)
        rsd_dem_mk = et.dem_rsd(t_mk, acts)
        err = float(np.mean(np.abs(rsd_mk - rsd_dem_mk)))
        resume[et.nom] = err
        ax.plot(t_dem / 100, rsd_dem, "k-", lw=0.9, alpha=0.7, label="DEM")
        ax.plot(t_mk / 100, rsd_mk, "o--", color=COLORS[et.nom], ms=4,
                label="Markov homogène")
        ax.set_title(f"{et.nom} — écart moyen {err:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("Temps (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("RSD teneur petites (–)")
    fig.suptitle(
        "Influence de la méthode de découpage : RSD DEM vs prédiction "
        "markovienne homogène ($\\tau = 1{,}57$ s, 2 blocs d'apprentissage)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "comparaison_methodes_rsd.png", dpi=200)
    plt.close(fig)
    with open(FIGDIR / "comparaison_methodes_table.txt", "w") as f:
        for k, v in resume.items():
            f.write(f"{k:<12} {v:8.4f}\n")
    print("comparaison ok", resume)
    return resume


def etude_start(etudes):
    """RSD Markov (appris avec/sans transitoire) vs DEM, prédiction dès 0 s."""
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    resume = {}
    for ax, (key, et) in zip(axes.flat, etudes.items()):
        cfg_tr = config_for(et.method, start_index=0)
        cfg_pm = config_for(et.method, start_index=START)
        rsd_tr, t_tr, acts = et.markov_rsd(cfg_tr, start_pred=0)
        rsd_pm, t_pm, _ = et.markov_rsd(cfg_pm, start_pred=0)
        t_dem = np.arange(0, N_T, 20)
        rsd_dem = et.dem_rsd(t_dem, acts)
        n = min(len(rsd_tr), len(rsd_pm))
        rsd_dem_mk = et.dem_rsd(t_tr[:n], acts)
        e_tr = float(np.mean(np.abs(rsd_tr[:n] - rsd_dem_mk)))
        e_pm = float(np.mean(np.abs(rsd_pm[:n] - rsd_dem_mk)))
        resume[et.nom] = (e_tr, e_pm)
        ax.plot(t_dem / 100, rsd_dem, "k-", lw=0.9, alpha=0.7, label="DEM")
        ax.plot(t_tr[:n] / 100, rsd_tr[:n], "s--", color="0.55", ms=4,
                label=f"Markov, start = 0 s (écart {e_tr:.3f})")
        ax.plot(t_pm[:n] / 100, rsd_pm[:n], "o--", color=COLORS[et.nom], ms=4,
                label=f"Markov, start = 1,57 s (écart {e_pm:.3f})")
        ax.axvspan(0, START / 100, color="0.85", alpha=0.8, zorder=0)
        ax.axvline(START / 100, color="k", ls="--", lw=1.2)
        ax.set_title(et.nom)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("Temps (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("RSD teneur petites (–)")
    fig.suptitle(
        "Justification du choix de start : RSD DEM vs prédictions markoviennes "
        "(propagées depuis $t = 0$ s) apprises avec ou sans le régime transitoire"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_start_rsd.png", dpi=200)
    plt.close(fig)
    print("start ok", resume)
    return resume


def etude_tau(etudes):
    et = etudes["physique"]
    taus = [10, 25, 50, 100, 157, 300, 500, 1000]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    t_dem = np.arange(START, N_T, 20)
    a = np.ones(et.n_states, bool)
    ax.plot(t_dem / 100, et.dem_rsd(t_dem), "k.-", ms=3, lw=1,
            label="RSD DEM (réel)")
    cmap = plt.get_cmap("viridis")
    for i, tau in enumerate(taus):
        cfg = config_for(et.method, tau=tau, step=tau,
                         dt=max(1, tau // 10))
        rsd_mk, t_mk, _ = et.markov_rsd(cfg)
        lw = 2.6 if tau == TAU else 1.4
        ax.plot(t_mk / 100, rsd_mk, color=cmap(i / len(taus)), lw=lw,
                label=f"Markov $\\tau$={tau / 100:g} s"
                      + (" (retenu)" if tau == TAU else ""))
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("RSD (–)")
    ax.set_title(
        "Influence du pas de temps de Markov $\\tau$ sur la cinétique prédite\n"
        "(découpage physique, 10 cellules, nlt=2, start=1,57 s, "
        "step=$\\tau$, dt=$\\tau$/10)"
    )
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_tau_rsd.png", dpi=200)
    plt.close(fig)
    print("tau ok")


def _erreur(et, cfg):
    rsd_mk, t_mk, acts = et.markov_rsd(cfg)
    rsd_dem = et.dem_rsd(t_mk, acts)
    return float(np.mean(np.abs(rsd_mk - rsd_dem)))


def etude_nlt(etudes):
    et = etudes["physique"]
    nlts = [1, 2, 3, 5, 8, 12, 18]
    errs = [_erreur(et, config_for(et.method, nlt=n)) for n in nlts]
    for n, e in zip(nlts, errs):
        print(f"  NLT={n} -> {e:.4f}")
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.plot(nlts, errs, "o-", color="#d62728", lw=2)
    ax.set_xlabel("Nombre de blocs d'apprentissage $NLT$")
    ax.set_ylabel(r"Erreur moyenne $|\mathrm{RSD}_{Markov} - \mathrm{RSD}_{DEM}|$ (–)")
    ax.set_title(
        "Influence de $NLT$ sur la qualité du modèle\n"
        "(découpage physique, 10 cellules, start=1,57 s, "
        "step=tau=1,57 s, dt=8 pas)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_nlt_erreur.png", dpi=200)
    plt.close(fig)
    print("nlt ok")
    return dict(zip(nlts, errs))


def etude_step(etudes):
    et = etudes["physique"]
    steps = [40, 80, 157, 314, 471]
    errs = [_erreur(et, config_for(et.method, nlt=3, step=s)) for s in steps]
    for s, e in zip(steps, errs):
        print(f"  step={s} -> {e:.4f}")
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.plot([s / 100 for s in steps], errs, "s-", color="#1f77b4", lw=2)
    ax.axvline(1.57, color="k", ls="--", lw=1.5)
    ax.text(1.65, max(errs) * 0.97, "step = $\\tau$ = 1,57 s", fontsize=10)
    ax.set_xlabel("Écart entre blocs $step$ (s)")
    ax.set_ylabel(r"Erreur moyenne $|\mathrm{RSD}_{Markov} - \mathrm{RSD}_{DEM}|$ (–)")
    ax.set_title(
        "Influence de $step$ sur la qualité du modèle\n"
        "(découpage physique, 10 cellules, nlt=3, start=1,57 s, "
        "tau=1,57 s, dt=8 pas)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_step_erreur.png", dpi=200)
    plt.close(fig)
    print("step ok")
    return dict(zip(steps, errs))


def etude_especes(etudes):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    resume = {}
    for ax, (key, et) in zip(axes.flat, etudes.items()):
        cfg = config_for(et.method)
        rsd_avec, t_mk, acts = et.markov_rsd(cfg)
        rsd_sans, _ = et.markov_rsd_single_P(cfg)
        n = min(len(rsd_avec), len(rsd_sans))
        rsd_dem_mk = et.dem_rsd(t_mk[:n], acts)
        e_avec = float(np.mean(np.abs(rsd_avec[:n] - rsd_dem_mk)))
        e_sans = float(np.mean(np.abs(rsd_sans[:n] - rsd_dem_mk)))
        resume[et.nom] = (e_sans, e_avec)
        t_dem = np.arange(START, N_T, 20)
        ax.plot(t_dem / 100, et.dem_rsd(t_dem, acts), "k-", lw=0.9,
                alpha=0.7, label="DEM")
        ax.plot(t_mk[:n] / 100, rsd_sans[:n], "s--", color="0.55", ms=4,
                label=f"sans distinction (écart {e_sans:.3f})")
        ax.plot(t_mk[:n] / 100, rsd_avec[:n], "o--", color=COLORS[et.nom],
                ms=4, label=f"avec distinction (écart {e_avec:.3f})")
        ax.set_title(et.nom)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    for ax in axes[-1]:
        ax.set_xlabel("Temps (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("RSD teneur petites (–)")
    fig.suptitle(
        "Matrice unique (sans distinction d'espèce) vs matrices par espèce "
        "($\\tau = 1{,}57$ s, 2 blocs)"
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_especes_rsd.png", dpi=200)
    plt.close(fig)
    with open(FIGDIR / "etude_especes_table.txt", "w") as f:
        f.write(f"{'méthode':<14}{'sans distinction':>18}"
                f"{'avec distinction':>18}\n")
        for k, (es, ea) in resume.items():
            f.write(f"{k:<14}{es:18.4f}{ea:18.4f}\n")
    print("especes ok", resume)
    return resume


def resultats_cylindrique(etudes):
    et = etudes["cylindrique"]
    cfg = config_for(et.method)

    # matrices par espèce (convention ligne-stochastique de la librairie)
    P_small = et.build_P(cfg, "small")
    P_large = et.build_P(cfg, "large")
    vmax = max(P_small.max(), P_large.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    # La librairie stocke P en convention ligne-stochastique
    # (P[i, j] = P(i -> j), S_next = S @ P) ; on transpose pour l'affichage
    # afin de retrouver la convention colonne du rapport
    # (colonnes = source j, lignes = arrivée i).
    for ax, P, ttl in ((axes[0], P_large.T, "Grandes particules (8 mm)"),
                       (axes[1], P_small.T, "Petites particules (4 mm)")):
        im = ax.imshow(P, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xlabel("Cellule source $j$")
        ax.set_ylabel("Cellule d'arrivée $i$")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, label="$P_{i,j}$")
    fig.suptitle("Matrices de transition — découpage cylindrique "
                 "(10 cellules, $\\tau = 1{,}57$ s)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "matrice_cylindrique_especes.png", dpi=200)
    plt.close(fig)

    # RSD DEM vs homogène vs inhomogène
    rsd_h, t_h, acts = et.markov_rsd(cfg)
    nlt_max = (N_T - START) // (2 * TAU)
    cfg_inh = config_for(et.method, nlt=nlt_max)
    rsd_i, t_i = et.markov_rsd_inhomogeneous(cfg_inh)
    t_dem = np.arange(START, N_T, 20)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(t_dem / 100, et.dem_rsd(t_dem, acts), "k-", lw=1, alpha=0.7,
            label="DEM")
    ax.plot(t_h / 100, rsd_h, "o--", color="#1f77b4", ms=4,
            label="Markov homogène")
    ax.plot(t_i / 100, rsd_i, "s--", color="#d62728", ms=4,
            label="Markov inhomogène")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("RSD de la teneur en petites particules (–)")
    ax.set_title("Découpage cylindrique : RSD DEM vs prédictions markoviennes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "rsd_cylindrique.png", dpi=200)
    plt.close(fig)
    print("cylindrique ok")


def figure_espace_caracteristiques(timestep_dict):
    """Positions + norme de vitesse (apprentissage vs prédiction)."""
    fit_ts = [t for t in range(START, START + 5 * TAU, 10)]
    pts_fit = np.concatenate([
        timestep_dict[t][["coordinates:0", "coordinates:1"]].to_numpy()
        for t in fit_ts])
    vn_fit = np.concatenate([np.linalg.norm(
        timestep_dict[t][["Velocity:0", "Velocity:1", "Velocity:2"]]
        .to_numpy(), axis=1) for t in fit_ts])
    pred_t = 4000
    pts_pred = timestep_dict[pred_t][
        ["coordinates:0", "coordinates:1"]].to_numpy()
    vn_pred = np.linalg.norm(timestep_dict[pred_t][
        ["Velocity:0", "Velocity:1", "Velocity:2"]].to_numpy(), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    vmax = np.percentile(vn_fit, 99)
    for ax, pts, vn, ttl in (
        (axes[0], pts_fit[::12], vn_fit[::12],
         "(a) Phase d'apprentissage (positions cumulées\n"
         "sur 5 tours du régime permanent)"),
        (axes[1], pts_pred, vn_pred,
         "(b) Phase de prédiction (instant $t = 40$ s)"),
    ):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.clip(vn, 0, vmax),
                        cmap="viridis", s=8, lw=0)
        ax.set_xlabel("$x$ (m)")
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=10.5)
    axes[0].set_ylabel("$y$ (m)")
    cb = fig.colorbar(sc, ax=axes, shrink=0.85)
    cb.set_label(r"$\|\mathbf{v}_p\|$ (m/s)")
    fig.suptitle(
        "Espace des caractéristiques du découpage physique : positions et "
        "norme de la vitesse des 1030 particules", fontsize=12)
    fig.savefig(FIGDIR / "espace_caracteristiques_physique.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("espace caracteristiques ok")




def _cfg_str(cfg):
    """Configuration explicite pour les légendes."""
    return (f"nlt={cfg.nlt}, start={cfg.start_index / 100:g} s, "
            f"step={cfg.step / 100:g} s, tau={cfg.tau / 100:g} s, "
            f"dt={cfg.dt} pas")


def matrices_par_methode(etudes):
    """Matrices de transition par espèce pour cartésien, Voronoï, physique
    (le cylindrique est déjà produit par resultats_cylindrique)."""
    from postprocessing.metrics import concentration_from_S  # noqa: F401

    for key in ("cartesien", "voronoi", "physique"):
        et = etudes[key]
        cfg = config_for(et.method)
        P_small = et.build_P(cfg, "small")
        P_large = et.build_P(cfg, "large")
        vmax = max(P_small.max(), P_large.max())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
        for ax, P, ttl in ((axes[0], P_large.T, "Grandes particules (8 mm)"),
                           (axes[1], P_small.T, "Petites particules (4 mm)")):
            im = ax.imshow(P, cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xlabel("Cellule source $j$")
            ax.set_ylabel("Cellule d'arrivée $i$")
            ax.set_title(ttl)
            fig.colorbar(im, ax=ax, label="$P_{i,j}$")
        fig.suptitle(f"Matrices de transition — découpage {et.nom.lower()} "
                     f"(10 cellules, {_cfg_str(cfg)})")
        fig.tight_layout()
        fig.savefig(FIGDIR / f"matrice_{key}_especes.png", dpi=200)
        plt.close(fig)
        print(f"matrices {key} ok")


def teneur_et_nombre(etudes):
    """Teneur locale et nombre de particules par cellule : DEM vs Markov
    (découpage de Voronoï, code librairie)."""
    from postprocessing.metrics import concentration_from_S

    et = etudes["voronoi"]
    cfg = config_for(et.method)
    trajs, t_mk, acts = et.markov_traj(cfg)
    act = acts["small"] & acts["large"]
    rows = np.searchsorted(et.times, t_mk)
    S_s_dem = et.S_matrices["small"][rows]
    S_l_dem = et.S_matrices["large"][rows]
    n = min(len(trajs["small"]), len(trajs["large"]), len(rows))

    C_dem = concentration_from_S(S_s_dem[:n], S_l_dem[:n])
    C_mk = concentration_from_S(trajs["small"][:n], trajs["large"][:n])
    N_dem = S_s_dem[:n] + S_l_dem[:n]
    N_mk = trajs["small"][:n] + trajs["large"][:n]
    t = t_mk[:n] / 100
    cells = np.where(act)[0]
    cmap = plt.get_cmap("viridis")

    # ── teneur locale ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, M, ttl in ((axes[0], C_dem, "(a) DEM"),
                       (axes[1], C_mk, "(b) Markov homogène")):
        for kk, c in enumerate(cells):
            ax.plot(t, M[:, c], color=cmap(kk / max(1, len(cells) - 1)),
                    lw=1.4, label=f"cellule {c}")
        ax.set_xlabel("Temps (s)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Teneur locale en petites particules (–)")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Teneur locale en petites particules par cellule — "
                 f"découpage de Voronoï ({_cfg_str(cfg)})")
    fig.tight_layout()
    fig.savefig(FIGDIR / "teneur_locale_cellules.png", dpi=200)
    plt.close(fig)

    # ── nombre de particules ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, M, ttl in ((axes[0], N_dem, "(a) DEM"),
                       (axes[1], N_mk, "(b) Markov homogène")):
        for kk, c in enumerate(cells):
            ax.plot(t, M[:, c], color=cmap(kk / max(1, len(cells) - 1)),
                    lw=1.4, label=f"cellule {c}")
        ax.set_xlabel("Temps (s)")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Nombre de particules par cellule")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Nombre de particules par cellule — "
                 f"découpage de Voronoï ({_cfg_str(cfg)})")
    fig.tight_layout()
    fig.savefig(FIGDIR / "nombre_particules_cellules.png", dpi=200)
    plt.close(fig)
    print("teneur + nombre ok")

    # teneurs par cellule au régime établi et tardif, pour les vues 3D
    return {"cells": cells, "C_dem": C_dem, "t": t}


def etude_nlt_erreur_relative(etudes):
    """Sensibilité au NLT par l'erreur relative sur la matrice de transition
    (grandeur de comparaison de Doucet et al. 2008) :
    E(NLT) = ||P(NLT) - P(NLT_ref)||_F / ||P(NLT_ref)||_F."""
    et = etudes["physique"]
    nlts = [1, 2, 3, 5, 8, 12, 18]
    nlt_ref = nlts[-1]
    P_ref = {sp: et.build_P(config_for(et.method, nlt=nlt_ref), sp)
             for sp in ("small", "large")}
    errs = {sp: [] for sp in ("small", "large")}
    for nlt in nlts:
        cfg = config_for(et.method, nlt=nlt)
        for sp in ("small", "large"):
            P = et.build_P(cfg, sp)
            e = (np.linalg.norm(P - P_ref[sp], "fro")
                 / np.linalg.norm(P_ref[sp], "fro"))
            errs[sp].append(e)
            print(f"  NLT={nlt} {sp}: E_rel={e:.4f}")
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.plot(nlts, errs["small"], "o-", color="#2166ac", lw=2,
            label="petites particules")
    ax.plot(nlts, errs["large"], "s-", color="#b2182b", lw=2,
            label="grandes particules")
    ax.set_xlabel("Nombre de blocs d'apprentissage $NLT$")
    ax.set_ylabel(r"Erreur relative $E(NLT) = \frac{\|\mathbf{P}^{(NLT)}"
                  r" - \mathbf{P}^{(réf)}\|_F}{\|\mathbf{P}^{(réf)}\|_F}$")
    ax.set_title(
        "Convergence de la matrice de transition avec $NLT$ "
        f"(découpage physique, 10 cellules,\n"
        f"start=1,57 s, step=tau=1,57 s, dt=8 pas ; référence NLT={nlt_ref})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_nlt_erreur_relative.png", dpi=200)
    plt.close(fig)
    print("nlt erreur relative ok")
    return errs, nlts


def dump_labels_3d(etudes):
    """Labels et teneurs par cellule pour les vues 3D (pyvista_js)."""
    out = {}
    from postprocessing.metrics import concentration_from_S
    for key, et in etudes.items():
        for t in (0, 157, 3000):
            row = et.idx_to_row[t]
            out[f"{key}_{t}"] = et.states_matrix[row]
        # teneur par cellule au regime etabli et tardif (pour annotations)
        for t in (157, 3000):
            row = et.idx_to_row[t]
            S_s = et.S_matrices["small"][row][None]
            S_l = et.S_matrices["large"][row][None]
            out[f"{key}_teneur_{t}"] = concentration_from_S(S_s, S_l)[0]
    np.savez(ROOT / "data" / "labels_librairie.npz", **out)
    print("labels 3D ok")


def etude_dt(etudes):
    """Choix du raffinage temporel dt : nombre de NaN dans la matrice.

    La matrice de transition conserve des NaN pour toute cellule occupée
    jamais observée comme source (dénominateur nul). On compte, pour chaque
    méthode et chaque valeur de dt, le nombre de colonnes de la convention
    du rapport (lignes de la convention de stockage) contenant des NaN,
    dans le pire cas des deux espèces. La matrice est conforme --- la
    condition d'homogénéisation est vérifiable sur toutes les colonnes ---
    lorsque ce nombre est nul.
    """
    dts = [157, 78, 39, 16, 8, 4, 2]
    resume = {}
    for key, et in etudes.items():
        vals = []
        for dt in dts:
            cfg = config_for(et.method, dt=dt)
            worst = 0
            for sp in ("small", "large"):
                P = et.build_P(cfg, sp)
                n_nan_cols = int(np.isnan(P).any(axis=1).sum())
                worst = max(worst, n_nan_cols)
            vals.append(worst)
            print(f"  {et.nom} dt={dt}: colonnes NaN={worst}")
        resume[et.nom] = vals

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for nom, vals in resume.items():
        ax.plot(dts, vals, "o-", color=COLORS[nom], lw=2, label=nom)
    ax.axhline(0, color="k", lw=0.8)
    ax.invert_xaxis()
    ax.set_xlabel("Raffinage temporel $dt$ (pas de sortie DEM)")
    ax.set_ylabel("Colonnes de $\\mathbf{P}$ contenant des NaN\n"
                  "(pire des deux espèces)")
    ax.set_title(
        "Conformité de la matrice de transition selon $dt$\n"
        "(10 cellules par méthode, nlt=2, start=1,57 s, step=tau=1,57 s)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_dt_nan.png", dpi=200)
    plt.close(fig)
    with open(FIGDIR / "etude_dt_table.txt", "w") as f:
        f.write("dt      " + "  ".join(f"{d:>5d}" for d in dts) + "\n")
        for nom, vals in resume.items():
            f.write(f"{nom:<12}" + "  ".join(f"{v:>5d}" for v in vals) + "\n")
    print("dt ok", resume)
    return resume


def matrices_annotees(etudes):
    """Matrices de transition annotées au centième.

    Pour chaque matrice : vérification de l'absence de NaN et de la
    condition d'homogénéisation (chaque colonne de la convention du rapport
    somme à un).
    """
    for key, et in etudes.items():
        cfg = config_for(et.method)
        P_small = et.build_P(cfg, "small")
        P_large = et.build_P(cfg, "large")
        for sp, P in (("small", P_small), ("large", P_large)):
            n_nan = int(np.isnan(P).sum())
            sums = P.sum(axis=1)  # convention stockage : lignes
            homog = np.allclose(sums[~np.isnan(sums)], 1.0, atol=1e-9)
            print(f"  {et.nom} {sp}: NaN={n_nan} | "
                  f"homogénéisation={'OK' if homog and n_nan == 0 else 'NON'}")
        vmax = max(np.nanmax(P_small), np.nanmax(P_large))
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
        for ax, P, ttl in ((axes[0], P_large.T, "Grandes particules (8 mm)"),
                           (axes[1], P_small.T, "Petites particules (4 mm)")):
            im = ax.imshow(P, cmap="YlOrRd", vmin=0, vmax=vmax)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    v = P[i, j]
                    txt = "NaN" if np.isnan(v) else f"{v:.2f}".replace("0.", ".")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                            color="black" if (np.isnan(v) or v < 0.55 * vmax)
                            else "white")
            ax.set_xticks(range(10))
            ax.set_yticks(range(10))
            ax.set_xlabel("Cellule source $j$")
            ax.set_ylabel("Cellule d'arrivée $i$")
            ax.set_title(ttl)
            fig.colorbar(im, ax=ax, label="$P_{i,j}$", shrink=0.85)
        fig.suptitle(f"Matrices de transition — découpage {et.nom.lower()} "
                     f"(10 cellules, nlt=2, start=1,57 s, step=tau=1,57 s, "
                     f"dt=8 pas) — probabilités arrondies au centième ; "
                     f"chaque colonne somme à un")
        fig.tight_layout()
        fig.savefig(FIGDIR / f"matrice_{key}_especes.png", dpi=200)
        plt.close(fig)
    print("matrices annotées ok")


def table_erreurs(etudes):
    """Figures d'écart |Markov - DEM| par méthode et par espèce (table 4)."""
    noms_sp = {"large": "grandes", "small": "petites"}
    for key in ("cartesien", "voronoi", "physique"):
        et = etudes[key]
        cfg = config_for(et.method)
        for sp in ("large", "small"):
            P_clean, act = clean_transition_matrix(et.build_P(cfg, sp))
            row0 = int(np.searchsorted(et.times, cfg.start_index))
            S0 = et.S_matrices[sp][row0].astype(float)
            traj, t_mk = propagate_markov(
                S0, P_clean, et.times, cfg.start_index, cfg.tau, act)
            rows = np.searchsorted(et.times, t_mk)
            S_dem = et.S_matrices[sp][rows]
            err = np.abs(traj - S_dem)
            rms = np.sqrt((err ** 2).mean(axis=1))

            fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                     width_ratios=[1, 1.25])
            axes[0].plot(t_mk / 100, rms, "-", color="#d62728", lw=2)
            axes[0].fill_between(t_mk / 100, 0, rms, color="#d62728",
                                 alpha=0.25)
            axes[0].set_xlabel("Temps (s)")
            axes[0].set_ylabel("Écart RMS (particules)")
            axes[0].set_title("Écart temporel RMS $|$Markov $-$ DEM$|$")
            axes[0].grid(alpha=0.3)
            im = axes[1].imshow(err.T, cmap="viridis", aspect="auto",
                                origin="lower",
                                extent=[t_mk[0] / 100, t_mk[-1] / 100,
                                        -0.5, err.shape[1] - 0.5])
            axes[1].set_xlabel("Temps (s)")
            axes[1].set_ylabel("Indice de cellule")
            axes[1].set_title("Écart absolu par cellule")
            fig.colorbar(im, ax=axes[1],
                         label="$|$Markov $-$ DEM$|$ (particules)")
            fig.suptitle(
                f"Découpage {et.nom.lower()} — {noms_sp[sp]} particules "
                f"(nlt=2, start=1,57 s, step=tau=1,57 s, dt=8 pas)",
                fontsize=11)
            fig.tight_layout()
            fig.savefig(FIGDIR / f"erreur_lib_{key}_{sp}.png", dpi=200)
            plt.close(fig)
            print(f"  erreur {key} {sp}: RMS moyen "
                  f"{rms[1:].mean():.2f} particules")
    print("table erreurs ok")


if __name__ == "__main__":
    timestep_dict = load_timestep_dict()
    figure_espace_caracteristiques(timestep_dict)

    print("\n🔍 Échantillonnage des coordonnées (librairie)…")
    sample_coords, s_velocities, _ = sample_coordinates(timestep_dict)

    permanent_rows = PERMANENT_START * N_PARTICLES_PER_TIMESTEP
    frame = make_frame(sample_coords, permanent_rows)

    etudes = {}
    for key in METHODES:
        print(f"\n══ Préparation {key} (fit + états, code librairie) ══")
        etudes[key] = EtudeMethode(key, timestep_dict, sample_coords,
                                   s_velocities, frame=frame)

    comparaison_methodes(etudes)
    etude_start(etudes)
    etude_tau(etudes)
    etude_nlt(etudes)
    etude_nlt_erreur_relative(etudes)
    etude_step(etudes)
    etude_especes(etudes)
    resultats_cylindrique(etudes)
    etude_dt(etudes)
    matrices_annotees(etudes)
    table_erreurs(etudes)
    teneur_et_nombre(etudes)
    dump_labels_3d(etudes)
    print("\n✅ toutes les figures écrites (calculs 100 % librairie) dans",
          FIGDIR)