"""Génère l'évolution de la teneur locale au cours du temps pour chaque méthode de découpage.

- Pour chaque méthode (cartésien, cylindrique, Voronoï, physique) :
  - vecteur d'état en teneur uniquement (pas de comptage)
  - vue du mélangeur discrétisé zoom cadré déjà existante (pv_teneur_*)
  - évolution temporelle de la teneur par cellule DEM vs Markov homogène
  - identification des cellules les plus / moins peuplées

Les figures produites :
- teneur_cartesien.png
- teneur_cylindrique.png
- teneur_voronoi.png (ou teneur_locale_cellules.png existante)
- teneur_physique.png
- et les versions librairie : teneur_*_lib.png avec légende normale et pointillés réduits
"""

from pathlib import Path
import sys
import gzip
import io

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 13,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 14,
})

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dem_mcm_coupling.partitioners import create_partitioner
from dem_mcm_coupling.run_sweep import (
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
from postprocessing.metrics import (
    clean_transition_matrix,
    propagate_markov,
    concentration_from_S,
)

FIGDIR = ROOT / "template-rapport-stage" / "figures"
FIGDIR.mkdir(exist_ok=True)

TAU = 157
START = 157
N_T = 6000
T_TOUR = 1.57

COLORS = {
    "Cartésien": "#fec44f",
    "Cylindrique": "#fe9929",
    "Voronoï": "#ec7014",
    "Physique": "#cc4c02",
}

GEO_METHODS = {"cartesien", "cylindrique"}

CELL_COLORS = [
    "#2ca02c",
    "#d62728",
    "#ff7f0e",
    "#e6b800",
    "#000000",
    "#1f77b4",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]

def add_tours_axis(ax):
    sec = ax.secondary_xaxis(
        "top", functions=(lambda s: s / T_TOUR, lambda n: n * T_TOUR))
    sec.set_xlabel("Nombre de tours du tambour")
    return sec

def make_frame(sample_coords, permanent_rows):
    pts = np.asarray(sample_coords[permanent_rows:], dtype=np.float64)
    c = pts.mean(axis=0)
    xy = pts[:, :2] - c[:2]
    cov = xy.T @ xy / len(xy)
    _, evecs = np.linalg.eigh(cov)
    R2 = evecs[:, ::-1]
    if np.linalg.det(R2) < 0:
        R2[:, 1] = -R2[:, 1]
    return c, R2

def transform_coords(coords, c, R2):
    out = np.asarray(coords, dtype=np.float64).copy()
    out[:, :2] = (out[:, :2] - c[:2]) @ R2
    out[:, 2] -= c[2]
    return out

def transform_dict(timestep_dict, c, R2):
    out = {}
    for idx, df in timestep_dict.items():
        xy = (df[["coordinates:0", "coordinates:1"]].to_numpy() - c[:2]) @ R2
        out[idx] = pd.DataFrame({
            "coordinates:0": xy[:, 0],
            "coordinates:1": xy[:, 1],
            "coordinates:2": df["coordinates:2"].to_numpy() - c[2],
        })
    return out

METHODES = {
    "cartesien": ("Cartésien", "cartesian", dict(nx=10, ny=1, nz=1)),
    "cylindrique": ("Cylindrique", "cylindrical",
                    dict(nr=1, ntheta=10, nz=1, radial_mode="equal_area")),
    "voronoi": ("Voronoï", "voronoi", dict(n_cells=10)),
    "physique": ("Physique", "physics",
                 dict(n_cells=10, velocity_weight=0.5)),
}

def load_timestep_dict():
    cols = ["Fichier_Source", "Diameter",
            "coordinates:0", "coordinates:1", "coordinates:2",
            "Velocity:0", "Velocity:1", "Velocity:2"]
    frames = []
    for path in sorted((ROOT / "data" / "chunks").glob("*.parquet.gz")):
        with gzip.open(path, "rb") as fh:
            frames.append(pd.read_parquet(io.BytesIO(fh.read()), columns=cols))
    df_full = pd.concat(frames, ignore_index=True)
    del frames
    timestep_dict = {}
    for source, group in df_full.groupby("Fichier_Source", sort=False):
        idx = int(str(source).replace("data_", "").replace(".csv", ""))
        timestep_dict[idx] = group.drop(columns="Fichier_Source").reset_index(drop=True)
    print(f"📦 {len(timestep_dict)} timesteps")
    return dict(sorted(timestep_dict.items()))

class EtudeMethode:
    def __init__(self, key, timestep_dict, sample_coords, s_velocities, frame=None):
        self.key = key
        self.nom, method, kwargs = METHODES[key]
        self.method = method
        self.timestep_dict = timestep_dict
        self.partitioner = create_partitioner(method, **kwargs)
        self.n_states = self.partitioner.n_cells
        cfg_fit = ExperimentConfig(method=method, method_kwargs=kwargs)
        permanent_rows = PERMANENT_START * N_PARTICLES_PER_TIMESTEP
        label_dict = timestep_dict
        fit_coords = sample_coords
        if frame is not None and key in GEO_METHODS:
            c, R2 = frame
            fit_coords = transform_coords(sample_coords, c, R2)
            label_dict = transform_dict(timestep_dict, c, R2)
            self.partitioner.fit(fit_coords[permanent_rows:])
        else:
            _fit_partitioner_for_sweep(
                self.partitioner, cfg_fit, fit_coords, s_velocities, permanent_rows,
            )
        self.species_masks = _detect_species(timestep_dict[min(timestep_dict)])
        self.states_matrix, self.sorted_indices = _compute_state_matrices(
            label_dict, self.partitioner, START
        )
        self.idx_to_row = {int(i): r for r, i in enumerate(self.sorted_indices)}
        self.S_matrices = _build_state_matrices(
            self.states_matrix, self.species_masks, self.n_states
        )
        self.times = np.asarray(self.sorted_indices)

    def _accumulate(self, pairs, mask):
        prev = np.concatenate(
            [self.states_matrix[self.idx_to_row[a]][mask] for a, b in pairs]
        )
        curr = np.concatenate(
            [self.states_matrix[self.idx_to_row[b]][mask] for a, b in pairs]
        )
        return prev, curr

    def build_P(self, config, species):
        pairs = _build_pairs(config, self.timestep_dict, self.idx_to_row)
        if not pairs:
            raise ValueError("Aucune paire valide")
        mask = (np.ones(N_PARTICLES_PER_TIMESTEP, bool) if species == "all"
                else self.species_masks[species])
        prev, curr = self._accumulate(pairs, mask)
        return compute_P_matrix_torch(prev, curr, self.n_states, "cpu").cpu().numpy()

    def markov_traj(self, config, start_pred=None):
        start_pred = config.start_index if start_pred is None else start_pred
        trajs, acts = {}, {}
        for sp in ("small", "large"):
            P_raw = self.build_P(config, sp)
            P_clean, activated = clean_transition_matrix(
                np.nan_to_num(P_raw, nan=0.0))
            row0 = np.searchsorted(self.times, start_pred)
            S0 = self.S_matrices[sp][row0].astype(float)
            traj, t_mk = propagate_markov(
                S0, P_clean, self.times, start_pred, config.tau, activated)
            trajs[sp], acts[sp] = traj, activated
        return trajs, t_mk, acts

def config_for(method, **kw):
    base = dict(nlt=2, tau=TAU, step=TAU, dt=8, start_index=START)
    base.update(kw)
    return ExperimentConfig(method=method, **base)

def plot_teneur(ax, t_dem, C_dem, t_mk, C_mk, cells):
    """Teneur uniquement, légende normale, pointillés réduits"""
    for c in cells:
        col = CELL_COLORS[int(c) % len(CELL_COLORS)]
        ax.plot(t_dem, C_dem[:, c], "-", color=col, lw=1.2, alpha=0.9)
        ax.plot(t_mk, C_mk[:, c], "o--", color=col, ms=3.5, lw=0.8,
                markeredgecolor="white", markeredgewidth=0.4, zorder=3, label=f"cellule {c}")
    # Légende normale
    ax.legend(ncol=5, fontsize=9, loc="upper right")

if __name__ == "__main__":
    timestep_dict = load_timestep_dict()
    sample_coords, s_velocities, _ = sample_coordinates(timestep_dict)
    permanent_rows = PERMANENT_START * N_PARTICLES_PER_TIMESTEP
    frame = make_frame(sample_coords, permanent_rows)

    etudes = {}
    for key in METHODES:
        print(f"\n══ Préparation {key} ══")
        etudes[key] = EtudeMethode(key, timestep_dict, sample_coords, s_velocities, frame=frame)

    for key, et in etudes.items():
        cfg = config_for(et.method)
        trajs, t_mk, acts = et.markov_traj(cfg)
        act = acts["small"] & acts["large"]
        cells = np.where(act)[0]
        n = min(len(trajs["small"]), len(trajs["large"]))
        C_mk = concentration_from_S(trajs["small"][:n], trajs["large"][:n])
        t_dem_idx = np.arange(START, N_T, 20)
        rows_dem = np.searchsorted(et.times, t_dem_idx)
        C_dem = concentration_from_S(et.S_matrices["small"][rows_dem],
                                     et.S_matrices["large"][rows_dem])

        # Analyse des cellules extrêmes au début et fin
        # début régime établi
        row_start = et.idx_to_row[START]
        C_start = concentration_from_S(et.S_matrices["small"][row_start][None],
                                       et.S_matrices["large"][row_start][None])[0]
        # fin
        row_end = et.idx_to_row[3000] if 3000 in et.idx_to_row else et.idx_to_row[et.times[-1]]
        C_end = concentration_from_S(et.S_matrices["small"][row_end][None],
                                     et.S_matrices["large"][row_end][None])[0]

        # identification
        most_start = int(np.argmax(C_start))
        least_start = int(np.argmin(C_start))
        most_end = int(np.argmax(C_end))
        least_end = int(np.argmin(C_end))

        print(f"{et.nom}: start most={most_start} ({C_start[most_start]:.2f}) least={least_start} ({C_start[least_start]:.2f}) | end most={most_end} ({C_end[most_end]:.2f}) least={least_end} ({C_end[least_end]:.2f})")

        fig, ax = plt.subplots(figsize=(12.5, 6.2))
        for c in cells:
            col = CELL_COLORS[int(c) % len(CELL_COLORS)]
            ax.plot(t_dem_idx/100, C_dem[:, c], "-", color=col, lw=1.2, alpha=0.8)
            ax.plot(t_mk[:n]/100, C_mk[:, c], "o--", color=col, ms=3.5, lw=0.8,
                    markeredgecolor="white", markeredgewidth=0.4, label=f"cellule {c}")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Teneur locale en petites particules (–)")
        add_tours_axis(ax)
        ax.grid(alpha=0.3)
        ax.set_title(f"Teneur locale par cellule – découpage {et.nom.lower()} (10 cellules, Markov homogène)\n"
                     f"Cellule la plus peuplée début: {most_start} (c={C_start[most_start]:.2f}), moins: {least_start} (c={C_start[least_start]:.2f}) | "
                     f"Fin plus: {most_end} (c={C_end[most_end]:.2f}), moins: {least_end} (c={C_end[least_end]:.2f})", fontsize=11)
        ax.legend(ncol=5, fontsize=9, loc="upper right")
        fig.tight_layout()
        # noms compatibles rapport
        if key == "voronoi":
            fig.savefig(FIGDIR / "teneur_locale_cellules.png", dpi=200)
            fig.savefig(FIGDIR / "teneur_voronoi.png", dpi=200)
        elif key == "physique":
            fig.savefig(FIGDIR / "teneur_physique.png", dpi=200)
            fig.savefig(FIGDIR / "teneur_nlt1_lib.png", dpi=200)
        elif key == "cartesien":
            fig.savefig(FIGDIR / "teneur_cartesien.png", dpi=200)
        elif key == "cylindrique":
            fig.savefig(FIGDIR / "teneur_cylindrique.png", dpi=200)
        # versions _lib pour cohérence
        fig.savefig(FIGDIR / f"teneur_{key}_lib.png", dpi=200)
        plt.close(fig)
        print(f"  -> {key} teneur ok")

    print("\n✅ Figures teneur par méthode générées")
