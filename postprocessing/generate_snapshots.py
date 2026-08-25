"""Génère des snapshots du mélangeur à différents tours pour toutes les méthodes.

Tours demandés : 1 (t=1.57s), fin-15 (t~36.45s), fin-5 (t~52.15s), fin (t=60s)
Pour chaque méthode : vue colorée par teneur locale, zoom cadré, face et profil.

Utilise les données DEM locales (chunks) + librairie dem_mcm_coupling.
"""

from pathlib import Path
import sys
import gzip, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dem_mcm_coupling.partitioners import create_partitioner
from dem_mcm_coupling.run_sweep import (
    ExperimentConfig, _compute_state_matrices, _build_state_matrices,
    _detect_species, _fit_partitioner_for_sweep, sample_coordinates,
    PERMANENT_START, N_PARTICLES_PER_TIMESTEP,
)
from postprocessing.metrics import concentration_from_S

FIGDIR = ROOT / "template-rapport-stage" / "figures"
FIGDIR.mkdir(exist_ok=True)

TAU = 157
START = 157
N_T = 6000
T_TOUR = 1.57

CELL_COLORS = [
    "#2ca02c","#d62728","#ff7f0e","#e6b800","#000000",
    "#1f77b4","#9467bd","#8c564b","#e377c2","#17becf"
]

def make_frame(sample_coords, permanent_rows):
    pts = np.asarray(sample_coords[permanent_rows:], dtype=np.float64)
    c = pts.mean(axis=0)
    xy = pts[:, :2] - c[:2]
    cov = xy.T @ xy / len(xy)
    _, evecs = np.linalg.eigh(cov)
    R2 = evecs[:, ::-1]
    if np.linalg.det(R2) < 0:
        R2[:,1] = -R2[:,1]
    return c,R2

def transform_coords(coords,c,R2):
    out = np.asarray(coords,dtype=np.float64).copy()
    out[:,:2] = (out[:,:2]-c[:2])@R2
    out[:,2] -= c[2]
    return out

def transform_dict(timestep_dict,c,R2):
    out={}
    for idx,df in timestep_dict.items():
        xy = (df[["coordinates:0","coordinates:1"]].to_numpy()-c[:2])@R2
        out[idx]=pd.DataFrame({
            "coordinates:0": xy[:,0],
            "coordinates:1": xy[:,1],
            "coordinates:2": df["coordinates:2"].to_numpy()-c[2],
        })
    return out

METHODES = {
    "cartesien": ("Cartésien", "cartesian", dict(nx=10, ny=1, nz=1)),
    "cylindrique": ("Cylindrique", "cylindrical", dict(nr=1, ntheta=10, nz=1, radial_mode="equal_area")),
    "voronoi": ("Voronoï", "voronoi", dict(n_cells=10)),
    "physique": ("Physique", "physics", dict(n_cells=10, velocity_weight=0.5)),
}

GEO_METHODS = {"cartesien","cylindrique"}

def load_timestep_dict():
    cols=["Fichier_Source","Diameter","coordinates:0","coordinates:1","coordinates:2","Velocity:0","Velocity:1","Velocity:2"]
    frames=[]
    for p in sorted((ROOT/"data"/"chunks").glob("*.parquet.gz")):
        with gzip.open(p,"rb") as fh:
            frames.append(pd.read_parquet(io.BytesIO(fh.read()), columns=cols))
    df_full=pd.concat(frames,ignore_index=True)
    d={}
    for src,g in df_full.groupby("Fichier_Source",sort=False):
        idx=int(str(src).replace("data_","").replace(".csv",""))
        d[idx]=g.drop(columns="Fichier_Source").reset_index(drop=True)
    return dict(sorted(d.items()))

class Etude:
    def __init__(self,key,timestep_dict,sample_coords,s_velocities,frame=None):
        self.key=key
        self.nom,method,kwargs=METHODES[key]
        self.method=method
        self.timestep_dict=timestep_dict
        self.partitioner=create_partitioner(method,**kwargs)
        self.n_states=self.partitioner.n_cells
        cfg_fit=ExperimentConfig(method=method,method_kwargs=kwargs)
        permanent_rows=PERMANENT_START*N_PARTICLES_PER_TIMESTEP
        label_dict=timestep_dict
        fit_coords=sample_coords
        if frame is not None and key in GEO_METHODS:
            c,R2=frame
            fit_coords=transform_coords(sample_coords,c,R2)
            label_dict=transform_dict(timestep_dict,c,R2)
            self.partitioner.fit(fit_coords[permanent_rows:])
        else:
            _fit_partitioner_for_sweep(self.partitioner,cfg_fit,fit_coords,s_velocities,permanent_rows)
        self.species_masks=_detect_species(timestep_dict[min(timestep_dict)])
        self.states_matrix,self.sorted_indices=_compute_state_matrices(label_dict,self.partitioner,START)
        self.idx_to_row={int(i):r for r,i in enumerate(self.sorted_indices)}
        self.S_matrices=_build_state_matrices(self.states_matrix,self.species_masks,self.n_states)
        self.times=np.asarray(self.sorted_indices)
        # coords for plotting: use transformed if geo
        self.plot_coords = {}
        for idx,df in timestep_dict.items():
            if key in GEO_METHODS and frame is not None:
                c,R2=frame
                pts=df[["coordinates:0","coordinates:1","coordinates:2"]].to_numpy()
                pts_t=transform_coords(pts,c,R2)
                self.plot_coords[idx]=pts_t
            else:
                self.plot_coords[idx]=df[["coordinates:0","coordinates:1","coordinates:2"]].to_numpy()

def generate_snapshots():
    timestep_dict=load_timestep_dict()
    sample_coords,s_velocities,_=sample_coordinates(timestep_dict)
    permanent_rows=PERMANENT_START*N_PARTICLES_PER_TIMESTEP
    frame=make_frame(sample_coords,permanent_rows)

    etudes={}
    for k in METHODES:
        print(f"Prep {k}")
        etudes[k]=Etude(k,timestep_dict,sample_coords,s_velocities,frame=frame)

    # instants demandés : tour 1, fin-15, fin-5, fin
    # fin = 60s = 6000, fin-15 = 60 -15*1.57=36.45s ~ 3645, fin-5=52.15s~5215
    # On prend les indices existants les plus proches
    target_times = {
        "tour1": 157,
        "fin-15": 3645,
        "fin-5": 5215,
        "fin": 5999,
    }
    # trouver indices existants proches
    available = sorted(timestep_dict.keys())
    def nearest(t):
        return min(available, key=lambda x: abs(x-t))

    snaps = {name: nearest(t) for name,t in target_times.items()}
    print("Snapshots indices:", snaps)

    for key,et in etudes.items():
        # concentration par cellule à chaque snapshot
        fig, axes = plt.subplots(2,4, figsize=(16,8), sharex=True, sharey=True)
        # 2 rows: face (x-y) and profil (z-y), 4 cols = 4 instants
        for col,(name, t_idx) in enumerate(snaps.items()):
            row = et.idx_to_row.get(t_idx)
            if row is None:
                continue
            S_s = et.S_matrices["small"][row][None]
            S_l = et.S_matrices["large"][row][None]
            C = concentration_from_S(S_s,S_l)[0]  # teneur par cellule
            labels = et.states_matrix[row]
            coords = et.plot_coords[t_idx]
            # face: x vs y
            ax_face = axes[0,col]
            for cell in range(et.n_states):
                mask = labels==cell
                if not np.any(mask):
                    continue
                cval = C[cell]
                ax_face.scatter(coords[mask,0], coords[mask,1], c=np.full(mask.sum(), cval), cmap="coolwarm", vmin=0, vmax=1, s=12, alpha=0.9)
            ax_face.set_title(f"{name} t={t_idx/100:.2f}s ({t_idx/100/T_TOUR:.1f} tours)\n"
                              f"plus {np.argmax(C)} c={C[np.argmax(C)]:.2f} | moins {np.argmin(C)} c={C[np.argmin(C)]:.2f}", fontsize=9)
            ax_face.set_aspect("equal")
            # profil: z vs y
            ax_prof = axes[1,col]
            for cell in range(et.n_states):
                mask = labels==cell
                if not np.any(mask):
                    continue
                cval = C[cell]
                ax_prof.scatter(coords[mask,2], coords[mask,1], c=np.full(mask.sum(), cval), cmap="coolwarm", vmin=0, vmax=1, s=12, alpha=0.9)
            ax_prof.set_aspect("equal")
        axes[0,0].set_ylabel("y (face x-y)")
        axes[1,0].set_ylabel("y (profil z-y)")
        for ax in axes[1]:
            ax.set_xlabel("x ou z (m)")
        fig.suptitle(f"Snapshots mélangeur – découpage {et.nom.lower()} – teneur locale (0=bleu,1=rouge) – zoom cadré", fontsize=13)
        fig.tight_layout()
        out_path = FIGDIR / f"snapshot_{key}_teneur.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"  -> {out_path}")

        # aussi générer les vues individuelles à ces instants pour compatibilité avec rapport (pv_teneur_*)
        for name,t_idx in snaps.items():
            row = et.idx_to_row.get(t_idx)
            if row is None:
                continue
            S_s = et.S_matrices["small"][row][None]
            S_l = et.S_matrices["large"][row][None]
            C = concentration_from_S(S_s,S_l)[0]
            labels = et.states_matrix[row]
            coords = et.plot_coords[t_idx]
            fig, ax = plt.subplots(1,2, figsize=(10,4.5), sharey=True)
            # face
            sc = ax[0].scatter(coords[:,0], coords[:,1], c=[C[l] for l in labels], cmap="coolwarm", vmin=0, vmax=1, s=14)
            ax[0].set_title(f"{name} face")
            ax[0].set_aspect("equal")
            # profil
            ax[1].scatter(coords[:,2], coords[:,1], c=[C[l] for l in labels], cmap="coolwarm", vmin=0, vmax=1, s=14)
            ax[1].set_title(f"{name} profil")
            ax[1].set_aspect("equal")
            fig.colorbar(sc, ax=ax, label="teneur petites")
            fig.suptitle(f"{et.nom} – {name} t={t_idx/100:.2f}s – teneur locale")
            fig.tight_layout()
            fig.savefig(FIGDIR / f"pv_teneur_{key}_{name}.png", dpi=200)
            plt.close(fig)

if __name__ == "__main__":
    generate_snapshots()
