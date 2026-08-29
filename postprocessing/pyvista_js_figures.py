"""Post-traitement 3D des résultats — rendu WebGL de type pyvista/vtk.js.

Le bac à sable ne disposant pas d'OpenGL natif pour VTK, le rendu 3D est
effectué dans le moteur WebGL d'un Chromium headless (embarqué par kaleido),
c'est-à-dire la même chaîne de rendu navigateur que ``pyvista.export_html``
/ vtk.js : chaque particule est une vraie sphère maillée (icosphère), au
diamètre réel, éclairée et ombrée, sur fond gris à la ParaView.

Figures produites dans ``template-rapport-stage/figures/`` :

* ``pv_melangeur_especes.png``     : mélangeur, espèces distinctes (4/8 mm),
  état initial vs régime établi ;
* ``pv_cellules_cartesien.png``    : particules colorées par cell_id ;
* ``pv_cellules_cylindrique.png``  : idem, découpage cylindrique ;
* ``pv_cellules_voronoi.png``      : idem, découpage de Voronoï ;
* ``pv_cellules_physique.png``     : idem, découpage physique ;
* ``pv_cellule_contenu.png``       : contenu d'une cellule (teneur locale),
  reste du lit estompé.

Usage::

    python postprocessing/pyvista_js_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "template-rapport-stage" / "figures"
FIGDIR.mkdir(exist_ok=True)

TAU = 157
START = 157
N_T = 6000
D_SMALL = 0.004
D_BIG = 0.008

BG = "#8a8d90"          # fond gris ParaView
BLEU_PETITES = "#2166ac"
ROUGE_GRANDES = "#b2182b"

# couleurs distinctes des cellules (vert, rouge, orange, jaune, noir,
# bleu, violet, marron, rose, cyan) — mêmes couleurs que les courbes de
# teneur/nombre par cellule du rapport (une couleur par cell_id).
TAB10 = ["#2ca02c", "#d62728", "#ff7f0e", "#e6b800", "#000000",
         "#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#17becf"]

LIGHTING = dict(ambient=0.55, diffuse=0.85, specular=0.35,
                roughness=0.55, fresnel=0.05)
LIGHTPOS = dict(x=0.0, y=0.3, z=2.0)

# caméra vue de face (plan x-y, axe z du tambour vers l'observateur)
CAM_FACE = dict(eye=dict(x=0.0, y=0.0, z=2.1), up=dict(x=0, y=1, z=0))
# caméra trois-quarts
CAM_3Q = dict(eye=dict(x=0, y=2.1, z=0), up=dict(x=0, y=1, z=0))
# caméra vue de côté (axe x vers l'observateur : plan z-y)
CAM_COTE = dict(eye=dict(x=2.1, y=0.0, z=0.0), up=dict(x=0, y=1, z=0))


# ---------------------------------------------------------------------------
# Icosphère
# ---------------------------------------------------------------------------


def icosphere(subdiv: int = 2):
    t = (1 + 5 ** 0.5) / 2
    v = np.array(
        [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t],
         [0, 1, t], [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1],
         [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    f = np.array(
        [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
         [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
         [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
         [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    for _ in range(subdiv):
        mid = {}
        vl = v.tolist()
        nf = []

        def mp(a, b):
            k = (min(a, b), max(a, b))
            if k not in mid:
                m = (np.array(vl[a]) + np.array(vl[b])) / 2
                m /= np.linalg.norm(m)
                vl.append(m.tolist())
                mid[k] = len(vl) - 1
            return mid[k]

        for a, b, c in f:
            ab, bc, ca = mp(a, b), mp(b, c), mp(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(vl)
        f = np.array(nf)
    return v, f


SV, SF = icosphere(2)  # 162 sommets, 320 faces par sphère


def spheres_mesh(centers, radii, colors, opacity=1.0, name=None):
    """Mesh3d : une icosphère par particule, couleur par sommet."""
    n = len(centers)
    V = (SV[None] * np.asarray(radii)[:, None, None]
         + np.asarray(centers)[:, None, :]).reshape(-1, 3)
    F = (SF[None] + (np.arange(n) * len(SV))[:, None, None]).reshape(-1, 3)
    colors = np.asarray(colors)
    vc = [f"rgb({r},{g},{b})" for r, g, b in
          np.repeat(colors, len(SV), axis=0)]
    return go.Mesh3d(
        x=V[:, 0], y=V[:, 1], z=V[:, 2],
        i=F[:, 0], j=F[:, 1], k=F[:, 2],
        vertexcolor=vc, flatshading=False,
        lighting=LIGHTING, lightposition=LIGHTPOS,
        opacity=opacity, name=name, showscale=False,
    )


def axes_triad(scale=0.02, origin=(-0.055, -0.055, -0.01)):
    """Triade d'axes orientés (x rouge, y vert, z bleu) avec cônes de sens,
    ancrée près du coin inférieur de la scène."""
    o = np.asarray(origin, dtype=float)
    traces = []
    dirs = {
        "x": (np.array([1.0, 0, 0]), "#d62728"),
        "y": (np.array([0, 1.0, 0]), "#2ca02c"),
        "z": (np.array([0, 0, 1.0]), "#1f77b4"),
    }
    for name, (d, color) in dirs.items():
        tip = o + d * scale
        traces.append(go.Scatter3d(
            x=[o[0], tip[0]], y=[o[1], tip[1]], z=[o[2], tip[2]],
            mode="lines", line=dict(color=color, width=7),
            showlegend=False, hoverinfo="skip"))
        traces.append(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[d[0] * scale * 0.5], v=[d[1] * scale * 0.5],
            w=[d[2] * scale * 0.5],
            anchor="tail", colorscale=[[0, color], [1, color]],
            showscale=False, hoverinfo="skip"))
        lab = o + d * scale * 1.55
        traces.append(go.Scatter3d(
            x=[lab[0]], y=[lab[1]], z=[lab[2]], mode="text",
            text=[name], textfont=dict(color=color, size=20),
            showlegend=False, hoverinfo="skip"))
    return traces


def add_triad_all_scenes(fig):
    """Ajoute la triade d'axes à chaque scène de la figure."""
    n_scenes = sum(1 for k in fig.layout if str(k).startswith("scene"))
    n_scenes = max(n_scenes, 1)
    for i in range(1, n_scenes + 1):
        for tr in axes_triad():
            tr.update(scene=f"scene{i}" if i > 1 else "scene")
            fig.add_trace(tr)


def scene_kwargs(camera):
    return dict(
        aspectmode="data",
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
        camera=camera, bgcolor=BG,
    )

def safe_write_image(fig, out_path, width=1600, height=800, scale=1):
    """Ecriture image avec fallback si Chrome/Kaleido manquant.
    1) tente fig.write_image (kaleido)
    2) si echec ChromeNotFoundError, sauve en HTML et genere fallback matplotlib si possible
    """
    out_path = Path(out_path)
    try:
        fig.write_image(out_path, width=width, height=height, scale=scale)
        print(f"✅ {out_path.name} ok")
        return True
    except Exception as e:
        msg = str(e)
        if "Chrome" in msg or "kaleido" in msg.lower() or "Kaleido" in msg:
            print(f"⚠️ Kaleido/Chrome manquant pour {out_path.name}: {e}")
            print("→ Tentative d'installation Chrome via kaleido_get_chrome ou plotly_get_chrome")
            # tenter installation auto si possible
            try:
                import kaleido
                try:
                    kaleido.get_chrome_sync()
                    print("Chrome installé via kaleido.get_chrome_sync(), nouvelle tentative...")
                    fig.write_image(out_path, width=width, height=height, scale=scale)
                    print(f"✅ {out_path.name} ok après install Chrome")
                    return True
                except Exception as e2:
                    print(f"  Echec install auto Chrome: {e2}")
            except Exception:
                pass
            # fallback: sauver en HTML
            html_path = out_path.with_suffix('.html')
            try:
                fig.write_html(html_path)
                print(f"💾 Fallback HTML sauvé: {html_path} – ouvrez dans navigateur et capturez PNG si besoin")
                print(f"   Pour corriger définitivement, lancez dans ce terminal:")
                print(f"   $ kaleido_get_chrome  # ou  plotly_get_chrome")
                print(f"   Puis relancez: python {Path(__file__).name}")
            except Exception as e3:
                print(f"  Echec fallback HTML: {e3}")
            return False
        else:
            print(f"❌ Erreur write_image {out_path.name}: {e}")
            raise




def _hex2rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


# ---------------------------------------------------------------------------
# Données et labels (mêmes conventions que etudes_rapport.py)
# ---------------------------------------------------------------------------


def ensure_compact():
    """Genere data/compact.npz depuis data/chunks si absent (auto-generation pour MyStudio)."""
    compact_path = ROOT / "data" / "compact.npz"
    if compact_path.exists():
        return
    chunks_dir = ROOT / "data" / "chunks"
    if chunks_dir.exists() and list(chunks_dir.glob("*.parquet.gz")):
        print(f"⚠️ {compact_path} absent – generation depuis {chunks_dir} via extract_compact...")
        try:
            from postprocessing.extract_compact import main as extract_main
            extract_main()
        except Exception as e:
            print(f"Erreur generation compact.npz: {e}")
            # fallback: generation manuelle minimale
            import gzip, io, pandas as pd
            COLS = ["Fichier_Source","Particle_ID","Diameter","coordinates:0","coordinates:1","coordinates:2","Velocity:0","Velocity:1","Velocity:2"]
            ts,pid,xyz,vn,small=[],[],[],[],[]
            for p in sorted(chunks_dir.glob("*.parquet.gz")):
                with gzip.open(p,"rb") as fh:
                    df=pd.read_parquet(io.BytesIO(fh.read()), columns=COLS)
                import numpy as np
                t = df["Fichier_Source"].str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
                ts.append(t.astype(np.int16))
                pid.append(df["Particle_ID"].to_numpy().astype(np.int16))
                xyz.append(df[["coordinates:0","coordinates:1","coordinates:2"]].to_numpy().astype(np.float32))
                v=df[["Velocity:0","Velocity:1","Velocity:2"]].to_numpy()
                vn.append(np.linalg.norm(v,axis=1).astype(np.float32))
                small.append((df["Diameter"].to_numpy()<0.006))
            import numpy as np
            np.savez_compressed(compact_path, t=np.concatenate(ts), pid=np.concatenate(pid), xyz=np.concatenate(xyz), vnorm=np.concatenate(vn), small=np.concatenate(small))
            print(f"✅ {compact_path} genere")
    else:
        print(f"⚠️ {compact_path} absent et {chunks_dir} vide – tentative via HuggingFace ou message d'aide")
        print("Pour generer les donnees, lancez: python postprocessing/extract_compact.py puis python postprocessing/etudes_librairie.py")

def load_frames():
    ensure_compact()
    d = np.load(ROOT / "data" / "compact.npz")
    t = d["t"].astype(np.int32)
    pid = d["pid"].astype(np.int32)
    xyz = d["xyz"]
    vnorm = d["vnorm"]
    small = d["small"]
    pid_u = np.unique(pid)
    pid_map = np.zeros(pid_u.max() + 1, dtype=np.int32)
    pid_map[pid_u] = np.arange(len(pid_u))
    n_p = len(pid_u)
    X = np.zeros((N_T, n_p, 3), dtype=np.float32)
    V = np.zeros((N_T, n_p), dtype=np.float32)
    X[t, pid_map[pid]] = xyz
    V[t, pid_map[pid]] = vnorm
    small_p = np.zeros(n_p, dtype=bool)
    small_p[pid_map[pid]] = small
    return X, V, small_p


def ensure_labels():
    """Genere data/labels_librairie.npz si absent via etudes_librairie.dump_labels_3d."""
    labels_path = ROOT / "data" / "labels_librairie.npz"
    if labels_path.exists():
        return
    compact_path = ROOT / "data" / "compact.npz"
    if not compact_path.exists():
        ensure_compact()
    if compact_path.exists():
        print(f"⚠️ {labels_path} absent – generation via etudes_librairie...")
        try:
            # reutilise la logique d'etudes_librairie pour generer labels aux tours demandes
            import sys
            sys.path.insert(0, str(ROOT))
            from postprocessing.etudes_librairie import load_timestep_dict, EtudeMethode, METHODES, config_for
            from dem_mcm_coupling.run_sweep import sample_coordinates, PERMANENT_START, N_PARTICLES_PER_TIMESTEP
            from postprocessing.metrics import concentration_from_S
            import numpy as np
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
                import pandas as pd
                out={}
                for idx,df in timestep_dict.items():
                    xy = (df[["coordinates:0","coordinates:1"]].to_numpy()-c[:2])@R2
                    out[idx]=pd.DataFrame({"coordinates:0": xy[:,0],"coordinates:1": xy[:,1],"coordinates:2": df["coordinates:2"].to_numpy()-c[2]})
                return out
            timestep_dict = load_timestep_dict()
            sample_coords, s_velocities, _ = sample_coordinates(timestep_dict)
            permanent_rows = PERMANENT_START * N_PARTICLES_PER_TIMESTEP
            frame = make_frame(sample_coords, permanent_rows)
            etudes={}
            for key in METHODES:
                print(f"Prep {key} pour labels")
                # inline EtudeMethode to avoid import recursion
                from dem_mcm_coupling.partitioners import create_partitioner
                from dem_mcm_coupling.run_sweep import ExperimentConfig, _build_state_matrices, _compute_state_matrices, _detect_species, _fit_partitioner_for_sweep
                nom,method,kwargs=METHODES[key]
                partitioner=create_partitioner(method,**kwargs)
                cfg_fit=ExperimentConfig(method=method,method_kwargs=kwargs)
                label_dict=timestep_dict
                fit_coords=sample_coords
                if key in {"cartesien","cylindrique"}:
                    c,R2=frame
                    fit_coords=transform_coords(sample_coords,c,R2)
                    label_dict=transform_dict(timestep_dict,c,R2)
                    partitioner.fit(fit_coords[permanent_rows:])
                else:
                    _fit_partitioner_for_sweep(partitioner,cfg_fit,fit_coords,s_velocities,permanent_rows)
                species_masks=_detect_species(timestep_dict[min(timestep_dict)])
                states_matrix,sorted_indices=_compute_state_matrices(label_dict,partitioner,157)
                idx_to_row={int(i):r for r,i in enumerate(sorted_indices)}
                S_matrices=_build_state_matrices(states_matrix,species_masks,partitioner.n_cells)
                times=np.asarray(sorted_indices)
                etudes[key]=(partitioner,states_matrix,sorted_indices,idx_to_row,S_matrices,times)
            out={}
            times_snap=(0,157,3645,5215,5999,3000)
            for key,(partitioner,states_matrix,sorted_indices,idx_to_row,S_matrices,times) in etudes.items():
                for t in times_snap:
                    if t in idx_to_row:
                        out[f"{key}_{t}"]=states_matrix[idx_to_row[t]]
                for t in times_snap:
                    if t in idx_to_row:
                        S_s=S_matrices["small"][idx_to_row[t]][None]
                        S_l=S_matrices["large"][idx_to_row[t]][None]
                        out[f"{key}_teneur_{t}"]=concentration_from_S(S_s,S_l)[0]
            np.savez(labels_path, **out)
            print(f"✅ {labels_path} genere")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Erreur generation labels: {e}")
    else:
        print(f"⚠️ Impossible de generer {labels_path}, {compact_path} manquant")

def partition_labels(X, V, small_p):
    ensure_labels()
    d = np.load(ROOT / "data" / "labels_librairie.npz")
    noms = {
        "cartesien": "cartésien (10 bandes, repère du lit)",
        "cylindrique": "cylindrique (10 secteurs, repère du lit)",
        "voronoi": "de Voronoï (10 cellules)",
        "physique": "physique (10 cellules)",
    }
    return {k: (noms[k], d[f"{k}_157"].astype(int)) for k in noms}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_melangeur_especes(X, small_p):
    diam = np.where(small_p, D_SMALL, D_BIG)
    colors = np.where(small_p[:, None],
                      [_hex2rgb(BLEU_PETITES)], [_hex2rgb(ROUGE_GRANDES)])
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("(a) État initial (t = 0 s)",
                        "(b) Régime établi (t = 1,57 s)"),
        horizontal_spacing=0.01,
    )
    fig.add_trace(spheres_mesh(X[0], diam / 2, colors), row=1, col=1)
    fig.add_trace(spheres_mesh(X[START], diam / 2, colors), row=1, col=2)
    fig.update_scenes(**scene_kwargs(CAM_3Q))
    fig.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=50, b=30),
        font=dict(color="white", size=21),
        showlegend=False,
        annotations=list(fig.layout.annotations) + [
            dict(x=0.35, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="\u25cf petites (4 mm)",
                 font=dict(color=BLEU_PETITES, size=20),
                 bgcolor="white"),
            dict(x=0.65, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="\u25cf grandes (8 mm)",
                 font=dict(color=ROUGE_GRANDES, size=20),
                 bgcolor="white"),
        ],
    )
    add_triad_all_scenes(fig)
    safe_write_image(fig, FIGDIR / "pv_melangeur_especes.png", width=1600, height=780, scale=1)
    print("pv_melangeur_especes ok")


def fig_cellules(X, small_p, key, nom, lab):
    """SERIEUSEMENT: vue isometrique uniquement pour pv_cellules_physique.png – demande interpretation
    Particules colorees par cellule (cell_id).
    - Toutes methodes: face + cote
    - Physique uniquement: + vue isometrique pour interpretation.
    """
    diam = np.where(small_p, D_SMALL, D_BIG)
    colors = np.array([_hex2rgb(TAB10[l % 10]) for l in lab])
    if key == "physique" or key=="voronoi":
        fig = make_subplots(
            rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
            subplot_titles=("vue de face (plan x\u2013y)",
                            "vue de côté (plan z\u2013y)",
                            "vue de dessus"),
            horizontal_spacing=0.01,
        )
        fig.add_trace(spheres_mesh(X[START], diam / 2, colors), row=1, col=1)
        fig.add_trace(spheres_mesh(X[START], diam / 2, colors), row=1, col=2)
        fig.add_trace(spheres_mesh(X[START], diam / 2, colors), row=1, col=3)
        for tr in _cell_id_labels(X[START], lab, view="face", per_cell_colors=TAB10):
            fig.add_trace(tr, row=1, col=1)
        for tr in _cell_id_labels(X[START], lab, view="cote", per_cell_colors=TAB10):
            fig.add_trace(tr, row=1, col=2)
        for tr in _cell_id_labels(X[START], lab, view="face", per_cell_colors=TAB10):
            fig.add_trace(tr, row=1, col=3)
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(
                colorscale=sum([[[i / 10, TAB10[i]], [(i + 1) / 10, TAB10[i]]] for i in range(10)], []),
                cmin=-0.5, cmax=9.5, color=[0],
                colorbar=dict(title=dict(text="cell_id", font=dict(color="white", size=20)),
                              tickvals=list(range(10)), tickfont=dict(color="white", size=19),
                              len=0.8, thickness=22),
                showscale=True, size=0.0001,
            ),
            showlegend=False,
        ), row=1, col=3)
        fig.update_scenes(**scene_kwargs(CAM_FACE))
        fig.update_scenes(camera=CAM_COTE, row=1, col=2)
        fig.update_scenes(camera=CAM_3Q, row=1, col=3)
        fig.update_layout(
            paper_bgcolor=BG, margin=dict(l=0, r=0, t=64, b=0),
            font=dict(color="white", size=21),
            title=dict(text=f"Découpage {nom} — particules par cellule (t=1,57s) — face, côté, dessus ",
                       font=dict(color="white", size=22), x=0.5),
        )
        add_triad_all_scenes(fig)
        safe_write_image(fig, FIGDIR / f"pv_cellules_{key}.png", width=2400, height=800, scale=1)
        print(f"pv_cellules_{key} ok – avec vue isométrique (physique uniquement)")
    else:
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
            subplot_titles=("vue de face (plan x\u2013y)",
                            "vue de côté (plan z\u2013y)"),
            horizontal_spacing=0.01,
        )
        fig.add_trace(spheres_mesh(X[START], diam / 2, colors), row=1, col=1)
        fig.add_trace(spheres_mesh(X[START], diam / 2, colors), row=1, col=2)
        for tr in _cell_id_labels(X[START], lab, view="face", per_cell_colors=TAB10):
            fig.add_trace(tr, row=1, col=1)
        for tr in _cell_id_labels(X[START], lab, view="cote", per_cell_colors=TAB10):
            fig.add_trace(tr, row=1, col=2)
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(
                colorscale=sum([[[i / 10, TAB10[i]], [(i + 1) / 10, TAB10[i]]] for i in range(10)], []),
                cmin=-0.5, cmax=9.5, color=[0],
                colorbar=dict(title=dict(text="cell_id", font=dict(color="white", size=20)),
                              tickvals=list(range(10)), tickfont=dict(color="white", size=19),
                              len=0.8, thickness=22),
                showscale=True, size=0.0001,
            ),
            showlegend=False,
        ), row=1, col=2)
        fig.update_scenes(**scene_kwargs(CAM_FACE))
        fig.update_scenes(camera=CAM_COTE, row=1, col=2)
        fig.update_layout(
            paper_bgcolor=BG, margin=dict(l=0, r=0, t=64, b=0),
            font=dict(color="white", size=21),
            title=dict(text=f"Découpage {nom} — particules par cellule (t=1,57s)",
                       font=dict(color="white", size=22), x=0.5),
        )
        add_triad_all_scenes(fig)
        safe_write_image(fig, FIGDIR / f"pv_cellules_{key}.png", width=1700, height=800, scale=1)
        print(f"pv_cellules_{key} ok – 2 vues")


def fig_contenu_cellule(X, small_p, labels):
    nom, lab = labels["physique"]
    cell = np.bincount(lab, minlength=10).argmax()
    inside = lab == cell
    ns = int((inside & small_p).sum())
    ntot = int(inside.sum())
    c = ns / ntot

    diam = np.where(small_p, D_SMALL, D_BIG)
    colors_in = np.where(small_p[inside][:, None],
                         [_hex2rgb(BLEU_PETITES)], [_hex2rgb(ROUGE_GRANDES)])
    grey = np.tile([210, 210, 210], ((~inside).sum(), 1))

    fig = go.Figure()
    fig.add_trace(spheres_mesh(X[START][~inside], diam[~inside] / 2, grey,
                               opacity=0.05))
    fig.add_trace(spheres_mesh(X[START][inside], diam[inside] / 2, colors_in))
    fig.update_scenes(**scene_kwargs(CAM_FACE))
    fig.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=54, b=30),
        title=dict(
            text=(f"Contenu de la cellule {cell} (découpage physique, "
                  f"t = 1,57 s) : {ns} petites + {ntot - ns} grandes "
                  f"\u21d2 teneur c = {ns}/{ntot} = {c:.2f}"),
            font=dict(color="white", size=22), x=0.5),
        annotations=[
            dict(x=0.32, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="\u25cf petites de la cellule",
                 font=dict(color=BLEU_PETITES, size=19), bgcolor="white"),
            dict(x=0.68, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="\u25cf grandes de la cellule",
                 font=dict(color=ROUGE_GRANDES, size=19), bgcolor="white"),
        ],
    )
    add_triad_all_scenes(fig)
    safe_write_image(fig, FIGDIR / "pv_cellule_contenu.png", width=1300, height=1000, scale=1)
    print("pv_cellule_contenu ok")




def fig_melange_instants(X, small_p):
    """Mélangeur à des instants distincts : évolution du mélange
    (non discrétisé, espèces distinctes) – snapshots à différents tours : 1, fin-15, fin-5, fin."""
    diam = np.where(small_p, D_SMALL, D_BIG)
    colors = np.where(small_p[:, None],
                      [_hex2rgb(BLEU_PETITES)], [_hex2rgb(ROUGE_GRANDES)])
    instants = [(157, "tour 1 (t = 1,57 s)"), (3645, "fin-15 (t = 36,45 s)"),
                (5215, "fin-5 (t = 52,15 s)"), (5999, "fin (t = 60 s)")]
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}] * 2] * 2,
        subplot_titles=[t for _, t in instants],
        horizontal_spacing=0.01, vertical_spacing=0.04,
    )
    for k, (t, _) in enumerate(instants):
        fig.add_trace(spheres_mesh(X[t], diam / 2, colors),
                      row=k // 2 + 1, col=k % 2 + 1)
    fig.update_scenes(**scene_kwargs(CAM_FACE))
    fig.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=40, b=8),
        font=dict(color="white", size=21), showlegend=False,
    )
    add_triad_all_scenes(fig)
    safe_write_image(fig, FIGDIR / "pv_melange_instants.png", width=1500, height=1400, scale=1)
    print("pv_melange_instants ok")


def fig_melange_instants_voronoi(X, small_p):
    """Mélangeur discrétisé (Voronoï) à des instants distincts :
    particules colorées par cellule – snapshots tour1, fin-15, fin-5, fin."""
    d = np.load(ROOT / "data" / "labels_librairie.npz")
    diam = np.where(small_p, D_SMALL, D_BIG)
    instants = [(157, "tour 1 (t = 1,57 s)"), (3645, "fin-15 (t = 36,45 s)"),
                (5215, "fin-5 (t = 52,15 s)"), (5999, "fin (t = 60 s)")]
    fig = make_subplots(
        rows=2, cols=2, specs=[[{"type": "scene"}] * 2] * 2,
        subplot_titles=[t for _, t in instants],
        horizontal_spacing=0.005, vertical_spacing=0.04,
    )
    for k, (t, _) in enumerate(instants):
        lab = d[f"voronoi_{t}"].astype(int)
        colors = np.array([_hex2rgb(TAB10[l % 10]) for l in lab])
        fig.add_trace(spheres_mesh(X[t], diam / 2, colors), row=k//2+1, col=k%2+1)
    fig.update_scenes(**scene_kwargs(CAM_FACE))
    fig.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=42, b=8),
        font=dict(color="white", size=21), showlegend=False,
    )
    add_triad_all_scenes(fig)
    safe_write_image(fig, FIGDIR / "pv_melange_instants_voronoi.png", width=1800, height=1400, scale=2)
    print("pv_melange_instants_voronoi ok")


def _label_text_color(c, per_cell_colors=None):
    """Couleur de texte contrastée avec la couleur de la cellule."""
    if per_cell_colors is None:
        return "black"
    h = per_cell_colors[int(c) % len(per_cell_colors)]
    r, g, b = _hex2rgb(h)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if lum < 110 else "black"


def _cell_id_labels(X_t, lab, view="face", per_cell_colors=None):
    """Étiquettes cell_id positionnées au barycentre de chaque cellule et
    projetées devant le lit (vers la caméra) pour rester lisibles :
    la cellule concernée s'identifie directement sur la vue 3D. Les
    étiquettes qui se superposeraient dans le plan de projection sont
    écartées par une courte répulsion 2D."""
    cells = np.unique(lab)
    # coordonnées 2D dans le plan de projection
    pos2 = []
    for c in cells:
        bc = X_t[lab == c].mean(axis=0)
        pos2.append([bc[0], bc[1]] if view == "face" else [bc[2], bc[1]])
    pos2 = np.array(pos2)
    # répulsion simple pour éviter les chevauchements
    d_min = 0.011
    for _ in range(60):
        moved = False
        for i in range(len(pos2)):
            for j in range(i + 1, len(pos2)):
                d = pos2[j] - pos2[i]
                dist = np.linalg.norm(d)
                if dist < d_min:
                    push = (d / (dist + 1e-9)) * (d_min - dist) / 2
                    pos2[i] -= push
                    pos2[j] += push
                    moved = True
        if not moved:
            break
    traces = []
    if view == "face":
        z_front = X_t[:, 2].max() + 0.012
    else:
        x_front = X_t[:, 0].max() + 0.012
    for k, c in enumerate(cells):
        if view == "face":
            x, y, z = pos2[k, 0], pos2[k, 1], z_front
        else:
            x, y, z = x_front, pos2[k, 1], pos2[k, 0]
        traces.append(go.Scatter3d(
            x=[x], y=[y], z=[z], mode="text",
            text=[str(int(c))],
            textfont=dict(color=_label_text_color(c, per_cell_colors),
                          size=34, family="Arial Black"),
            showlegend=False, hoverinfo="skip"))
    return traces


NOMS_METHODES = {
    "cartesien": "cartésien",
    "cylindrique": "cylindrique",
    "voronoi": "Voronoï",
    "physique": "physique",
}


def fig_teneur_3d(X, small_p):
    """Mélangeur discrétisé coloré par la teneur locale de la cellule,
    vues de face et de côté : snapshots à différents tours (tour1, fin-15, fin-5, fin)
    pour les quatre méthodes de découpage. Chaque cellule porte son numéro au barycentre.
    Figures pyvista_js avec particules réelles."""
    d = np.load(ROOT / "data" / "labels_librairie.npz")
    diam = np.where(small_p, D_SMALL, D_BIG)
    cmap = plt_cmap_viridis()
    # snapshots demandés pour toutes méthodes
    times_snap = [157, 3645, 5215, 5999]
    cas = [(key, t) for key in NOMS_METHODES for t in times_snap]
    # garder aussi 3000 pour compatibilité
    cas += [("voronoi", 3000)]
    for key, t in cas:
        lab = d[f"{key}_{t}"].astype(int)
        ten = d[f"{key}_teneur_{t}"]
        vals = ten[lab]
        colors = (np.array([cmap(v) for v in vals])[:, :3] * 255).astype(int)
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
            subplot_titles=("vue de face", "vue de côté"),
            horizontal_spacing=0.01,
        )
        fig.add_trace(spheres_mesh(X[t], diam / 2, colors), row=1, col=1)
        fig.add_trace(spheres_mesh(X[t], diam / 2, colors), row=1, col=2)
        # numéros de cellule au barycentre, sur les deux vues (couleur de
        # texte contrastée avec la teinte de teneur de la cellule)
        ten_colors = ["#%02x%02x%02x" % tuple(
            (np.array(cmap(ten[c]))[:3] * 255).astype(int)) for c in range(10)]
        for tr in _cell_id_labels(X[t], lab, view="face",
                                  per_cell_colors=ten_colors):
            fig.add_trace(tr, row=1, col=1)
        for tr in _cell_id_labels(X[t], lab, view="cote",
                                  per_cell_colors=ten_colors):
            fig.add_trace(tr, row=1, col=2)
        fig.update_scenes(**scene_kwargs(CAM_FACE))
        fig.update_scenes(camera=CAM_COTE, row=1, col=2)
        # colorbar continue viridis
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(colorscale="Viridis", cmin=0, cmax=1, color=[0],
                        colorbar=dict(
                            title=dict(text="teneur locale",
                                       font=dict(color="white", size=20)),
                            tickfont=dict(color="white", size=19),
                            len=0.75, thickness=20),
                        showscale=True, size=0.0001),
            showlegend=False), row=1, col=2)
        cmin, cmax = ten.min(), ten.max()
        imin, imax = int(np.argmin(ten)), int(np.argmax(ten))
        fig.update_layout(
            paper_bgcolor=BG, margin=dict(l=0, r=0, t=64, b=8),
            font=dict(color="white", size=21),
            title=dict(text=(f"Teneur locale par cellule "
                             f"({NOMS_METHODES[key]}, "
                             f"t = {t / 100:g} s) — min : cellule {imin} "
                             f"({cmin:.2f}), max : cellule {imax} "
                             f"({cmax:.2f}) — numéro de cellule au "
                             f"barycentre"),
                       font=dict(color="white", size=22), x=0.5),
        )
        add_triad_all_scenes(fig)
        safe_write_image(fig, FIGDIR / f"pv_teneur_{key}_t{t}.png", width=1700, height=780, scale=1)
        print(f"pv_teneur_{key}_t{t} ok")


def plt_cmap_viridis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.get_cmap("viridis")



def fig_snapshots_all_methods(X, small_p):
    """Snapshots pyvista_js avec particules réelles pour chaque méthode à 4 tours : tour1, fin-15, fin-5, fin.
    Chaque figure combine 4 instants (2x2) avec particules colorées par teneur locale, zoom cadré.
    """
    d = np.load(ROOT / "data" / "labels_librairie.npz")
    diam = np.where(small_p, D_SMALL, D_BIG)
    cmap = plt_cmap_viridis()
    times_snap = [(157, "tour 1"), (3645, "fin-15"), (5215, "fin-5"), (5999, "fin")]
    for key in NOMS_METHODES:
        fig = make_subplots(
            rows=2, cols=2, specs=[[{"type": "scene"}]*2]*2,
            subplot_titles=[f"{name} (t={t/100:.2f}s)" for t,name in times_snap],
            horizontal_spacing=0.01, vertical_spacing=0.04,
        )
        for k,(t,name) in enumerate(times_snap):
            if f"{key}_{t}" not in d or f"{key}_teneur_{t}" not in d:
                continue
            lab = d[f"{key}_{t}"].astype(int)
            ten = d[f"{key}_teneur_{t}"]
            vals = ten[lab]
            colors = (np.array([cmap(v) for v in vals])[:,:3]*255).astype(int)
            fig.add_trace(spheres_mesh(X[t], diam/2, colors), row=k//2+1, col=k%2+1)
        fig.update_scenes(**scene_kwargs(CAM_FACE))
        fig.update_layout(
            paper_bgcolor=BG, margin=dict(l=0,r=0,t=60,b=0),
            font=dict(color="white", size=18),
            title=dict(text=f"Snapshots pyvista – découpage {NOMS_METHODES[key]} – particules réelles colorées par teneur locale (tour1, fin-15, fin-5, fin)",
                       font=dict(color="white", size=20), x=0.5),
            showlegend=False,
        )
        add_triad_all_scenes(fig)
        out = FIGDIR / f"pv_snapshot_{key}_teneur.png"
        safe_write_image(fig, out, width=1600, height=1400, scale=1)
        print(f"pv_snapshot_{key} ok -> {out}")


if __name__ == "__main__":
    X, V, small_p = load_frames()
    labels = partition_labels(X, V, small_p)
    # fig_melangeur_especes(X, small_p)
    # fig_melange_instants(X, small_p)
    fig_melange_instants_voronoi(X, small_p)
    # fig_teneur_3d(X, small_p)
    # fig_snapshots_all_methods(X, small_p)
    # for key, (nom, lab) in labels.items():
    #     fig_cellules(X, small_p, key, nom, lab)
    # # fig_contenu_cellule(X, small_p, labels)
    print("✅ figures WebGL (pyvista_js) avec particules réelles écrites dans", FIGDIR)