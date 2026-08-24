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

# palette tab10 pour les cell_id
TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
         "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

LIGHTING = dict(ambient=0.55, diffuse=0.85, specular=0.35,
                roughness=0.55, fresnel=0.05)
LIGHTPOS = dict(x=0.0, y=0.3, z=2.0)

# caméra vue de face (plan x-y, axe z du tambour vers l'observateur)
CAM_FACE = dict(eye=dict(x=0.0, y=0.0, z=2.1), up=dict(x=0, y=1, z=0))
# caméra trois-quarts
CAM_3Q = dict(eye=dict(x=1.3, y=0.9, z=1.5), up=dict(x=0, y=1, z=0))
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


def scene_kwargs(camera):
    return dict(
        aspectmode="data",
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
        camera=camera, bgcolor=BG,
    )


def _hex2rgb(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


# ---------------------------------------------------------------------------
# Données et labels (mêmes conventions que etudes_rapport.py)
# ---------------------------------------------------------------------------


def load_frames():
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


def partition_labels(X, V, small_p):
    """Labels de cellule au régime établi, produits par la librairie.

    Les labels sont calculés par ``etudes_librairie.py`` (partitionneurs de
    ``dem_mcm_coupling.partitioners``, fit de ``run_sweep``) et sauvegardés
    dans ``data/labels_librairie.npz`` — aucune réimplémentation ici.
    """
    d = np.load(ROOT / "data" / "labels_librairie.npz")
    noms = {
        "cartesien": "cartésien (5 \u00d7 2 \u00d7 1)",
        "cylindrique": "cylindrique (n\u1d63 = 2, n\u03b8 = 5)",
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
        font=dict(color="white", size=15),
        showlegend=False,
        annotations=list(fig.layout.annotations) + [
            dict(x=0.35, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="\u25cf petites (4 mm)",
                 font=dict(color=BLEU_PETITES, size=16),
                 bgcolor="white"),
            dict(x=0.65, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="\u25cf grandes (8 mm)",
                 font=dict(color=ROUGE_GRANDES, size=16),
                 bgcolor="white"),
        ],
    )
    fig.write_image(FIGDIR / "pv_melangeur_especes.png",
                    width=1600, height=780, scale=1)
    print("pv_melangeur_especes ok")


def fig_cellules(X, small_p, key, nom, lab):
    """Particules colorées par cellule (cell_id), vue de face ET de côté :
    les cellules k-means se partagent aussi selon l'axe z du tambour,
    invisible en vue de face seule."""
    diam = np.where(small_p, D_SMALL, D_BIG)
    colors = np.array([_hex2rgb(TAB10[l % 10]) for l in lab])
    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
        subplot_titles=("vue de face (plan x\u2013y)",
                        "vue de côté (plan z\u2013y)"),
        horizontal_spacing=0.01,
    )
    fig.add_trace(spheres_mesh(X[START], diam / 2, colors), row=1, col=1)
    fig.add_trace(spheres_mesh(X[START], diam / 2, colors), row=1, col=2)
    # barre de couleurs discrète simulée par un scatter invisible
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None], mode="markers",
        marker=dict(
            colorscale=sum([[[i / 10, TAB10[i]], [(i + 1) / 10, TAB10[i]]]
                            for i in range(10)], []),
            cmin=-0.5, cmax=9.5,
            color=[0],
            colorbar=dict(title=dict(text="cell_id", font=dict(color="white")),
                          tickvals=list(range(10)),
                          tickfont=dict(color="white"),
                          len=0.8, thickness=22),
            showscale=True, size=0.0001,
        ),
        showlegend=False,
    ), row=1, col=2)
    fig.update_scenes(**scene_kwargs(CAM_FACE))
    fig.update_scenes(camera=CAM_COTE, row=1, col=2)
    fig.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=64, b=0),
        font=dict(color="white", size=14),
        title=dict(text=f"Découpage {nom} — particules colorées par cellule "
                        f"(t = 1,57 s)",
                   font=dict(color="white", size=17), x=0.5),
    )
    fig.write_image(FIGDIR / f"pv_cellules_{key}.png",
                    width=1700, height=800, scale=1)
    print(f"pv_cellules_{key} ok")


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
            font=dict(color="white", size=16), x=0.5),
        annotations=[
            dict(x=0.32, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="\u25cf petites de la cellule",
                 font=dict(color=BLEU_PETITES, size=15), bgcolor="white"),
            dict(x=0.68, y=0.02, xref="paper", yref="paper", showarrow=False,
                 text="\u25cf grandes de la cellule",
                 font=dict(color=ROUGE_GRANDES, size=15), bgcolor="white"),
        ],
    )
    fig.write_image(FIGDIR / "pv_cellule_contenu.png",
                    width=1300, height=1000, scale=1)
    print("pv_cellule_contenu ok")




def fig_melange_instants(X, small_p):
    """Mélangeur à des instants distincts : évolution du mélange
    (non discrétisé, espèces distinctes)."""
    diam = np.where(small_p, D_SMALL, D_BIG)
    colors = np.where(small_p[:, None],
                      [_hex2rgb(BLEU_PETITES)], [_hex2rgb(ROUGE_GRANDES)])
    instants = [(0, "t = 0 s"), (157, "t = 1,57 s"),
                (1000, "t = 10 s"), (3000, "t = 30 s")]
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
        font=dict(color="white", size=14), showlegend=False,
    )
    fig.write_image(FIGDIR / "pv_melange_instants.png",
                    width=1500, height=1400, scale=1)
    print("pv_melange_instants ok")


def fig_melange_instants_voronoi(X, small_p):
    """Mélangeur discrétisé (Voronoï) à des instants distincts :
    particules colorées par cellule."""
    d = np.load(ROOT / "data" / "labels_librairie.npz")
    diam = np.where(small_p, D_SMALL, D_BIG)
    instants = [(0, "t = 0 s"), (157, "t = 1,57 s"), (3000, "t = 30 s")]
    fig = make_subplots(
        rows=1, cols=3, specs=[[{"type": "scene"}] * 3],
        subplot_titles=[t for _, t in instants],
        horizontal_spacing=0.005,
    )
    for k, (t, _) in enumerate(instants):
        lab = d[f"voronoi_{t}"].astype(int)
        colors = np.array([_hex2rgb(TAB10[l % 10]) for l in lab])
        fig.add_trace(spheres_mesh(X[t], diam / 2, colors), row=1, col=k + 1)
    fig.update_scenes(**scene_kwargs(CAM_FACE))
    fig.update_layout(
        paper_bgcolor=BG, margin=dict(l=0, r=0, t=42, b=8),
        font=dict(color="white", size=14), showlegend=False,
    )
    fig.write_image(FIGDIR / "pv_melange_instants_voronoi.png",
                    width=1800, height=680, scale=1)
    print("pv_melange_instants_voronoi ok")


def fig_teneur_3d(X, small_p):
    """Mélangeur discrétisé (Voronoï) coloré par la teneur locale de la
    cellule, vues de face et de côté, régime établi et instant tardif.
    Met en évidence les cellules sensibles (teneur haute/basse)."""
    d = np.load(ROOT / "data" / "labels_librairie.npz")
    diam = np.where(small_p, D_SMALL, D_BIG)
    cmap = plt_cmap_viridis()
    for t in (157, 3000):
        lab = d[f"voronoi_{t}"].astype(int)
        ten = d[f"voronoi_teneur_{t}"]
        vals = ten[lab]
        colors = (np.array([cmap(v) for v in vals])[:, :3] * 255).astype(int)
        fig = make_subplots(
            rows=1, cols=2, specs=[[{"type": "scene"}] * 2],
            subplot_titles=("vue de face", "vue de côté"),
            horizontal_spacing=0.01,
        )
        fig.add_trace(spheres_mesh(X[t], diam / 2, colors), row=1, col=1)
        fig.add_trace(spheres_mesh(X[t], diam / 2, colors), row=1, col=2)
        fig.update_scenes(**scene_kwargs(CAM_FACE))
        fig.update_scenes(camera=CAM_COTE, row=1, col=2)
        # colorbar continue viridis
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(colorscale="Viridis", cmin=0, cmax=1, color=[0],
                        colorbar=dict(
                            title=dict(text="teneur locale",
                                       font=dict(color="white")),
                            tickfont=dict(color="white"),
                            len=0.75, thickness=20),
                        showscale=True, size=0.0001),
            showlegend=False), row=1, col=2)
        cmin, cmax = ten.min(), ten.max()
        imin, imax = int(np.argmin(ten)), int(np.argmax(ten))
        fig.update_layout(
            paper_bgcolor=BG, margin=dict(l=0, r=0, t=64, b=8),
            font=dict(color="white", size=14),
            title=dict(text=(f"Teneur locale par cellule (Voronoï, "
                             f"t = {t / 100:g} s) — min : cellule {imin} "
                             f"({cmin:.2f}), max : cellule {imax} "
                             f"({cmax:.2f})"),
                       font=dict(color="white", size=16), x=0.5),
        )
        fig.write_image(FIGDIR / f"pv_teneur_voronoi_t{t}.png",
                        width=1700, height=780, scale=1)
        print(f"pv_teneur_voronoi_t{t} ok")


def plt_cmap_viridis():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.get_cmap("viridis")


if __name__ == "__main__":
    X, V, small_p = load_frames()
    labels = partition_labels(X, V, small_p)
    fig_melangeur_especes(X, small_p)
    fig_melange_instants(X, small_p)
    fig_melange_instants_voronoi(X, small_p)
    fig_teneur_3d(X, small_p)
    for key, (nom, lab) in labels.items():
        fig_cellules(X, small_p, key, nom, lab)
    fig_contenu_cellule(X, small_p, labels)
    print("✅ figures WebGL écrites dans", FIGDIR)
