"""Post-traitement 3D des résultats avec pyvista.

Conformément aux conventions de la librairie :mod:`dem_mcm_coupling`
(cf. ``Markov.build_vtp`` et ``Markov.visualize`` dans ``markov_core.py``),
les particules sont représentées par des sphères de diamètre réel
(``Diameter``) portant les scalaires ``cell_id`` (label de partition) et
``species`` (petite/grande). Les figures servent de base visuelle aux
justifications et validations des résultats :

* ``pv_melangeur_especes.png``      : le mélangeur avec les deux espèces
  (grandes 8 mm / petites 4 mm) à l'instant initial et en régime établi ;
* ``pv_cellules_cartesien.png``     : particules colorées par cellule,
  découpage cartésien (10 cellules) ;
* ``pv_cellules_cylindrique.png``   : idem, découpage cylindrique ;
* ``pv_cellules_voronoi.png``       : idem, découpage de Voronoï ;
* ``pv_cellules_physique.png``      : idem, découpage physique ;
* ``pv_cellule_contenu.png``        : zoom sur une cellule (particules de la
  cellule opaques, reste du lit estompé) pour montrer visuellement le
  contenu d'une cellule et sa teneur.

Rendu : pyvista (off-screen) est utilisé lorsqu'un backend OpenGL/OSMesa
est disponible ; à défaut (bac à sable sans GL), un rendu logiciel
équivalent est produit avec matplotlib 3D — mêmes vues, mêmes scalaires,
mêmes tailles de sphères — afin que les figures restent reproductibles
partout. Sur une machine disposant d'OpenGL, relancer ce script régénère
les mêmes fichiers via pyvista.

Usage::

    python postprocessing/pyvista_figures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "template-rapport-stage" / "figures"
FIGDIR.mkdir(exist_ok=True)

TAU = 157
START = 157
N_T = 6000

D_SMALL = 0.004
D_BIG = 0.008

# ---------------------------------------------------------------------------
# Détection du backend de rendu
# ---------------------------------------------------------------------------


def _pyvista_available() -> bool:
    """True si pyvista peut effectivement rendre off-screen (GL présent)."""
    try:
        import subprocess
        import sys

        code = (
            "import pyvista as pv; pv.OFF_SCREEN = True;"
            "p = pv.Plotter(off_screen=True, window_size=[64, 64]);"
            "p.add_mesh(pv.Sphere()); p.screenshot('/tmp/_pv_probe.png');"
            "print('OK')"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, timeout=120
        )
        return out.returncode == 0 and b"OK" in out.stdout
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Données
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
    """Labels de cellule par méthode à l'instant START (10 cellules)."""
    import sys

    sys.path.insert(0, str(ROOT / "postprocessing"))
    from etudes_rapport import Cartesian, Cylindrical, Physics, Voronoi

    fit_ts = np.arange(START, START + 5 * TAU, 10)
    fit_pts = X[fit_ts].reshape(-1, 3)
    fit_vn = V[fit_ts].reshape(-1)

    labels = {}
    cart = Cartesian(5, 2, 1).fit(fit_pts)
    labels["cartesien"] = ("Cartésien", cart.states(X[START]))
    cyl = Cylindrical(2, 5, 1).fit(fit_pts)
    labels["cylindrique"] = ("Cylindrique", cyl.states(X[START]))
    vor = Voronoi(10).fit(fit_pts)
    labels["voronoi"] = ("Voronoï", vor.states(X[START]))
    phy = Physics(10, vw=0.5)
    phy.fit(fit_pts, fit_vn)
    labels["physique"] = ("Physique", phy.states(X[START], V[START]))
    return labels


# ---------------------------------------------------------------------------
# Rendu pyvista (utilisé quand un backend GL est disponible)
# ---------------------------------------------------------------------------


def _render_pyvista(coords, diameters, scalars, cmap, fname, title,
                    clim=None, annotations=None):
    """Screenshot pyvista : sphères aux diamètres réels, scalaire coloré.

    Convention identique à ``Markov.build_vtp`` : ``pv.PolyData(coords)``
    porte les scalaires, et le glyphage par sphères utilise ``Diameter``.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    cloud = pv.PolyData(np.asarray(coords, dtype=float))
    cloud["scalars"] = np.asarray(scalars, dtype=float)
    cloud["Diameter"] = np.asarray(diameters, dtype=float)

    geom = pv.Sphere(theta_resolution=12, phi_resolution=12)
    glyphs = cloud.glyph(scale="Diameter", geom=geom, orient=False)

    p = pv.Plotter(off_screen=True, window_size=[1400, 1000])
    p.add_mesh(glyphs, scalars="scalars", cmap=cmap, clim=clim,
               smooth_shading=True,
               scalar_bar_args=dict(title=title))
    if annotations:
        for text, pos in annotations:
            p.add_point_labels([pos], [text], font_size=18,
                               point_size=1, always_visible=True)
    p.view_xy()
    p.camera.zoom(1.25)
    p.screenshot(str(FIGDIR / fname))
    p.close()


# ---------------------------------------------------------------------------
# Rendu matplotlib 3D (repli logiciel, mêmes vues et scalaires)
# ---------------------------------------------------------------------------


def _render_mpl(coords, diameters, scalars, cmap, fname, title,
                discrete=False, alpha=None, suptitle=None, elev=12,
                azim=-60):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    s = (np.asarray(diameters) * 3600) ** 2 / 16  # aire ∝ diamètre²
    order = np.argsort(coords[:, 1])  # peintre : arrière vers avant
    sc = ax.scatter(coords[order, 0], coords[order, 2], coords[order, 1],
                    c=np.asarray(scalars)[order], cmap=cmap, s=s[order],
                    alpha=alpha if alpha is not None else 0.95,
                    edgecolors="k", linewidths=0.15, depthshade=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08)
    cb.set_label(title)
    if discrete:
        cb.set_ticks(np.unique(scalars))
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$z$ (m)")
    ax.set_zlabel("$y$ (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 1))
    if suptitle:
        ax.set_title(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_melangeur_especes(X, small_p, use_pv):
    """Mélangeur avec espèces distinctes, état initial vs régime établi."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    diam = np.where(small_p, D_SMALL, D_BIG)
    if use_pv:
        for t, tag in ((0, "t0"), (START, "start")):
            _render_pyvista(X[t], diam, small_p.astype(float), "coolwarm",
                            f"pv_melangeur_especes_{tag}.png", "espèce")
        return

    fig = plt.figure(figsize=(13, 6))
    for k, (t, ttl) in enumerate(
        ((0, "(a) État initial ($t = 0$ s) : espèces séparées"),
         (START, "(b) Régime établi ($t = 1{,}57$ s) : lit en écoulement"))
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        s = (diam * 3600) ** 2 / 16
        order = np.argsort(X[t][:, 1])
        colors = np.where(small_p[order], "#1f4e9c", "#c23b3b")
        ax.scatter(X[t][order, 0], X[t][order, 2], X[t][order, 1],
                   c=colors, s=s[order], alpha=0.95,
                   edgecolors="k", linewidths=0.15, depthshade=True)
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$z$ (m)")
        ax.set_zlabel("$y$ (m)")
        ax.view_init(elev=12, azim=-60)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(ttl, fontsize=11)
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", mfc="#1f4e9c", mec="k",
                   ms=7, label="petites (4 mm)"),
        plt.Line2D([0], [0], marker="o", ls="", mfc="#c23b3b", mec="k",
                   ms=11, label="grandes (8 mm)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=11)
    fig.suptitle("Mélangeur : les 1030 particules aux diamètres réels, "
                 "espèces distinguées par la couleur", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGDIR / "pv_melangeur_especes.png", dpi=200)
    plt.close(fig)


def fig_cellules_par_methode(X, small_p, labels, use_pv):
    """Une figure par méthode : particules colorées par cellule (cell_id)."""
    diam = np.where(small_p, D_SMALL, D_BIG)
    for key, (nom, lab) in labels.items():
        fname = f"pv_cellules_{key}.png"
        if use_pv:
            _render_pyvista(X[START], diam, lab, "tab10", fname, "cell_id",
                            clim=[0, 9])
        else:
            _render_mpl(X[START], diam, lab, "tab10", fname, "cell id",
                        discrete=True,
                        suptitle=f"Découpage {nom} : particules colorées "
                                 f"par cellule ($t = 1{{,}}57$ s, 10 cellules)")


def fig_contenu_cellule(X, small_p, labels, use_pv):
    """Zoom sur une cellule du découpage physique : contenu et teneur."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nom, lab = labels["physique"]
    # cellule la plus peuplée
    cell = np.bincount(lab, minlength=10).argmax()
    inside = lab == cell
    ns = int((inside & small_p).sum())
    ntot = int(inside.sum())
    c = ns / ntot

    diam = np.where(small_p, D_SMALL, D_BIG)
    if use_pv:
        scal = np.where(inside, small_p.astype(float) + 1.0, 0.0)
        _render_pyvista(X[START], diam, scal, "viridis",
                        "pv_cellule_contenu.png",
                        f"cellule {cell} (teneur {c:.2f})")
        return

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    s = (diam * 3600) ** 2 / 16
    order = np.argsort(X[START][:, 1])
    xo, so, io_, sp = (X[START][order], s[order], inside[order],
                       small_p[order])
    # reste du lit estompé
    ax.scatter(xo[~io_, 0], xo[~io_, 2], xo[~io_, 1], c="0.8",
               s=so[~io_], alpha=0.25, linewidths=0, depthshade=True)
    # particules de la cellule : petites bleues / grandes rouges
    m_s = io_ & sp
    m_b = io_ & ~sp
    ax.scatter(xo[m_s, 0], xo[m_s, 2], xo[m_s, 1], c="#1f4e9c",
               s=so[m_s], alpha=1.0, edgecolors="k", linewidths=0.2)
    ax.scatter(xo[m_b, 0], xo[m_b, 2], xo[m_b, 1], c="#c23b3b",
               s=so[m_b], alpha=1.0, edgecolors="k", linewidths=0.2)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$z$ (m)")
    ax.set_zlabel("$y$ (m)")
    ax.view_init(elev=12, azim=-60)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(
        f"Contenu de la cellule {cell} (découpage physique, $t = 1{{,}}57$ s) : "
        f"{ns} petites + {ntot - ns} grandes $\\Rightarrow$ teneur "
        f"$c_{{{cell}}} = {ns}/{ntot} = {c:.2f}$",
        fontsize=11,
    )
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", mfc="#1f4e9c", mec="k",
                   ms=7, label="petites de la cellule"),
        plt.Line2D([0], [0], marker="o", ls="", mfc="#c23b3b", mec="k",
                   ms=11, label="grandes de la cellule"),
        plt.Line2D([0], [0], marker="o", ls="", mfc="0.8", mec="0.8",
                   ms=8, label="reste du lit"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "pv_cellule_contenu.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    use_pv = _pyvista_available()
    print("backend pyvista :", "OpenGL disponible" if use_pv
          else "indisponible -> repli matplotlib 3D (mêmes vues)")
    X, V, small_p = load_frames()
    labels = partition_labels(X, V, small_p)
    fig_melangeur_especes(X, small_p, use_pv)
    fig_cellules_par_methode(X, small_p, labels, use_pv)
    fig_contenu_cellule(X, small_p, labels, use_pv)
    print("✅ figures 3D écrites dans", FIGDIR)
