"""Génération des schémas de principe du rapport de stage.

Produit dans ``template-rapport-stage/figures/`` :

* ``schema_tambour.png``            : géométrie du tambour tournant (vue de
  face + vue de côté) avec ses dimensions et la vitesse de rotation ;
* ``schema_forces_tambour.png``     : bilan des actions mécaniques (forces
  normales, tangentielles, pesanteur) sur une particule du lit granulaire ;
* ``schema_decoupage_cartesien.png``: principe du découpage cartésien ;
* ``schema_decoupage_cylindrique.png`` : principe du découpage cylindrique ;
* ``schema_decoupage_voronoi.png``  : principe des découpages de Voronoï et
  physique (cellules k-moyennes autour de centroïdes) ;
* ``schema_parametres_temporels.png`` : chronologie des paramètres du modèle
  (start, tau, step, dt, NLT) sur l'axe des instants DEM ;
* ``schema_tau_tour.png``           : correspondance tau <-> un tour complet
  du tambour à 4 rad/s.

Usage::

    python postprocessing/figures_rapport_schemas.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle, Wedge

FIGDIR = Path(__file__).resolve().parents[1] / "template-rapport-stage" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Vitesse de rotation du tambour (rad/s) et pas d'extraction DEM (s)
OMEGA = 4.0
DT_DEM = 0.01
TAU_S = 2 * np.pi / OMEGA  # ~1.57 s : un tour complet
TAU_STEPS = int(round(TAU_S / DT_DEM))  # ~157 pas DEM

BLEU = "#1f77b4"
VERT = "#2ca08c"
ROUGE = "#d62728"
ORANGE = "#ff7f0e"
GRIS = "#555555"


def _lit_granulaire(ax, R=1.0, angle_deg=35.0, color=VERT, n=400, seed=3):
    """Dessine un lit granulaire incliné (régime rolling) dans le cercle."""
    rng = np.random.default_rng(seed)
    a = np.deg2rad(angle_deg)
    pts = []
    while len(pts) < n:
        x, y = rng.uniform(-R, R, 2)
        if x * x + y * y < (0.97 * R) ** 2 and y < -0.15 * R + np.tan(a) * (-x):
            pts.append((x, y))
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, c=color, alpha=0.75, lw=0)
    return a


def schema_tambour():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.2))

    # ---------------- vue de face ----------------
    R = 1.0
    ax1.add_patch(Circle((0, 0), R, fill=False, ec=BLEU, lw=2.5, zorder=3))
    _lit_granulaire(ax1, R)
    # rayon
    ax1.annotate(
        "", xy=(R * np.cos(np.pi / 4), R * np.sin(np.pi / 4)), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.6),
    )
    ax1.text(0.32, 0.47, r"$R$", fontsize=15, color=GRIS)
    ax1.plot(0, 0, "k+", ms=10)
    # fleche de rotation
    ax1.add_patch(Arc((0, 0), 2.6, 2.6, theta1=110, theta2=170,
                      color=ROUGE, lw=2.2))
    ax1.annotate(
        "", xy=(1.3 * np.cos(np.deg2rad(172)), 1.3 * np.sin(np.deg2rad(172))),
        xytext=(1.3 * np.cos(np.deg2rad(160)), 1.3 * np.sin(np.deg2rad(160))),
        arrowprops=dict(arrowstyle="-|>", color=ROUGE, lw=2.2),
    )
    ax1.text(-1.55, 1.05, r"$\omega = 4\ \mathrm{rad/s}$",
             fontsize=13, color=ROUGE)
    # zones active / passive
    ax1.text(0.05, 0.02, "zone active\n(surface)", fontsize=10,
             ha="center", color="k",
             bbox=dict(fc="white", ec="none", alpha=0.7))
    ax1.text(-0.42, -0.62, "zone passive\n(cœur du lit)", fontsize=10,
             ha="center", color="k",
             bbox=dict(fc="white", ec="none", alpha=0.7))
    ax1.set_title("(a) Vue de face (plan $x$–$y$)")
    ax1.set_xlim(-1.75, 1.75)
    ax1.set_ylim(-1.45, 1.55)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # ---------------- vue de cote ----------------
    L, D = 1.6, 2.0
    ax2.add_patch(Rectangle((-L / 2, -D / 2), L, D, fill=False,
                            ec=BLEU, lw=2.5))
    ax2.plot([-L / 2 - 0.25, L / 2 + 0.25], [0, 0], "-.", color=GRIS, lw=1)
    # cotes
    ax2.annotate("", xy=(L / 2, -D / 2 - 0.22), xytext=(-L / 2, -D / 2 - 0.22),
                 arrowprops=dict(arrowstyle="<|-|>", color=GRIS, lw=1.4))
    ax2.text(0, -D / 2 - 0.42, r"$L$", fontsize=15, ha="center", color=GRIS)
    ax2.annotate("", xy=(L / 2 + 0.28, D / 2), xytext=(L / 2 + 0.28, -D / 2),
                 arrowprops=dict(arrowstyle="<|-|>", color=GRIS, lw=1.4))
    ax2.text(L / 2 + 0.40, 0, r"$D = 2R$", fontsize=15, va="center",
             color=GRIS, rotation=90)
    # axe z
    ax2.annotate("", xy=(L / 2 + 0.85, 0), xytext=(L / 2 + 0.55, 0),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax2.text(L / 2 + 0.9, 0.05, r"$z$ (axe de rotation)", fontsize=11)
    # remplissage
    ax2.add_patch(Rectangle((-L / 2, -D / 2), L, 0.75, color=VERT, alpha=0.5))
    ax2.text(0, -D / 2 + 0.34, "lit de particules", fontsize=10, ha="center")
    ax2.set_title("(b) Vue de côté (plan $x$–$z$)")
    ax2.set_xlim(-1.7, 2.6)
    ax2.set_ylim(-1.8, 1.6)
    ax2.set_aspect("equal")
    ax2.axis("off")

    fig.suptitle(
        "Géométrie du tambour tournant — les limites $(R,\\ L)$ de la partie "
        "utile sont recalées sur l'enveloppe des positions DEM (phase de fit)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_tambour.png", dpi=200)
    plt.close(fig)


def schema_forces():
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    R = 1.0
    ax.add_patch(Circle((0, 0), R, fill=False, ec=BLEU, lw=2.5))
    a = _lit_granulaire(ax, R)

    # particule etudiee sur la surface libre
    px, py = 0.18, 0.02
    ax.add_patch(Circle((px, py), 0.09, fc=ORANGE, ec="k", zorder=5))
    # particule voisine en contact
    qx, qy = 0.33, -0.10
    ax.add_patch(Circle((qx, qy), 0.09, fc="#cccccc", ec="k", zorder=4))

    # normale au contact
    nvec = np.array([px - qx, py - qy])
    nvec = nvec / np.linalg.norm(nvec)
    tvec = np.array([-nvec[1], nvec[0]])

    def fleche(x0, y0, dx, dy, color, label, dlx=0.03, dly=0.03):
        ax.annotate("", xy=(x0 + dx, y0 + dy), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.4),
                    zorder=6)
        ax.text(x0 + dx + dlx, y0 + dy + dly, label, fontsize=14,
                color=color, zorder=6)

    fleche(px, py, 0.42 * nvec[0], 0.42 * nvec[1], ROUGE,
           r"$\mathbf{F}_{n,ij}$")
    fleche(px, py, 0.40 * tvec[0], 0.40 * tvec[1], "#8c1aff",
           r"$\mathbf{F}_{t,ij}$", dlx=-0.30, dly=0.05)
    fleche(px, py, 0.0, -0.5, "k", r"$m_i\,\mathbf{g}$", dlx=0.04, dly=-0.06)
    # moment de rotation de la particule
    ax.add_patch(Arc((px, py), 0.34, 0.34, theta1=250, theta2=170,
                     color=VERT, lw=1.8, zorder=6))
    ax.text(px - 0.33, py + 0.16, r"$\mathbf{M}_{r,ij}$", fontsize=13,
            color=VERT, zorder=6)

    # rotation du tambour
    ax.add_patch(Arc((0, 0), 2.55, 2.55, theta1=115, theta2=165,
                     color=BLEU, lw=2.0))
    ax.annotate(
        "", xy=(1.27 * np.cos(np.deg2rad(167)), 1.27 * np.sin(np.deg2rad(167))),
        xytext=(1.27 * np.cos(np.deg2rad(157)), 1.27 * np.sin(np.deg2rad(157))),
        arrowprops=dict(arrowstyle="-|>", color=BLEU, lw=2.0))
    ax.text(-1.52, 1.02, r"$\omega$", fontsize=15, color=BLEU)

    # frottement paroi
    wx, wy = -R * np.cos(np.deg2rad(40)), -R * np.sin(np.deg2rad(40))
    tw = np.array([np.sin(np.deg2rad(40)), -np.cos(np.deg2rad(40))])
    fleche(wx, wy, 0.34 * tw[0], 0.34 * tw[1], "#b8860b",
           r"$\mathbf{F}_{t,\mathrm{paroi}}$", dlx=-0.15, dly=-0.16)

    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.65, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Actions mécaniques sur une particule du lit : contact normal et\n"
        "tangentiel inter-grains, frottement à la paroi entraînante, pesanteur",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_forces_tambour.png", dpi=200)
    plt.close(fig)


def schema_cartesien():
    """Principe du découpage cartésien : dix bandes parallèles à la
    surface libre du lit. Les limites de bandes (segments parallèles) se
    prolongent au-delà du cercle : côté ciel, les cellules marginales ne
    rencontrent que peu ou pas de grains."""
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    R = 1.0
    a = np.deg2rad(35)
    surf = lambda x: -0.15 * R - np.tan(a) * x   # surface libre du lit

    ax.add_patch(Circle((0, 0), R, fill=False, ec=BLEU, lw=2.5, zorder=3))
    _lit_granulaire(ax, R)

    nbands = 10
    # bornes des bandes : 11 droites parallèles à la surface libre,
    # de la surface libre (légèrement au-dessus) jusqu'à la paroi basse
    cmin, cmax = -0.10 * R, -1.02 * R
    cs = np.linspace(cmin, cmax, nbands + 1)
    xl = np.array([-1.35 * R, 1.25 * R])
    for c in cs:
        ax.plot(xl, c - np.tan(a) * xl, "k-", lw=2.4, zorder=4)
    # numérotation des cellules, au centre de chaque bande (trace à x fixe)
    x_fix = -0.15
    for k in range(nbands):
        cm = 0.5 * (cs[k] + cs[k + 1])
        p = np.array([x_fix, cm - np.tan(a) * x_fix])
        ax.text(p[0], p[1], str(k), fontsize=13, ha="center", va="center",
                color="k",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.12),
                zorder=6)
    # surface libre en rouge
    ax.plot(xl, -0.15 * R - np.tan(a) * xl, "-", color=ROUGE, lw=2.2,
            zorder=5)
    ax.annotate("surface libre du lit", xy=(-0.72, surf(-0.72)),
                xytext=(-1.52, 0.52), fontsize=11.5, color=ROUGE,
                arrowprops=dict(arrowstyle="->", color=ROUGE, lw=1.4))
    ax.annotate("cellules marginales\n(au-dessus de la surface) :\npeu ou "
                "pas de grains", xy=(-1.13, 0.85), xytext=(-1.50, 1.22),
                fontsize=10.5, color=GRIS, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.4))

    ax.set_xlim(-1.55, 1.5)
    ax.set_ylim(-1.35, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Découpage cartésien : dix bandes parallèles à la surface libre\n"
        "(les limites de bandes se prolongent hors de l'enceinte du tambour)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_decoupage_cartesien.png", dpi=200)
    plt.close(fig)


def schema_cylindrique():
    """Principe du découpage cylindrique.

    (a) coupe transverse : dix secteurs angulaires dans le repère du lit
    (configuration du mémoire), limites de cellules en traits forts ;
    (b) vue de côté : découpage axial optionnel en z (non retenu : nz = 1).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))

    # ---- (a) coupe transverse : 10 secteurs angulaires (repère du lit) ---
    R = 1.0
    a = np.deg2rad(35)
    ax1.add_patch(Circle((0, 0), R, fill=False, ec=BLEU, lw=2.5, zorder=3))
    _lit_granulaire(ax1, R)
    # origine du repère du lit : barycentre approximatif du lit
    Ob = np.array([0.18 * R, -0.62 * R])
    ntheta = 10
    # base angulaire légèrement tournée (axes du lit)
    theta0 = np.deg2rad(100)
    thetas = theta0 + np.linspace(0, 2 * np.pi, ntheta + 1)
    clip_cercle = Circle((0, 0), R, fill=False, transform=ax1.transData)
    for th in thetas[:-1]:
        d = np.array([np.cos(th), np.sin(th)])
        ln, = ax1.plot([Ob[0], Ob[0] + 2.2 * R * d[0]],
                       [Ob[1], Ob[1] + 2.2 * R * d[1]], "k-", lw=2.2,
                       zorder=4)
        ln.set_clip_path(clip_cercle)
    # numérotation des secteurs : rayon adapté à la distance au bord
    for k in range(ntheta):
        thm = 0.5 * (thetas[k] + thetas[k + 1])
        d = np.array([np.cos(thm), np.sin(thm)])
        # distance du barycentre au cercle le long de d
        tmax = (-np.dot(Ob, d) + np.sqrt(
            max(np.dot(Ob, d) ** 2 + R * R - np.dot(Ob, Ob), 0)))
        p = Ob + 0.62 * tmax * d
        ax1.text(p[0], p[1], str(k), fontsize=12.5, ha="center",
                 va="center", color="k",
                 bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.12),
                 zorder=5)
    # arc d'un intervalle angulaire
    a1, a2_ = thetas[0], thetas[1]
    ax1.add_patch(Arc(Ob, 0.62, 0.62, angle=np.rad2deg(a1), theta1=0,
                      theta2=np.rad2deg(a2_ - a1), color=ROUGE, lw=2.2))
    amid = a1 + (a2_ - a1) / 2
    ax1.text(Ob[0] + 0.42 * np.cos(amid) - 0.02,
             Ob[1] + 0.42 * np.sin(amid) + 0.10,
             r"$\Delta\theta$", fontsize=13, color=ROUGE)
    # repère : barycentre du lit
    ax1.plot(Ob[0], Ob[1], "k+", ms=14, mew=2.2, zorder=6)
    ax1.annotate("barycentre du lit", xy=Ob, xytext=Ob + [-1.18, -0.30],
                 fontsize=10, color=GRIS, va="center",
                 arrowprops=dict(arrowstyle="->", color=GRIS, lw=1.2))
    ax1.set_title("(a) Coupe transverse : dix secteurs angulaires dans le\n"
                  "repère du lit — limites de cellules en traits forts",
                  fontsize=12)
    ax1.set_xlim(-1.75, 1.75)
    ax1.set_ylim(-1.5, 1.4)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # ---- (b) vue de côté : tranches axiales en z --------------------------
    L, D = 1.8, 2.0
    nz = 4
    cmap = plt.get_cmap("viridis")
    zs = np.linspace(-L / 2, L / 2, nz + 1)
    for i in range(nz):
        ax2.add_patch(Rectangle((zs[i], -D / 2), zs[i + 1] - zs[i], D,
                                fc=cmap(i / nz), ec="k", lw=2.0, alpha=0.45))
    ax2.plot([-L / 2 - 0.25, L / 2 + 0.25], [0, 0], "-.", color=GRIS, lw=1)
    ax2.annotate("", xy=(zs[1], -D / 2 - 0.22), xytext=(zs[0], -D / 2 - 0.22),
                 arrowprops=dict(arrowstyle="<|-|>", color=GRIS, lw=1.4))
    ax2.text(0.5 * (zs[0] + zs[1]), -D / 2 - 0.45, r"$\Delta z$",
             fontsize=13, ha="center", color=GRIS)
    ax2.text(L / 2 + 0.35, 0.05, r"$z$", fontsize=13)
    ax2.annotate("", xy=(L / 2 + 0.3, 0), xytext=(L / 2 + 0.05, 0),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax2.text(0, D / 2 + 0.18, "découpage axial optionnel (non retenu : "
             "$n_z = 1$)", fontsize=10.5, ha="center", color=GRIS)
    ax2.set_title("(b) Vue de côté : tranches axiales en $z$", fontsize=12)
    ax2.set_xlim(-1.6, 1.9)
    ax2.set_ylim(-1.7, 1.4)
    ax2.set_aspect("equal")
    ax2.axis("off")

    fig.suptitle(
        "Découpage cylindrique : cellules $(r,\\ \\theta,\\ z)$ épousant la "
        "géométrie du tambour — numérotation radiale $\\to$ angulaire "
        "$\\to$ axiale",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_decoupage_cylindrique.png", dpi=200)
    plt.close(fig)


def _kmeans_manuel(features, K, rng, n_iter=60):
    """Petit k-means illustratif (initialisation Forgy, graine fixée)."""
    centres = features[rng.choice(len(features), K, replace=False)].copy()
    for _ in range(n_iter):
        d = ((features[:, None, :] - centres[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for kk in range(K):
            if (lab == kk).any():
                centres[kk] = features[lab == kk].mean(0)
    d = ((features[:, None, :] - centres[None]) ** 2).sum(-1)
    return d.argmin(1), centres


def _trace_voronoi(ax, pts, lab, K, R, lw=1.6):
    """Frontières effectives des cellules : champ de labels propagé au
    voisin le plus proche sur une grille, puis lignes de niveau. Le ciel
    (trop loin de toute particule) est masqué."""
    from scipy.spatial import cKDTree

    g = np.linspace(-R, R, 420)
    Xg, Yg = np.meshgrid(g, g)
    mask = (Xg ** 2 + Yg ** 2) < R * R
    dist, idx = cKDTree(pts[:, :2]).query(np.c_[Xg[mask], Yg[mask]])
    Z = np.full(Xg.shape, np.nan)
    Zg = np.zeros(mask.sum())
    Zg[dist < 0.22 * R] = lab[idx[dist < 0.22 * R]]
    Z[mask] = np.where(dist < 0.22 * R, Zg, np.nan)
    ax.contour(Xg, Yg, Z, levels=np.arange(K) + 0.5, colors="k",
               linewidths=lw, zorder=4)


def schema_voronoi():
    """Découpages statistiques : (a) Voronoï (positions) vs (b) physique
    (positions + norme de la vitesse) — cellules et centroïdes distincts."""
    rng = np.random.default_rng(7)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4))

    # nuage de points en forme de lit granulaire
    R = 1.0
    a = np.deg2rad(35)
    pts = []
    while len(pts) < 900:
        x, y = rng.uniform(-R, R, 2)
        if x * x + y * y < (0.97 * R) ** 2 and y < -0.15 * R + np.tan(a) * (-x):
            pts.append((x, y))
    pts = np.array(pts)

    # champ de vitesse synthétique (régime roulant) : rapide en surface
    # libre, quasi nul dans le cœur passif
    surface = pts[:, 1] + np.tan(a) * pts[:, 0]   # > -0.15R côté ciel
    profondeur = np.clip(-0.15 * R - surface, 0, None)  # 0 en surface
    h_echelle = 0.35 * R
    vitesse = 0.9 * np.exp(-profondeur / h_echelle) * (1 + 0.1 * rng.normal(
        size=len(pts)))

    K = 8
    # (a) k-means sur les seules positions
    lab_a, cen_a = _kmeans_manuel(pts[:, :2], K, rng)
    # (b) k-means sur [x, y, v] standardisés, vitesse pondérée 0,5
    feats = np.column_stack([pts[:, 0], pts[:, 1], vitesse])
    mean, std = feats.mean(0), feats.std(0)
    z = (feats - mean) / std
    z[:, 2] *= 0.5
    lab_b, cen_b = _kmeans_manuel(z, K, rng)
    cen_b_xy = cen_b[:, :2] * std[:2] + mean[:2]   # reprojection (x, y)

    cmap = plt.get_cmap("tab10")
    for ax, lab, cen_xy, title in (
            (ax1, lab_a, cen_a[:, :2],
             "(a) Découpage de Voronoï "
             "($\\mathbf{z}_p = [x_p\\ y_p\\ z_p]^T$)"),
            (ax2, lab_b, cen_b_xy,
             "(b) Découpage physique "
             "($\\mathbf{z}_p = [x_p\\ y_p\\ z_p\\ "
             "\\|\\mathbf{v}_p\\|]^T$)")):
        ax.add_patch(Circle((0, 0), R, fill=False, ec=BLEU, lw=2.2))
        ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(l % 10) for l in lab],
                   s=9, alpha=0.55, lw=0)
        _trace_voronoi(ax, pts, lab, K, R)
        ax.scatter(cen_xy[:, 0], cen_xy[:, 1], marker="X", s=160,
                   c="k", zorder=5, label=r"centres $\boldsymbol{\mu}_k$")
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=11)

    fig.suptitle(
        "Découpages statistiques par k-moyennes : chaque particule est "
        "affectée à la cellule dont le centroïde est le plus proche",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_decoupage_voronoi.png", dpi=200)
    plt.close(fig)


def schema_parametres_temporels():
    fig, ax = plt.subplots(figsize=(11.5, 4.6))

    start, tau, step = 157, 157, 157
    nlt = 2
    tmax = start + nlt * (step + tau) + 120

    # axe des temps
    ax.annotate("", xy=(tmax + 60, 0), xytext=(-40, 0),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))
    ax.text(tmax + 65, -0.05, "temps (s)", fontsize=10)

    # regime transitoire
    ax.axvspan(0, start, ymin=0.42, ymax=0.58, color="#dddddd")
    # label centré dans le rectangle (ymin/ymax sont des fractions d'axes)
    y_lo, y_hi = -0.55, 1.5
    y_mid_tr = y_lo + 0.50 * (y_hi - y_lo)
    ax.text(start / 2, y_mid_tr, "régime\ntransitoire", ha="center",
            va="center", fontsize=10, color=GRIS)
    ax.axvline(start, color=ROUGE, lw=2)
    ax.text(start, -0.32, "start = 1,57 s\n(début du régime permanent)",
            ha="center", fontsize=10, color=ROUGE)

    # blocs d'apprentissage
    y = 0.75
    for k in range(nlt):
        b0 = start + k * (step + tau)
        ax.axvspan(b0, b0 + step, ymin=0.62, ymax=0.78,
                   color=VERT, alpha=0.35)
        # label centré dans le rectangle du bloc (ymin/ymax sont des
        # fractions d'axes : on convertit en coordonnées données)
        y_lo, y_hi = -0.55, 1.5
        y_mid = y_lo + 0.70 * (y_hi - y_lo)
        ax.text(b0 + step / 2, y_mid, f"bloc {k + 1}", ha="center",
                va="center", fontsize=11, color="#1e6f5c", weight="bold")
        # une seule paire de transition (t, t+tau) par bloc
        t0 = b0
        ax.annotate("", xy=(t0 + tau, y - 0.13),
                    xytext=(t0, y - 0.13),
                    arrowprops=dict(arrowstyle="-|>", color=BLEU, lw=1.8))
        ax.text(t0 + tau / 2, y - 0.24, "une paire $(t,\\ t+\\tau)$",
                fontsize=9.5, color=BLEU, ha="center")
        ax.text(b0 + step + tau / 2, y - 0.48, r"$\tau = 1{,}57$ s", fontsize=10,
                color=BLEU, ha="center")

    # step entre deux blocs
    b0, b1 = start, start + step + tau
    ax.annotate("", xy=(b1, 1.22), xytext=(b0, 1.22),
                arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=1.6))
    ax.text((b0 + b1) / 2, 1.28, "step + $\\tau$ (décalage entre blocs)",
            ha="center", fontsize=10, color=ORANGE)

    ax.set_xlim(-60, tmax + 200)
    ax.set_ylim(-0.55, 1.5)
    ax.axis("off")
    ax.set_title(
        "Paramètres temporels de l'apprentissage : à partir de start, chaque "
        "bloc fournit une paire de transition $(t,\\ t+\\tau)$ ;\nles NLT "
        "blocs successifs, décalés de step, sont moyennés (chaîne homogène) "
        "ou conservés séparément (chaîne inhomogène)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_parametres_temporels.png", dpi=200)
    plt.close(fig)


def schema_tau_tour():
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    R = 1.0
    ax.add_patch(Circle((0, 0), R, fill=False, ec=BLEU, lw=2.5))
    _lit_granulaire(ax, R)
    ax.plot(0, 0, "k+", ms=10)

    # tour complet
    ax.add_patch(Arc((0, 0), 2.7, 2.7, theta1=-80, theta2=250,
                     color=ROUGE, lw=2.4))
    th = np.deg2rad(-83)
    ax.annotate("", xy=(1.35 * np.cos(th), 1.35 * np.sin(th)),
                xytext=(1.35 * np.cos(th + 0.18), 1.35 * np.sin(th + 0.18)),
                arrowprops=dict(arrowstyle="-|>", color=ROUGE, lw=2.4))
    ax.text(0, 1.52, "un tour complet", fontsize=12, ha="center", color=ROUGE)

    ax.text(
        0, -1.72,
        r"$\tau \;=\; \dfrac{2\pi}{\omega} \;=\; \dfrac{2\pi}{4}"
        r" \;\approx\; 1{,}57\ \mathrm{s}$ (soit 157 pas DEM"
        r" de $\Delta t_{\mathrm{DEM}} = 0{,}01\ \mathrm{s}$)",
        fontsize=13, ha="center",
    )
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-2.05, 1.75)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Pas de temps de Markov : un tour de tambour à "
                 "$\\omega = 4$ rad/s", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_tau_tour.png", dpi=200)
    plt.close(fig)


def schema_construction_markov():
    """Organigramme de la construction du modèle de Markov."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11.5, 5.6))

    def boite(x, y, w, h, titre, corps, fc="#eaf2fb", ec=BLEU):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.06",
                                    fc=fc, ec=ec, lw=1.6))
        ax.text(x + w / 2, y + h - 0.16, titre, ha="center", va="top",
                fontsize=10.5, weight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.14, corps, ha="center", va="center",
                fontsize=8.8)

    def fleche(x0, y0, x1, y1, label="", dy=0.1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.8))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, fontsize=8.5,
                    ha="center", color=GRIS)

    # rangée du haut : de la DEM aux vecteurs d'état
    boite(0.0, 2.6, 2.6, 1.5, "1. Données DEM",
          "positions, vitesses,\ndiamètres\n(6000 instants)")
    boite(3.4, 2.6, 2.9, 1.5, "2. Ajustement du\npartitionneur",
          "entrainement sur le régime\npermanent des\ncoordonnées des \nparticules")
    boite(7.1, 2.6, 2.9, 1.5, "3. Labélisation",
          "label $l_p(t_k)$ de chaque\nparticule à chaque instant")
    boite(10.8, 2.6, 3.0, 1.5, "4. Vecteurs d'état",
          "Calcul du nombre de particules de\n chaque espèces\npar cellule\net par espèce\n$S_i(t_k)$")
    fleche(2.6, 3.35, 3.4, 3.35)
    fleche(6.3, 3.35, 7.1, 3.35)
    fleche(10.0, 3.35, 10.8, 3.35)

    # rangée du bas : matrice puis prédiction puis validation
    boite(10.8, 0.3, 3.0, 1.5, "5. Matrice(s) de\ntransition",
          "une paire $(t,\\ t+\\tau)$ par bloc,\nmoyennée sur NLT blocs\n"
          "ou une matrice par bloc")
    boite(6.4, 0.3, 3.4, 1.5, "6. Prédiction",
          "$\\mathbf{C}_{k+1} = \\mathbf{P}\\,\\mathbf{C}_k$ (homogène)\n"
          "$\\mathbf{C}_{k+1} = \\mathbf{P}^{(k)}\\mathbf{C}_k$ (inhomogène)")
    boite(1.6, 0.3, 3.6, 1.5, "7. Validation vs DEM",
          "teneur locale en petites\nparticules, RSD,\nécart $|$Markov $-$ DEM$|$",
          fc="#fdeeee", ec=ROUGE)
    fleche(12.3, 2.6, 12.3, 1.8, "statistique des transitions entre cellules\n($start$, $\\tau$, $step$, $NLT$)", dy=0.0)
    fleche(10.8, 1.05, 9.8, 1.05)
    fleche(6.4, 1.05, 5.2, 1.05, "comparaison à la\nréférence DEM", dy=0.35)

    ax.set_xlim(-0.3, 14.2)
    ax.set_ylim(-0.1, 4.5)
    ax.axis("off")
    ax.set_title("Algorithme de construction et d'exploitation du modèle de Markov",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_construction_markov.png", dpi=200)
    plt.close(fig)


def schema_teneur_locale():
    """Teneur locale en petites particules dans un découpage cartésien."""
    rng = np.random.default_rng(11)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    nx, ny = 3, 2
    W, H = 3.0, 2.0
    # populations (petites, grandes) par cellule
    pops = [[(6, 2), (3, 5), (1, 7)], [(5, 5), (2, 6), (7, 1)]]
    for j in range(ny):
        for i in range(nx):
            x0, y0 = i * W, j * H
            ax.add_patch(Rectangle((x0, y0), W, H, fill=False, ec="k", lw=1.6))
            ns, nb = pops[j][i]
            # petites (bleues) et grandes (rouges)
            for _ in range(ns):
                ax.add_patch(Circle((x0 + rng.uniform(0.3, W - 0.3),
                                     y0 + rng.uniform(0.3, H - 0.3)),
                                    0.09, fc="#1f4e9c", ec="none", alpha=0.9))
            for _ in range(nb):
                ax.add_patch(Circle((x0 + rng.uniform(0.35, W - 0.35),
                                     y0 + rng.uniform(0.35, H - 0.35)),
                                    0.17, fc="#c23b3b", ec="none", alpha=0.85))
            c = ns / (ns + nb)
            ax.text(x0 + W / 2, y0 + H - 0.28,
                    f"$c_{{{j * nx + i}}} = \\frac{{{ns}}}{{{ns + nb}}} = {c:.2f}$",
                    ha="center", fontsize=11,
                    bbox=dict(fc="white", ec="0.6", alpha=0.9, pad=2.5))
    # légende
    ax.add_patch(Circle((0.35, -0.55), 0.09, fc="#1f4e9c", ec="none"))
    ax.text(0.55, -0.55, "petite particule (4 mm)", va="center", fontsize=10)
    ax.add_patch(Circle((4.1, -0.55), 0.17, fc="#c23b3b", ec="none"))
    ax.text(4.35, -0.55, "grande particule (8 mm)", va="center", fontsize=10)
    ax.text(9.0, -0.55,
            r"$c_i = \dfrac{n_{\mathrm{petites},i}}{n_{\mathrm{petites},i} + n_{\mathrm{grandes},i}}$",
            va="center", ha="right", fontsize=12)
    ax.set_xlim(-0.3, nx * W + 0.3)
    ax.set_ylim(-1.05, ny * H + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Teneur locale en petites particules $c_i$\n"
                 "dans un découpage cartésien", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_teneur_locale.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    # schema_tambour() : désactivé — la version validée dans le dépôt
    # (révisée manuellement) ne doit pas être écrasée.
    schema_construction_markov()
    schema_teneur_locale()
    schema_forces()
    schema_cartesien()
    schema_cylindrique()
    schema_voronoi()
    schema_parametres_temporels()
    schema_tau_tour()
    print(f"✅ Schémas écrits dans {FIGDIR}")