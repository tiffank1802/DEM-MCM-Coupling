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
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    R = 1.0
    ax.add_patch(Circle((0, 0), R, fill=False, ec=BLEU, lw=2.5, zorder=3))
    _lit_granulaire(ax, R)
    # grille cartesienne sur la partie utile (moitie basse)
    nx, ny = 6, 3
    xs = np.linspace(-R, R, nx + 1)
    ys = np.linspace(-R, 0.05, ny + 1)
    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], "k-", lw=1.4, zorder=4)
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], "k-", lw=1.4, zorder=4)
    # numerotation lexicographique x -> y
    k = 0
    for j in range(ny):
        for i in range(nx):
            cx = 0.5 * (xs[i] + xs[i + 1])
            cy = 0.5 * (ys[j] + ys[j + 1])
            if cx * cx + cy * cy < 1.15 * R * R:
                ax.text(cx, cy, str(k), fontsize=14, ha="center",
                        va="center", color="k",
                        bbox=dict(fc="white", ec="none", alpha=0.65,
                                  pad=0.15), zorder=5)
            k += 1
    ax.text(0, 0.55, "ciel (peu de particules)", fontsize=13,
            ha="center", color=BLEU)
    ax.annotate(r"pas $\Delta x = \frac{x_{max}-x_{min}}{n_x}$",
                xy=(xs[1], 0.05), xytext=(-1.6, 0.85),
                fontsize=13, arrowprops=dict(arrowstyle="->", color=GRIS))
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.45, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Découpage cartésien : cellules parallélépipédiques de volume "
        "constant,\nnumérotation lexicographique $x \\to y \\to z$",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "schema_decoupage_cartesien.png", dpi=200)
    plt.close(fig)


def schema_cylindrique():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4))

    # -------- coupe transverse : secteurs r, theta --------
    R = 1.0
    nr, ntheta = 3, 8
    radii = np.linspace(0, R, nr + 1)
    thetas = np.linspace(0, 360, ntheta + 1)
    cmap = plt.get_cmap("viridis")
    k = 0
    for i in range(nr):
        for j in range(ntheta):
            ax1.add_patch(Wedge((0, 0), radii[i + 1], thetas[j],
                                thetas[j + 1], width=radii[i + 1] - radii[i],
                                fc=cmap((k % 24) / 24), ec="k",
                                lw=2.4, alpha=0.35))
            k += 1
    # Nuage de points : les particules restent visibles et les frontières
    # géométriques sont superposées en trait fort.
    _lit_granulaire(ax1, R, n=420, seed=12)
    for rr in radii[1:-1]:
        ax1.add_patch(Circle((0, 0), rr, fill=False, ec="black", lw=2.4,
                             zorder=4))
    for th in np.deg2rad(thetas):
        ax1.plot([0, R*np.cos(th)], [0, R*np.sin(th)], color="black",
                 lw=2.0, zorder=4)
    # rayons limites
    ax1.annotate("", xy=(radii[1] * np.cos(np.pi / 3),
                         radii[1] * np.sin(np.pi / 3)), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))
    ax1.text(0.06, 0.26, r"$r_1$", fontsize=13)
    ax1.annotate("", xy=(R * np.cos(np.pi / 8), R * np.sin(np.pi / 8)),
                 xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.6))
    ax1.text(0.62, 0.14, r"$r_{max}$", fontsize=13)
    ax1.add_patch(Arc((0, 0), 1.15, 1.15, theta1=0, theta2=45,
                      color=ROUGE, lw=2))
    ax1.text(0.62, 0.32, r"$\Delta\theta$", fontsize=13, color=ROUGE)
    ax1.set_title("(a) Coupe transverse : intervalles en $r$ et $\\theta$")
    ax1.set_xlim(-1.4, 1.6)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # -------- vue de cote : tranches en z --------
    L, D = 1.8, 2.0
    rng = np.random.default_rng(13)
    zpts = rng.uniform(-L / 2, L / 2, 420)
    ypts = rng.uniform(-D / 2, 0.15, 420)
    ax2.scatter(zpts, ypts, s=10, c=VERT, alpha=0.75, edgecolors="none",
                zorder=3)
    nz = 4
    zs = np.linspace(-L / 2, L / 2, nz + 1)
    for i in range(nz):
        ax2.add_patch(Rectangle((zs[i], -D / 2), zs[i + 1] - zs[i], D,
                                fc=cmap(i / nz), ec="k", lw=2.4, alpha=0.3))
    ax2.plot([-L / 2 - 0.25, L / 2 + 0.25], [0, 0], "-.", color=GRIS, lw=1)
    ax2.annotate("", xy=(zs[1], -D / 2 - 0.22), xytext=(zs[0], -D / 2 - 0.22),
                 arrowprops=dict(arrowstyle="<|-|>", color=GRIS, lw=1.4))
    ax2.text(0.5 * (zs[0] + zs[1]), -D / 2 - 0.45, r"$\Delta z$",
             fontsize=13, ha="center", color=GRIS)
    ax2.text(L / 2 + 0.35, 0.05, r"$z$", fontsize=13)
    ax2.annotate("", xy=(L / 2 + 0.3, 0), xytext=(L / 2 + 0.05, 0),
                 arrowprops=dict(arrowstyle="-|>", color="k", lw=1.4))
    ax2.set_title("(b) Vue de côté : tranches axiales en $z$")
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


def schema_voronoi():
    rng = np.random.default_rng(7)
    rng_physique=np.random.default_rng(8)
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

    # k-means "manuel" simple pour l'illustration
    K = 8
    centres = pts[rng.choice(len(pts), K, replace=False)]
    centres_physique = pts[rng_physique.choice(len(pts), K, replace=False)]
    for _ in range(30):
        d = ((pts[:, None, :] - centres[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        d_physique = ((pts[:, None, :] - centres_physique[None]) ** 2).sum(-1)
        lab_physique = d_physique.argmin(1)
        for kk in range(K):
            if (lab == kk).any():
                centres[kk] = pts[lab == kk].mean(0)
            if (lab_physique == kk).any():
                centres_physique[kk] = pts[lab_physique == kk].mean(0)


    cmap = plt.get_cmap("tab10")
    for ax, title in ((ax1, "(a) Découpage de Voronoï "
                            "($\\mathbf{z}_p = [x_p\\ y_p\\ z_p]^T$)"),
                      (ax2, "(b) Découpage physique "
                            "($\\mathbf{z}_p = [x_p\\ y_p\\ z_p\\ "
                            "\\|\\mathbf{v}_p\\|]^T$)")):
        ax.add_patch(Circle((0, 0), R, fill=False, ec=BLEU, lw=2.2))
        if ax==ax1:
            ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(l % 10) for l in lab],
                    s=12, alpha=0.8, lw=0)
            ax.scatter(centres[:, 0], centres[:, 1], marker="X", s=160,
                   c="k", zorder=5, label=r"centres $\boldsymbol{\mu}_k$")
        if ax==ax2:
            ax.scatter(pts[:, 0], pts[:, 1], c=[cmap(l % 10) for l in lab_physique],
                    s=12, alpha=0.8, lw=0)
            ax.scatter(centres_physique[:, 0], centres_physique[:, 1], marker="X", s=160,
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
    dt = 31  # espacement illustratif des paires dans un bloc
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
        # paires (t, t+tau) echantillonnees tous les dt
        for i, t0 in enumerate(range(b0, b0 + step - 1, dt)):
            ax.annotate("", xy=(t0 + tau, y - 0.13 - 0.05 * i),
                        xytext=(t0, y - 0.13 - 0.05 * i),
                        arrowprops=dict(arrowstyle="-|>", color=BLEU,
                                        lw=1.2, alpha=0.8))
        ax.text(b0 + step + tau / 2, y - 0.48, r"$\tau = 1{,}57$ s", fontsize=10,
                color=BLEU, ha="center")

    # step entre deux blocs
    b0, b1 = start, start + step + tau
    ax.annotate("", xy=(b1, 1.22), xytext=(b0, 1.22),
                arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=1.6))
    ax.text((b0 + b1) / 2, 1.28, "step + $\\tau$ (décalage entre blocs)",
            ha="center", fontsize=10, color=ORANGE)
    # dt dans un bloc
    ax.annotate("", xy=(start + dt, 0.5), xytext=(start, 0.5),
                arrowprops=dict(arrowstyle="<|-|>", color="#8c1aff", lw=1.5))
    ax.text(start + dt / 2, 0.55, "dt", ha="center", fontsize=10,
            color="#8c1aff")

    ax.set_xlim(-60, tmax + 200)
    ax.set_ylim(-0.55, 1.5)
    ax.axis("off")
    ax.set_title(
        "Paramètres temporels de l'apprentissage : à partir de start, chaque "
        "bloc fournit des paires $(t,\\ t+\\tau)$ échantillonnées tous les "
        "dt ;\nles NLT blocs successifs, décalés de step, sont moyennés "
        "(chaîne homogène) ou conservés séparément (chaîne inhomogène)",
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
          "fit sur le régime\npermanent\n(limites, centres)")
    boite(7.1, 2.6, 2.9, 1.5, "3. Labélisation",
          "label $l_p(t_k)$ de chaque\nparticule à chaque instant")
    boite(10.8, 2.6, 3.0, 1.5, "4. Vecteurs d'état",
          "comptage par cellule\net par espèce\n$S_i(t_k)$")
    fleche(2.6, 3.35, 3.4, 3.35)
    fleche(6.3, 3.35, 7.1, 3.35)
    fleche(10.0, 3.35, 10.8, 3.35)

    # rangée du bas : matrice puis prédiction puis validation
    boite(10.8, 0.3, 3.0, 1.5, "5. Matrice(s) de\ntransition",
          "paires $(t,\\ t+\\tau)$,\nmoyenne sur NLT blocs\nou une matrice par bloc")
    boite(6.4, 0.3, 3.4, 1.5, "6. Prédiction",
          "$\\mathbf{S}_{k+1} = \\mathbf{P}\\,\\mathbf{S}_k$ (homogène)\n"
          "$\\mathbf{S}_{k+1} = \\mathbf{P}^{(k)}\\mathbf{S}_k$ (inhomogène)")
    boite(1.6, 0.3, 3.6, 1.5, "7. Validation vs DEM",
          "teneur locale en petites\nparticules, RSD,\nécart $|$Markov $-$ DEM$|$",
          fc="#fdeeee", ec=ROUGE)
    fleche(12.3, 2.6, 12.3, 1.8, "apprentissage\n($start$, $\\tau$, $step$, $dt$, $NLT$)", dy=0.0)
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