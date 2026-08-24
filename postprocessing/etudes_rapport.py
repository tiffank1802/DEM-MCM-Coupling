"""Études quantitatives du rapport à partir des données DEM réelles.

Charge ``data/compact.npz`` (produit par ``extract_compact.py``) et génère
dans ``template-rapport-stage/figures/`` les figures des annexes et des
résultats :

* ``etude_start_rsd.png``       : RSD DEM de la teneur en petites particules
  au début de simulation, pour les 4 découpages, avec la ligne start=157 ;
* ``etude_tau_rsd.png``         : influence de tau sur le RSD prédit ;
* ``etude_nlt_erreur.png``      : influence de NLT sur l'erreur de prédiction ;
* ``etude_step_erreur.png``     : influence de step sur l'erreur de prédiction ;
* ``matrice_cylindrique_reelle_*.png`` : matrices de transition cylindriques
  par espèce ;
* ``rsd_cylindrique.png``       : RSD DEM vs Markov (homogène / inhomogène)
  pour le découpage cylindrique ;
* ``comparaison_methodes_rsd.png`` : RSD DEM vs Markov pour les 4 découpages.

Conventions identiques au package ``dem_mcm_coupling`` : matrice colonne-
stochastique (éq. du rapport), tau = step = start = 157, apprentissage sur le
régime permanent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "template-rapport-stage" / "figures"
FIGDIR.mkdir(exist_ok=True)

TAU = 157
START = 157
N_T = 6000
N_P = 1030

# ---------------------------------------------------------------------------
# Chargement compact -> tableaux (n_t, n_p)
# ---------------------------------------------------------------------------


def load_arrays():
    d = np.load(ROOT / "data" / "compact.npz")
    t = d["t"].astype(np.int32)
    pid = d["pid"].astype(np.int32)
    xyz = d["xyz"]
    vnorm = d["vnorm"]
    small = d["small"]

    # réindexer en (timestep, particule)
    pid_u = np.unique(pid)
    pid_map = np.zeros(pid_u.max() + 1, dtype=np.int32)
    pid_map[pid_u] = np.arange(len(pid_u))
    row = t
    col = pid_map[pid]

    X = np.zeros((N_T, N_P, 3), dtype=np.float32)
    V = np.zeros((N_T, N_P), dtype=np.float32)
    X[row, col] = xyz
    V[row, col] = vnorm
    small_p = np.zeros(N_P, dtype=bool)
    small_p[col] = small
    return X, V, small_p


# ---------------------------------------------------------------------------
# Partitionneurs (implémentations compactes, mêmes conventions que le package)
# ---------------------------------------------------------------------------


class Cartesian:
    name = "Cartésien"

    def __init__(self, nx=3, ny=3, nz=1):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.n_cells = nx * ny * nz

    def fit(self, pts):
        self.mins = pts.min(0)
        self.maxs = pts.max(0) + 1e-9
        return self

    def states(self, pts):
        rel = (pts.astype(np.float64) - self.mins) / (self.maxs - self.mins)
        ix = np.clip((rel[:, 0] * self.nx).astype(np.int32), 0, self.nx - 1)
        iy = np.clip((rel[:, 1] * self.ny).astype(np.int32), 0, self.ny - 1)
        iz = np.clip((rel[:, 2] * self.nz).astype(np.int32), 0, self.nz - 1)
        return ix + iy * self.nx + iz * self.nx * self.ny


class Cylindrical:
    name = "Cylindrique"

    def __init__(self, nr=3, ntheta=4, nz=1):
        self.nr, self.ntheta, self.nz = nr, ntheta, nz
        self.n_cells = nr * ntheta * nz

    def fit(self, pts):
        self.cx = pts[:, 0].mean()
        self.cy = pts[:, 1].mean()
        r = np.hypot(pts[:, 0] - self.cx, pts[:, 1] - self.cy)
        self.rmax = r.max() + 1e-9
        # mode aire constante : r_k = rmax * sqrt(k / nr)
        self.r_edges = self.rmax * np.sqrt(np.arange(self.nr + 1) / self.nr)
        self.zmin = pts[:, 2].min()
        self.zmax = pts[:, 2].max() + 1e-9
        return self

    def states(self, pts):
        x = pts[:, 0] - self.cx
        y = pts[:, 1] - self.cy
        r = np.hypot(x, y)
        th = np.arctan2(y, x) + np.pi  # [0, 2pi)
        ir = np.clip(
            np.searchsorted(self.r_edges, r, side="right") - 1, 0, self.nr - 1
        )
        it = np.clip(
            (th / (2 * np.pi + 1e-12) * self.ntheta).astype(np.int32),
            0,
            self.ntheta - 1,
        )
        iz = np.clip(
            ((pts[:, 2] - self.zmin) / (self.zmax - self.zmin) * self.nz).astype(
                np.int32
            ),
            0,
            self.nz - 1,
        )
        return ir + it * self.nr + iz * self.nr * self.ntheta


class Voronoi:
    name = "Voronoï"

    def __init__(self, k=10, seed=42):
        self.k = k
        self.seed = seed
        self.n_cells = k

    def fit(self, pts):
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed)
        sub = pts[:: max(1, len(pts) // 200_000)]
        km.fit(sub)
        self.centres = km.cluster_centers_.astype(np.float32)
        return self

    def states(self, pts):
        d = ((pts[:, None, :] - self.centres[None]) ** 2).sum(-1)
        return d.argmin(1).astype(np.int32)


class Physics:
    name = "Physique"

    def __init__(self, k=10, vw=0.5, seed=42):
        self.k = k
        self.vw = vw
        self.seed = seed
        self.n_cells = k

    def fit(self, pts, vn):
        from sklearn.cluster import KMeans

        z = self._feat(pts, vn, fit=True)
        km = KMeans(n_clusters=self.k, n_init=10, random_state=self.seed)
        sub = z[:: max(1, len(z) // 200_000)]
        km.fit(sub)
        self.centres = km.cluster_centers_.astype(np.float32)
        return self

    def _feat(self, pts, vn, fit=False):
        if fit:
            self.p_mean, self.p_std = pts.mean(0), pts.std(0) + 1e-9
            self.v_mean, self.v_std = vn.mean(), vn.std() + 1e-9
        zp = (pts - self.p_mean) / self.p_std
        zv = ((vn - self.v_mean) / self.v_std * self.vw)[:, None]
        return np.hstack([zp, zv]).astype(np.float32)

    def states(self, pts, vn):
        z = self._feat(pts, vn)
        d = ((z[:, None, :] - self.centres[None]) ** 2).sum(-1)
        return d.argmin(1).astype(np.int32)


# ---------------------------------------------------------------------------
# Noyau markovien
# ---------------------------------------------------------------------------


def transition_matrix(s_prev, s_curr, n):
    """Matrice colonne-stochastique P[i, j] = P(j -> i) sur une liste de paires."""
    P = np.zeros((n, n))
    np.add.at(P, (s_curr, s_prev), 1.0)
    denom = P.sum(0)
    denom[denom == 0] = 1.0
    return P / denom


def learn_P(states_t, start, tau, step, nlt, dt, n):
    """Moyenne des matrices apprises sur nlt blocs (chaîne homogène)."""
    mats = []
    for k in range(nlt):
        b0 = start + k * (step + tau)
        if b0 + tau >= N_T:
            break
        s_prev, s_curr = [], []
        for t0 in range(b0, min(b0 + step, N_T - tau), dt):
            s_prev.append(states_t[t0])
            s_curr.append(states_t[t0 + tau])
        if s_prev:
            mats.append(
                transition_matrix(
                    np.concatenate(s_prev), np.concatenate(s_curr), n
                )
            )
    return np.mean(mats, axis=0), len(mats)


def learn_P_blocks(states_t, start, tau, step, nlt, dt, n):
    """Une matrice par bloc (chaîne inhomogène)."""
    out = []
    for k in range(nlt):
        b0 = start + k * (step + tau)
        if b0 + tau >= N_T:
            break
        s_prev, s_curr = [], []
        for t0 in range(b0, min(b0 + step, N_T - tau), dt):
            s_prev.append(states_t[t0])
            s_curr.append(states_t[t0 + tau])
        if s_prev:
            out.append(
                transition_matrix(
                    np.concatenate(s_prev), np.concatenate(s_curr), n
                )
            )
    return out


def counts(states, n, mask=None):
    if mask is not None:
        states = states[mask]
    return np.bincount(states, minlength=n).astype(float)


def rsd_conc(S_small, S_all):
    """RSD de la teneur en petites particules sur les cellules actives."""
    active = S_all > 0
    c = np.where(active, S_small / np.maximum(S_all, 1e-12), np.nan)
    cbar = np.nansum(S_small) / max(np.nansum(S_all), 1e-12)
    sd = np.sqrt(np.nanmean((c - cbar) ** 2))
    return sd / max(cbar, 1e-12)


def dem_rsd_series(states_t, small_p, n, times):
    out = np.empty(len(times))
    for i, t in enumerate(times):
        s = states_t[t]
        out[i] = rsd_conc(counts(s, n, small_p), counts(s, n))
    return out


def markov_rsd_series(P_small, P_all, S0_small, S0_all, n_steps):
    rs, Ss, Sa = [], S0_small.copy(), S0_all.copy()
    rs.append(rsd_conc(Ss, Sa))
    for _ in range(n_steps):
        Ss = P_small @ Ss
        Sa = P_all @ Sa
        rs.append(rsd_conc(Ss, Sa))
    return np.array(rs)


def markov_rsd_series_inh(Ps_small, Ps_all, S0_small, S0_all, n_steps):
    rs, Ss, Sa = [], S0_small.copy(), S0_all.copy()
    rs.append(rsd_conc(Ss, Sa))
    for k in range(n_steps):
        Pk_s = Ps_small[min(k, len(Ps_small) - 1)]
        Pk_a = Ps_all[min(k, len(Ps_all) - 1)]
        Ss = Pk_s @ Ss
        Sa = Pk_a @ Sa
        rs.append(rsd_conc(Ss, Sa))
    return np.array(rs)


# ---------------------------------------------------------------------------
# Préparation des labels par méthode
# ---------------------------------------------------------------------------


def compute_all_states(X, V, small_p):
    """Ajuste les 4 partitionneurs sur le régime permanent, labellise tout.

    Le nombre de cellules est identique pour les quatre méthodes (10 cellules,
    soit ~100 particules par cellule) afin que la comparaison ne soit pas
    biaisée par la résolution du maillage.
    """
    fit_ts = np.arange(START, START + 5 * TAU, 10)  # régime permanent
    fit_pts = X[fit_ts].reshape(-1, 3)
    fit_vn = V[fit_ts].reshape(-1)

    parts = {}

    cart = Cartesian(5, 2, 1).fit(fit_pts)  # 10 cellules
    parts["Cartésien"] = (cart, None)

    cyl = Cylindrical(2, 5, 1).fit(fit_pts)  # 10 cellules
    parts["Cylindrique"] = (cyl, None)

    vor = Voronoi(10).fit(fit_pts)  # 10 cellules
    parts["Voronoï"] = (vor, None)

    phy = Physics(10, vw=0.5)  # 10 cellules
    phy.fit(fit_pts, fit_vn)
    parts["Physique"] = (phy, "vel")

    states = {}
    for name, (p, needs_v) in parts.items():
        print("labels", name)
        st = np.empty((N_T, N_P), dtype=np.int32)
        for t in range(N_T):
            if needs_v:
                st[t] = p.states(X[t], V[t])
            else:
                st[t] = p.states(X[t])
        states[name] = (st, p.n_cells)
    return states


# ---------------------------------------------------------------------------
# Études
# ---------------------------------------------------------------------------

COLORS = {
    "Cartésien": "#1f77b4",
    "Cylindrique": "#d62728",
    "Voronoï": "#2ca02c",
    "Physique": "#ff7f0e",
}


def etude_start(states, small_p):
    """Justification du start : RSD Markov vs DEM, prédiction depuis t = 0 s.

    Pour chaque méthode de découpage, deux chaînes homogènes sont comparées :
    l'une apprise en incluant le régime transitoire (start = 0 s), l'autre
    apprise sur le régime permanent (start = 1,57 s). Les deux prédictions
    sont propagées depuis l'instant initial t = 0 s, ce qui rend visible le
    biais introduit par un apprentissage sur le régime transitoire.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    n_steps = (N_T - 1) // TAU
    t_probe = (np.arange(n_steps + 1) * TAU).clip(max=N_T - 1)
    for ax, (name, (st, n)) in zip(axes.flat, states.items()):
        st_small = st[:, small_p]
        # apprentissage incluant le régime transitoire (start = 0)
        P_all_tr, _ = learn_P(st, 0, TAU, TAU, 2, 8, n)
        P_small_tr, _ = learn_P(st_small, 0, TAU, TAU, 2, 8, n)
        # apprentissage sur le régime permanent (start = 157)
        P_all_pm, _ = learn_P(st, START, TAU, TAU, 2, 8, n)
        P_small_pm, _ = learn_P(st_small, START, TAU, TAU, 2, 8, n)
        # prédiction propagée depuis l'instant initial t = 0 s
        S0_all = counts(st[0], n)
        S0_small = counts(st[0], n, small_p)
        r_tr = markov_rsd_series(P_small_tr, P_all_tr, S0_small, S0_all, n_steps)
        r_pm = markov_rsd_series(P_small_pm, P_all_pm, S0_small, S0_all, n_steps)
        t_mk = np.arange(n_steps + 1) * TAU / 100  # secondes
        t_dem = np.arange(0, N_T, 20)
        r_dem = dem_rsd_series(st, small_p, n, t_dem)
        r_dem_probe = dem_rsd_series(st, small_p, n, t_probe)
        e_tr = np.mean(np.abs(r_tr - r_dem_probe))
        e_pm = np.mean(np.abs(r_pm - r_dem_probe))
        ax.plot(t_dem / 100, r_dem, "k-", lw=0.9, alpha=0.7, label="DEM")
        ax.plot(t_mk, r_tr, "s--", color="0.55", ms=4,
                label=f"Markov, start = 0 s (écart {e_tr:.3f})")
        ax.plot(t_mk, r_pm, "o--", color=COLORS[name], ms=4,
                label=f"Markov, start = 1,57 s (écart {e_pm:.3f})")
        ax.axvspan(0, START / 100, color="0.85", alpha=0.8, zorder=0)
        ax.axvline(START / 100, color="k", ls="--", lw=1.2)
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
        ax.text(START / 200, ax.get_ylim()[1] * 0.55, "régime\ntransitoire",
                ha="center", fontsize=8, color="0.35", rotation=90)
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

    # tableau des valeurs du RSD DEM (temps en secondes)
    rows = []
    probe = [0, 50, 100, 157, 300, 600, 1000]
    for name, (st, n) in states.items():
        r = dem_rsd_series(st, small_p, n, probe)
        rows.append((name, r))
    with open(FIGDIR / "etude_start_table.txt", "w") as f:
        f.write("t(s)    " + "  ".join(f"{t / 100:>6.2f}" for t in probe) + "\n")
        for name, r in rows:
            f.write(f"{name:<12}" + "  ".join(f"{x:6.3f}" for x in r) + "\n")
    print("start ok")


def etude_tau(states, small_p):
    """Influence de tau sur le RSD prédit (découpage physique)."""
    st, n = states["Physique"]
    times = np.arange(START, N_T, 20)
    r_dem = dem_rsd_series(st, small_p, n, times)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(times / 100, r_dem, "k.-", ms=3, lw=1, label="RSD DEM (réel)")
    cmap = plt.get_cmap("viridis")
    taus = [10, 25, 50, 100, 157, 300, 500, 1000]
    for i, tau in enumerate(taus):
        P_s, _ = learn_P(st, START, tau, tau, 2, max(1, tau // 10), n)
        # matrice petites / toutes especes
        st_small = st[:, small_p]
        P_small, _ = learn_P(st_small, START, tau, tau, 2, max(1, tau // 10), n)
        n_steps = (N_T - START) // tau
        S0_all = counts(st[START], n)
        S0_small = counts(st[START], n, small_p)
        r_mk = markov_rsd_series(P_small, P_s, S0_small, S0_all, n_steps)
        t_mk = (START + np.arange(n_steps + 1) * tau) / 100
        lw = 2.6 if tau == TAU else 1.4
        ax.plot(t_mk, r_mk, color=cmap(i / len(taus)), lw=lw,
                label=f"Markov $\\tau$={tau / 100:g} s"
                + (" (retenu)" if tau == TAU else ""))
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("RSD (–)")
    ax.set_title(
        "Influence du pas de temps de Markov $\\tau$ sur la cinétique prédite "
        "(découpage physique, 10 cellules, teneur en petites)"
    )
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_tau_rsd.png", dpi=200)
    plt.close(fig)
    print("tau ok")


def _erreur_pred(st, small_p, n, nlt, step, dt=8):
    """Erreur moyenne |RSD_markov - RSD_dem| sur la fenêtre de prédiction."""
    st_small = st[:, small_p]
    P_all, nb = learn_P(st, START, TAU, step, nlt, dt, n)
    P_small, _ = learn_P(st_small, START, TAU, step, nlt, dt, n)
    n_steps = (N_T - START) // TAU
    S0_all = counts(st[START], n)
    S0_small = counts(st[START], n, small_p)
    r_mk = markov_rsd_series(P_small, P_all, S0_small, S0_all, n_steps)
    times = START + np.arange(n_steps + 1) * TAU
    times = times[times < N_T]
    r_dem = dem_rsd_series(st, small_p, n, times)
    m = min(len(r_mk), len(r_dem))
    return np.mean(np.abs(r_mk[:m] - r_dem[:m])), nb


def etude_nlt(states, small_p):
    st, n = states["Physique"]
    nlts = [1, 2, 3, 5, 8, 12, 18]
    errs = []
    for nlt in nlts:
        e, nb = _erreur_pred(st, small_p, n, nlt, TAU)
        errs.append(e)
        print(f"  NLT={nlt} ({nb} blocs) -> {e:.2f}")
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.plot(nlts, errs, "o-", color="#d62728", lw=2)
    ax.set_xlabel("Nombre de blocs d'apprentissage $NLT$")
    ax.set_ylabel(r"Erreur moyenne $|\mathrm{RSD}_{Markov} - \mathrm{RSD}_{DEM}|$ (–)")
    ax.set_title(
        "Influence de $NLT$ sur la qualité du modèle "
        "(découpage physique, 10 cellules, $\\tau = step = 1{,}57$ s)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_nlt_erreur.png", dpi=200)
    plt.close(fig)
    print("nlt ok")


def etude_step(states, small_p):
    st, n = states["Physique"]
    steps = [40, 80, 157, 314, 471]
    errs = []
    for s in steps:
        e, nb = _erreur_pred(st, small_p, n, 3, s)
        errs.append(e)
        print(f"  step={s} ({nb} blocs) -> {e:.2f}")
    steps_s = [s / 100 for s in steps]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    ax.plot(steps_s, errs, "s-", color="#1f77b4", lw=2)
    ax.axvline(1.57, color="k", ls="--", lw=1.5)
    ax.text(1.65, max(errs) * 0.97, "step = $\\tau$ = 1,57 s", fontsize=10)
    ax.set_xlabel("Écart entre blocs $step$ (s)")
    ax.set_ylabel(r"Erreur moyenne $|\mathrm{RSD}_{Markov} - \mathrm{RSD}_{DEM}|$ (–)")
    ax.set_title(
        "Influence de $step$ sur la qualité du modèle "
        "(découpage physique, 10 cellules, $NLT = 3$, $\\tau = 1{,}57$ s)"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGDIR / "etude_step_erreur.png", dpi=200)
    plt.close(fig)
    print("step ok")


def resultats_cylindrique(states, small_p):
    st, n = states["Cylindrique"]
    st_small = st[:, small_p]
    st_big = st[:, ~small_p]

    # matrices par espece
    P_small, _ = learn_P(st_small, START, TAU, TAU, 2, 8, n)
    P_big, _ = learn_P(st_big, START, TAU, TAU, 2, 8, n)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, P, ttl in ((axes[0], P_big, "Grandes particules (8 mm)"),
                       (axes[1], P_small, "Petites particules (4 mm)")):
        im = ax.imshow(P, cmap="viridis", vmin=0, vmax=max(P_big.max(), P_small.max()))
        ax.set_xlabel("Cellule source $j$")
        ax.set_ylabel("Cellule d'arrivée $i$")
        ax.set_title(ttl)
        fig.colorbar(im, ax=ax, label="$P_{i,j}$")
    fig.suptitle("Matrices de transition — découpage cylindrique (10 cellules, $\\tau = 1{,}57$ s)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "matrice_cylindrique_especes.png", dpi=200)
    plt.close(fig)

    # RSD DEM vs Markov homogene vs inhomogene
    P_all, _ = learn_P(st, START, TAU, TAU, 2, 8, n)
    n_steps = (N_T - START) // TAU
    S0_all = counts(st[START], n)
    S0_small = counts(st[START], n, small_p)
    r_h = markov_rsd_series(P_small, P_all, S0_small, S0_all, n_steps)

    nlt_max = (N_T - START) // (2 * TAU)
    Ps_small = learn_P_blocks(st_small, START, TAU, TAU, nlt_max, 8, n)
    Ps_all = learn_P_blocks(st, START, TAU, TAU, nlt_max, 8, n)
    r_i = markov_rsd_series_inh(Ps_small, Ps_all, S0_small, S0_all, n_steps)

    times_mk = (START + np.arange(n_steps + 1) * TAU)
    times_dem = np.arange(START, N_T, 20)
    r_dem = dem_rsd_series(st, small_p, n, times_dem)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.plot(times_dem / 100, r_dem, "k-", lw=1, alpha=0.7, label="DEM")
    ax.plot(times_mk / 100, r_h, "o--", color="#1f77b4", ms=4,
            label="Markov homogène")
    ax.plot(times_mk / 100, r_i, "s--", color="#d62728", ms=4,
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


def comparaison_methodes(states, small_p):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    err_summary = {}
    for ax, (name, (st, n)) in zip(axes.flat, states.items()):
        st_small = st[:, small_p]
        P_all, _ = learn_P(st, START, TAU, TAU, 2, 8, n)
        P_small, _ = learn_P(st_small, START, TAU, TAU, 2, 8, n)
        n_steps = (N_T - START) // TAU
        S0_all = counts(st[START], n)
        S0_small = counts(st[START], n, small_p)
        r_mk = markov_rsd_series(P_small, P_all, S0_small, S0_all, n_steps)
        t_mk = (START + np.arange(n_steps + 1) * TAU) / 100
        t_dem = np.arange(START, N_T, 20)
        r_dem = dem_rsd_series(st, small_p, n, t_dem)
        # erreur aux memes instants
        r_dem_mk = dem_rsd_series(st, small_p, n,
                                  (START + np.arange(n_steps + 1) * TAU)
                                  .clip(max=N_T - 1))
        err = np.mean(np.abs(r_mk - r_dem_mk))
        err_summary[name] = err
        ax.plot(t_dem / 100, r_dem, "k-", lw=0.9, alpha=0.7, label="DEM")
        ax.plot(t_mk, r_mk, "o--", color=COLORS[name], ms=4,
                label="Markov homogène")
        ax.set_title(f"{name} — écart moyen {err:.3f}")
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
        for k, v in err_summary.items():
            f.write(f"{k:<12} {v:8.4f}\n")
    print("comparaison ok", err_summary)


if __name__ == "__main__":
    print("chargement…")
    X, V, small_p = load_arrays()
    print("petites:", small_p.sum(), "grandes:", (~small_p).sum())
    states = compute_all_states(X, V, small_p)
    del X, V
    etude_start(states, small_p)
    etude_tau(states, small_p)
    etude_nlt(states, small_p)
    etude_step(states, small_p)
    resultats_cylindrique(states, small_p)
    comparaison_methodes(states, small_p)
    print("✅ toutes les figures écrites dans", FIGDIR)


def etude_especes(states, small_p):
    """Comparaison : matrice unique (sans distinction d'espèce) vs matrices
    distinctes par espèce, pour les quatre découpages.

    Dans le cas « sans distinction », une seule matrice apprise sur toutes
    les particules propage à la fois le comptage des petites et le comptage
    total ; dans le cas « avec distinction », chaque espèce est propagée par
    sa propre matrice.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
    n_steps = (N_T - START) // TAU
    t_probe = (START + np.arange(n_steps + 1) * TAU).clip(max=N_T - 1)
    resume = {}
    for ax, (name, (st, n)) in zip(axes.flat, states.items()):
        st_small = st[:, small_p]
        # matrices
        P_all, _ = learn_P(st, START, TAU, TAU, 2, 8, n)         # toutes especes
        P_small, _ = learn_P(st_small, START, TAU, TAU, 2, 8, n)  # petites seules
        S0_all = counts(st[START], n)
        S0_small = counts(st[START], n, small_p)
        # avec distinction : chaque espece suit sa matrice
        r_avec = markov_rsd_series(P_small, P_all, S0_small, S0_all, n_steps)
        # sans distinction : la matrice unique P_all propage les deux comptages
        r_sans = markov_rsd_series(P_all, P_all, S0_small, S0_all, n_steps)
        t_mk = (START + np.arange(n_steps + 1) * TAU) / 100
        t_dem = np.arange(START, N_T, 20)
        r_dem = dem_rsd_series(st, small_p, n, t_dem)
        r_dem_probe = dem_rsd_series(st, small_p, n, t_probe)
        e_avec = np.mean(np.abs(r_avec - r_dem_probe))
        e_sans = np.mean(np.abs(r_sans - r_dem_probe))
        resume[name] = (e_sans, e_avec)
        ax.plot(t_dem / 100, r_dem, "k-", lw=0.9, alpha=0.7, label="DEM")
        ax.plot(t_mk, r_sans, "s--", color="0.55", ms=4,
                label=f"sans distinction (écart {e_sans:.3f})")
        ax.plot(t_mk, r_avec, "o--", color=COLORS[name], ms=4,
                label=f"avec distinction (écart {e_avec:.3f})")
        ax.set_title(name)
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
        f.write(f"{'méthode':<14}{'sans distinction':>18}{'avec distinction':>18}\n")
        for k, (es, ea) in resume.items():
            f.write(f"{k:<14}{es:18.4f}{ea:18.4f}\n")
    print("especes ok", resume)


def figure_espace_caracteristiques(X, V, small_p):
    """Espace des caractéristiques du découpage physique : positions (x, y)
    et norme de vitesse, sur la fenêtre d'apprentissage et un instant de la
    fenêtre de prédiction."""
    fit_ts = np.arange(START, START + 5 * TAU, 10)
    pred_t = 4000  # un instant de la phase de prédiction (40 s)

    pts_fit = X[fit_ts].reshape(-1, 3)
    vn_fit = V[fit_ts].reshape(-1)
    pts_pred = X[pred_t]
    vn_pred = V[pred_t]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    vmax = np.percentile(vn_fit, 99)
    for ax, pts, vn, ttl in (
        (axes[0], pts_fit[::12], vn_fit[::12],
         "(a) Phase d'apprentissage (positions cumulées\nsur 5 tours du régime permanent)"),
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
        "norme de la vitesse des 1030 particules",
        fontsize=12,
    )
    fig.savefig(FIGDIR / "espace_caracteristiques_physique.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("espace caracteristiques ok")
