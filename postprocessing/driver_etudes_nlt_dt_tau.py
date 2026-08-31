# -*- coding: utf-8 -*-
"""Driver des études ajoutées après la revue du tuteur (août 2026).

Deux livrables :

1. **Étude d'influence du NLT pour le découpage cylindrique** — miroir
   exact de :func:`etudes_librairie.etude_nlt_erreur_relative` (protocole
   dt = 1 pas, référence NLT = 18, erreur relative de Frobenius de Doucet
   et al. 2008), appliqué à la méthode ``cylindrique`` au lieu de
   ``physique``.  Figure : ``etude_nlt_erreur_relative_cylindrique.png``.

2. **Matrices de transition apprises avec dt = tau** (une paire de
   transition par tour de tambour) pour les quatre découpages, destinées
   au corps du texte — les matrices dt = 8 pas sont reléguées en annexe.
   Une paire de transition s'étalant sur exactement un tour, le schéma
   d'apprentissage illustré par ``schema_reservoir_temps.png`` s'applique
   tel quel.  Figures : ``matrice_<cle>_especes_dt_tau.png``.

Toutes les matrices et les grandeurs chiffrées sont aussi imprimées sur
la sortie standard (versionnée dans ``resultats_etudes_nlt_dt_tau.txt``)
pour alimenter les commentaires du rapport.

Le calcul s'appuie sur la même chaîne numérique que les études du rapport
(:mod:`etudes_librairie`), avec deux adaptations à banc d'essai sans GPU :
``torch`` est remplacé par un module factice pour l'import et
``compute_P_matrix_torch`` est remplacée par une transcription numpy
strictement équivalente (mêmes conventions, mêmes NaN).
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# --- torch factice (uniquement pour satisfaire l'import de run_sweep) ----
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.Tensor = type("Tensor", (), {})  # requis par scipy.stats
    _fake_torch.__spec__ = None
    sys.modules["torch"] = _fake_torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import postprocessing.etudes_librairie as el


class _FakeTensor:
    """Mini-chaîne d'accès .cpu().numpy() compatible avec le site d'appel."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


def _compute_P_matrix_numpy(states_prev, states_curr, n_states,
                            device="cpu", species_labels=None):
    """Transcription numpy exacte de run_sweep.compute_P_matrix_torch.

    transitions[i, j] = nombre de paires (i -> j) ; ligne-stochastique ;
    les lignes jamais observées comme source restent NaN (0/0)."""
    s_prev = np.asarray(states_prev, dtype=np.int64).ravel()
    s_curr = np.asarray(states_curr, dtype=np.int64).ravel()
    n = min(len(s_prev), len(s_curr))
    s_prev, s_curr = s_prev[:n], s_curr[:n]
    transitions = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(transitions, (s_prev, s_curr), 1.0)
    denominator = np.bincount(s_prev, minlength=n_states).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        P = transitions / denominator[:, None]
    return _FakeTensor(P)


el.compute_P_matrix_torch = _compute_P_matrix_numpy

REPDIR = _ROOT / "template-rapport-stage" / "figures"


def etude_nlt_cylindrique(et):
    """Influence du NLT sur le découpage cylindrique (10 secteurs)."""
    nlts = [1, 2, 3, 5, 8, 12, 18]
    nlt_ref = nlts[-1]
    print("\n=== Étude NLT — découpage CYLINDRIQUE (dt=1 pas, réf NLT=18) ===")
    t0 = time.time()
    P_ref = {sp: et.build_P(el.config_for(et.method, nlt=nlt_ref, dt=1), sp)
             for sp in ("small", "large")}
    errs = {sp: [] for sp in ("small", "large")}
    for nlt in nlts:
        cfg = el.config_for(et.method, nlt=nlt, dt=1)
        for sp in ("small", "large"):
            P = et.build_P(cfg, sp)
            e = (np.nan_to_num(P - P_ref[sp], nan=0.0) ** 2).sum() ** 0.5
            e /= np.nan_to_num(P_ref[sp], nan=0.0).__pow__(2).sum() ** 0.5
            errs[sp].append(float(e))
            print(f"  NLT={nlt:>2} {sp:>5}: E_rel={e:.4f}")
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
        "(découpage cylindrique, 10 cellules,\n"
        f"start=1,57 s, step=tau=1,57 s, dt=1 pas ; référence NLT={nlt_ref})"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPDIR / "etude_nlt_erreur_relative_cylindrique.png", dpi=200)
    plt.close(fig)
    print(f"  [figure etude_nlt_erreur_relative_cylindrique.png écrite "
          f"en {time.time()-t0:.0f} s]")
    return nlts, errs


def matrices_dt_tau(etudes):
    """Matrices annotées (convention du rapport) apprises avec dt = tau."""
    print("\n=== Matrices de transition apprises avec dt = tau (157 pas) ===")
    for key, et in etudes.items():
        cfg = el.config_for(et.method)  # dt = tau = 157 par défaut
        P_small = et.build_P(cfg, "small")
        P_large = et.build_P(cfg, "large")
        for sp, P in (("small", P_small), ("large", P_large)):
            n_nan = int(np.isnan(P).sum())
            sums = P.sum(axis=1)
            homog = n_nan == 0 and np.allclose(sums[~np.isnan(sums)], 1.0,
                                               atol=1e-9)
            print(f"\n  {et.nom} {sp}: NaN={n_nan} | "
                  f"homogénéisation={'OK' if homog else 'NON'}")
            print("    P =")
            with np.printoptions(precision=3, suppress=True, linewidth=200):
                print(P)
            print(f"    diagonale = "
                  f"{np.round(np.diag(np.nan_to_num(P)), 3)}")
        vmax = max(np.nanmax(P_small), np.nanmax(P_large))
        fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.2))
        for ax, P, ttl in ((axes[0], P_large.T, "Grandes particules (8 mm)"),
                           (axes[1], P_small.T, "Petites particules (4 mm)")):
            im = ax.imshow(P, cmap="YlOrRd", vmin=0, vmax=vmax)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    v = P[i, j]
                    txt = "NaN" if np.isnan(v) else f"{v:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                            color="black" if (np.isnan(v) or v < 0.55 * vmax)
                            else "white")
            ax.set_xticks(range(P.shape[0]))
            ax.set_yticks(range(P.shape[0]))
            ax.set_xlabel("Cellule source $j$")
            ax.set_ylabel("Cellule d'arrivée $i$")
            ax.set_title(ttl)
        fig.colorbar(im, ax=axes, label="$P_{i,j}$", shrink=0.85,
                     fraction=0.046, pad=0.02)
        fig.suptitle(f"{et.nom} — apprentissage avec $dt = \\tau$ "
                     "(une paire de transition par tour)", fontsize=12)
        fig.savefig(REPDIR / f"matrice_{key}_especes_dt_tau.png", dpi=200,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  [figure matrice_{key}_especes_dt_tau.png écrite]")


def main():
    print("chargement des données ...")
    timestep_dict = el.load_timestep_dict()
    sample_coords, s_velocities, _ = el.sample_coordinates(timestep_dict)
    permanent_rows = el.PERMANENT_START * el.N_PARTICLES_PER_TIMESTEP
    frame = el.make_frame(sample_coords, permanent_rows)
    print("fit des quatre découpages + états ...")
    etudes = {k: el.EtudeMethode(k, timestep_dict, sample_coords,
                                 s_velocities, frame=frame)
              for k in ("cartesien", "cylindrique", "voronoi", "physique")}
    etude_nlt_cylindrique(etudes["cylindrique"])
    matrices_dt_tau(etudes)
    print("\nTERMINÉ")


if __name__ == "__main__":
    main()
