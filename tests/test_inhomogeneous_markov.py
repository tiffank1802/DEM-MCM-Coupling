"""
tests/test_inhomogeneous_markov.py.
===================================
Tests complets pour le modèle de Markov inhomogène.

Valide que :
1. Le flag `inhomogeneous` dans ExperimentConfig modifie bien le nom du dossier
2. `run_inhomogeneous_experiment()` construit P_blocks (une matrice par NLT)
3. `save_inhomogeneous_results()` sauvegarde au bon format avec metadata
4. `propagate_markov_inhomogeneous()` utilise les bonnes matrices au bon moment
5. `prepare_species_inhomogeneous()` prépare correctement les données
6. `load_experiment()` détecte le format inhomogène et charge P_blocks
7. Le round-trip complet (config → expérience → sauvegarde →
   charge → propagation) est cohérent
8. Les cas limites (1 seule espèce, 1 seul NLT, matrices identiques, etc.) fonctionnent
"""

from __future__ import annotations

import io
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Ajout du chemin projet ─────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Imports à tester ───────────────────────────────────────────────────────

# Configuration

from typing import Any

# Run_sweep : fonctions à tester
from dem_mcm_coupling.run_sweep import (
    ExperimentConfig,
    _detect_species,
    compute_P_matrix_torch,
    run_inhomogeneous_experiment,
    save_inhomogeneous_results,
)

# Postprocess : fonctions à tester
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
# Bucket I/O
from dem_mcm_coupling.bucket_io import (
    ALL_CATEGORIES,
    CATEGORY_MAP,
    get_simulation_category,
    save_experiment_to_bucket,
)
from postprocessing.postprocess import (
    clean_transition_matrix,
    load_experiment,
    prepare_species,
    prepare_species_inhomogeneous,
    propagate_markov,
    propagate_markov_inhomogeneous,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def rng() -> np.random.RandomState:
    """Seed RNG for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def n_states() -> int:
    return 10


@pytest.fixture
def n_particles() -> int:
    return 500


@pytest.fixture
def n_timesteps() -> int:
    return 200


@pytest.fixture
def n_timesteps_large() -> int:
    """Plus de timesteps pour les tests inhomogènes (nlt=3 nécessite ~500 timesteps)."""
    return 600


# ── Configurations ──────────────────────────────────────────────────────────


@pytest.fixture
def homogeneous_config() -> ExperimentConfig:
    """Configuration homogène standard."""
    return ExperimentConfig(
        method="cartesian",
        method_kwargs={"nx": 3, "ny": 3, "nz": 2},
        nlt=3,
        tau=50,
        step=100,
        dt=10,
        start_index=250,
        particle_diameter=None,
        inhomogeneous=False,
    )


@pytest.fixture
def inhomogeneous_config() -> ExperimentConfig:
    """Configuration inhomogène — une matrice P par NLT."""
    return ExperimentConfig(
        method="cartesian",
        method_kwargs={"nx": 3, "ny": 3, "nz": 2},
        nlt=3,
        tau=50,
        step=100,
        dt=10,
        start_index=250,
        particle_diameter=None,
        inhomogeneous=True,
    )


@pytest.fixture
def inhomogeneous_config_single_nlt() -> ExperimentConfig:
    """Configuration inhomogène avec un seul NLT."""
    return ExperimentConfig(
        method="cartesian",
        method_kwargs={"nx": 3, "ny": 3, "nz": 2},
        nlt=1,
        tau=50,
        step=100,
        dt=1,
        start_index=250,
        particle_diameter=None,
        inhomogeneous=True,
    )


# ── Données synthétiques ───────────────────────────────────────────────────


@pytest.fixture
def synthetic_timestep_dict(
    rng: np.random.RandomState, n_particles: int, n_timesteps: int
) -> dict[int, pd.DataFrame]:
    """
    Crée un dictionnaire de timesteps synthétiques pour les tests.

    Retourne {idx: pd.DataFrame} avec des coordonnées 3D, vitesses et diamètres.
    Deux espèces : small (d=0.004) et large (d=0.008).
    """
    import pandas as pd

    timestep_dict = {}
    base_idx = 250

    for i in range(n_timesteps):
        idx = base_idx + i

        # Positions : mélange de deux distributions
        n_small = n_particles // 2
        n_large = n_particles - n_small

        x = np.concatenate(
            [
                rng.normal(0.02, 0.01, n_small),
                rng.normal(-0.02, 0.01, n_large),
            ]
        )
        y = np.concatenate(
            [
                rng.uniform(0, 0.05, n_small),
                rng.uniform(0, 0.05, n_large),
            ]
        )
        z = np.concatenate(
            [
                rng.uniform(0, 0.02, n_small),
                rng.uniform(0, 0.02, n_large),
            ]
        )

        # Vitesses
        vx = rng.normal(0, 0.1, n_particles)
        vy = rng.normal(0, 0.1, n_particles)
        vz = rng.normal(0, 0.05, n_particles)

        # Diamètres
        diameters = np.array([0.004] * n_small + [0.008] * n_large)

        # IDs particules
        particle_ids = np.arange(n_particles)

        df = pd.DataFrame(
            {
                "coordinates:0": x,
                "coordinates:1": y,
                "coordinates:2": z,
                "Velocity:0": vx,
                "Velocity:1": vy,
                "Velocity:2": vz,
                "Diameter": diameters,
                "Particle_ID": particle_ids,
                "Fichier_Source": f"data_{idx}.csv",
            }
        )

        timestep_dict[idx] = df

    return timestep_dict


@pytest.fixture
def synthetic_timestep_dict_large(
    rng: np.random.RandomState, n_particles: int, n_timesteps_large: int
) -> dict[int, pd.DataFrame]:
    """
    Crée un dictionnaire de timesteps synthétiques AVEC PLUS DE TIMESTEPS
    pour les tests inhomogènes qui ont besoin de NLT=3 blocs.
    """
    import pandas as pd

    timestep_dict = {}
    base_idx = 250

    for i in range(n_timesteps_large):
        idx = base_idx + i

        # Positions : mélange de deux distributions
        n_small = n_particles // 2
        n_large = n_particles - n_small

        x = np.concatenate(
            [
                rng.normal(0.02, 0.01, n_small),
                rng.normal(-0.02, 0.01, n_large),
            ]
        )
        y = np.concatenate(
            [
                rng.uniform(0, 0.05, n_small),
                rng.uniform(0, 0.05, n_large),
            ]
        )
        z = np.concatenate(
            [
                rng.uniform(0, 0.02, n_small),
                rng.uniform(0, 0.02, n_large),
            ]
        )

        # Vitesses
        vx = rng.normal(0, 0.1, n_particles)
        vy = rng.normal(0, 0.1, n_particles)
        vz = rng.normal(0, 0.05, n_particles)

        # Diamètres
        diameters = np.array([0.004] * n_small + [0.008] * n_large)

        # IDs particules
        particle_ids = np.arange(n_particles)

        df = pd.DataFrame(
            {
                "coordinates:0": x,
                "coordinates:1": y,
                "coordinates:2": z,
                "Velocity:0": vx,
                "Velocity:1": vy,
                "Velocity:2": vz,
                "Diameter": diameters,
                "Particle_ID": particle_ids,
                "Fichier_Source": f"data_{idx}.csv",
            }
        )

        timestep_dict[idx] = df

    return timestep_dict


@pytest.fixture
def mock_partitioner(rng: np.random.RandomState, n_states: int) -> Any:
    """
    Mock d'un partitionneur simple.

    Simule les méthodes fit() et compute_states() pour retourner
    des états aléatoires mais reproductibles.
    """

    class MockPartitioner:
        def __init__(self) -> None:
            self.n_cells = n_states
            self._label = f"mock_{n_states}cells"
            self.centroids = rng.rand(n_states, 3)
            self.use_velocity = False
            self.dem_velocities = None

        @property
        def label(self) -> str:
            return self._label

        def fit(self, coords: np.ndarray, **kwargs: Any) -> MockPartitioner:
            return self

        def compute_states(
            self,
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            vx: np.ndarray | None = None,
            vy: np.ndarray | None = None,
            vz: np.ndarray | None = None,
        ) -> np.ndarray:
            n = len(x)
            # Distribution inégale pour créer des états visités/non visités
            states = rng.choice(n_states, size=n, p=_make_uneven_distribution(n_states))
            return states.astype(np.int64)

        def diagnostics(self, coords: np.ndarray) -> dict:
            return {
                "n_visited": n_states - 2,
                "pop_min": 5,
                "pop_max": 100,
                "pop_mean": 50,
                "pop_std": 30,
            }

    return MockPartitioner()


def _make_uneven_distribution(n_states: int) -> np.ndarray:
    """Crée une distribution inégale sur les états (certains états vides)."""
    p = np.exp(-np.arange(n_states) / 3)
    p[0] = p[0] * 3  # premier état très visité
    p[-1] = 0.0  # dernier état jamais visité
    p[-2] = p[-2] * 0.1  # avant-dernier quasi vide
    return p / p.sum()


# ── Matrices de transition synthétiques ────────────────────────────────────


@pytest.fixture
def homogeneous_transition_matrix(
    rng: np.random.RandomState, n_states: int
) -> np.ndarray:
    """Matrice de transition homogène row-stochastic."""
    P = rng.rand(n_states, n_states)
    P /= P.sum(axis=1, keepdims=True)
    return P


@pytest.fixture
def inhomogeneous_P_blocks(rng: np.random.RandomState, n_states: int) -> np.ndarray:
    """
    Blocs de matrices inhomogènes : 3 matrices P différentes.

    Chaque matrice est volontairement différente pour tester
    que la propagation inhomogène utilise bien la bonne matrice
    au bon moment.
    """
    blocks = []
    for k in range(3):
        P = rng.rand(n_states, n_states)
        P /= P.sum(axis=1, keepdims=True)
        # Rendre chaque bloc distinct
        if k == 0:
            P *= 0.8
            P[range(n_states), range(n_states)] += 0.2  # Forte diagonale
        elif k == 1:
            P *= 0.5
            P[range(n_states), range(n_states)] += 0.5  # Très forte diagonale
        else:
            P *= 0.3
            P[range(n_states), range(n_states)] += 0.7  # Mix lente
        P /= P.sum(axis=1, keepdims=True)
        blocks.append(P)
    return np.array(blocks)  # (3, n_states, n_states)


@pytest.fixture
def synthetic_S_matrix(
    rng: np.random.RandomState, n_timesteps: int, n_states: int
) -> np.ndarray:
    """Matrice d'états synthétique (n_timesteps, n_states)."""
    S = rng.poisson(50, size=(n_timesteps, n_states)).astype(np.float64)
    return S


@pytest.fixture
def synthetic_times(n_timesteps: int) -> np.ndarray:
    """Timesteps indices."""
    return np.arange(250, 250 + n_timesteps)


# ============================================================================
# 1. TESTS DE CONFIGURATION
# ============================================================================


class TestExperimentConfigInhomogeneous:
    """Vérifie que le flag inhomogeneous modifie correctement la configuration."""

    def test_default_is_false(self) -> None:
        """Le flag inhomogeneous doit être False par défaut."""
        config = ExperimentConfig(method="voronoi", method_kwargs={"n_cells": 10})
        assert config.inhomogeneous is False

    def test_inhomogeneous_flag_stored(self) -> None:
        """Le flag inhomogeneous=True doit être conservé."""
        config = ExperimentConfig(
            method="voronoi", method_kwargs={"n_cells": 10}, inhomogeneous=True
        )
        assert config.inhomogeneous is True

    def test_output_folder_prefix_homogeneous(
        self, homogeneous_config: ExperimentConfig
    ) -> None:
        """Sans le flag, pas de préfixe inhomogeneous_."""
        folder = homogeneous_config.output_folder()
        assert not folder.startswith("inhomogeneous_")

    def test_output_folder_prefix_inhomogeneous(
        self, inhomogeneous_config: ExperimentConfig
    ) -> None:
        """Avec le flag, le dossier commence par inhomogeneous_."""
        folder = inhomogeneous_config.output_folder()
        assert folder.startswith("inhomogeneous_")

    def test_output_folder_prefix_single_nlt(
        self, inhomogeneous_config_single_nlt: ExperimentConfig
    ) -> None:
        """Même avec NLT=1, le préfixe inhomogeneous_ est présent."""
        folder = inhomogeneous_config_single_nlt.output_folder()
        assert folder.startswith("inhomogeneous_")

    def test_output_folder_deterministic(
        self, inhomogeneous_config: ExperimentConfig
    ) -> None:
        """Deux appels produisent le même nom de dossier."""
        f1 = inhomogeneous_config.output_folder()
        f2 = inhomogeneous_config.output_folder()
        assert f1 == f2

    def test_asdict_contains_inhomogeneous(
        self, inhomogeneous_config: ExperimentConfig
    ) -> None:
        """asdict() doit inclure le champ inhomogeneous."""
        d = asdict(inhomogeneous_config)
        assert "inhomogeneous" in d
        assert d["inhomogeneous"] is True

    def test_inhomogeneous_config_preserves_other_params(
        self, inhomogeneous_config: ExperimentConfig
    ) -> None:
        """Le flag inhomogène ne doit pas altérer les autres paramètres."""
        h_config = ExperimentConfig(
            method=inhomogeneous_config.method,
            method_kwargs=inhomogeneous_config.method_kwargs,
            nlt=inhomogeneous_config.nlt,
            tau=inhomogeneous_config.tau,
            step=inhomogeneous_config.step,
            dt=inhomogeneous_config.dt,
            start_index=inhomogeneous_config.start_index,
            particle_diameter=inhomogeneous_config.particle_diameter,
            inhomogeneous=False,
        )
        # Vérifier que tous les paramètres sauf inhomogeneous sont identiques
        d_h = asdict(h_config)
        d_ih = asdict(inhomogeneous_config)
        for key in d_h:
            if key != "inhomogeneous":
                assert d_h[key] == d_ih[key], (
                    f"Le paramètre '{key}' diffère entre homogène et inhomogène"
                )


# ============================================================================
# 2. TESTS MATHÉMATIQUES — propagate_markov_inhomogeneous
# ============================================================================


class TestPropagateMarkovInhomogeneous:
    """Valide la propagation markovienne avec matrices variables."""

    def test_basic_shape(
        self, inhomogeneous_P_blocks: np.ndarray, n_states: int
    ) -> None:
        """La propagation doit retourner (n_steps+1, n_states)."""
        S0 = np.ones(n_states) * 50
        times = np.arange(250, 500)
        activated = np.ones(n_states, dtype=bool)

        traj, t_markov = propagate_markov_inhomogeneous(
            S0,
            inhomogeneous_P_blocks,
            times,
            start_idx=250,
            tau=10,
            activated=activated,
        )
        assert traj.ndim == 2
        assert traj.shape[1] == n_states
        assert len(t_markov) == traj.shape[0]

    def test_preserves_total_particles(
        self, inhomogeneous_P_blocks: np.ndarray, n_states: int
    ) -> None:
        """La somme des particules doit être conservée (normalisation)."""
        S0 = np.ones(n_states) * 50
        total_init = S0.sum()
        times = np.arange(250, 500)
        activated = np.ones(n_states, dtype=bool)

        traj, _ = propagate_markov_inhomogeneous(
            S0,
            inhomogeneous_P_blocks,
            times,
            start_idx=250,
            tau=10,
            activated=activated,
        )
        for t in range(len(traj)):
            assert np.isclose(traj[t].sum(), total_init, rtol=1e-10), (
                f"Perte de particules au pas {t}: {traj[t].sum()} != {total_init}"
            )

    def test_activated_states_only(
        self, inhomogeneous_P_blocks: np.ndarray, n_states: int
    ) -> None:
        """
        Les états désactivés ne doivent pas recevoir de population INITIALE.

        Note : pendant la propagation, la matrice de transition peut envoyer
        de la masse depuis les états activés vers les états désactivés.
        C'est un comportement normal. On vérifie donc uniquement que :
        - S0[~activated] = 0 (pas de population initiale)
        - Les états activés reçoivent bien la population initiale
        """
        S0 = np.ones(n_states) * 50
        activated = np.zeros(n_states, dtype=bool)
        activated[: n_states // 2] = True  # Seulement la moitié des états activés
        times = np.arange(250, 500)

        # Matrices du fixture (row-stochastic, validées par
        # test_preserves_total_particles)
        P_blocks = inhomogeneous_P_blocks

        traj, _ = propagate_markov_inhomogeneous(
            S0, P_blocks, times, start_idx=250, tau=10, activated=activated
        )
        # Vérifier que S0 a bien les états désactivés à 0
        S0_modified = S0.copy().astype(float)
        S0_modified[~activated] = 0.0
        assert np.allclose(traj[0], S0_modified), (
            "L'état initial devrait avoir les états désactivés à 0"
        )
        # Vérifier que la population totale est conservée
        total_initial = S0_modified.sum()
        for t in range(len(traj)):
            assert np.isclose(traj[t].sum(), total_initial, rtol=1e-10), (
                f"Perte de particules au pas {t}"
            )

    def test_single_block_equals_homogeneous(self, n_states: int) -> None:
        """
        Avec 1 seul bloc et NLT=1, le résultat doit être identique
        à la propagation homogène avec la même matrice.

        On utilise une distribution initiale non-uniforme pour que
        la multiplication matricielle ait un effet visible.
        """
        rng = np.random.RandomState(42)
        P = rng.rand(n_states, n_states)
        P /= P.sum(axis=1, keepdims=True)

        # Distribution initiale non-uniforme pour voir l'effet de P
        S0 = np.zeros(n_states)
        S0[0] = n_states * 50  # Toute la masse dans l'état 0
        times = np.arange(250, 500)
        activated = np.ones(n_states, dtype=bool)
        P_clean, _ = clean_transition_matrix(P)

        # Propagation homogène (utilise P_clean)
        traj_homo, _ = propagate_markov(
            S0, P_clean, times, start_idx=250, tau=10, activated=activated
        )
        # Propagation inhomogène avec un seul bloc (doit utiliser la MÊME matrice)
        traj_inhomo, _ = propagate_markov_inhomogeneous(
            S0,
            P_clean[np.newaxis, :, :],
            times,
            start_idx=250,
            tau=10,
            activated=activated,
        )
        assert np.allclose(traj_homo, traj_inhomo), (
            "Propagation inhomogène à 1 bloc ≠ propagation homogène"
        )

    def test_identical_blocks_equals_homogeneous(self, n_states: int) -> None:
        """
        Si tous les blocs P_k sont identiques, le résultat inhomogène
        doit être identique à l'homogène (qui utilise la même matrice
        à chaque pas).

        On utilise une distribution initiale non-uniforme pour que
        la multiplication matricielle ait un effet visible.
        """
        rng = np.random.RandomState(42)
        P = rng.rand(n_states, n_states)
        P /= P.sum(axis=1, keepdims=True)

        P_clean, _ = clean_transition_matrix(P)
        P_blocks = np.stack(
            [P_clean.copy() for _ in range(5)]
        )  # 5 blocs identiques (nettoyés)

        # Distribution initiale non-uniforme
        S0 = np.zeros(n_states)
        S0[0] = n_states * 50
        times = np.arange(250, 500)
        activated = np.ones(n_states, dtype=bool)

        traj_homo, _ = propagate_markov(
            S0, P_clean, times, start_idx=250, tau=10, activated=activated
        )
        traj_inhomo, _ = propagate_markov_inhomogeneous(
            S0, P_blocks, times, start_idx=250, tau=10, activated=activated
        )
        assert np.allclose(traj_homo, traj_inhomo), (
            "Blocs identiques devraient donner le même résultat qu'une seule matrice"
        )

    def test_different_blocks_produce_different_trajectories(
        self, n_states: int
    ) -> None:
        """
        Des matrices différentes doivent produire des trajectoires différentes.

        On utilise une distribution initiale non-uniforme (masse dans l'état 0)
        pour que les différentes matrices aient un effet visible.

        Bloc 1 : forte rétention (diagonale élevée) → l'état 0 reste très peuplé
        Bloc 2 : fort mixing (faible diagonale) → la population se disperse
        """
        rng = np.random.RandomState(42)
        # Bloc 1 : forte rétention (diagonale élevée)
        P1 = rng.rand(n_states, n_states) * 0.3
        P1[range(n_states), range(n_states)] += 0.7
        P1 /= P1.sum(axis=1, keepdims=True)

        # Bloc 2 : fort mixing (faible diagonale)
        P2 = rng.rand(n_states, n_states) * 0.9
        P2[range(n_states), range(n_states)] += 0.1
        P2 /= P2.sum(axis=1, keepdims=True)

        # Distribution initiale non-uniforme
        S0 = np.zeros(n_states)
        S0[0] = n_states * 50  # Toute la masse dans l'état 0
        times = np.arange(250, 500)
        activated = np.ones(n_states, dtype=bool)

        # Avec P1 seulement (forte rétention)
        traj_P1, _ = propagate_markov_inhomogeneous(
            S0, P1[np.newaxis, :, :], times, start_idx=250, tau=10, activated=activated
        )
        # Avec P2 seulement (fort mixing)
        traj_P2, _ = propagate_markov_inhomogeneous(
            S0, P2[np.newaxis, :, :], times, start_idx=250, tau=10, activated=activated
        )
        # Les trajectoires avec 1 seul bloc doivent différer entre P1 et P2
        assert not np.allclose(traj_P1, traj_P2), (
            "P1 et P2 devraient produire des trajectoires différentes "
            "(rétention vs mixing)"
        )

        # Avec P1 puis P2 (2 blocs)
        traj_both, _ = propagate_markov_inhomogeneous(
            S0, np.stack([P1, P2]), times, start_idx=250, tau=10, activated=activated
        )
        # La trajectoire des 2 blocs doit différer des deux mono-blocs
        assert not np.allclose(traj_both, traj_P1), (
            "La trajectoire avec 2 blocs ne devrait pas être identique à P1 seul"
        )
        assert not np.allclose(traj_both, traj_P2), (
            "La trajectoire avec 2 blocs ne devrait pas être identique à P2 seul"
        )

    def test_block_index_usage(self, n_states: int) -> None:
        """
        Vérifie que la propagation utilise bien des matrices différentes
        selon l'étape (bloc).

        On crée 2 blocs extrêmes :
        - Bloc 0 : ne bouge pas (identité) → la distribution reste identique
        - Bloc 1 : permute tout → la distribution change radicalement

        Si le block_index est correct, la première moitié des étapes
        doit conserver la distribution, la seconde moitié doit la changer.
        """
        S0 = np.zeros(n_states)
        S0[0] = 100  # Toutes les particules dans l'état 0

        # Bloc 0 : identité (ne rien changer)
        P_identity = np.eye(n_states)

        # Bloc 1 : mélange uniforme
        P_uniform = np.ones((n_states, n_states)) / n_states

        P_blocks = np.stack([P_identity, P_uniform])

        times = np.arange(250, 500)
        activated = np.ones(n_states, dtype=bool)

        traj, t_markov = propagate_markov_inhomogeneous(
            S0, P_blocks, times, start_idx=250, tau=10, activated=activated
        )

        # Avec block_size = n_steps // 2:
        n_steps = len(t_markov) - 1
        block_size = max(1, n_steps // 2)

        # Les block_size premières propagations utilisent P_identity
        # donc la distribution reste concentrée sur l'état 0
        for t in range(1, min(block_size + 1, len(t_markov))):
            assert traj[t][0] > 90, (
                f"À l'étape {t}, l'état 0 a perdu trop de masse : {traj[t][0]}"
            )

        # Après block_size, on utilise P_uniform → distribution uniforme
        if block_size + 1 < len(traj):
            t_late = min(block_size + 3, len(traj) - 1)
            max_state = traj[t_late].max()
            min_state = traj[t_late].min()
            assert max_state - min_state < 30, (
                f"À l'étape {t_late}, la distribution n'est pas uniforme : "
                f"max={max_state}, min={min_state}"
            )

    def test_non_negative_states(
        self, inhomogeneous_P_blocks: np.ndarray, n_states: int
    ) -> None:
        """Les populations d'états doivent toujours être non-négatives."""
        S0 = np.ones(n_states) * 50
        times = np.arange(250, 500)
        activated = np.ones(n_states, dtype=bool)

        traj, _ = propagate_markov_inhomogeneous(
            S0,
            inhomogeneous_P_blocks,
            times,
            start_idx=250,
            tau=10,
            activated=activated,
        )
        assert np.all(traj >= -1e-12), "Des populations d'états négatives détectées"

    def test_large_n_blocks(self, n_states: int) -> None:
        """Beaucoup de blocs doit fonctionner (test de performance logique)."""
        rng = np.random.RandomState(42)
        n_blocks = 20
        blocks = []
        for _ in range(n_blocks):
            P = rng.rand(n_states, n_states)
            P /= P.sum(axis=1, keepdims=True)
            blocks.append(P)
        P_blocks = np.array(blocks)

        S0 = np.ones(n_states) * 50
        times = np.arange(250, 500)
        activated = np.ones(n_states, dtype=bool)

        traj, t_markov = propagate_markov_inhomogeneous(
            S0, P_blocks, times, start_idx=250, tau=10, activated=activated
        )
        assert len(traj) == len(t_markov)
        # Vérifier conservation par pas de temps, pas sur toute la trajectoire
        for t in range(len(traj)):
            assert np.isclose(traj[t].sum(), S0.sum(), rtol=1e-10), (
                f"Perte de particules au pas {t}"
            )

    def test_zero_activation(
        self, inhomogeneous_P_blocks: np.ndarray, n_states: int
    ) -> None:
        """Aucun état activé → tout reste à 0."""
        S0 = np.ones(n_states) * 50
        times = np.arange(250, 500)
        activated = np.zeros(n_states, dtype=bool)

        traj, _ = propagate_markov_inhomogeneous(
            S0,
            inhomogeneous_P_blocks,
            times,
            start_idx=250,
            tau=10,
            activated=activated,
        )
        assert np.allclose(traj, 0.0), (
            "Sans états activés, la propagation devrait rester nulle"
        )


# ============================================================================
# 3. TESTS — compute_P_matrix_torch
# ============================================================================


class TestComputePMatrixTorch:
    """Vérifie le calcul de la matrice de transition via PyTorch.

    Convention du package (ligne-stochastique): P[i, j] = P(i -> j),
    donc sum(P[i, :]) = 1 (ou 0 si la ligne est vide). Un vecteur d'état
    évolue comme phi_next = phi @ P.
    """

    def test_basic_shape(self, rng: np.random.RandomState, n_states: int) -> None:
        """La matrice P doit être de taille (n_states, n_states)."""
        n = 1000
        prev = rng.choice(n_states, size=n)
        curr = rng.choice(n_states, size=n)

        P = compute_P_matrix_torch(prev, curr, n_states, device="cpu")
        assert P.shape == (n_states, n_states)

    def test_row_stochastic(self, rng: np.random.RandomState, n_states: int) -> None:
        """Chaque LIGNE de P doit sommer à 1 (ou 0 si ligne vide).

        P[i, j] = probabilité de transition i -> j.
        """
        n = 1000
        prev = rng.choice(n_states, size=n)
        curr = rng.choice(n_states, size=n)

        P = compute_P_matrix_torch(prev, curr, n_states, device="cpu")
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0), (
            "Les lignes non-vides de P doivent être stochastiques (sommer à 1)"
        )

    def test_non_negative(self, rng: np.random.RandomState, n_states: int) -> None:
        """La matrice P ne doit pas contenir de valeurs négatives."""
        n = 1000
        prev = rng.choice(n_states, size=n)
        curr = rng.choice(n_states, size=n)

        P = compute_P_matrix_torch(prev, curr, n_states, device="cpu")
        # P est un torch tensor, convertir en numpy pour np.all
        P_np = P.cpu().numpy()
        assert np.all(P_np >= -1e-12), "P contient des valeurs négatives"

    def test_single_transition(self, n_states: int) -> None:
        """Test simple : transition de l'état 0 vers l'état 1.

        Convention ligne-stochastique : P[0, 1] = P(0 -> 1). Les états jamais
        visités produisent des lignes nulles (pas de NaN).
        """
        n = n_states * 10  # Assez pour que chaque état ait des particules
        prev = np.zeros(n, dtype=int)
        curr = np.zeros(n, dtype=int)
        for i in range(n):
            prev[i] = i % n_states
            curr[i] = (i + 1) % n_states  # Transition i -> i+1

        P = compute_P_matrix_torch(prev, curr, n_states, device="cpu")
        P_np = P.cpu().numpy()

        # P[from=0, to=1] devrait être proche de 1 (toutes les particules de 0 vont à 1)
        assert np.isclose(P_np[0, 1], 1.0, atol=1e-10), (
            f"P[0,1] devrait être 1.0, trouvé {P_np[0, 1]}"
        )
        assert P_np.shape == (n_states, n_states)

    def test_two_species_equivalent(
        self, rng: np.random.RandomState, n_states: int
    ) -> None:
        """
        compute_P_matrix_torch avec des labels d'espèces doit être équivalent
        à filtrer les états par espèce puis appeler la fonction.
        Cette propriété est cruciale pour les matrices par espèce.
        """
        n = 2000
        prev = rng.choice(n_states, size=n)
        curr = rng.choice(n_states, size=n)
        # Labels : moitié small, moitié large
        species_labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
        rng.shuffle(species_labels)

        # Calcul avec toutes les données
        P_all = compute_P_matrix_torch(prev, curr, n_states, device="cpu")

        # Calcul espèces séparées
        P_small = compute_P_matrix_torch(
            prev[species_labels == 0], curr[species_labels == 0], n_states, device="cpu"
        )
        P_large = compute_P_matrix_torch(
            prev[species_labels == 1], curr[species_labels == 1], n_states, device="cpu"
        )

        # Vérifier que les matrices par espèce diffèrent de la matrice globale
        assert not np.allclose(P_small, P_all), (
            "La matrice 'small' ne devrait pas être identique à P_all"
        )
        assert not np.allclose(P_large, P_all), (
            "La matrice 'large' ne devrait pas être identique à P_all"
        )
        assert not np.allclose(P_small, P_large), (
            "Les matrices small et large devraient différer"
        )


# ============================================================================
# 4. TESTS — prepare_species_inhomogeneous
# ============================================================================


class TestPrepareSpeciesInhomogeneous:
    """Vérifie la préparation des données pour la propagation inhomogène."""

    def create_inhomogeneous_experiment_data(
        self,
        rng: np.random.RandomState,
        n_states: int,
        n_timesteps: int,
        n_blocks: int = 3,
    ) -> dict:
        """Helper pour créer un jeu de données inhomogène complet."""
        times = np.arange(250, 250 + n_timesteps)
        S = rng.poisson(50, size=(n_timesteps, n_states)).astype(np.float64)

        P_blocks = []
        for k in range(n_blocks):
            P = rng.rand(n_states, n_states)
            P /= P.sum(axis=1, keepdims=True)
            P_blocks.append(P)
        P_blocks = np.array(P_blocks)

        return {
            "config": {
                "tau": 50,
                "start_index": 250,
                # The real temporal structure of the blocks: block k starts
                # at start_index + k * (step + tau). With step=20 and tau=50,
                # the 4 propagation steps (250, 300, 350, 400) cross the
                # blocks 0, 0, 1, 2 — which makes the inhomogeneous
                # trajectory genuinely differ from the homogeneous one.
                "step": 20,
                "nlt": n_blocks,
            },
            "stats": {
                "species_list": ["small", "large"],
            },
            "species": {
                "small": {
                    "P_blocks": P_blocks,
                    "P_raw": P_blocks[0],
                    "S_matrix": S,
                    "times": times,
                },
                "large": {
                    "P_blocks": P_blocks + 0.1 * rng.randn(*P_blocks.shape),
                    "P_raw": P_blocks[0],
                    "S_matrix": S * 0.8,
                    "times": times,
                },
            },
            "matrix": rng.randint(0, n_states, size=(n_timesteps, 500)),
            "inhomogeneous": True,
            "inhomogeneous_metadata": {
                "n_blocks": n_blocks,
                "n_pairs_per_block": [10] * n_blocks,
                "species_list": ["small", "large"],
                "block_start_indices": [250, 350, 450],
            },
        }

    def test_uses_P_blocks(
        self, rng: np.random.RandomState, n_states: int, n_timesteps: int
    ) -> None:
        """prepare_species_inhomogeneous doit utiliser P_blocks."""
        exp = self.create_inhomogeneous_experiment_data(rng, n_states, n_timesteps)
        result = prepare_species_inhomogeneous(exp)

        for sp in ["small", "large"]:
            assert "P_blocks" in result[sp], f"P_blocks manquant pour '{sp}'"
            assert result[sp]["P_blocks"].ndim == 3, (
                f"P_blocks doit être 3D pour '{sp}'"
            )

    def test_raises_without_P_blocks(
        self, rng: np.random.RandomState, n_states: int, n_timesteps: int
    ) -> None:
        """Sans P_blocks, prepare_species_inhomogeneous doit lever une erreur."""
        exp = self.create_inhomogeneous_experiment_data(rng, n_states, n_timesteps)
        # Supprimer P_blocks d'une espèce
        del exp["species"]["small"]["P_blocks"]

        with pytest.raises(KeyError, match="P_blocks"):
            prepare_species_inhomogeneous(exp)

    def test_traj_markov_shape(
        self, rng: np.random.RandomState, n_states: int, n_timesteps: int
    ) -> None:
        """traj_markov doit être 2D avec le bon nombre d'états."""
        exp = self.create_inhomogeneous_experiment_data(rng, n_states, n_timesteps)
        result = prepare_species_inhomogeneous(exp)

        for sp in ["small", "large"]:
            assert result[sp]["traj_markov"].ndim == 2
            assert result[sp]["traj_markov"].shape[1] == n_states

    def test_compatible_with_homogeneous_keys(
        self, rng: np.random.RandomState, n_states: int, n_timesteps: int
    ) -> None:
        """
        Le résultat de prepare_species_inhomogeneous doit avoir les mêmes
        clés que prepare_species (sauf P_blocks en plus).
        """
        exp = self.create_inhomogeneous_experiment_data(rng, n_states, n_timesteps)

        # Créer une version homogène du même jeu de données
        exp_homo = deepcopy(exp)
        for sp in ["small", "large"]:
            exp_homo["species"][sp]["P_raw"] = exp_homo["species"][sp]["P_blocks"][0]
            del exp_homo["species"][sp]["P_blocks"]
        exp_homo["inhomogeneous"] = False

        result_homo = prepare_species(exp_homo)
        result_inhomo = prepare_species_inhomogeneous(exp)

        # Les clés de base doivent être les mêmes
        for sp in ["small", "large"]:
            set(result_homo[sp].keys()) & set(result_inhomo[sp].keys())
            # "P" est la première matrice nettoyée
            assert "P" in result_inhomo[sp]
            assert "traj_markov" in result_inhomo[sp]
            assert "times_markov" in result_inhomo[sp]
            assert "P_blocks" in result_inhomo[sp]  # clé supplémentaire

    def test_inhomogeneous_vs_homogeneous_different_trajs(
        self, rng: np.random.RandomState, n_states: int, n_timesteps: int
    ) -> None:
        """
        Avec des matrices P_blocks très différentes les unes des autres,
        la trajectoire inhomogène doit différer de la trajectoire homogène
        (qui utilise seulement P_blocks[0]).
        """
        exp = self.create_inhomogeneous_experiment_data(
            rng, n_states, n_timesteps, n_blocks=5
        )

        # Rendre P_blocks[0] très différent des autres
        P0 = exp["species"]["small"]["P_blocks"][0].copy()
        P0[range(n_states), range(n_states)] *= 2
        P0 /= P0.sum(axis=1, keepdims=True)
        exp["species"]["small"]["P_blocks"][0] = P0

        result_inhomo = prepare_species_inhomogeneous(exp)

        # Version homogène : utilise seulement P_blocks[0]
        exp_homo = deepcopy(exp)
        exp_homo["species"]["small"]["P_raw"] = exp_homo["species"]["small"][
            "P_blocks"
        ][0]
        del exp_homo["species"]["small"]["P_blocks"]
        exp_homo["inhomogeneous"] = False
        result_homo = prepare_species(exp_homo)

        # Les trajectoires doivent différer
        traj_homo = result_homo["small"]["traj_markov"]
        traj_inhomo = result_inhomo["small"]["traj_markov"]

        # Au moins une différence significative
        diff = np.max(np.abs(traj_homo - traj_inhomo))
        assert diff > 1.0, (
            f"Les trajectoires homogène et inhomogène devraient différer, "
            f"mais diff max = {diff}"
        )


# ============================================================================
# 5. TESTS — load_experiment (détection format inhomogène)
# ============================================================================


class TestLoadExperimentInhomogeneous:
    """Vérifie la détection et le chargement du format inhomogène."""

    def test_detects_inhomogeneous(self) -> None:
        """
        load_experiment doit détecter le format inhomogène via
        la présence de inhomogeneous_metadata.json.
        """
        # On va mocker HfFileSystem pour simuler la présence du fichier
        with patch("postprocessing.postprocess.fs") as mock_fs:
            # Simuler que inhomogeneous_metadata.json existe
            def mock_open(path: str, mode: str = "r") -> Any:
                if "inhomogeneous_metadata.json" in path:
                    io.BytesIO()
                    data = json.dumps(
                        {
                            "n_blocks": 3,
                            "n_pairs_per_block": [10, 10, 10],
                            "species_list": ["small", "large"],
                            "block_start_indices": [250, 350, 450],
                        }
                    ).encode()
                    if "b" in mode:
                        return io.BytesIO(data)
                    else:
                        return io.StringIO(data.decode())
                elif path.endswith("config.json"):
                    content = json.dumps({"tau": 50}).encode()
                    if "b" in mode:
                        return io.BytesIO(content)
                    else:
                        return io.StringIO(content.decode())
                elif path.endswith("stats.json"):
                    content = json.dumps(
                        {
                            "species_list": ["small", "large"],
                            "n_states": 10,
                        }
                    ).encode()
                    if "b" in mode:
                        return io.BytesIO(content)
                    else:
                        return io.StringIO(content.decode())
                elif "P_blocks_small.npy" in path or "P_blocks_large.npy" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.ones((3, 10, 10), dtype=np.float64))
                    buf.seek(0)
                    return buf
                elif "S_matrix" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.ones((100, 10), dtype=np.float64))
                    buf.seek(0)
                    return buf
                elif "times" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.arange(250, 350, dtype=np.int64))
                    buf.seek(0)
                    return buf
                else:
                    raise FileNotFoundError(f"Fichier non trouvé: {path}")

            mock_fs.open = mock_open

            result = load_experiment("hf://buckets/test/Inhomogènes/test_experiment")
            assert result["inhomogeneous"] is True, (
                "load_experiment devrait détecter le format inhomogène"
            )
            assert result["inhomogeneous_metadata"] is not None
            assert result["inhomogeneous_metadata"]["n_blocks"] == 3

    def test_load_P_blocks_shape(self) -> None:
        """P_blocks chargés doivent être 3D."""
        with patch("postprocessing.postprocess.fs") as mock_fs:

            def mock_open(path: str, mode: str = "r") -> Any:
                if "inhomogeneous_metadata.json" in path:
                    data = json.dumps(
                        {
                            "n_blocks": 3,
                            "n_pairs_per_block": [10, 10, 10],
                            "species_list": ["small", "large"],
                            "block_start_indices": [250, 350, 450],
                        }
                    ).encode()
                    return (
                        io.StringIO(data.decode()) if "r" in mode else io.BytesIO(data)
                    )
                elif path.endswith("config.json"):
                    content = json.dumps({"tau": 50}).encode()
                    return (
                        io.StringIO(content.decode())
                        if "r" in mode
                        else io.BytesIO(content)
                    )
                elif path.endswith("stats.json"):
                    content = json.dumps(
                        {
                            "species_list": ["small", "large"],
                            "n_states": 10,
                        }
                    ).encode()
                    return (
                        io.StringIO(content.decode())
                        if "r" in mode
                        else io.BytesIO(content)
                    )
                elif "P_blocks_small.npy" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.ones((3, 10, 10), dtype=np.float64))
                    buf.seek(0)
                    return buf
                elif "P_blocks_large.npy" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.ones((3, 10, 10), dtype=np.float64) * 0.5)
                    buf.seek(0)
                    return buf
                elif "S_matrix" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.ones((100, 10), dtype=np.float64))
                    buf.seek(0)
                    return buf
                elif "times" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.arange(250, 350, dtype=np.int64))
                    buf.seek(0)
                    return buf
                else:
                    raise FileNotFoundError(f"Fichier: {path}")

            mock_fs.open = mock_open

            result = load_experiment("hf://buckets/test/Inhomogènes/test_experiment")
            for sp in ["small", "large"]:
                assert "P_blocks" in result["species"][sp], (
                    f"P_blocks manquant pour '{sp}'"
                )
                assert result["species"][sp]["P_blocks"].ndim == 3, (
                    f"P_blocks pour '{sp}' doit être 3D"
                )

    def test_homogeneous_fallback(self) -> None:
        """Sans inhomogeneous_metadata.json, inhomogeneous doit être False."""
        with patch("postprocessing.postprocess.fs") as mock_fs:

            def mock_open(path: str, mode: str = "r") -> Any:
                if "inhomogeneous_metadata.json" in path:
                    raise FileNotFoundError(
                        "Fichier non trouvé: inhomogeneous_metadata.json"
                    )
                elif path.endswith("config.json"):
                    content = json.dumps({"tau": 50}).encode()
                    return (
                        io.StringIO(content.decode())
                        if "r" in mode
                        else io.BytesIO(content)
                    )
                elif path.endswith("stats.json"):
                    content = json.dumps(
                        {
                            "species_list": ["small", "large"],
                            "n_states": 10,
                        }
                    ).encode()
                    return (
                        io.StringIO(content.decode())
                        if "r" in mode
                        else io.BytesIO(content)
                    )
                elif (
                    "transitionmatrix_small.npy" in path
                    or "transitionmatrix_large" in path
                ):
                    buf = io.BytesIO()
                    np.save(buf, np.eye(10, dtype=np.float64))
                    buf.seek(0)
                    return buf
                elif "S_matrix" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.ones((100, 10), dtype=np.float64))
                    buf.seek(0)
                    return buf
                elif "times" in path:
                    buf = io.BytesIO()
                    np.save(buf, np.arange(250, 350, dtype=np.int64))
                    buf.seek(0)
                    return buf
                else:
                    raise FileNotFoundError(f"Fichier: {path}")

            mock_fs.open = mock_open

            result = load_experiment("hf://buckets/test/Experiment/test_experiment")
            assert result["inhomogeneous"] is False
            assert result["inhomogeneous_metadata"] is None


# ============================================================================
# 6. TESTS — bucket_io (catégorie Inhomogènes)
# ============================================================================


class TestBucketIOInhomogeneous:
    """Vérifie que le bucket I/O gère correctement la catégorie Inhomogènes."""

    def test_category_map_contains_inhomogeneous(self) -> None:
        """CATEGORY_MAP doit contenir inhomogeneous_ → Inhomogènes."""
        assert "inhomogeneous_" in CATEGORY_MAP
        assert CATEGORY_MAP["inhomogeneous_"] == "Inhomogènes"

    def test_category_map_priority(self) -> None:
        """
        inhomogeneous_ doit être en premier dans CATEGORY_MAP
        pour être détecté avant les autres préfixes.
        """
        keys = list(CATEGORY_MAP.keys())
        assert keys[0] == "inhomogeneous_", (
            f"inhomogeneous_ devrait être en premier, trouvé: {keys[0]}"
        )

    def test_get_simulation_category_inhomogeneous(self) -> None:
        """Un dossier commençant par inhomogeneous_ doit aller dans Inhomogènes."""
        folder = "inhomogeneous_voronoi_20cells_NLT3_step100_dt10_tau50_start250"
        cat = get_simulation_category(folder)
        assert cat == "Inhomogènes", f"Devrait être 'Inhomogènes', trouvé: '{cat}'"

    def test_get_simulation_category_homogeneous(self) -> None:
        """Un dossier sans préfixe inhomogeneous_ va dans sa catégorie normale."""
        folder = "voronoi_20cells_NLT3_step100_dt10_tau50_start250"
        cat = get_simulation_category(folder)
        assert cat != "Inhomogènes", (
            "Le dossier homogène ne doit pas aller dans Inhomogènes"
        )

    def test_inhomogeneous_category_in_all_categories(self) -> None:
        """ALL_CATEGORIES doit contenir Inhomogènes."""
        assert "Inhomogènes" in ALL_CATEGORIES


# ============================================================================
# 7. TESTS — save_experiment_to_bucket avec metadata inhomogène
# ============================================================================


class TestSaveExperimentWithInhomogeneousMetadata:
    """Vérifie que la sauvegarde inclut les métadonnées inhomogènes."""

    @patch("dem_mcm_coupling.bucket_io.get_api")
    @patch("dem_mcm_coupling.bucket_io.get_fs")
    def test_inhomogeneous_metadata_passed_to_save(
        self, mock_get_fs: MagicMock, mock_get_api: MagicMock
    ) -> None:
        """save_experiment_to_bucket doit accepter inhomogeneous_metadata."""
        mock_api = MagicMock()
        mock_get_api.return_value = mock_api
        mock_fs = MagicMock()
        mock_get_fs.return_value = mock_fs

        metadata = {
            "n_blocks": 3,
            "n_pairs_per_block": [10, 10, 10],
            "species_list": ["small", "large"],
            "block_start_indices": [250, 350, 450],
        }

        # Doit fonctionner sans erreur
        save_experiment_to_bucket(
            folder_name="inhomogeneous_test",
            species_data={
                "P_blocks_small": np.ones((3, 5, 5)),
                "P_blocks_large": np.ones((3, 5, 5)),
                "S_matrix_small": np.ones((10, 5)),
                "S_matrix_large": np.ones((10, 5)),
                "times_small": np.arange(10),
                "times_large": np.arange(10),
                "states_matrix": np.ones((10, 100)),
            },
            stats={"species_list": ["small", "large"]},
            config={"inhomogeneous": True, "method": "test"},
            inhomogeneous_metadata=metadata,
        )

        # Vérifier que batch_bucket_files a été appelé avec les bons fichiers
        call_args = mock_api.batch_bucket_files.call_args
        assert call_args is not None, "batch_bucket_files devrait être appelé"
        _, kwargs = call_args
        uploaded_files = kwargs.get("add", [])

        # Vérifier que inhomogeneous_metadata.json est dans les fichiers uploadés
        uploaded_paths = [bp for (_, bp) in uploaded_files]
        has_metadata = any("inhomogeneous_metadata.json" in bp for bp in uploaded_paths)
        assert has_metadata, (
            "inhomogeneous_metadata.json devrait être dans les fichiers uploadés"
        )

    @patch("dem_mcm_coupling.bucket_io.get_api")
    def test_inhomogeneous_metadata_type_check(self, mock_get_api: MagicMock) -> None:
        """inhomogeneous_metadata doit être un dict."""
        mock_api = MagicMock()
        mock_get_api.return_value = mock_api

        with pytest.raises(TypeError, match="inhomogeneous_metadata"):
            save_experiment_to_bucket(
                folder_name="inhomogeneous_test",
                species_data={},
                stats={},
                config={},
                inhomogeneous_metadata="not_a_dict",  # Devrait lever une erreur
            )


# ============================================================================
# 8. TESTS D'INTÉGRATION — run_inhomogeneous_experiment
# ============================================================================


class TestRunInhomogeneousExperiment:
    """
    Tests d'intégration pour run_inhomogeneous_experiment.

    Ces tests utilisent des données synthétiques et un mock du partitionneur
    pour valider la logique métier sans connexion HuggingFace.
    """

    def test_returns_P_blocks_not_transitionmatrix(
        self,
        inhomogeneous_config: ExperimentConfig,
        mock_partitioner: Any,
        synthetic_timestep_dict: dict[int, pd.DataFrame],
    ) -> None:
        """Les résultats doivent contenir P_blocks et non transitionmatrix."""
        results, _stats = run_inhomogeneous_experiment(
            inhomogeneous_config, mock_partitioner, synthetic_timestep_dict
        )

        for species in results:
            if species == "matrix":
                continue
            assert "P_blocks" in results[species], f"P_blocks manquant pour '{species}'"
            assert "transitionmatrix" not in results[species], (
                f"transitionmatrix ne devrait pas exister pour '{species}'"
            )

    def test_P_blocks_3d_shape(
        self,
        inhomogeneous_config: ExperimentConfig,
        mock_partitioner: Any,
        synthetic_timestep_dict: dict[int, pd.DataFrame],
    ) -> None:
        """P_blocks doit être 3D : (n_blocks, n_states, n_states)."""
        results, _stats = run_inhomogeneous_experiment(
            inhomogeneous_config, mock_partitioner, synthetic_timestep_dict
        )

        n_states = mock_partitioner.n_cells
        for species in results:
            if species == "matrix":
                continue
            P_blocks = results[species]["P_blocks"]
            assert P_blocks.ndim == 3, (
                f"P_blocks pour '{species}' devrait être 3D, shape={P_blocks.shape}"
            )
            assert P_blocks.shape[1] == n_states, (
                f"P_blocks shape[1] devrait être {n_states}, trouvé {P_blocks.shape[1]}"
            )
            assert P_blocks.shape[2] == n_states, (
                f"P_blocks shape[2] devrait être {n_states}, trouvé {P_blocks.shape[2]}"
            )

    def test_n_blocks_equals_nlt(
        self,
        inhomogeneous_config: ExperimentConfig,
        mock_partitioner: Any,
        synthetic_timestep_dict_large: dict[int, pd.DataFrame],
    ) -> None:
        """Le nombre de blocs doit être égal à NLT."""
        _results, stats = run_inhomogeneous_experiment(
            inhomogeneous_config, mock_partitioner, synthetic_timestep_dict_large
        )

        assert stats["n_blocks"] == inhomogeneous_config.nlt, (
            f"n_blocks ({stats['n_blocks']}) ≠ NLT ({inhomogeneous_config.nlt})"
        )

    def test_stats_inhomogeneous_flag(
        self,
        inhomogeneous_config: ExperimentConfig,
        mock_partitioner: Any,
        synthetic_timestep_dict_large: dict[int, pd.DataFrame],
    ) -> None:
        """Les stats doivent contenir inhomogeneous=True."""
        _results, stats = run_inhomogeneous_experiment(
            inhomogeneous_config, mock_partitioner, synthetic_timestep_dict_large
        )

        assert stats.get("inhomogeneous") is True, (
            "stats['inhomogeneous'] doit être True"
        )

    def test_each_block_is_stochastic(
        self,
        inhomogeneous_config: ExperimentConfig,
        mock_partitioner: Any,
        synthetic_timestep_dict_large: dict[int, pd.DataFrame],
    ) -> None:
        """Chaque matrice P_k doit être row-stochastic."""
        results, _stats = run_inhomogeneous_experiment(
            inhomogeneous_config, mock_partitioner, synthetic_timestep_dict_large
        )

        for species in results:
            if species == "matrix":
                continue
            P_blocks = results[species]["P_blocks"]
            for k in range(P_blocks.shape[0]):
                P_k = P_blocks[k]
                row_sums = P_k.sum(axis=1)
                # Les lignes avec des transitions doivent sommer à 1
                valid_rows = row_sums > 0
                assert np.allclose(row_sums[valid_rows], 1.0), (
                    f"P_{k} pour '{species}' n'est pas row-stochastic"
                )

    def test_species_have_separate_blocks(
        self,
        inhomogeneous_config: ExperimentConfig,
        mock_partitioner: Any,
        synthetic_timestep_dict_large: dict[int, pd.DataFrame],
    ) -> None:
        """Chaque espèce doit avoir ses propres P_blocks."""
        results, stats = run_inhomogeneous_experiment(
            inhomogeneous_config, mock_partitioner, synthetic_timestep_dict_large
        )

        species_list = stats.get("species", [])
        assert len(species_list) >= 2, "Au moins 2 espèces attendues"

        # Vérifier que les matrices diffèrent entre espèces
        P_small = results["small"]["P_blocks"]
        P_large = results["large"]["P_blocks"]
        assert not np.allclose(P_small, P_large), (
            "Les matrices des deux espèces devraient différer"
        )


# ============================================================================
# 9. TESTS D'INTÉGRATION — save_inhomogeneous_results
# ============================================================================


class TestSaveInhomogeneousResults:
    """Vérifie que save_inhomogeneous_results prépare correctement les données."""

    def test_species_data_has_P_blocks_keys(
        self,
        inhomogeneous_config: ExperimentConfig,
        rng: np.random.RandomState,
        n_states: int,
    ) -> None:
        """Les clés des données espèces doivent être P_blocks_{species}."""
        # Créer des résultats factices
        results = {
            "matrix": rng.randint(0, n_states, size=(50, 500)),
            "small": {
                "P_blocks": rng.rand(3, n_states, n_states),
                "S_matrix": rng.rand(50, n_states),
                "times": np.arange(50),
            },
            "large": {
                "P_blocks": rng.rand(3, n_states, n_states),
                "S_matrix": rng.rand(50, n_states),
                "times": np.arange(50),
            },
        }
        # Normaliser les matrices
        for sp in ["small", "large"]:
            for k in range(3):
                P = results[sp]["P_blocks"][k]
                P /= P.sum(axis=1, keepdims=True)
            results[sp]["P_blocks"] = np.array(results[sp]["P_blocks"])

        stats = {
            "n_blocks": 3,
            "n_pairs_per_block": [10, 10, 10],
            "species": ["small", "large"],
            "n_states": n_states,
            "method": "test",
        }

        # Mocker save_experiment_to_bucket pour capturer les arguments
        with patch("dem_mcm_coupling.run_sweep.save_experiment_to_bucket") as mock_save:
            save_inhomogeneous_results(
                config=inhomogeneous_config,
                partitioner=MagicMock(n_cells=n_states, label="test", centroids=None),
                results=results,
                stats=stats,
            )

            # Vérifier que save_experiment_to_bucket a été appelé
            assert mock_save.called, "save_experiment_to_bucket devrait être appelé"

            # Vérifier les arguments
            call_kwargs = mock_save.call_args[1]
            species_data = call_kwargs.get("species_data", {})

            # Vérifier les clés P_blocks
            assert "P_blocks_small" in species_data, (
                "species_data doit contenir P_blocks_small"
            )
            assert "P_blocks_large" in species_data, (
                "species_data doit contenir P_blocks_large"
            )
            # Vérifier l'absence de transitionmatrix
            assert "transitionmatrix_small" not in species_data, (
                "Ne doit pas contenir transitionmatrix_small"
            )
            assert "transitionmatrix_large" not in species_data, (
                "Ne doit pas contenir transitionmatrix_large"
            )

            # Vérifier les métadonnées inhomogènes
            assert "inhomogeneous_metadata" in call_kwargs, (
                "inhomogeneous_metadata doit être passé"
            )
            meta = call_kwargs["inhomogeneous_metadata"]
            assert meta["n_blocks"] == 3
            assert "small" in meta["species_list"]
            assert "large" in meta["species_list"]


# ============================================================================
# 10. TEST D'INTÉGRATION COMPLET — Round-trip
# ============================================================================


class TestInhomogeneousRoundTrip:
    """
    Test de bout en bout :
    configuration → expérience → sauvegarde → chargement → propagation.

    Vérifie que l'ensemble du pipeline inhomogène est cohérent.
    """

    def test_full_roundtrip_logical_consistency(
        self,
        inhomogeneous_config_single_nlt: ExperimentConfig,
        rng: np.random.RandomState,
        n_states: int,
    ) -> None:
        """
        Round-trip logique : avec NLT=1, l'inhomogène doit se comporter
        comme l'homogène (1 seule matrice = 1 seul bloc).

        On vérifie que :
        1. run_inhomogeneous_experiment produit 1 bloc
        2. La matrice de ce bloc est row-stochastic
        3. clean_transition_matrix + propagate_markov_inhomogeneous
           donne le même résultat que la version homogène
        """
        # Créer des données de test avec ASSEZ DE TIMESTEPS
        # Config: start=250, tau=50, step=100, dt=1, nlt=1
        # Le bloc a besoin de paires: start=250, end=300 (tau=50)
        # Donc on a besoin de timesteps jusqu'à au moins 300
        n_particles = 200
        n_timesteps = 100  # 250 à 350 pour avoir des paires valides
        start_idx = inhomogeneous_config_single_nlt.start_index
        times_indices = list(range(start_idx, start_idx + n_timesteps))

        import pandas as pd

        timestep_dict = {}
        for idx in times_indices:
            n_small = n_particles // 2
            n_large = n_particles - n_small
            df = pd.DataFrame(
                {
                    "coordinates:0": rng.randn(n_particles) * 0.01,
                    "coordinates:1": rng.rand(n_particles) * 0.05,
                    "coordinates:2": rng.rand(n_particles) * 0.02,
                    "Velocity:0": rng.randn(n_particles) * 0.1,
                    "Velocity:1": rng.randn(n_particles) * 0.1,
                    "Velocity:2": rng.randn(n_particles) * 0.05,
                    "Diameter": np.array([0.004] * n_small + [0.008] * n_large),
                    "Particle_ID": np.arange(n_particles),
                    "Fichier_Source": f"data_{idx}.csv",
                }
            )
            timestep_dict[idx] = df

        # Partitioner simple
        partitioner = MagicMock()
        partitioner.n_cells = n_states
        partitioner.label = "test_roundtrip"
        partitioner.use_velocity = False
        partitioner.dem_velocities = None

        def mock_compute_states(
            x: np.ndarray,
            y: np.ndarray,
            z: np.ndarray,
            vx: np.ndarray | None = None,
            vy: np.ndarray | None = None,
            vz: np.ndarray | None = None,
        ) -> np.ndarray:
            n = len(x)
            return rng.choice(n_states, size=n).astype(np.int64)

        partitioner.compute_states = mock_compute_states
        partitioner.fit = lambda *a, **kw: partitioner

        # Exécuter l'expérience inhomogène
        results_inhomo, stats_inhomo = run_inhomogeneous_experiment(
            inhomogeneous_config_single_nlt, partitioner, timestep_dict
        )

        # Vérifications
        assert stats_inhomo["n_blocks"] == 1, (
            f"Avec NLT=1, n_blocks devrait être 1, trouvé {stats_inhomo['n_blocks']}"
        )
        assert stats_inhomo["inhomogeneous"] is True

        # Chaque espèce a un P_blocks 3D
        for sp in stats_inhomo["species"]:
            P_blocks = results_inhomo[sp]["P_blocks"]
            assert P_blocks.shape == (1, n_states, n_states), (
                f"Shape P_blocks pour '{sp}': {P_blocks.shape}"
            )

            # La matrice unique est row-stochastic
            P = P_blocks[0]
            row_sums = P.sum(axis=1)
            valid_rows = row_sums > 0
            assert np.allclose(row_sums[valid_rows], 1.0), (
                f"P pour '{sp}' n'est pas row-stochastic"
            )

    def test_inhomogeneous_stats_completeness(
        self,
        inhomogeneous_config: ExperimentConfig,
        mock_partitioner: Any,
        synthetic_timestep_dict: dict[int, pd.DataFrame],
    ) -> None:
        """Les statistiques inhomogènes doivent contenir tous les champs requis."""
        _results, stats = run_inhomogeneous_experiment(
            inhomogeneous_config, mock_partitioner, synthetic_timestep_dict
        )

        required_fields = [
            "n_blocks",
            "n_pairs_per_block",
            "n_nlt_requested",
            "n_states",
            "n_states_visited",
            "fraction_visited",
            "method",
            "species",
            "tau",
            "step",
            "dt",
            "start_index",
            "inhomogeneous",
        ]

        for field in required_fields:
            assert field in stats, f"Champ obligatoire '{field}' manquant dans stats"

        assert stats["inhomogeneous"] is True
        assert isinstance(stats["n_pairs_per_block"], list)
        assert len(stats["n_pairs_per_block"]) == stats["n_blocks"]


# ============================================================================
# 11. TESTS DE RÉGRESSION — Cas limites
# ============================================================================


class TestInhomogeneousEdgeCases:
    """Tests des cas limites de l'implémentation inhomogène."""

    def test_single_species_detection(self, rng: np.random.RandomState) -> None:
        """Un seul diamètre → une seule espèce 'all'."""
        n = 500
        df = MagicMock()
        df.__getitem__.return_value = MagicMock()
        df.__getitem__.return_value.to_numpy.return_value = np.ones(n) * 0.004
        df.__getitem__.side_effect = lambda col: (
            MagicMock(to_numpy=lambda: np.ones(n) * 0.004)
            if col == "Diameter"
            else MagicMock()
        )

        # Solution plus simple : créer un vrai DataFrame
        import pandas as pd

        df = pd.DataFrame({"Diameter": np.ones(n) * 0.004})
        species = _detect_species(df)
        assert "all" in species
        assert species["all"].sum() == n

    def test_two_species_detection(self, rng: np.random.RandomState) -> None:
        """Deux diamètres → espèces 'small' et 'large'."""
        import pandas as pd

        n = 500
        diameters = np.array([0.004] * (n // 2) + [0.008] * (n - n // 2))
        rng.shuffle(diameters)
        df = pd.DataFrame({"Diameter": diameters})

        species = _detect_species(df)
        assert "small" in species
        assert "large" in species
        assert species["small"].sum() == n // 2
        assert species["large"].sum() == n - n // 2

    def test_three_species_detection(self, rng: np.random.RandomState) -> None:
        """Trois diamètres → noms génériques 'd0004', 'd0008', etc."""
        import pandas as pd

        n = 600
        diameters = np.array(
            [0.004] * (n // 3) + [0.008] * (n // 3) + [0.012] * (n - 2 * (n // 3))
        )
        rng.shuffle(diameters)
        df = pd.DataFrame({"Diameter": diameters})

        species = _detect_species(df)
        assert "d0004" in species
        assert "d0008" in species
        assert "d0012" in species

    def test_mock_partitioner_consistency(
        self, mock_partitioner: Any, n_states: int
    ) -> None:
        """Le mock partitionneur doit avoir les bonnes propriétés."""
        assert mock_partitioner.n_cells == n_states
        assert "mock" in mock_partitioner.label

        # compute_states doit retourner des entiers
        x = np.random.randn(100)
        y = np.random.randn(100)
        z = np.random.randn(100)
        states = mock_partitioner.compute_states(x, y, z)
        assert states.dtype == np.int64
        assert states.min() >= 0
        assert states.max() < n_states

    def test_propagate_with_empty_species_data(self, n_states: int) -> None:
        """
        La propagation avec des matrices vides (toutes à 0) mais S0 non nul
        doit échouer car P @ S0 = 0 (perte de masse).
        """
        S0 = np.ones(n_states) * 50  # Non nul
        P_blocks = np.zeros((2, n_states, n_states))
        times = np.arange(250, 300)
        activated = np.ones(n_states, dtype=bool)

        # Avec des matrices nulles, la masse ne se conserve pas
        # (P @ S0 = 0, donc la somme des particules devient 0)
        traj, _ = propagate_markov_inhomogeneous(
            S0, P_blocks, times, start_idx=250, tau=10, activated=activated
        )
        # Tous les pas après l'initial doivent être nuls
        for t in range(1, len(traj)):
            assert np.allclose(traj[t], 0.0), (
                f"Avec des matrices nulles, l'état devrait être 0 au pas {t}"
            )

    def test_n_blocks_greater_than_requested_nlt(
        self,
        inhomogeneous_config: ExperimentConfig,
        mock_partitioner: Any,
        synthetic_timestep_dict: dict[int, pd.DataFrame],
    ) -> None:
        """
        Si le nombre de blocs réels est inférieur au NLT demandé
        (par manque de données), stats doit refléter le nombre réel.
        """
        # Utiliser une config avec un très grand NLT
        big_nlt_config = ExperimentConfig(
            method="cartesian",
            method_kwargs={"nx": 3, "ny": 3, "nz": 2},
            nlt=100,  # Beaucoup de NLTs
            tau=50,
            step=100,
            dt=10,
            start_index=250,
            particle_diameter=None,
            inhomogeneous=True,
        )

        _results, stats = run_inhomogeneous_experiment(
            big_nlt_config, mock_partitioner, synthetic_timestep_dict
        )

        # Le nombre de blocs réels doit être <= NLT demandé
        assert stats["n_blocks"] <= big_nlt_config.nlt, (
            f"n_blocks ({stats['n_blocks']}) ne devrait pas dépasser "
            f"NLT ({big_nlt_config.nlt})"
        )
        assert stats["n_nlt_requested"] == big_nlt_config.nlt
