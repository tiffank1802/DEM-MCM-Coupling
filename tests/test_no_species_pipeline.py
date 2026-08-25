"""Tests of the no-species pipeline (no particle-size mask).

The initial model assumption: large and small particles share the same
kinetics. The no-species pipeline builds ONE state vector and ONE transition
matrix from the whole particle population (no diameter mask), saves them
under the ``nospecies_`` prefix (routed to the ``nospecies_simulations/``
bucket folder), and the postprocessing compares the masked vs unmasked
predictions to quantify the influence of the species distinction.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from dem_mcm_coupling.bucket_io import (
    ALL_CATEGORIES,
    CATEGORY_MAP,
    get_simulation_category,
)
from dem_mcm_coupling.run_sweep import (
    ExperimentConfig,
    run_experiment,
    run_no_species_experiment,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def n_states() -> int:
    return 10


@pytest.fixture
def rng() -> np.random.RandomState:
    return np.random.RandomState(42)


def _make_timestep_dict(
    rng: np.random.RandomState, n_timesteps: int = 60, two_species: bool = True
) -> dict[int, pd.DataFrame]:
    """Synthetic DEM data (n_timesteps, 100 particles, 2 diameters)."""
    n_particles = 100
    timestep_dict: dict[int, pd.DataFrame] = {}
    for i in range(n_timesteps):
        idx = 250 + i
        if two_species:
            diameters = np.array([0.004] * 50 + [0.008] * 50)
        else:
            diameters = np.full(n_particles, 0.004)
        df = pd.DataFrame(
            {
                "coordinates:0": rng.normal(0.02, 0.01, n_particles),
                "coordinates:1": rng.uniform(0, 0.05, n_particles),
                "coordinates:2": rng.uniform(0, 0.02, n_particles),
                "Velocity:0": rng.normal(0, 0.1, n_particles),
                "Velocity:1": rng.normal(0, 0.1, n_particles),
                "Velocity:2": rng.normal(0, 0.05, n_particles),
                "Diameter": diameters,
                "Particle_ID": np.arange(n_particles),
                "Fichier_Source": f"data_{idx}.csv",
            }
        )
        timestep_dict[idx] = df
    return timestep_dict


def _make_partitioner(rng: np.random.RandomState, n_states: int) -> MagicMock:
    """MagicMock partitioner assigning random states."""
    partitioner = MagicMock()
    partitioner.n_cells = n_states
    partitioner.label = "cartesian_nx3_ny3_nz2"
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
        return rng.choice(n_states, size=len(x)).astype(np.int64)

    partitioner.compute_states = mock_compute_states
    partitioner.fit = lambda *a, **kw: partitioner
    return partitioner


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig(
        method="cartesian",
        method_kwargs={"nx": 3, "ny": 3, "nz": 2},
        nlt=1,
        tau=10,
        step=20,
        dt=1,
        start_index=250,
        particle_diameter=None,
        inhomogeneous=False,
    )


# ============================================================================
# BUCKET ROUTING OF THE NEW FOLDER
# ============================================================================


class TestNoSpeciesBucketCategory:
    def test_category_map_entry(self) -> None:
        assert CATEGORY_MAP["nospecies_"] == "nospecies_simulations"

    def test_inhomogeneous_remains_first(self) -> None:
        # The most specific prefix must stay first (ordering contract).
        assert next(iter(CATEGORY_MAP)) == "inhomogeneous_"

    def test_category_listed(self) -> None:
        assert "nospecies_simulations" in ALL_CATEGORIES

    def test_get_simulation_category(self) -> None:
        folder = "nospecies_voronoi_10cells_NLT2_step20_dt1_tau50_start250"
        assert get_simulation_category(folder) == "nospecies_simulations"

    def test_masked_folders_unaffected(self) -> None:
        folder = "voronoi_10cells_NLT2_step20_dt1_tau50_start250"
        assert get_simulation_category(folder) == "voronoi_simulations"


# ============================================================================
# METHOD DETECTION OF THE NOSPECIES FOLDERS
# ============================================================================


class TestNoSpeciesMethodDetection:
    def test_detect_method_strips_prefix(self) -> None:
        from dem_mcm_coupling.analyze_results import MarkovAnalyzer

        analyzer = MarkovAnalyzer.__new__(MarkovAnalyzer)
        folder = "nospecies_voronoi_10cells_NLT2_step20_dt1_tau50_start250"
        assert analyzer._detect_method(folder) == "voronoi"

    def test_detect_method_masked_unchanged(self) -> None:
        from dem_mcm_coupling.analyze_results import MarkovAnalyzer

        analyzer = MarkovAnalyzer.__new__(MarkovAnalyzer)
        folder = "voronoi_10cells_NLT2_step20_dt1_tau50_start250"
        assert analyzer._detect_method(folder) == "voronoi"


# ============================================================================
# THE NO-SPECIES EXPERIMENT
# ============================================================================


class TestRunNoSpeciesExperiment:
    def test_single_all_species(
        self, rng: np.random.RandomState, config: ExperimentConfig, n_states: int
    ) -> None:
        timestep_dict = _make_timestep_dict(rng)
        partitioner = _make_partitioner(rng, n_states)

        results, stats = run_no_species_experiment(config, partitioner, timestep_dict)

        # Une seule espèce "all" : aucune distinction de taille.
        assert set(results) == {"matrix", "all"}
        assert stats["species"] == ["all"]
        assert stats["species_masks_applied"] is False

        P = results["all"]["P"]
        S = results["all"]["S_matrix"]
        assert P.shape == (n_states, n_states)
        assert S.shape == (len(timestep_dict), n_states)

        # Matrice stochastique en lignes.
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums[row_sums > 0], 1.0)

        # Le vecteur d'état compte TOUTES les particules (aucun masque).
        np.testing.assert_allclose(S.sum(axis=1), 100.0)

    def test_equals_masked_single_diameter(
        self, rng: np.random.RandomState, config: ExperimentConfig, n_states: int
    ) -> None:
        """Sans distinction de diamètre dans les données, les deux pipelines
        (avec et sans masque) doivent produire exactement la même matrice."""
        timestep_dict = _make_timestep_dict(rng, two_species=False)

        partitioner_masked = _make_partitioner(np.random.RandomState(0), n_states)
        partitioner_nospecies = _make_partitioner(np.random.RandomState(0), n_states)

        results_masked, stats_masked = run_experiment(
            config, partitioner_masked, timestep_dict
        )
        results_nospecies, _stats_ns = run_no_species_experiment(
            config, partitioner_nospecies, timestep_dict
        )

        # Un seul diamètre → _detect_species retourne "all" : même expérience.
        assert stats_masked["species"] == ["all"]
        assert stats_masked["species_masks_applied"] is True
        np.testing.assert_allclose(
            results_masked["all"]["P"], results_nospecies["all"]["P"]
        )
        np.testing.assert_allclose(
            results_masked["all"]["S_matrix"], results_nospecies["all"]["S_matrix"]
        )

    def test_nospecies_differs_from_species_masks(
        self, rng: np.random.RandomState, config: ExperimentConfig, n_states: int
    ) -> None:
        """Avec deux diamètres, la matrice sans masque diffère des matrices
        par espèce : c'est précisément l'influence de la distinction."""
        timestep_dict = _make_timestep_dict(rng)

        partitioner_masked = _make_partitioner(np.random.RandomState(0), n_states)
        partitioner_nospecies = _make_partitioner(np.random.RandomState(0), n_states)

        results_masked, _ = run_experiment(config, partitioner_masked, timestep_dict)
        results_nospecies, _ = run_no_species_experiment(
            config, partitioner_nospecies, timestep_dict
        )

        assert set(results_masked) == {"matrix", "small", "large"}
        P_small = results_masked["small"]["P"]
        P_large = results_masked["large"]["P"]
        P_all = results_nospecies["all"]["P"]

        assert not np.allclose(P_small, P_large)
        assert not np.allclose(P_all, P_small)
        assert not np.allclose(P_all, P_large)

        # La matrice sans masque est la moyenne pondérée par les effectifs :
        # P_all ≈ (n_small * P_small + n_large * P_large) / n_total.
        n_small = 50
        n_large = 50
        P_weighted = (n_small * P_small + n_large * P_large) / (n_small + n_large)
        # L'écart résiduel vient du fait que les matrices par espèce sont
        # normalisées par ligne séparément.
        assert np.abs(P_all - P_weighted).max() < 0.5

    def test_missing_start_raises(
        self, rng: np.random.RandomState, config: ExperimentConfig, n_states: int
    ) -> None:
        timestep_dict = _make_timestep_dict(rng)
        partitioner = _make_partitioner(rng, n_states)
        config.start_index = 9999
        with pytest.raises(KeyError, match="absent from the data"):
            run_no_species_experiment(config, partitioner, timestep_dict)
