"""Tests of the :mod:`dem_mcm_coupling.markov_core` model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dem_mcm_coupling._config import StateVectorData
from dem_mcm_coupling.data import InMemoryDataSource
from dem_mcm_coupling.markov_core import Markov


def _make_timestep_df(idx: int, n_particles: int = 100) -> pd.DataFrame:
    rng = np.random.RandomState(idx)
    return pd.DataFrame(
        {
            "coordinates:0": rng.uniform(-0.04, 0.04, n_particles),
            "coordinates:1": rng.uniform(-0.05, 0.05, n_particles),
            "coordinates:2": rng.uniform(-0.02, 0.25, n_particles),
            "Velocity:0": rng.randn(n_particles) * 0.1,
            "Velocity:1": rng.randn(n_particles) * 0.1,
            "Velocity:2": rng.randn(n_particles) * 0.05,
            "Diameter": np.full(n_particles, 0.004),
            "Particle_ID": np.arange(n_particles),
            "Fichier_Source": f"data_{idx}.csv",
        }
    )


@pytest.fixture
def timestep_dict() -> dict[int, pd.DataFrame]:
    return {idx: _make_timestep_df(idx) for idx in (250, 300, 350)}


class TestMarkovInit:
    def test_creates_partitioner(self) -> None:
        mk = Markov(method="voronoi", method_kwargs={"n_cells": 8})
        assert mk.partitioner.n_cells == 8
        assert mk.method == "voronoi"

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown partitioning method"):
            Markov(method="not_a_method")

    def test_repr(self) -> None:
        mk = Markov(method="cartesian", method_kwargs={"nx": 2, "ny": 2, "nz": 2})
        assert "Markov(" in repr(mk)


class TestMarkovPipeline:
    def test_full_workflow_with_in_memory_source(
        self, timestep_dict: dict[int, pd.DataFrame]
    ) -> None:
        source = InMemoryDataSource(timesteps=timestep_dict)
        mk = Markov(
            method="cartesian",
            method_kwargs={"nx": 2, "ny": 2, "nz": 2},
            data_source=source,
        )

        mk.load_dem_data()
        assert set(mk.datas) == {250, 300, 350}

        coords = mk.get_coords([250, 300])
        assert coords.shape == (200, 3)

        mk.fit_partitioner(coords)
        assert mk.partitioner.n_cells == 8

        state0 = mk.build_initial_state_vector(250)
        assert isinstance(state0, StateVectorData)
        assert state0.total_particles == 100
        assert np.isclose(state0.phi.sum(), 100)

    def test_get_coords_missing_timestep_raises(
        self, timestep_dict: dict[int, pd.DataFrame]
    ) -> None:
        mk = Markov(data_source=InMemoryDataSource(timesteps=timestep_dict))
        mk.load_dem_data()
        with pytest.raises(KeyError, match="Timestep 42 unavailable"):
            mk.get_coords([42])


class TestPropagation:
    def test_mass_conservation(self) -> None:
        rng = np.random.RandomState(0)
        n_states = 10
        P = rng.rand(n_states, n_states)
        P /= P.sum(axis=1, keepdims=True)  # row-stochastic

        mk = Markov(method="cartesian", method_kwargs={"nx": 2, "ny": 2, "nz": 2})
        phi0 = rng.rand(n_states) * 100
        traj = mk.propagate_markov(phi0, P, n_steps=50)

        assert traj.states.shape == (51, n_states)
        for t in range(51):
            assert np.isclose(traj.states[t].sum(), phi0.sum(), rtol=1e-6)

    def test_dimension_mismatch_raises(self) -> None:
        mk = Markov(method="cartesian")
        P = np.eye(4)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            mk.propagate_markov(np.ones(3), P, n_steps=10)


class TestStateVectorData:
    def test_validate_normalization(self) -> None:
        ok = StateVectorData(
            phi=np.array([1.0, 2.0, 3.0]), timestamp=0, total_particles=6
        )
        bad = StateVectorData(phi=np.array([1.0, 2.0]), timestamp=0, total_particles=6)
        assert ok.validate_normalization()
        assert not bad.validate_normalization()
