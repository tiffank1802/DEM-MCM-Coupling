"""Tests of the data-source abstraction (``dem_mcm_coupling.data``).

Validates that every backend — local directory, in-memory, Hugging Face —
implements the same :class:`DataSource` interface, and that the URI factory
builds the right backend.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from dem_mcm_coupling.data import (
    DataSource,
    DataSourceError,
    DemSnapshot,
    HuggingFaceDataSource,
    InMemoryDataSource,
    LocalDataSource,
)
from dem_mcm_coupling.utils import data_source_from_uri

# ============================================================================
# FIXTURES
# ============================================================================


def _make_timestep_df(idx: int, n_particles: int = 50) -> pd.DataFrame:
    """Build a minimal DEM data frame for one timestep."""
    rng = np.random.RandomState(idx)
    n_small = n_particles // 2
    return pd.DataFrame(
        {
            "coordinates:0": rng.rand(n_particles) * 0.04,
            "coordinates:1": rng.rand(n_particles) * 0.05,
            "coordinates:2": rng.rand(n_particles) * 0.02,
            "Velocity:0": rng.randn(n_particles) * 0.1,
            "Velocity:1": rng.randn(n_particles) * 0.1,
            "Velocity:2": rng.randn(n_particles) * 0.05,
            "Diameter": np.array([0.004] * n_small + [0.008] * (n_particles - n_small)),
            "Particle_ID": np.arange(n_particles),
            "Fichier_Source": f"data_{idx}.csv",
        }
    )


@pytest.fixture
def timestep_dict() -> dict[int, pd.DataFrame]:
    return {idx: _make_timestep_df(idx) for idx in (250, 300, 350)}


@pytest.fixture
def experiment_payload() -> dict:
    """Minimal homogeneous experiment matching the bucket format."""
    rng = np.random.RandomState(0)
    P = rng.rand(4, 4)
    P /= P.sum(axis=1, keepdims=True)
    return {
        "stats": {"species_list": ["small", "large"], "particle_diameter": None},
        "config": {"method": "voronoi", "nlt": 2},
        "species": {
            "small": {
                "P": P,
                "S_matrix": np.ones((3, 4)),
                "times": np.array([250, 300, 350]),
            },
            "large": {
                "P": P,
                "S_matrix": np.ones((3, 4)) * 2,
                "times": np.array([250, 300, 350]),
            },
        },
    }


# ============================================================================
# INTERFACE CONSISTENCY
# ============================================================================


def test_all_backends_implement_datasource() -> None:
    """Every concrete backend must subclass DataSource."""
    for cls in (HuggingFaceDataSource, LocalDataSource, InMemoryDataSource):
        assert issubclass(cls, DataSource)


def test_data_source_from_uri(tmp_path) -> None:
    """The URI factory must map schemes to the right backend."""
    assert isinstance(data_source_from_uri("memory://"), InMemoryDataSource)
    assert isinstance(
        data_source_from_uri("hf://ktongue/DEM_MCM"), HuggingFaceDataSource
    )
    assert isinstance(data_source_from_uri(str(tmp_path)), LocalDataSource)


# ============================================================================
# IN-MEMORY SOURCE
# ============================================================================


class TestInMemoryDataSource:
    def test_read_timesteps_roundtrip(
        self, timestep_dict: dict[int, pd.DataFrame]
    ) -> None:
        source = InMemoryDataSource(timesteps=timestep_dict)
        loaded = source.read_timesteps()
        assert list(loaded) == [250, 300, 350]
        assert loaded[250].equals(timestep_dict[250])

    def test_read_timesteps_subset(
        self, timestep_dict: dict[int, pd.DataFrame]
    ) -> None:
        source = InMemoryDataSource(timesteps=timestep_dict)
        loaded = source.read_timesteps([250, 999])
        assert list(loaded) == [250]

    def test_write_read_experiment(self, experiment_payload: dict) -> None:
        source = InMemoryDataSource()
        source.add_experiment("voronoi_test", experiment_payload)
        assert source.list_experiments() == ["voronoi_test"]

        loaded = source.read_experiment("voronoi_test")
        assert loaded["stats"]["species_list"] == ["small", "large"]
        assert np.allclose(
            loaded["species"]["small"]["P"], experiment_payload["species"]["small"]["P"]
        )

    def test_read_missing_experiment_raises(self) -> None:
        source = InMemoryDataSource()
        with pytest.raises(DataSourceError, match="Unknown experiment"):
            source.read_experiment("missing")

    def test_write_experiment_builds_payload(self) -> None:
        source = InMemoryDataSource()
        source.write_experiment(
            folder_name="inh_test",
            stats={"n_blocks": 2},
            config={"method": "voronoi"},
            species_data={"P_blocks_small": np.ones((2, 3, 3))},
            inhomogeneous_metadata={"n_blocks": 2},
        )
        loaded = source.read_experiment("inh_test")
        assert loaded["inhomogeneous"] is True
        assert loaded["inhomogeneous_metadata"] == {"n_blocks": 2}
        assert loaded["species"]["P_blocks_small"].shape == (2, 3, 3)

    def test_snapshot_helper(self, timestep_dict: dict[int, pd.DataFrame]) -> None:
        source = InMemoryDataSource(timesteps=timestep_dict)
        snap = source.snapshot(300)
        assert isinstance(snap, DemSnapshot)
        assert snap.timestep == 300
        assert snap.coordinates.shape == (50, 3)
        assert snap.velocities.shape == (50, 3)
        assert snap.diameters.shape == (50,)

    def test_snapshot_missing_raises(
        self, timestep_dict: dict[int, pd.DataFrame]
    ) -> None:
        source = InMemoryDataSource(timesteps=timestep_dict)
        with pytest.raises(DataSourceError, match="not found"):
            source.snapshot(42)

    def test_sample_coordinates_stacks(
        self, timestep_dict: dict[int, pd.DataFrame]
    ) -> None:
        source = InMemoryDataSource(timesteps=timestep_dict)
        coords, velocities, diameters = source.sample_coordinates()
        assert coords.shape == (150, 3)
        assert velocities.shape == (150, 3)
        assert diameters.shape == (150,)


# ============================================================================
# LOCAL SOURCE
# ============================================================================


class TestLocalDataSource:
    def test_read_timesteps_from_parquet(
        self, tmp_path, timestep_dict: dict[int, pd.DataFrame]
    ) -> None:
        df_full = pd.concat(timestep_dict.values(), ignore_index=True)
        df_full.to_parquet(tmp_path / "simulation_complete.parquet")

        source = LocalDataSource(tmp_path)
        loaded = source.read_timesteps()
        assert list(loaded) == [250, 300, 350]

    def test_missing_parquet_raises(self, tmp_path) -> None:
        source = LocalDataSource(tmp_path)
        with pytest.raises(DataSourceError, match="Parquet file not found"):
            source.read_timesteps()

    def test_write_read_experiment(self, tmp_path, experiment_payload: dict) -> None:
        source = LocalDataSource(tmp_path)
        species_data = {}
        for sp in ("small", "large"):
            species_data[f"transitionmatrix_{sp}"] = experiment_payload["species"][sp][
                "P"
            ]
            species_data[f"S_matrix_{sp}"] = experiment_payload["species"][sp][
                "S_matrix"
            ]
            species_data[f"times_{sp}"] = experiment_payload["species"][sp]["times"]
        source.write_experiment(
            folder_name="voronoi_local",
            stats=experiment_payload["stats"],
            config=experiment_payload["config"],
            species_data=species_data,
        )
        assert source.list_experiments() == ["voronoi_local"]

        loaded = source.read_experiment("voronoi_local")
        assert loaded["stats"] == experiment_payload["stats"]
        assert np.allclose(
            loaded["species"]["small"]["P"],
            experiment_payload["species"]["small"]["P"],
        )

    def test_read_missing_experiment_raises(self, tmp_path) -> None:
        source = LocalDataSource(tmp_path)
        with pytest.raises(DataSourceError, match="Experiment folder not found"):
            source.read_experiment("missing")


# ============================================================================
# HUGGING FACE SOURCE
# ============================================================================


class TestHuggingFaceDataSource:
    def test_parquet_path_follows_prefix(self) -> None:
        source = HuggingFaceDataSource(particle_diameter=0.004)
        assert source.parquet_path.endswith("_Good/SMALL/simulation_complete.parquet")

    def test_read_timesteps_delegates_to_bucket_io(
        self, timestep_dict: dict[int, pd.DataFrame]
    ) -> None:
        with (
            patch(
                "dem_mcm_coupling.data.huggingface.load_parquet_as_timestep_dict",
                return_value=timestep_dict,
            ) as mock_load,
            patch("dem_mcm_coupling.data.huggingface.bucket_io.get_fs") as mock_fs,
        ):
            source = HuggingFaceDataSource()
            loaded = source.read_timesteps([250, 300])
            assert list(loaded) == [250, 300]
            mock_load.assert_called_once_with(
                parquet_path=source.parquet_path, fs=mock_fs.return_value
            )

    def test_read_timesteps_wraps_errors(self) -> None:
        with (
            patch(
                "dem_mcm_coupling.data.huggingface.load_parquet_as_timestep_dict",
                side_effect=OSError("connection lost"),
            ),
            patch("dem_mcm_coupling.data.huggingface.bucket_io.get_fs"),
        ):
            source = HuggingFaceDataSource()
            with pytest.raises(DataSourceError, match="connection lost"):
                source.read_timesteps()

    def test_read_experiment_delegates_to_bucket_io(
        self, experiment_payload: dict
    ) -> None:
        with patch(
            "dem_mcm_coupling.data.huggingface.bucket_io.load_experiment_from_bucket",
            return_value=experiment_payload,
        ) as mock_load:
            source = HuggingFaceDataSource()
            loaded = source.read_experiment("voronoi_test")
            assert loaded is experiment_payload
            mock_load.assert_called_once_with(
                "voronoi_test", bucket_prefix=source.prefix
            )

    def test_read_experiment_wraps_file_not_found(self) -> None:
        with patch(
            "dem_mcm_coupling.data.huggingface.bucket_io.load_experiment_from_bucket",
            side_effect=FileNotFoundError("missing"),
        ):
            source = HuggingFaceDataSource()
            with pytest.raises(DataSourceError, match="missing"):
                source.read_experiment("missing")

    def test_write_experiment_delegates_to_bucket_io(self) -> None:
        with patch(
            "dem_mcm_coupling.data.huggingface.bucket_io.save_experiment_to_bucket"
        ) as mock_save:
            source = HuggingFaceDataSource()
            source.write_experiment(
                folder_name="voronoi_test",
                stats={"particle_diameter": 0.004, "species_list": ["small"]},
                config={"method": "voronoi"},
                species_data={"transitionmatrix_small": np.eye(3)},
            )
            mock_save.assert_called_once()
            kwargs = mock_save.call_args.kwargs
            assert kwargs["folder_name"] == "voronoi_test"
            assert kwargs["particle_diameter"] == 0.004
