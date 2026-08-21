"""Tests of the post-processing pipeline functions.

Validates:

* :func:`fig_mesh` — the VTK time series carries an **evolving** particle
  state (like the positions) and the ``.pvd`` collection stores physical
  times in seconds;
* the DEM/Markov error functions are **homogenised by the particle count**
  (scale-invariant relative errors);
* :func:`fig_matrix_components_evolution` plots every ``p_ij`` component on
  a **single shared scale** and overlays a fitted interpolation law.
"""

from __future__ import annotations

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from postprocessing import postprocess as pp
from postprocessing import postprocess_inhomogeneous as ppi

# ============================================================================
# FIXTURES
# ============================================================================


def _particle_frame(t: int, n_particles: int, offset: float = 0.0) -> pd.DataFrame:
    """Build a DEM frame at timestep ``t`` with stable particle ids."""
    rng = np.random.RandomState(t + 1)
    return pd.DataFrame(
        {
            "coordinates:0": rng.uniform(-0.04, 0.04, n_particles) + offset,
            "coordinates:1": rng.uniform(-0.05, 0.05, n_particles),
            "coordinates:2": rng.uniform(-0.02, 0.25, n_particles),
            "Diameter": np.full(n_particles, 0.004),
            "Particle_ID": np.arange(n_particles),
            "Fichier_Source": f"data_{t}.csv",
        }
    )


@pytest.fixture
def mesh_experiment() -> dict:
    """Minimal experiment with a 2-timestep particle-state matrix.

    At t=250 every particle is in state 0; at t=251 every particle is in
    state 1: the state vector must therefore change between the two frames.
    """
    n_particles = 8
    times = np.array([250, 251])
    matrix = np.zeros((2, n_particles), dtype=int)
    matrix[1] = 1
    return {
        "config": {"start_index": 250, "tau": 1},
        "species": {"small": {"times": times}},
        "matrix": matrix,
        "stats": {"species_list": ["small"]},
    }


# ============================================================================
# fig_mesh — VTK TIME SERIES WITH EVOLVING STATES
# ============================================================================


class TestFigMeshVtkTimeSeries:
    def test_states_evolve_like_positions(
        self, mesh_experiment: dict, tmp_path
    ) -> None:
        out_img = tmp_path / "img"
        out_files = tmp_path / "files"
        out_img.mkdir()
        out_files.mkdir()

        df_start = _particle_frame(250, 8)
        # The particles move between the two frames.
        timestep_dict = {
            250: _particle_frame(250, 8, offset=0.0),
            251: _particle_frame(251, 8, offset=0.01),
        }

        # Keep the frame files on disk instead of zipping them, and skip the
        # off-screen rendering (no GL context in headless CI environments).
        import pyvista as pv

        with (
            patch.object(pp.shutil, "make_archive"),
            patch.object(pp.shutil, "rmtree"),
            patch.object(pv.Plotter, "screenshot"),
        ):
            pp.fig_mesh(
                mesh_experiment,
                df_start,
                "test_mesh",
                out_img,
                out_files,
                timestep_dict=timestep_dict,
                frame_stride=1,
            )

        series_dir = out_files / "_tmp_vtp_series_test_mesh"
        assert series_dir.is_dir()

        frame0 = pv.read(str(series_dir / "frame_0000.vtp"))
        frame1 = pv.read(str(series_dir / "frame_0001.vtp"))

        states0 = np.asarray(frame0.point_data["partition_state"])
        states1 = np.asarray(frame1.point_data["partition_state"])

        # The state vector is inserted in the mesh and evolves with time:
        # frame at t=250 → all state 0, frame at t=251 → all state 1.
        assert states0.max() == 0
        assert states1.min() == 1
        # The frozen reference label stays identical on every frame.
        ref0 = np.asarray(frame0.point_data["partition_label_start"])
        ref1 = np.asarray(frame1.point_data["partition_label_start"])
        np.testing.assert_array_equal(ref0, ref1)

        # Positions also move between the frames.
        pos0 = np.asarray(frame0.points)
        pos1 = np.asarray(frame1.points)
        assert not np.allclose(pos0, pos1)

    def test_pvd_contains_physical_seconds(
        self, mesh_experiment: dict, tmp_path
    ) -> None:
        out_img = tmp_path / "img"
        out_files = tmp_path / "files"
        out_img.mkdir()
        out_files.mkdir()

        import pyvista as pv

        with (
            patch.object(pp.shutil, "make_archive"),
            patch.object(pp.shutil, "rmtree"),
            patch.object(pv.Plotter, "screenshot"),
        ):
            pp.fig_mesh(
                mesh_experiment,
                _particle_frame(250, 8),
                "test_mesh",
                out_img,
                out_files,
                timestep_dict={
                    250: _particle_frame(250, 8),
                    251: _particle_frame(251, 8),
                },
                frame_stride=1,
            )

        pvd_path = out_files / "_tmp_vtp_series_test_mesh" / "series_test_mesh.pvd"
        pvd_text = pvd_path.read_text()
        # 1 timestep = 0.01 s → physical times in the collection file.
        assert 'timestep="2.50"' in pvd_text
        assert 'timestep="2.51"' in pvd_text
        assert pvd_text.count("<DataSet") == 2


# ============================================================================
# ERROR FUNCTIONS — HOMOGENISED BY THE PARTICLE COUNT
# ============================================================================


class TestErrorNormalization:
    def test_abs_error_scale_invariance(self) -> None:
        """Scaling both populations leaves the normalised error unchanged."""
        rng = np.random.RandomState(0)
        S_dem = rng.uniform(5, 30, size=(6, 4))
        S_markov = rng.uniform(5, 30, size=(6, 4))
        times_dem = np.arange(250, 256)
        times_markov = np.arange(250, 256)
        activated = np.ones(4, dtype=bool)

        _, err = pp.calculate_abs_error_over_time(
            S_dem, S_markov, times_dem, times_markov, activated, normalize=True
        )
        _, err_scaled = pp.calculate_abs_error_over_time(
            3.0 * S_dem,
            3.0 * S_markov,
            times_dem,
            times_markov,
            activated,
            normalize=True,
        )
        np.testing.assert_allclose(err, err_scaled)

    def test_abs_error_in_probability_bounds(self) -> None:
        rng = np.random.RandomState(1)
        S_dem = rng.uniform(5, 30, size=(6, 4))
        S_markov = rng.uniform(5, 30, size=(6, 4))
        times = np.arange(250, 256)
        activated = np.ones(4, dtype=bool)
        _, err = pp.calculate_abs_error_over_time(
            S_dem, S_markov, times, times, activated, normalize=True
        )
        # L1 between two probability vectors is bounded by 2.
        assert (err >= 0).all() and (err <= 2.0).all()

    def test_discrepancy_per_cell_normalized_and_scale_invariant(self) -> None:
        rng = np.random.RandomState(2)
        S_dem = rng.uniform(5, 30, size=(8, 5))
        S_markov = rng.uniform(5, 30, size=(8, 5))
        times_dem = np.arange(250, 258)
        times_markov = np.arange(250, 258)
        activated = np.ones(5, dtype=bool)

        result = pp.calculate_discrepancy_per_cell(
            S_dem, S_markov, times_dem, times_markov, activated, normalize=True
        )
        assert len(result) == 5
        _disc_cell, disc_time, _times_aligned, rmse_cell, diff_per_step = result

        # Homogenised by the particle count → fractions in [0, 1].
        assert (diff_per_step >= 0).all() and (diff_per_step <= 1.0).all()
        assert (rmse_cell >= 0).all() and (rmse_cell <= 1.0).all()
        assert (disc_time >= 0).all() and (disc_time <= 1.0).all()

        # Scaling both populations by a constant does not change the result.
        result_scaled = pp.calculate_discrepancy_per_cell(
            2.5 * S_dem,
            2.5 * S_markov,
            times_dem,
            times_markov,
            activated,
            normalize=True,
        )
        for a, b in zip(result, result_scaled):
            np.testing.assert_allclose(a, b)

    def test_discrepancy_without_normalization_is_scale_dependent(self) -> None:
        rng = np.random.RandomState(3)
        S_dem = rng.uniform(5, 30, size=(8, 5))
        S_markov = rng.uniform(5, 30, size=(8, 5))
        times = np.arange(250, 258)
        activated = np.ones(5, dtype=bool)

        raw = pp.calculate_discrepancy_per_cell(
            S_dem, S_markov, times, times, activated, normalize=False
        )
        scaled = pp.calculate_discrepancy_per_cell(
            2.0 * S_dem, 2.0 * S_markov, times, times, activated, normalize=False
        )
        assert not np.allclose(raw[4], scaled[4])


# ============================================================================
# p_ij COMPONENTS — SHARED SCALE + INTERPOLATION LAW
# ============================================================================


class TestMatrixComponentsEvolution:
    def test_shared_scale_and_fitted_law(self, tmp_path) -> None:
        P_blocks = np.zeros((3, 4, 4))
        P_blocks[:, 0, 1] = [0.30, 0.25, 0.20]
        P_blocks[:, 2, 3] = [0.50, 0.55, 0.60]
        sp_data = {"P_blocks": P_blocks}
        block_times = np.array([2.5, 3.5, 4.5])

        with patch.object(ppi.plt, "close"), patch.object(ppi.plt, "savefig"):
            ppi.fig_matrix_components_evolution(
                "small",
                sp_data,
                "inhomogeneous_test_NLT3",
                tmp_path,
                block_times=block_times,
            )

        fig = plt.gcf()
        axes = fig.get_axes()
        assert len(axes) >= 2, "two significant components expected"

        # Échelle commune : tous les sous-graphiques partagent le même ylim.
        ylims = [tuple(ax.get_ylim()) for ax in axes]
        assert all(ylims[0] == y for y in ylims), f"scales differ: {ylims}"

        # Une loi d'interpolation est superposée sur chaque composante.
        legend_labels = [t.get_text() for t in axes[0].get_legend().get_texts()]
        assert any("Ajustement degré" in label for label in legend_labels)
        assert any("mesuré" in label for label in legend_labels)

        # Le titre de chaque sous-figure explicite le degré du fit.
        assert "fit degré" in axes[0].get_title()
        assert axes[0].get_xlabel() == "Temps (s)"

    def test_single_significant_component(self, tmp_path) -> None:
        P_blocks = np.zeros((3, 4, 4))
        P_blocks[:, 1, 2] = [0.1, 0.15, 0.2]
        sp_data = {"P_blocks": P_blocks}

        with patch.object(ppi.plt, "close"), patch.object(ppi.plt, "savefig"):
            ppi.fig_matrix_components_evolution(
                "small", sp_data, "inhomogeneous_test_NLT3", tmp_path
            )
        fig = plt.gcf()
        assert len(fig.get_axes()) == 1

    def test_no_significant_component_skipped(self, tmp_path) -> None:
        P_blocks = np.zeros((3, 4, 4))
        with patch.object(ppi.plt, "close"):
            result = ppi.fig_matrix_components_evolution(
                "small", {"P_blocks": P_blocks}, "inhomogeneous_test_NLT3", tmp_path
            )
        assert result is None
