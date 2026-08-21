"""Tests of the scientific figures and the global colour code.

Validates that:

* every figure function returns a matplotlib figure and saves its PNG;
* the colour code is respected: a partitioning method always maps to the
  same colour, species keep their DEM/Markov tones;
* time axes are expressed in seconds.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from postprocessing import figures, style
from postprocessing.metrics import propagate_markov

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def species_data() -> dict[str, dict]:
    """Prepared per-species data of a small 2-species experiment."""
    rng = np.random.RandomState(0)
    n_states, _, n_dem = 8, 100, 20

    P = rng.rand(n_states, n_states) * 0.02
    P[np.arange(n_states), np.arange(n_states)] += 1.0
    P /= P.sum(axis=1, keepdims=True)

    times = np.arange(250, 250 + n_dem)
    activated = np.ones(n_states, dtype=bool)

    S_small = np.zeros((n_dem, n_states))
    S_large = np.zeros((n_dem, n_states))
    S_small[0, :4] = 20.0
    S_large[0, 4:] = 20.0
    for t in range(1, n_dem):
        S_small[t] = S_small[t - 1] @ P
        S_large[t] = S_large[t - 1] @ P

    def _prepare(S: np.ndarray) -> dict:
        traj, times_markov = propagate_markov(S[0], P, times, 250, 2, activated)
        return {
            "P": P,
            "P_raw": P,
            "S_matrix": S,
            "times": times,
            "S_dem": S,
            "times_dem": times,
            "traj_markov": traj,
            "times_markov": times_markov,
            "activated": activated,
        }

    return {"small": _prepare(S_small), "large": _prepare(S_large)}


# ============================================================================
# COLOUR CODE
# ============================================================================


class TestColorCode:
    def test_method_colors_are_stable(self) -> None:
        """The same method must always map to the same colour."""
        for name in [
            "voronoi_125cells_NLT3_step100_dt1_tau50_start157",
            "voronoi_10cells_NLT2_step20_dt2_tau100_start157",
            "cartesian_nx3_ny3_nz3_NLT3_step100_dt1_tau50_start157",
            "cylindrical_nr5_nth8_nz5_equal_area_NLT2_step20_dt2_tau100_start157",
        ]:
            assert (
                style.method_color(name) == style.METHOD_COLORS[style.method_of(name)]
            )

    def test_distinct_methods_have_distinct_colors(self) -> None:
        assert style.method_color(
            "voronoi_10cells_NLT2_step20_dt2_tau100_start157"
        ) != style.method_color("cartesian_nx3_ny3_nz3_NLT3_step100_dt1_tau50_start157")

    def test_unknown_method_fallback(self) -> None:
        assert style.method_color("mystery_method_1cells") == style.UNKNOWN_METHOD_COLOR

    def test_species_colors(self) -> None:
        assert (
            style.species_colors("small")["dem"] == style.SPECIES_COLORS["small"]["dem"]
        )
        assert (
            style.species_colors("large")["markov"]
            == style.SPECIES_COLORS["large"]["markov"]
        )

    def test_seconds_conversion(self) -> None:
        times = np.array([250, 300, 350])
        np.testing.assert_allclose(style.timesteps_to_seconds(times), [2.5, 3.0, 3.5])


# ============================================================================
# FIGURES
# ============================================================================


class TestFigures:
    def test_rsd_scientific(self, species_data: dict[str, dict], tmp_path) -> None:
        fig, axes = figures.fig_rsd_scientific(
            species_data, "voronoi_8cells_NLT2_step20_dt2_tau50_start250", tmp_path
        )
        assert fig is not None
        assert axes.shape == (1, 2)
        assert (tmp_path / "scientific_rsd.png").exists()

        # DEM line of the small species keeps the DEM blue tone.
        dem_line = axes[0][0].get_lines()[0]
        assert dem_line.get_color() == style.SPECIES_COLORS["small"]["dem"]

    def test_transition_matrix_scientific(
        self, species_data: dict[str, dict], tmp_path
    ) -> None:
        fig, ax = figures.fig_transition_matrix_scientific(
            "small",
            species_data["small"],
            "voronoi_8cells_NLT2_step20_dt2_tau50_start250",
            tmp_path,
        )
        assert fig is not None
        assert (tmp_path / "scientific_matrix_small.png").exists()
        # Row convention: y axis is the source states.
        assert "source" in ax.get_ylabel().lower()
        assert "destination" in ax.get_xlabel().lower()

    def test_stationary_distribution_scientific(
        self, species_data: dict[str, dict], tmp_path
    ) -> None:
        fig, _ax = figures.fig_stationary_distribution_scientific(
            "small",
            species_data["small"],
            "voronoi_8cells_NLT2_step20_dt2_tau50_start250",
            tmp_path,
        )
        assert fig is not None
        assert (tmp_path / "scientific_stationary_small.png").exists()

    def test_concentration_scientific(
        self, species_data: dict[str, dict], tmp_path
    ) -> None:
        fig, axes = figures.fig_concentration_scientific(
            species_data, "voronoi_8cells_NLT2_step20_dt2_tau50_start250", tmp_path
        )
        assert fig is not None
        assert axes.shape == (3,)
        assert (tmp_path / "scientific_concentration.png").exists()

    def test_concentration_scientific_needs_two_species(
        self, species_data: dict[str, dict], tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="2 species"):
            figures.fig_concentration_scientific(
                {"small": species_data["small"]}, "voronoi_8cells", tmp_path
            )

    def test_compare_methods_rsd(self, species_data: dict[str, dict], tmp_path) -> None:
        all_data = {
            "voronoi_8cells_NLT2_step20_dt2_tau50_start250": species_data,
            "cartesian_nx2_ny2_nz2_NLT2_step20_dt2_tau50_start250": species_data,
        }
        fig, ax = figures.fig_compare_methods_rsd(all_data, tmp_path)
        assert fig is not None
        assert (tmp_path / "scientific_compare_methods.png").exists()

        # Each experiment line keeps the colour of its partitioning method.
        line_colors = [line.get_color() for line in ax.get_lines()]
        assert style.METHOD_COLORS["voronoi"] in line_colors
        assert style.METHOD_COLORS["cartesian"] in line_colors

    def test_compare_methods_rsd_empty_raises(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="at least one"):
            figures.fig_compare_methods_rsd({}, tmp_path)
