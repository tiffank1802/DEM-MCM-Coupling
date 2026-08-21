"""Global visual style for the post-processing figures.

Centralises the **colour code** used across every figure so that a given
partitioning method (voronoi, cartesian, cylindrical, ...) always keeps the
same colour, and a given particle species always keeps the same
colour/style, throughout the whole pipeline.

Colour conventions:

* **Partitioning methods** — :data:`METHOD_COLORS` (shared with
  :mod:`dem_mcm_coupling.analyze_results`): one fixed colour per method.
* **Particle species** — :data:`SPECIES_COLORS`: blue for ``small``, orange
  for ``large``; each species has a ``dem`` tone (solid line) and a
  ``markov`` tone (dashed line with markers).
* **DEM reference curves** — dark grey, solid; they are always the physical
  reference in comparison figures.

The :func:`apply_scientific_style` rcParams give every figure a clean,
publication-oriented layout (visible grid, no top/right spines, serif-less
fonts, 300-dpi saving).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

from dem_mcm_coupling._config import TIMESTEP_TO_SECONDS
from dem_mcm_coupling.analyze_results import METHOD_COLORS

matplotlib.use("Agg")

#: Colour of the DEM reference curves in comparison figures.
DEM_REFERENCE_COLOR = "#555555"

#: Colour of the "unknown method" fallback.
UNKNOWN_METHOD_COLOR = "#7f7f7f"

#: Per-species colours: ``dem`` tone = solid DEM curve, ``markov`` tone =
#: dashed Markov curve with markers.
SPECIES_COLORS: dict[str, dict[str, str]] = {
    "small": {"dem": "#2196F3", "markov": "#E53935"},
    "large": {"dem": "#FF9800", "markov": "#43A047"},
}
DEFAULT_COLOR: dict[str, str] = {"dem": "#607D8B", "markov": "#9C27B0"}


def method_of(experiment_name: str) -> str:
    """Extract the partitioning method from an experiment folder name.

    Args:
        experiment_name: Experiment name, e.g.
            ``"voronoi_125cells_NLT3_step100_dt1_tau50_start157"``.

    Returns:
        The method name (``"unknown"`` when undetermined).
    """
    for method in METHOD_COLORS:
        if experiment_name.startswith(method):
            return method
    return "unknown"


def method_color(experiment_name: str) -> str:
    """Return the fixed colour of the partitioning method of an experiment.

    Args:
        experiment_name: Experiment folder name.

    Returns:
        The hexadecimal colour of the method (see :data:`METHOD_COLORS`).
    """
    return METHOD_COLORS.get(method_of(experiment_name), UNKNOWN_METHOD_COLOR)


def species_colors(species: str) -> dict[str, str]:
    """Return the ``{"dem", "markov"}`` colour pair of a species.

    Args:
        species: Species name (``"small"``, ``"large"``, ...).

    Returns:
        The colour pair; :data:`DEFAULT_COLOR` for unknown species.
    """
    return SPECIES_COLORS.get(species, DEFAULT_COLOR)


def apply_scientific_style() -> None:
    """Apply the global scientific matplotlib style to every new figure."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "#f8f9fa",
            "axes.grid": True,
            "grid.color": "white",
            "grid.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.framealpha": 0.9,
            "lines.linewidth": 1.6,
            "font.family": "DejaVu Sans",
        }
    )


def timesteps_to_seconds(times: Any) -> Any:
    """Convert raw DEM timestep indices into physical seconds.

    One timestep corresponds to :data:`TIMESTEP_TO_SECONDS` (0.01 s) of
    simulated time, so ``t_seconds = timestep_index * 0.01``.

    Args:
        times: Array-like of timestep indices (or centiseconds).

    Returns:
        The same shape expressed in seconds.
    """
    import numpy as np

    return np.asarray(times, dtype=float) * TIMESTEP_TO_SECONDS


def annotate_mixing_times(
    ax: Any,
    times_seconds: Any,
    rsd: Any,
    fractions: tuple[float, float] = (0.5, 0.1),
) -> None:
    """Annotate the t50/t90 mixing times of an RSD curve.

    The mixing time at fraction ``f`` is the first time where the RSD falls
    below ``f`` times its initial value (``t50``: RSD ÷ 2, ``t90``: RSD ÷
    10). Vertical dashed lines and text labels are added to the axis.

    Args:
        ax: Matplotlib axis.
        times_seconds: Time axis in seconds.
        rsd: RSD curve.
        fractions: Fractions of the initial RSD to annotate.
    """
    import numpy as np

    rsd = np.asarray(rsd, dtype=float)
    times = np.asarray(times_seconds, dtype=float)
    rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
    label = {0.5: r"$t_{50}$", 0.1: r"$t_{90}$"}

    for fraction in fractions:
        hit = np.where(rsd < fraction * rsd_0)[0]
        if len(hit) == 0:
            continue
        t_star = times[hit[0]]
        ax.axvline(t_star, color=DEM_REFERENCE_COLOR, lw=1.0, ls="--", alpha=0.7)
        ax.annotate(
            f"{label.get(fraction, str(fraction))} = {t_star:.1f} s",
            xy=(t_star, fraction * rsd_0),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=DEM_REFERENCE_COLOR,
        )


def save_figure(fig: Any, out_dir: str | Path, filename: str) -> Path:
    """Save a figure with a uniform, publication-quality setup.

    Args:
        fig: Matplotlib figure.
        out_dir: Destination directory (created if needed).
        filename: Output file name (``.png``).

    Returns:
        The path of the saved file.
    """
    out_path = Path(out_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"   💾 {filename}")
    return out_path


def experiment_label(experiment_name: str) -> str:
    """Return a compact human-readable label for an experiment name.

    Args:
        experiment_name: Experiment folder name.

    Returns:
        The method plus the number of cells, e.g. ``"voronoi (125 cells)"``.
    """
    match = re.search(r"(\d+)cells", experiment_name)
    n_cells = int(match.group(1)) if match else None
    method = method_of(experiment_name)
    suffix = f" ({n_cells} cells)" if n_cells else ""
    return f"{method}{suffix}"
