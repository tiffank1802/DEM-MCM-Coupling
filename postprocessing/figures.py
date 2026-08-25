"""Publication-oriented scientific figures of the post-processing pipeline.

Every figure produced here follows the global colour code defined in
:mod:`postprocessing.style`:

* a partitioning method always keeps the same colour (one colour per method
  across the whole pipeline);
* the ``small`` species is blue and the ``large`` species is orange, with a
  solid line for the DEM reference and a dashed line with markers for the
  Markov prediction;
* time axes keep the **raw DEM timesteps** (no unit conversion);
* mixing times ``t50``/``t90`` are annotated directly on the RSD curves.

All functions return ``(fig, axes)`` so that they can be introspected and
tested, and save their PNG with :func:`postprocessing.style.save_figure`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from postprocessing import metrics, style

style.apply_scientific_style()

#: Time-series style of the DEM reference.
DEM_STYLE: dict[str, Any] = {"linestyle": "-", "linewidth": 2.0, "alpha": 0.9}

#: Time-series style of the Markov prediction.
MARKOV_STYLE: dict[str, Any] = {
    "linestyle": "--",
    "linewidth": 1.6,
    "marker": "o",
    "markersize": 4,
    "alpha": 0.9,
}


def _common_time_window(species_data: dict[str, dict]) -> tuple[int, int]:
    """Return the common time window (in timesteps) of all species."""
    lengths = [len(d["times_markov"]) for d in species_data.values()]
    return 0, min(lengths)


def fig_rsd_scientific(
    species_data: dict[str, dict], short_name: str, out_dir: Path
) -> tuple[Any, Any]:
    """RSD vs time, DEM reference against the Markov prediction.

    One subplot per species, annotated with the ``t50``/``t90`` mixing times
    of the DEM curve, with the partitioning-method colour as a title accent.

    Args:
        species_data: Prepared per-species data (see
            :func:`postprocessing.postprocess.prepare_species`).
        short_name: Experiment folder name.
        out_dir: Destination directory.

    Returns:
        Tuple ``(fig, axes)``.
    """
    species_list = list(species_data)
    n = len(species_list)
    method_color = style.method_color(short_name)

    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.2), squeeze=False)
    fig.suptitle(
        f"RSD — {style.experiment_label(short_name)}",
        fontweight="bold",
        color=method_color,
    )

    for i, sp in enumerate(species_list):
        d = species_data[sp]
        colors = style.species_colors(sp)
        ax = axes[0][i]

        rsd_dem = metrics.rsd_from_S(d["S_dem"], d["activated"])
        rsd_markov = metrics.rsd_from_S(d["traj_markov"], d["activated"])
        t_dem = d["times_dem"]
        t_markov = d["times_markov"]

        ax.plot(t_dem, rsd_dem, color=colors["dem"], label=f"DEM — {sp}", **DEM_STYLE)
        ax.plot(
            t_markov,
            rsd_markov,
            color=colors["markov"],
            label=f"Markov — {sp}",
            **MARKOV_STYLE,
        )

        style.annotate_mixing_times(ax, t_dem, rsd_dem)

        ax.set_title(f"Species '{sp}'")
        ax.set_xlabel("Time (timestep)")
        ax.set_ylabel("RSD (-)")
        ax.set_ylim(bottom=0)
        ax.legend()

    fig.tight_layout()
    style.save_figure(fig, out_dir, "scientific_rsd.png")
    return fig, axes


def fig_transition_matrix_scientific(
    sp: str, sp_data: dict, short_name: str, out_dir: Path
) -> tuple[Any, Any]:
    """Annotated transition-matrix heatmap.

    The matrix follows the row convention: the **rows** are the source
    states ``i`` and the **columns** the destination states ``j``, so the
    cell ``(i, j)`` displays ``P(i → j)``.

    Args:
        sp: Species name.
        sp_data: Prepared species data (must contain ``"P"``).
        short_name: Experiment folder name.
        out_dir: Destination directory.

    Returns:
        Tuple ``(fig, ax)``.
    """
    P = np.asarray(sp_data["P"], dtype=float)
    n = P.shape[0]
    vmax = np.percentile(P[P > 0], 98) if (P > 0).any() else 1.0
    method_color = style.method_color(short_name)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.55 + 2), max(5, n * 0.55 + 2)))
    im = ax.imshow(
        P,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("P(i → j)", fontsize=10)

    ax.set_title(
        f"Transition matrix — {sp}\n{style.experiment_label(short_name)}",
        fontweight="bold",
        color=method_color,
    )
    ax.set_xlabel("Destination state j")
    ax.set_ylabel("Source state i")
    ax.set_xticks(range(n))
    ax.set_xticklabels(range(n), fontsize=7, rotation=90)
    ax.set_yticks(range(n))
    ax.set_yticklabels(range(n), fontsize=7)
    ax.tick_params(which="both", bottom=False, left=False)

    # Row-stochasticity reminder.
    row_sums = P.sum(axis=1)
    ax.text(
        0.99,
        0.02,
        f"Σ_j P(i → j) = 1  (visited rows)\n"
        f"min {row_sums[row_sums > 0].min():.4f} · max "
        f"{row_sums[row_sums > 0].max():.4f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "grey"},
    )

    fig.tight_layout()
    style.save_figure(fig, out_dir, f"scientific_matrix_{sp}.png")
    return fig, ax


def fig_stationary_distribution_scientific(
    sp: str, sp_data: dict, short_name: str, out_dir: Path
) -> tuple[Any, Any]:
    """Stationary distribution ``π`` of the transition matrix.

    ``π`` is the dominant left eigenvector of ``P`` (``π P = π``); the RSD of
    ``π`` over the activated cells measures how far the long-time state is
    from uniformity.

    Args:
        sp: Species name.
        sp_data: Prepared species data (must contain ``"P"`` and
            ``"activated"``).
        short_name: Experiment folder name.
        out_dir: Destination directory.

    Returns:
        Tuple ``(fig, ax)``.
    """
    P = np.asarray(sp_data["P"], dtype=float)
    activated = np.asarray(sp_data["activated"], dtype=bool)
    pi = metrics.stationary_distribution(P)
    method_color = style.method_color(short_name)

    pi_active = pi[activated]
    rsd_pi = float(pi_active.std() / pi_active.mean()) if pi_active.mean() > 0 else 0.0

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(
        np.arange(len(pi)),
        pi,
        color=method_color,
        alpha=0.8,
        edgecolor="white",
        label="π (stationary)",
    )
    ax.axhline(
        1 / activated.sum(),
        color=style.DEM_REFERENCE_COLOR,
        lw=1.2,
        ls="--",
        label="Uniform over activated cells",
    )
    ax.set_title(
        f"Stationary distribution — {sp}\n{style.experiment_label(short_name)}",
        fontweight="bold",
        color=method_color,
    )
    ax.set_xlabel("State i")
    ax.set_ylabel("π(i)")
    ax.legend()
    ax.text(
        0.99,
        0.98,
        f"RSD(π) = {rsd_pi:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "grey"},
    )

    fig.tight_layout()
    style.save_figure(fig, out_dir, f"scientific_stationary_{sp}.png")
    return fig, ax


def fig_concentration_scientific(
    species_data: dict[str, dict], short_name: str, out_dir: Path
) -> tuple[Any, Any]:
    """Three-panel figure of the binary mixing metrics.

    Panels: concentration RSD (with ``t50``/``t90`` annotations), normalised
    mixing entropy and intensity of segregation (Danckwerts) — DEM reference
    against the Markov prediction.

    Args:
        species_data: Prepared per-species data of at least two species.
        short_name: Experiment folder name.
        out_dir: Destination directory.

    Returns:
        Tuple ``(fig, axes)``.
    """
    sps = list(species_data)
    if len(sps) < 2:
        raise ValueError("fig_concentration_scientific needs 2 species")

    sp_a, sp_b = sps[0], sps[1]
    da, db = species_data[sp_a], species_data[sp_b]
    n_m = min(len(da["times_markov"]), len(db["times_markov"]))
    n_d = min(len(da["times_dem"]), len(db["times_dem"]))
    method_color = style.method_color(short_name)

    rsd_d = metrics.rsd_concentration(
        da["S_dem"][:n_d], db["S_dem"][:n_d], da["activated"], db["activated"]
    )
    rsd_m = metrics.rsd_concentration(
        da["traj_markov"][:n_m],
        db["traj_markov"][:n_m],
        da["activated"],
        db["activated"],
    )
    ent_d = metrics.entropy_concentration(
        da["S_dem"][:n_d], db["S_dem"][:n_d], da["activated"], db["activated"]
    )
    ent_m = metrics.entropy_concentration(
        da["traj_markov"][:n_m],
        db["traj_markov"][:n_m],
        da["activated"],
        db["activated"],
    )
    seg_d = metrics.intensity_of_segregation(
        da["S_dem"][:n_d], db["S_dem"][:n_d], da["activated"], db["activated"]
    )
    seg_m = metrics.intensity_of_segregation(
        da["traj_markov"][:n_m],
        db["traj_markov"][:n_m],
        da["activated"],
        db["activated"],
    )

    t_d = da["times_dem"][:n_d]
    t_m = da["times_markov"][:n_m]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    fig.suptitle(
        f"Binary mixing metrics — {style.experiment_label(short_name)}",
        fontweight="bold",
        color=method_color,
    )

    panels = [
        (
            axes[0],
            rsd_d,
            rsd_m,
            "Concentration RSD",
            "RSD (-)",
            style.annotate_mixing_times,
        ),
        (axes[1], ent_d, ent_m, "Mixing entropy (normalised)", "H (-) ∈ [0, 1]", None),
        (
            axes[2],
            seg_d,
            seg_m,
            "Intensity of segregation (Danckwerts)",
            "I (-) ∈ [0, 1]",
            None,
        ),
    ]
    for ax, series_d, series_m, title, ylabel, annotator in panels:
        ax.plot(
            t_d, series_d, color=style.DEM_REFERENCE_COLOR, label="DEM", **DEM_STYLE
        )
        ax.plot(t_m, series_m, color=method_color, label="Markov", **MARKOV_STYLE)
        if annotator is not None:
            annotator(ax, t_d, series_d)
        ax.set_title(title)
        ax.set_xlabel("Time (timestep)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend()

    fig.tight_layout()
    style.save_figure(fig, out_dir, "scientific_concentration.png")
    return fig, axes


def fig_compare_methods_rsd(
    all_species_data: dict[str, dict],
    out_dir: Path,
    species: str | None = None,
) -> tuple[Any, Any]:
    """Compare the concentration RSD across partitioning methods.

    **Colour code**: each experiment keeps the fixed colour of its
    partitioning method (see :data:`postprocessing.style.METHOD_COLORS`), so
    a method is always represented by the same colour in every comparison
    figure of the pipeline.

    Args:
        all_species_data: Mapping ``experiment name -> prepared species
            data`` (several experiments, possibly different methods).
        out_dir: Destination directory.
        species: Optional species to plot (first available when ``None``).

    Returns:
        Tuple ``(fig, ax)``.
    """
    if not all_species_data:
        raise ValueError("fig_compare_methods_rsd needs at least one experiment")

    fig, ax = plt.subplots(figsize=(10, 5.6))

    first_sp: str | None = None
    for name, species_data in all_species_data.items():
        sp = species if species in species_data else next(iter(species_data))
        first_sp = first_sp or sp
        d = species_data[sp]
        rsd = metrics.rsd_from_S(d["traj_markov"], d["activated"])
        t = d["times_markov"]
        ax.plot(
            t,
            rsd,
            color=style.method_color(name),
            label=style.experiment_label(name),
            **MARKOV_STYLE,
        )

    ax.set_title(
        f"Concentration RSD across partitioning methods — species '{first_sp}'",
        fontweight="bold",
    )
    ax.set_xlabel("Time (timestep)")
    ax.set_ylabel("RSD (-)")
    ax.set_ylim(bottom=0)
    ax.legend(title="Method")
    fig.tight_layout()

    style.save_figure(fig, out_dir, "scientific_compare_methods.png")
    return fig, ax
