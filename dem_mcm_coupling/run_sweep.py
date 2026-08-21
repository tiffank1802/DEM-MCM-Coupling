"""Markovian sweep pipelines (homogeneous and inhomogeneous).

Runs every experiment configuration of a given partitioning method: builds
the transition matrix (or matrices) from DEM state trajectories and uploads
the results to the configured data source.

Usage::

    python -m dem_mcm_coupling.run_sweep --method voronoi
    python -m dem_mcm_coupling.run_sweep --method voronoi --list
    python -m dem_mcm_coupling.run_sweep --method cartesian --inhomogeneous

From Python::

    from dem_mcm_coupling.run_sweep import run_markov_sweep
    run_markov_sweep("cylindrical")
"""

from __future__ import annotations

import argparse
import traceback
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dem_mcm_coupling import partitioners as part
from dem_mcm_coupling._config import BUCKET_ID
from dem_mcm_coupling.bucket_io import save_experiment_to_bucket
from dem_mcm_coupling.data.base import DataSource
from dem_mcm_coupling.partitioners import REGISTRY, create_partitioner
from dem_mcm_coupling.utils import load_parquet_as_timestep_dict

# =============================================================================
# GENERAL CONFIGURATION
# =============================================================================

#: Default local output directory (kept for CLI compatibility).
BASE_OUTPUT_DIR = "RaffinageTemporel"

#: Default parquet path of the complete DEM simulation (bucket root).
DEFAULT_PARQUET_PATH = f"hf://buckets/{BUCKET_ID}/simulation_complete.parquet"

#: Timestep from which the DEM simulation is considered stationary:
#: ``PERMANENT_START * N_PARTICLES`` particles rows are skipped when fitting.
PERMANENT_START = 250
N_PARTICLES_PER_TIMESTEP = 1030


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration of a single Markov experiment.

    Attributes:
        method: Partitioning method identifier.
        method_kwargs: Arguments forwarded to the partitioner.
        nlt: Number of learning timesteps (NLT blocks).
        tau: Lag (in timesteps) between the ``start`` and ``end`` of each
            transition pair.
        step: Distance between two consecutive block starts (when NLT > 1).
        dt: Temporal refinement inside each step (pairs are sampled every
            ``dt`` timesteps).
        start_index: First timestep index used for learning.
        particle_diameter: Optional diameter filter (``0.004``, ``0.008``,
            ``None``).
        inhomogeneous: When ``True``, one transition matrix per NLT block is
            built (inhomogeneous chain).
    """

    method: str = "cartesian"
    method_kwargs: dict[str, Any] = field(default_factory=dict)
    nlt: int = 2
    tau: int = 157  # gap between start and end of each pair
    step: int = 157  # distance between two main starts (when NLT > 1)
    dt: int | None = None  # temporal refinement inside each step
    start_index: int = 157
    particle_diameter: float | None = None
    inhomogeneous: bool = False  # if True, one P matrix per NLT block

    def __post_init__(self) -> None:
        if self.method_kwargs is None:
            self.method_kwargs = {}
        if self.dt is None:
            # Default refinement: 5 learning pairs per step.
            self.dt = max(1, self.step // 100)

    def output_folder(self, sample_coords: np.ndarray | None = None) -> str:
        """Return the experiment folder name (no path).

        For partitioners whose label depends on the data (adaptive,
        multizone), the partitioner is fitted on ``sample_coords`` first.

        Args:
            sample_coords: Optional coordinates used to fit the partitioner
                before reading its label.

        Returns:
            The folder name, e.g.
            ``voronoi_125cells_NLT2_step157_dt1_tau157_start157``.
        """
        partitioner = create_partitioner(self.method, **self.method_kwargs)
        if sample_coords is not None:
            partitioner.fit(sample_coords)

        folder_name = (
            f"{partitioner.label}_NLT{self.nlt}_step{self.step}_"
            f"dt{self.dt}_tau{self.tau}_start{self.start_index}"
        )

        if self.particle_diameter is not None:
            diameter_str = str(self.particle_diameter).replace(".", "")
            folder_name += f"_d{diameter_str}"

        if self.inhomogeneous:
            folder_name = f"inhomogeneous_{folder_name}"

        return folder_name


# =============================================================================
# CONFIGURATIONS PER METHOD
# =============================================================================


def get_configs(
    method: str, particle_diameter: float | None = None
) -> list[ExperimentConfig]:
    """Return the list of experiment configurations of a method.

    Args:
        method: Partitioning method identifier.
        particle_diameter: Optional diameter filter applied to every
            configuration.

    Returns:
        List of :class:`ExperimentConfig` (spatial sweep + temporal sweep,
        deduplicated).

    Raises:
        ValueError: If ``method`` is unknown.
    """
    configs: list[ExperimentConfig] = []

    print(f"   🔍 Generating configs for {method}...")
    if particle_diameter is not None:
        print(f"   🎯 Diameter filter: {particle_diameter}")

    # ── 1. Spatial discretisation sweep (default temporal parameters) ──────

    if method == "cartesian":
        for n in [2, 3, 4, 5]:
            configs.append(
                ExperimentConfig(
                    method="cartesian",
                    method_kwargs={"nx": n, "ny": n, "nz": n},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "cylindrical":
        for nr in [3, 4, 5, 6]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": nr,
                        "ntheta": 3,
                        "nz": 3,
                        "radial_mode": "equal_area",
                    },
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "dbscan":
        for min_samples in [3, 4, 5, 6]:
            configs.append(
                ExperimentConfig(
                    method="dbscan",
                    method_kwargs={"min_samples": min_samples},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "voronoi":
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="voronoi",
                    method_kwargs={"n_cells": nc},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "quantile":
        for n in [2, 3, 4, 5, 6]:
            configs.append(
                ExperimentConfig(
                    method="quantile",
                    method_kwargs={"nx": n, "ny": n, "nz": 1},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "octree":
        for mp in [20, 40, 80, 16, 32, 64, 28, 50, 60, 70, 100]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": mp, "max_depth": 2},
                    particle_diameter=particle_diameter,
                )
            )
        for md in [2, 1]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": 100, "max_depth": md},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "physics":
        # n_cells sweep with the default velocity weight.
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="physics",
                    method_kwargs={"n_cells": nc, "velocity_weight": 0.5},
                    particle_diameter=particle_diameter,
                )
            )
        # velocity_weight sweep (importance of the velocity in the clustering).
        for vw in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
            configs.append(
                ExperimentConfig(
                    method="physics",
                    method_kwargs={"n_cells": 30, "velocity_weight": vw},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "physics_full_vel":
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="physics_full_vel",
                    method_kwargs={"n_cells": nc, "velocity_weight": 0.5},
                    particle_diameter=particle_diameter,
                )
            )
        for vw in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="physics_full_vel",
                    method_kwargs={"n_cells": 30, "velocity_weight": vw},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "spectral":
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={
                        "n_cells": nc,
                        "velocity_weight": 0.5,
                        "n_neighbors": 15,
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for vw in [0.1, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={
                        "n_cells": 20,
                        "velocity_weight": vw,
                        "n_neighbors": 15,
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for k in [5, 10, 15, 20, 30]:
            configs.append(
                ExperimentConfig(
                    method="spectral",
                    method_kwargs={
                        "n_cells": 20,
                        "velocity_weight": 0.5,
                        "n_neighbors": k,
                    },
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "gmm":
        for nc in [10, 15, 20, 25, 30]:
            configs.append(
                ExperimentConfig(
                    method="gmm",
                    method_kwargs={"n_cells": nc, "velocity_weight": 0.5},
                    particle_diameter=particle_diameter,
                )
            )
        for vw in [0.1, 0.3, 0.5, 0.7, 1.0]:
            configs.append(
                ExperimentConfig(
                    method="gmm",
                    method_kwargs={"n_cells": 20, "velocity_weight": vw},
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "spectral_biclustering":
        for n_cells, n_col in [(20, 3), (30, 3), (40, 4), (50, 4)]:
            configs.append(
                ExperimentConfig(
                    method="spectral_biclustering",
                    method_kwargs={
                        "n_cells": n_cells,
                        "n_col_clusters": n_col,
                        "method": "log",
                        "velocity_weight": 0.6,
                    },
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "adaptive":
        for y_q in [0.5, 0.6, 0.7, 0.8, 0.9]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": y_q,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 1,
                            "ntheta": 36,
                            "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nr in [2, 3, 1]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": nr,
                            "ntheta": 30,
                            "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nz in [1, 2]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 1,
                            "ntheta": 30,
                            "nz": nz,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nth in [
            1,
            4,
            8,
            12,
            16,
            20,
            30,
            21,
            23,
            22,
            35,
            37,
            39,
            40,
            50,
            60,
            70,
            80,
            90,
            10,
            12,
            23,
            40,
        ]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 1,
                            "ntheta": nth,
                            "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for n_top in [1, 2, 3, 4]:
            top_method = "single" if n_top == 1 else "cylindrical"
            top_kwargs: dict[str, Any] = (
                {}
                if n_top == 1
                else {
                    "nr": 1,
                    "ntheta": n_top,
                    "nz": 1,
                    "radial_mode": "equal_area",
                }
            )
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": n_top,
                        "top_method": top_method,
                        "top_kwargs": top_kwargs,
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 1,
                            "ntheta": 30,
                            "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                    particle_diameter=particle_diameter,
                )
            )
        for nc in [10, 20, 30, 50, 64, 15, 35, 25, 50]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "y_split": 0.90,
                        "y_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "voronoi",
                        "bottom_kwargs": {"n_cells": nc},
                    },
                    particle_diameter=particle_diameter,
                )
            )

    elif method == "multizone":
        pass  # no default sweep defined yet

    elif method == "single":
        configs.append(
            ExperimentConfig(
                method="single",
                method_kwargs={},
                particle_diameter=particle_diameter,
            )
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    spatial_count = len(configs)
    print(f"   📊 Spatial configs for {method}: {spatial_count}")

    # ── 2. Temporal sweep (default spatial parameters) ──────────────────────

    default_spatial_kwargs = _get_default_kwargs(method)
    print(f"   🕒 Adding temporal sweeps with: {default_spatial_kwargs}")

    temporal_configs: list[ExperimentConfig] = []

    for nlt in [1, 2, 3, 5]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                nlt=nlt,
                particle_diameter=particle_diameter,
            )
        )

    for step in [20, 30, 40]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                step=step,
                particle_diameter=particle_diameter,
            )
        )

    step_ref = 20
    for dt in [1, 2, 3, 4]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                step=step_ref,
                dt=dt,
                particle_diameter=particle_diameter,
            )
        )

    for tau in [20, 50, 100, 200]:
        temporal_configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_spatial_kwargs,
                tau=tau,
                particle_diameter=particle_diameter,
            )
        )

    recommended_configs = [
        ExperimentConfig(
            method=method,
            method_kwargs=default_spatial_kwargs,
            nlt=3,
            step=100,
            dt=1,
            tau=50,
            particle_diameter=particle_diameter,
        ),
        ExperimentConfig(
            method=method,
            method_kwargs=default_spatial_kwargs,
            nlt=5,
            step=20,
            dt=2,
            tau=100,
            particle_diameter=particle_diameter,
        ),
        ExperimentConfig(
            method=method,
            method_kwargs=default_spatial_kwargs,
            nlt=2,
            step=20,
            dt=1,
            tau=100,
            particle_diameter=particle_diameter,
        ),
    ]
    temporal_configs.extend(recommended_configs)
    print(f"   🕒 Temporal configs generated: {len(temporal_configs)}")

    # ── 3. Smart combination and deduplication ──────────────────────────────

    all_configs = configs + temporal_configs
    print(
        f"   🔗 Total before deduplication: {len(all_configs)} "
        f"({spatial_count} spatial + {len(temporal_configs)} temporal)"
    )

    seen: set[str] = set()
    unique: list[ExperimentConfig] = []
    duplicates = 0

    for config in all_configs:
        if config.method in ["adaptive", "multizone"]:
            key = (
                f"{config.method}_{config.method_kwargs}_NLT{config.nlt}"
                f"_step{config.step}_dt{config.dt}_tau{config.tau}"
                f"_start{config.start_index}"
            )
        else:
            key = config.output_folder()

        if key not in seen:
            seen.add(key)
            unique.append(config)
        else:
            duplicates += 1

    print(
        f"   🔄 Deduplication: {len(all_configs)} → {len(unique)} "
        f"({duplicates} duplicates removed)"
    )
    return unique


def _get_default_kwargs(method: str) -> dict[str, Any]:
    """Return the default discretisation parameters of a method.

    Used by the temporal sweeps, which fix the spatial discretisation.
    """
    defaults: dict[str, dict[str, Any]] = {
        "cartesian": {"nx": 5, "ny": 5, "nz": 5},
        "cylindrical": {
            "nr": 3,
            "ntheta": 8,
            "nz": 1,
            "radial_mode": "equal_area",
        },
        "voronoi": {"n_cells": 40},
        "quantile": {"nx": 5, "ny": 5, "nz": 5},
        "octree": {"max_particles": 100, "max_depth": 1},
        "physics": {"n_cells": 30, "velocity_weight": 0.5},
        "physics_full_vel": {"n_cells": 30, "velocity_weight": 0.5},
        "spectral": {
            "n_cells": 20,
            "velocity_weight": 0.5,
            "n_neighbors": 15,
            "max_samples": 5000,
        },
        "gmm": {"n_cells": 20, "velocity_weight": 0.5},
        "spectral_biclustering": {
            "n_cells": 30,
            "n_col_clusters": 3,
            "method": "log",
            "velocity_weight": 0.6,
        },
        "adaptive": {
            "y_split": 0.90,
            "y_split_mode": "quantile",
            "n_cells_top": 1,
            "top_method": "single",
            "top_kwargs": {},
            "bottom_method": "voronoi",
            "bottom_kwargs": {"n_cells": 100},
        },
        "multizone": {
            "y_mode": "quantile",
            "zones": [
                {
                    "y_min": 0.0,
                    "y_max": 0.75,
                    "method": "cylindrical",
                    "kwargs": {
                        "nr": 2,
                        "ntheta": 2,
                        "nz": 1,
                        "radial_mode": "equal_area",
                    },
                },
                {
                    "y_min": 0.75,
                    "y_max": 1.0,
                    "method": "single",
                    "kwargs": {},
                },
            ],
        },
        "single": {},
    }
    return defaults.get(method, {})


# =============================================================================
# TRANSITION MATRIX (TORCH)
# =============================================================================

#: Methods whose partitioners consume velocity features when computing states.
_METHODS_USING_VELOCITY: frozenset[str] = frozenset(
    {
        "physics",
        "physics_full_vel",
        "spectral",
        "spectral_biclustering",
        "gmm",
        "dbscan",
    }
)

#: Methods that are also *fitted* with velocities in the sweeps. (Spectral
#: clustering and GMM are fitted on positions only, by historical choice.)
_METHODS_FITTED_WITH_VELOCITY: frozenset[str] = frozenset(
    {"physics", "physics_full_vel", "spectral_biclustering", "dbscan"}
)


def compute_P_matrix_torch(
    states_prev: np.ndarray | torch.Tensor,
    states_curr: np.ndarray | torch.Tensor,
    n_states: int,
    device: str = "cpu",
    species_labels: np.ndarray | None = None,
) -> torch.Tensor:
    """Compute one transition matrix ``P`` from state transition pairs.

    **Convention** (used everywhere in this package): ``P[i, j]`` is the
    probability to jump from state ``i`` to state ``j``, rows are stochastic
    (``P.sum(axis=1) == 1``) and a state vector evolves as
    ``phi_next = phi @ P``.

    The computation is fully vectorised: every ``(prev, curr)`` pair is
    one-hot encoded and the joint counts are accumulated with one matrix
    product.

    Args:
        states_prev: Previous state of each transition, shape ``(n_pairs,)``.
        states_curr: Next state of each transition, shape ``(n_pairs,)``.
        n_states: Total number of states.
        device: Torch device (``"cpu"`` or ``"cuda"``).
        species_labels: Unused, kept for API compatibility.

    Returns:
        The row-stochastic transition matrix as a ``torch.float64`` tensor of
        shape ``(n_states, n_states)``. Rows never visited are all zeros.
    """
    if isinstance(states_prev, np.ndarray):
        states_prev = torch.from_numpy(states_prev)
    if isinstance(states_curr, np.ndarray):
        states_curr = torch.from_numpy(states_curr)

    s_prev = states_prev.to(device).long()
    s_curr = states_curr.to(device).long()

    n = min(len(s_prev), len(s_curr))
    s_prev = s_prev[:n]
    s_curr = s_curr[:n]

    phi_prev = (s_prev.unsqueeze(1) == torch.arange(n_states, device=device)).float()
    phi_curr = (s_curr.unsqueeze(1) == torch.arange(n_states, device=device)).float()

    # transitions[i, j] = number of pairs (i -> j).
    transitions = phi_prev.T @ phi_curr
    # denominator[i] = number of pairs starting from state i.
    denominator = phi_prev.sum(dim=0)

    # Row-stochastic: P[i, j] = #(i -> j) / #(i).
    P = transitions / denominator.unsqueeze(1)
    P[denominator == 0] = 0.0

    return P.to(torch.float64)


# =============================================================================
# SAVING
# =============================================================================


def _collect_partitioner_data(partitioner: part.BasePartitioner) -> dict[str, Any]:
    """Gather the serialisable attributes of a fitted partitioner.

    Args:
        partitioner: The fitted partitioner.

    Returns:
        Dictionary of arrays/values to persist.
    """
    partitioner_data: dict[str, Any] = {}

    if hasattr(partitioner, "centroids") and partitioner.centroids is not None:
        partitioner_data["centroids"] = partitioner.centroids
    if hasattr(partitioner, "_r_edges") and partitioner._r_edges is not None:
        partitioner_data["r_edges"] = partitioner._r_edges
    if hasattr(partitioner, "_leaves") and partitioner._leaves:
        partitioner_data["leaves"] = np.array(partitioner._leaves)
    x_edges = getattr(partitioner, "_x_edges", None)
    if x_edges is not None:
        partitioner_data["x_edges"] = x_edges
        partitioner_data["y_edges"] = getattr(partitioner, "_y_edges", None)
        partitioner_data["z_edges"] = getattr(partitioner, "_z_edges", None)
    if isinstance(partitioner, part.PhysicsAwarePartitioner):
        if partitioner._mean is not None:
            partitioner_data["mean"] = partitioner._mean
        if partitioner._std is not None:
            partitioner_data["std"] = partitioner._std
        partitioner_data["physics_params"] = {
            "n_features": partitioner._n_features,
            "velocity_weight": partitioner.velocity_weight,
            "velocity_mode": partitioner.velocity_mode,
        }

    partitioner_data["partitioner_meta"] = {
        "type": type(partitioner).__name__,
        "label": partitioner.label,
        "n_cells": partitioner.n_cells,
    }
    return partitioner_data


def _species_arrays_from_results(results: dict[str, Any]) -> dict[str, np.ndarray]:
    """Convert per-species result dictionaries into ``.npy`` arrays.

    Args:
        results: Output of :func:`run_experiment`.

    Returns:
        Mapping ``file_stem -> array``.
    """
    species_data: dict[str, np.ndarray] = {}
    if "matrix" in results:
        species_data["states_matrix"] = results["matrix"]
    for species, data in results.items():
        if species == "matrix":
            continue
        species_data[f"transitionmatrix_{species}"] = data["P"]
        species_data[f"S_matrix_{species}"] = data["S_matrix"]
        species_data[f"times_{species}"] = data["times"]
    return species_data


def save_results(
    config: ExperimentConfig,
    partitioner: part.BasePartitioner,
    results: dict[str, Any],
    stats: dict[str, Any],
    image_data: dict[str, bytes] | None = None,
    folder_name: str | None = None,
    data_source: DataSource | None = None,
) -> None:
    """Save a homogeneous experiment (one transition matrix per species).

    Args:
        config: Experiment configuration.
        partitioner: The fitted partitioner.
        results: Output of :func:`run_experiment` (``"matrix"`` + one dict
            per species with ``"P"``, ``"S_matrix"``, ``"times"``).
        stats: Statistics dictionary.
        image_data: Optional images to store.
        folder_name: Optional folder name (computed from ``config`` when
            ``None``).
        data_source: Optional destination source (Hugging Face bucket by
            default).
    """
    if folder_name is None:
        folder_name = config.output_folder()

    if data_source is not None:
        data_source.write_experiment(
            folder_name=folder_name,
            stats={**stats, "species_list": [k for k in results if k != "matrix"]},
            config=asdict(config),
            species_data=_species_arrays_from_results(results),
            partitioner_data=_collect_partitioner_data(partitioner),
            image_data=image_data,
            inhomogeneous_metadata=None,
        )
    else:
        save_experiment_to_bucket(
            folder_name=folder_name,
            species_data=_species_arrays_from_results(results),
            stats={**stats, "species_list": [k for k in results if k != "matrix"]},
            config=asdict(config),
            partitioner_data=_collect_partitioner_data(partitioner),
            image_data=image_data,
            particle_diameter=config.particle_diameter,
        )

    bucket_name = (
        "BIG"
        if config.particle_diameter == 0.008
        else "SMALL"
        if config.particle_diameter == 0.004
        else "Experiments"
    )
    print(
        f"   💾 Bucket: {bucket_name}/{folder_name}/ "
        f"({[k for k in results if k != 'matrix']})"
    )


def sample_coordinates(
    timestep_dict: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack coordinates, velocities and diameters over every timestep.

    Args:
        timestep_dict: Mapping ``timestep_index -> DataFrame``.

    Returns:
        Tuple ``(coordinates, velocities, diameters)``.
    """
    all_coords: list[np.ndarray] = []
    all_velocities: list[np.ndarray] = []
    all_diameters: list[np.ndarray] = []

    for idx in sorted(timestep_dict):
        df = timestep_dict[idx]
        all_coords.append(
            np.column_stack(
                (
                    df["coordinates:0"].to_numpy(),
                    df["coordinates:1"].to_numpy(),
                    df["coordinates:2"].to_numpy(),
                )
            )
        )
        all_velocities.append(
            np.column_stack(
                (
                    df["Velocity:0"].to_numpy(),
                    df["Velocity:1"].to_numpy(),
                    df["Velocity:2"].to_numpy(),
                )
            )
        )
        all_diameters.append(df["Diameter"].to_numpy())

    print(f"   📏 Diameters loaded: {sum(len(d) for d in all_diameters)} particles")
    return (
        np.vstack(all_coords),
        np.vstack(all_velocities),
        np.concatenate(all_diameters),
    )


def _detect_species(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Auto-detect particle species by diameter.

    Args:
        df: Data frame of one timestep.

    Returns:
        Mapping ``species name -> boolean mask`` (aligned with ``df`` rows).
        A single diameter yields the ``"all"`` species; two diameters yield
        ``"small"``/``"large"``; more yield generic ``"d0004"``-style names.
    """
    diameters = df["Diameter"].to_numpy()
    unique_diams = np.sort(np.unique(diameters))

    if len(unique_diams) == 1:
        print(f"   ⚠️  Single diameter found ({unique_diams[0]}) — no species split")
        return {"all": np.ones(len(diameters), dtype=bool)}

    if len(unique_diams) == 2:
        labels = ["small", "large"]
    else:
        labels = [f"d{str(d).replace('.', '')}" for d in unique_diams]
        print(f"   i  {len(unique_diams)} diameters detected: {unique_diams}")

    species_masks: dict[str, np.ndarray] = {}
    for label, diam in zip(labels, unique_diams):
        mask = diameters == diam
        species_masks[label] = mask
        print(f"   ✅ Species '{label}' (d={diam:.4f}): {mask.sum()} particles")
    return species_masks


# =============================================================================
# EXPERIMENT RUN (HOMOGENEOUS)
# =============================================================================


def _compute_state_matrices(
    timestep_dict: dict[int, pd.DataFrame],
    partitioner: part.BasePartitioner,
    start_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the flat states and the ``(n_timesteps, n_particles)`` matrix.

    States are computed on every timestep starting from ``start_base``.
    Velocity features are attached to the partitioner when it is
    physics-aware.

    Args:
        timestep_dict: Mapping ``timestep_index -> DataFrame``.
        partitioner: The fitted partitioner.
        start_base: First timestep to consider.

    Returns:
        Tuple ``(states_matrix, sorted_indices)`` where ``states_matrix`` has
        shape ``(n_timesteps, n_particles)``.
    """
    sorted_indices = sorted(timestep_dict)
    n_timesteps = len(sorted_indices)
    n_particles = len(timestep_dict[sorted_indices[0]])

    uses_velocity = isinstance(
        partitioner,
        (
            part.PhysicsAwarePartitioner,
            part.SpectralBiclusteringPartitioner,
            part.GaussianMixturePartitioner,
            part.DBSCANPartitioner,
            part.SpectralClusteringPartitioner,
        ),
    )

    all_x, all_y, all_z = [], [], []
    all_vx, all_vy, all_vz = [], [], []
    for idx in sorted_indices:
        df = timestep_dict[idx]
        all_x.append(df["coordinates:0"].to_numpy())
        all_y.append(df["coordinates:1"].to_numpy())
        all_z.append(df["coordinates:2"].to_numpy())
        if uses_velocity:
            all_vx.append(df["Velocity:0"].to_numpy())
            all_vy.append(df["Velocity:1"].to_numpy())
            all_vz.append(df["Velocity:2"].to_numpy())

    coords_x = np.concatenate(all_x)
    coords_y = np.concatenate(all_y)
    coords_z = np.concatenate(all_z)

    if uses_velocity:
        vx_all = np.concatenate(all_vx)
        vy_all = np.concatenate(all_vy)
        vz_all = np.concatenate(all_vz)
        partitioner.use_velocity = True
        partitioner.dem_velocities = np.column_stack([vx_all, vy_all, vz_all])
        states_flat = partitioner.compute_states(
            coords_x, coords_y, coords_z, vx_all, vy_all, vz_all
        )
    else:
        states_flat = partitioner.compute_states(coords_x, coords_y, coords_z)

    states_matrix = states_flat.reshape(n_timesteps, n_particles)
    print(f"   ✅ states_matrix: {states_matrix.shape}")
    return states_matrix, np.array(sorted_indices)


def _build_state_matrices(
    states_matrix: np.ndarray,
    species_masks: dict[str, np.ndarray],
    n_states: int,
) -> dict[str, np.ndarray]:
    """Count particles per state and per timestep for each species.

    Args:
        states_matrix: State matrix of shape ``(n_timesteps, n_particles)``.
        species_masks: Mapping ``species -> boolean mask``.
        n_states: Number of states.

    Returns:
        Mapping ``species -> S matrix`` of shape ``(n_timesteps, n_states)``.
    """
    n_timesteps = states_matrix.shape[0]
    S_matrices: dict[str, np.ndarray] = {}
    for species, mask in species_masks.items():
        states_species = states_matrix[:, mask]
        S = np.zeros((n_timesteps, n_states), dtype=np.float64)
        for t in range(n_timesteps):
            S[t] = np.bincount(states_species[t], minlength=n_states)
        S_matrices[species] = S
        print(
            f"   ✅ S_matrix '{species}': {S.shape} | "
            f"sum t=0: {S[0].sum():.0f} particles ({mask.sum()} expected)"
        )
    return S_matrices


def _build_pairs(
    config: ExperimentConfig,
    timestep_dict: dict[int, pd.DataFrame],
    idx_to_row: dict[int, int],
) -> list[tuple[int, int]]:
    """Build the list of ``(start, end)`` transition pairs of a config.

    Args:
        config: Experiment configuration.
        timestep_dict: Mapping ``timestep_index -> DataFrame``.
        idx_to_row: Mapping ``timestep_index -> row in states_matrix``.

    Returns:
        List of ``(idx_prev, idx_curr)`` pairs.
    """
    start_base = config.start_index
    tau = config.tau
    step = config.step
    dt = config.dt or 1

    all_pairs: list[tuple[int, int]] = []
    for nlt_idx in range(config.nlt):
        current_start_base = start_base + nlt_idx * (step + tau)
        max_end_possible = max(timestep_dict)
        max_start_possible = max_end_possible - tau

        if current_start_base > max_start_possible:
            print(
                f"   ⚠️  Block {nlt_idx + 1} ignored "
                f"(start={current_start_base} > max={max_start_possible})"
            )
            break

        if nlt_idx == config.nlt - 1:
            remaining_range = max_start_possible - current_start_base
            n_apprentissages = min((step + tau) // dt, remaining_range // dt) + 1
        else:
            n_apprentissages = (step + tau) // dt

        for i in range(n_apprentissages):
            start_idx = current_start_base + i * dt
            end_idx = start_idx + tau
            if start_idx not in idx_to_row or end_idx not in idx_to_row:
                break
            all_pairs.append((start_idx, end_idx))

    return all_pairs


def run_experiment(
    config: ExperimentConfig,
    partitioner: part.BasePartitioner,
    timestep_dict: dict[int, pd.DataFrame],
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the transition matrices of an experiment, one per species.

    The species are auto-detected from the particle diameters: each species
    gets its own state matrix and its own transition matrix (a particle
    species mask is applied).

    Args:
        config: Experiment configuration.
        partitioner: The fitted partitioner.
        timestep_dict: Mapping ``timestep_index -> DataFrame``.
        device: Torch device.

    Returns:
        Tuple ``(results, stats)`` where ``results`` holds the
        ``"matrix"`` (raw state matrix) and one entry per species
        (``"P"``, ``"S_matrix"``, ``"times"``).

    Raises:
        KeyError: If ``config.start_index`` is absent from the data.
        ValueError: If no transition pair can be built.
    """
    try:
        df_init = timestep_dict[config.start_index]
    except KeyError:
        raise KeyError(
            f"Timestep start_base={config.start_index} absent from the data"
        ) from None

    # Auto-detect the species from the diameters and apply one mask per
    # species.
    species_masks = _detect_species(df_init)
    return _run_experiment_with_masks(
        config,
        partitioner,
        timestep_dict,
        device,
        species_masks,
        species_masks_applied=True,
    )


def run_no_species_experiment(
    config: ExperimentConfig,
    partitioner: part.BasePartitioner,
    timestep_dict: dict[int, pd.DataFrame],
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build ONE transition matrix from every particle (no species mask).

    Sibling of :func:`run_experiment` for the **initial assumption** of the
    model: large and small particles share the same kinetics. No diameter
    mask is applied — the state vector and the transition matrix are built
    from the whole particle population, stored under the single ``"all"``
    species.

    Args:
        config: Experiment configuration.
        partitioner: The fitted partitioner.
        timestep_dict: Mapping ``timestep_index -> DataFrame``.
        device: Torch device.

    Returns:
        Tuple ``(results, stats)`` — ``results`` holds ``"matrix"`` and the
        single ``"all"`` species entry (``"P"``, ``"S_matrix"``,
        ``"times"``).

    Raises:
        KeyError: If ``config.start_index`` is absent from the data.
        ValueError: If no transition pair can be built.
    """
    try:
        df_init = timestep_dict[config.start_index]
    except KeyError:
        raise KeyError(
            f"Timestep start_base={config.start_index} absent from the data"
        ) from None

    # No species distinction: every particle belongs to the "all" species.
    species_masks = {"all": np.ones(len(df_init), dtype=bool)}
    return _run_experiment_with_masks(
        config,
        partitioner,
        timestep_dict,
        device,
        species_masks,
        species_masks_applied=False,
    )


def _run_experiment_with_masks(
    config: ExperimentConfig,
    partitioner: part.BasePartitioner,
    timestep_dict: dict[int, pd.DataFrame],
    device: str,
    species_masks: dict[str, np.ndarray],
    species_masks_applied: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shared core of the masked and no-species pipelines.

    Builds the state matrices and the transition matrices from the provided
    species masks.

    Args:
        config: Experiment configuration.
        partitioner: The fitted partitioner.
        timestep_dict: Mapping ``timestep_index -> DataFrame``.
        device: Torch device.
        species_masks: Mapping ``species name -> boolean mask`` aligned with
            the particle rows.
        species_masks_applied: Whether the masks distinguish particle
            species (recorded in the statistics for later analysis).

    Returns:
        Tuple ``(results, stats)`` (see :func:`run_experiment`).

    Raises:
        ValueError: If no transition pair can be built.
    """
    n_states = partitioner.n_cells
    tau = config.tau
    start_base = config.start_index

    print(
        f"   📐 Configuration: NLT={config.nlt}, step={config.step}, dt={config.dt}, tau={tau}"
    )
    print(
        f"   📦 {len(timestep_dict)} timesteps available "
        f"(index {min(timestep_dict)} → {max(timestep_dict)})"
    )
    if not species_masks_applied:
        print(
            "   🧬 Aucun masque d'espèce : les grosses et petites particules "
            "partagent la même cinétique (une seule matrice de transition)."
        )

    sorted_indices = sorted(timestep_dict)
    n_timesteps = len(sorted_indices)
    idx_to_row = {idx: row for row, idx in enumerate(sorted_indices)}

    print(f"   🔧 State computation: {n_timesteps} timesteps (from t={start_base})...")
    states_matrix, _ = _compute_state_matrices(timestep_dict, partitioner, start_base)
    S_matrices = _build_state_matrices(states_matrix, species_masks, n_states)

    all_pairs = _build_pairs(config, timestep_dict, idx_to_row)
    if not all_pairs:
        raise ValueError("No valid pair with these parameters")

    print(
        f"   📊 {len(all_pairs)} pairs | "
        f"data_{all_pairs[0][0]}→{all_pairs[0][1]} … "
        f"data_{all_pairs[-1][0]}→{all_pairs[-1][1]}"
    )

    # Accumulate the (prev, curr) states per species.
    accumulators: dict[str, dict[str, np.ndarray]] = {
        species: {
            "prev": np.empty(0, dtype=np.int64),
            "curr": np.empty(0, dtype=np.int64),
        }
        for species in species_masks
    }
    for idx_prev, idx_curr in tqdm(all_pairs, desc="   Pairs", leave=False):
        row_prev = idx_to_row[idx_prev]
        row_curr = idx_to_row[idx_curr]
        for species, mask in species_masks.items():
            accumulators[species]["prev"] = np.concatenate(
                (accumulators[species]["prev"], states_matrix[row_prev][mask])
            )
            accumulators[species]["curr"] = np.concatenate(
                (accumulators[species]["curr"], states_matrix[row_curr][mask])
            )

    results: dict[str, Any] = {"matrix": states_matrix}

    for species in species_masks:
        print(f"\n   📐 Matrix P — species '{species}'...")
        P = (
            compute_P_matrix_torch(
                accumulators[species]["prev"],
                accumulators[species]["curr"],
                n_states,
                device,
            )
            .cpu()
            .numpy()
        )

        n_visited = int((P.sum(axis=1) > 0).sum())
        S_mat = S_matrices[species]
        print(
            f"      {n_states} states | {n_visited} visited | "
            f"P(stay)={np.diag(P).mean():.4f} | "
            f"S0 sum={S_mat[0].sum():.0f}"
        )

        results[species] = {
            "P": P,
            "S_matrix": S_mat,
            "times": np.array(sorted_indices),
        }

    n_paires_par_bloc = (config.step + tau) // (config.dt or 1)
    n_blocs_complets = len(all_pairs) // n_paires_par_bloc
    n_paires_dernier_bloc = len(all_pairs) % n_paires_par_bloc

    first_species = next(iter(species_masks))
    P_ref = results[first_species]["P"]
    row_sums = P_ref.sum(axis=1)
    visited = row_sums > 0

    stats = {
        "n_pairs_used": len(all_pairs),
        "n_nlt_requested": config.nlt,
        "n_blocs_complets": n_blocs_complets,
        "n_paires_dernier_bloc": n_paires_dernier_bloc,
        "n_states": n_states,
        "n_states_visited": int(visited.sum()),
        "n_states_empty": int((~visited).sum()),
        "fraction_visited": round(float(visited.sum()) / n_states, 4),
        "diagonal_mean": float(np.diag(P_ref).mean()),
        "diagonal_std": float(np.diag(P_ref).std()),
        "method": config.method,
        "species": list(species_masks),
        "species_masks_applied": species_masks_applied,
        "n_timesteps": n_timesteps,
        "tau": tau,
        "step": config.step,
        "dt": config.dt,
        "raffinage_ratio": n_paires_par_bloc,
        "plage_temporelle": int(all_pairs[-1][1] - all_pairs[0][0]),
        "start_index": config.start_index,
        "first_pair": list(all_pairs[0]),
        "last_pair": list(all_pairs[-1]),
        "particle_diameter": config.particle_diameter,
    }

    return results, stats


# =============================================================================
# SWEEP ORCHESTRATION (HOMOGENEOUS)
# =============================================================================


def _load_timestep_dict(data_source: DataSource | None) -> dict[int, pd.DataFrame]:
    """Load the DEM timesteps from a data source (or the default bucket).

    Args:
        data_source: Optional explicit source.

    Returns:
        Mapping ``timestep_index -> DataFrame``.
    """
    if data_source is not None:
        return data_source.read_timesteps()

    from dem_mcm_coupling.bucket_io import get_fs

    return load_parquet_as_timestep_dict(DEFAULT_PARQUET_PATH, fs=get_fs())


def _fit_partitioner_for_sweep(
    partitioner: part.BasePartitioner,
    config: ExperimentConfig,
    sample_coords: np.ndarray,
    s_velocities: np.ndarray,
    permanent_start: int,
) -> None:
    """Fit a partitioner on the stationary phase of the DEM data.

    Velocity-aware methods receive the velocities of the stationary phase.

    Args:
        partitioner: The partitioner to fit.
        config: Experiment configuration.
        sample_coords: Stacked coordinates over all timesteps.
        s_velocities: Stacked velocities over all timesteps.
        permanent_start: Row offset of the stationary phase.
    """
    if config.method in _METHODS_FITTED_WITH_VELOCITY:
        partitioner.use_velocity = True
        partitioner.dem_velocities = s_velocities[permanent_start:, :]
        partitioner.fit(sample_coords[permanent_start:, :])
    elif config.method == "cartesian":
        partitioner.fit(sample_coords)
    else:
        # Fit on the stationary phase of the simulation.
        partitioner.fit(sample_coords[permanent_start:, :])


def run_markov_sweep(
    method: str,
    configs: list[ExperimentConfig] | None = None,
    particle_diameter: float | None = None,
    base_dir: str = BASE_OUTPUT_DIR,
    data_source: DataSource | None = None,
) -> list[dict[str, Any]]:
    """Run the homogeneous sweep of one (or all) partitioning method(s).

    Args:
        method: Partitioning method, or ``"all"`` for every method.
        configs: Optional explicit list of configurations.
        particle_diameter: Optional diameter filter for the generated
            configurations.
        base_dir: Local output directory (legacy parameter, kept for CLI
            compatibility).
        data_source: Optional data source; the Hugging Face bucket is used
            by default.

    Returns:
        List of ``{"config", "stats", "success"[, "error"]}`` dictionaries.
    """
    del base_dir  # kept for API compatibility

    print("=" * 70)
    print(f"  MARKOVIAN SWEEP — method: {method.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    print("\n📦 Loading the parquet file...")
    timestep_dict = _load_timestep_dict(data_source)

    print("\n🔍 Sampling the coordinates for the fit...")
    sample_coords, s_velocities, _all_diameters = sample_coordinates(timestep_dict)
    print(f"   {len(sample_coords)} points sampled")

    methods = list(REGISTRY) if method == "all" else [method]

    if configs is None:
        all_configs = [
            c
            for m in methods
            for c in get_configs(m, particle_diameter=particle_diameter)
        ]
    else:
        all_configs = configs

    print(f"\n📋 {len(all_configs)} experiments to run:")
    print("-" * 70)

    results_summary: list[dict[str, Any]] = []
    permanent_start = PERMANENT_START * N_PARTICLES_PER_TIMESTEP

    for i, config in enumerate(all_configs):
        try:
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            print("   🔧 Fitting the partitioner...")

            _fit_partitioner_for_sweep(
                partitioner, config, sample_coords, s_velocities, permanent_start
            )

            if config.method in [
                "adaptive",
                "multizone",
                *sorted(_METHODS_FITTED_WITH_VELOCITY),
            ]:
                folder_name = config.output_folder(sample_coords=sample_coords)
            else:
                folder_name = config.output_folder()
            print(f"\n[{i + 1}/{len(all_configs)}] {folder_name}")

            results, stats = run_experiment(
                config, partitioner, timestep_dict, str(device)
            )

            save_results(
                config=config,
                partitioner=partitioner,
                results=results,
                stats=stats,
                image_data=None,
                folder_name=folder_name,
                data_source=data_source,
            )

            results_summary.append(
                {"config": asdict(config), "stats": stats, "success": True}
            )
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} states | "
                f"P(stay)={stats['diagonal_mean']:.4f} | "
                f"species={stats['species']} | "
                f"pairs={stats['n_pairs_used']}"
            )
        except Exception as exc:
            print(f"   ❌ Error: {exc}")
            traceback.print_exc()
            results_summary.append(
                {
                    "config": asdict(config),
                    "stats": None,
                    "success": False,
                    "error": str(exc),
                }
            )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok = [r for r in results_summary if r["success"]]
    ko = [r for r in results_summary if not r["success"]]
    print(f"\n✅ Success: {len(ok)}/{len(results_summary)}")
    if ko:
        print(f"❌ Failed: {len(ko)}")
        for r in ko:
            print(f"   - {r['config']['method']}: {r.get('error', '?')}")

    summary_data = {
        "method": method,
        "total": len(results_summary),
        "success": len(ok),
        "failed": len(ko),
        "results": results_summary,
    }
    try:
        if data_source is not None:
            data_source.write_experiment(
                folder_name=f"_summary_{method}",
                stats=summary_data,
                config={"type": "summary", "method": method},
            )
        else:
            save_experiment_to_bucket(
                folder_name=f"_summary_{method}",
                species_data={},
                stats=summary_data,
                config={"type": "summary", "method": method},
            )
        print(f"\n💾 Summary saved: _summary_{method}/")
    except Exception as exc:
        print(f"\n⚠️  Could not save the summary: {exc}")

    print("✨ Done!")
    return results_summary


# =============================================================================
# NO-SPECIES PIPELINE (no particle-size mask, single transition matrix)
# =============================================================================


def run_no_species_sweep(
    method: str,
    configs: list[ExperimentConfig] | None = None,
    base_dir: str = BASE_OUTPUT_DIR,
    data_source: DataSource | None = None,
) -> list[dict[str, Any]]:
    """Run the no-species sweep: one transition matrix for every particle.

    Sibling of :func:`run_markov_sweep` for the **initial assumption** of
    the model: the particle size is ignored, so large and small particles
    share the same kinetics. The state vector and the transition matrix are
    built without any species mask and saved under the ``nospecies_``
    prefix, which routes them to the ``nospecies_simulations/`` folder of
    the bucket.

    Args:
        method: Partitioning method, or ``"all"`` for every method.
        configs: Optional explicit list of configurations.
        base_dir: Local output directory (legacy parameter, kept for CLI
            compatibility).
        data_source: Optional data source; the Hugging Face bucket is used
            by default.

    Returns:
        List of ``{"config", "stats", "success"[, "error"]}`` dictionaries.
    """
    del base_dir  # kept for API compatibility

    print("=" * 70)
    print(f"  NO-SPECIES SWEEP (no mask) — method: {method.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    print("\n📦 Loading the parquet file...")
    timestep_dict = _load_timestep_dict(data_source)

    print("\n🔍 Sampling the coordinates for the fit...")
    sample_coords, s_velocities, _all_diameters = sample_coordinates(timestep_dict)
    print(f"   {len(sample_coords)} points sampled")

    methods = list(REGISTRY) if method == "all" else [method]

    if configs is None:
        # The no-species pipeline uses the WHOLE particle population: no
        # diameter filter is applied.
        all_configs = [
            c for m in methods for c in get_configs(m, particle_diameter=None)
        ]
    else:
        all_configs = configs

    print(f"\n📋 {len(all_configs)} no-species experiments to run:")
    print("-" * 70)

    results_summary: list[dict[str, Any]] = []
    permanent_start = PERMANENT_START * N_PARTICLES_PER_TIMESTEP

    for i, config in enumerate(all_configs):
        try:
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            print("   🔧 Fitting the partitioner...")

            _fit_partitioner_for_sweep(
                partitioner, config, sample_coords, s_velocities, permanent_start
            )

            if config.method in [
                "adaptive",
                "multizone",
                *sorted(_METHODS_FITTED_WITH_VELOCITY),
            ]:
                base_folder = config.output_folder(sample_coords=sample_coords)
            else:
                base_folder = config.output_folder()
            # The `nospecies_` prefix routes the folder to the
            # `nospecies_simulations/` bucket category.
            folder_name = f"nospecies_{base_folder}"
            print(f"\n[{i + 1}/{len(all_configs)}] {folder_name}")

            results, stats = run_no_species_experiment(
                config, partitioner, timestep_dict, str(device)
            )

            save_results(
                config=config,
                partitioner=partitioner,
                results=results,
                stats=stats,
                image_data=None,
                folder_name=folder_name,
                data_source=data_source,
            )

            results_summary.append(
                {"config": asdict(config), "stats": stats, "success": True}
            )
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} states | "
                f"P(stay)={stats['diagonal_mean']:.4f} | "
                f"species={stats['species']} | "
                f"pairs={stats['n_pairs_used']}"
            )
        except Exception as exc:
            print(f"   ❌ Error: {exc}")
            traceback.print_exc()
            results_summary.append(
                {
                    "config": asdict(config),
                    "stats": None,
                    "success": False,
                    "error": str(exc),
                }
            )

    print("\n" + "=" * 70)
    print("SUMMARY (no species)")
    print("=" * 70)
    ok = [r for r in results_summary if r["success"]]
    ko = [r for r in results_summary if not r["success"]]
    print(f"\n✅ Success: {len(ok)}/{len(results_summary)}")
    if ko:
        print(f"❌ Failed: {len(ko)}")
        for r in ko:
            print(f"   - {r['config']['method']}: {r.get('error', '?')}")

    summary_data = {
        "method": method,
        "total": len(results_summary),
        "success": len(ok),
        "failed": len(ko),
        "results": results_summary,
        "nospecies": True,
    }
    try:
        if data_source is not None:
            data_source.write_experiment(
                folder_name=f"_summary_nospecies_{method}",
                stats=summary_data,
                config={"type": "summary", "method": method, "nospecies": True},
            )
        else:
            save_experiment_to_bucket(
                folder_name=f"_summary_nospecies_{method}",
                species_data={},
                stats=summary_data,
                config={"type": "summary", "method": method, "nospecies": True},
            )
        print(f"\n💾 Summary saved: _summary_nospecies_{method}/")
    except Exception as exc:
        print(f"\n⚠️  Could not save the summary: {exc}")

    print("✨ Done!")
    return results_summary


# =============================================================================
# INHOMOGENEOUS PIPELINE (sibling of the homogeneous one above)
# =============================================================================


def _build_inhomogeneous_blocks(
    config: ExperimentConfig,
    timestep_dict: dict[int, pd.DataFrame],
    idx_to_row: dict[int, int],
) -> list[list[tuple[int, int]]]:
    """Build the transition pairs grouped by NLT block.

    Args:
        config: Experiment configuration.
        timestep_dict: Mapping ``timestep_index -> DataFrame``.
        idx_to_row: Mapping ``timestep_index -> row in states_matrix``.

    Returns:
        List of blocks; each block is a list of ``(idx_prev, idx_curr)``
        pairs.
    """
    start_base = config.start_index
    tau = config.tau
    step = config.step
    dt = config.dt or 1

    blocks: list[list[tuple[int, int]]] = []
    for nlt_idx in range(config.nlt):
        block_pairs: list[tuple[int, int]] = []
        # `step` is the distance between the end of a block and the start of
        # the next one.
        current_start_base = start_base + nlt_idx * (step + tau)
        max_end_possible = max(timestep_dict)
        max_start_possible = max_end_possible - tau

        if current_start_base > max_start_possible:
            print(
                f"   ⚠️  Block {nlt_idx + 1} ignored "
                f"(start={current_start_base} > max={max_start_possible})"
            )
            break

        if nlt_idx == config.nlt - 1:
            remaining_range = max_start_possible - current_start_base
            n_apprentissages = min(tau // dt, remaining_range // dt) + 1
        else:
            n_apprentissages = tau // dt

        for i in range(n_apprentissages):
            start_idx = current_start_base + i * dt
            end_idx = start_idx + tau
            if start_idx not in idx_to_row or end_idx not in idx_to_row:
                break
            block_pairs.append((start_idx, end_idx))

        if block_pairs:
            blocks.append(block_pairs)
            print(
                f"   📦 Block {len(blocks)}: {len(block_pairs)} pairs "
                f"[{block_pairs[0][0]}→{block_pairs[0][1]} … "
                f"{block_pairs[-1][0]}→{block_pairs[-1][1]}]"
            )

    return blocks


def run_inhomogeneous_experiment(
    config: ExperimentConfig,
    partitioner: part.BasePartitioner,
    timestep_dict: dict[int, pd.DataFrame],
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one transition matrix ``P_k`` per NLT block (inhomogeneous).

    Sibling of :func:`run_experiment` — same preparation logic, but returns
    a list of matrices (one per NLT block) instead of a single one.

    Args:
        config: Experiment configuration.
        partitioner: The fitted partitioner.
        timestep_dict: Mapping ``timestep_index -> DataFrame``.
        device: Torch device.

    Returns:
        Tuple ``(results, stats)``; ``results`` holds ``"matrix"`` and, per
        species, ``{"P_blocks", "S_matrix", "times"}``.

    Raises:
        KeyError: If ``config.start_index`` is absent from the data.
        ValueError: If no valid block can be built.
    """
    n_states = partitioner.n_cells
    tau = config.tau
    start_base = config.start_index

    print(
        f"   📐 Inhomogeneous configuration: NLT={config.nlt}, "
        f"step={config.step}, dt={config.dt}, tau={tau}"
    )
    print(
        f"   📦 {len(timestep_dict)} timesteps available "
        f"(index {min(timestep_dict)} → {max(timestep_dict)})"
    )

    try:
        df_init = timestep_dict[start_base]
    except KeyError:
        raise KeyError(
            f"Timestep start_base={start_base} absent from the data"
        ) from None

    species_masks = _detect_species(df_init)

    sorted_indices = sorted(timestep_dict)
    n_timesteps = len(sorted_indices)
    idx_to_row = {idx: row for row, idx in enumerate(sorted_indices)}

    print(f"   🔧 State computation: {n_timesteps} timesteps (from t={start_base})...")
    states_matrix, _ = _compute_state_matrices(timestep_dict, partitioner, start_base)
    S_matrices = _build_state_matrices(states_matrix, species_masks, n_states)

    # ── Construction per NLT block: each block produces its own P_k ────────

    blocks = _build_inhomogeneous_blocks(config, timestep_dict, idx_to_row)
    if not blocks:
        raise ValueError("No valid pair — cannot build the blocks")

    P_blocks_by_species: dict[str, list[np.ndarray]] = {
        species: [] for species in species_masks
    }

    for block_idx, block_pairs in enumerate(blocks):
        accum: dict[str, dict[str, np.ndarray]] = {
            species: {
                "prev": np.empty(0, dtype=np.int64),
                "curr": np.empty(0, dtype=np.int64),
            }
            for species in species_masks
        }

        for idx_prev, idx_curr in block_pairs:
            row_prev = idx_to_row[idx_prev]
            row_curr = idx_to_row[idx_curr]
            for species, mask in species_masks.items():
                accum[species]["prev"] = np.concatenate(
                    (accum[species]["prev"], states_matrix[row_prev][mask])
                )
                accum[species]["curr"] = np.concatenate(
                    (accum[species]["curr"], states_matrix[row_curr][mask])
                )

        for species in species_masks:
            P_k = (
                compute_P_matrix_torch(
                    accum[species]["prev"],
                    accum[species]["curr"],
                    n_states,
                    device,
                )
                .cpu()
                .numpy()
            )
            P_blocks_by_species[species].append(P_k)

        n_visited_k = int(
            (
                P_blocks_by_species[next(iter(species_masks))][block_idx].sum(axis=1)
                > 0
            ).sum()
        )
        print(f"      P_{block_idx}: {n_states} states, {n_visited_k} visited")

    # Assemble the results dictionary.
    results: dict[str, Any] = {"matrix": states_matrix}
    for species in species_masks:
        results[species] = {
            # (n_blocks, n_states, n_states)
            "P_blocks": np.array(P_blocks_by_species[species]),
            "S_matrix": S_matrices[species],
            "times": np.array(sorted_indices),
        }

    # Enriched statistics.
    first_species = next(iter(species_masks))
    n_visited_total = int(
        (results[first_species]["P_blocks"].sum(axis=(0, 2)) > 0).sum()
    )

    stats = {
        "n_blocks": len(blocks),
        "n_pairs_per_block": [len(b) for b in blocks],
        "n_nlt_requested": config.nlt,
        "n_states": n_states,
        "n_states_visited": n_visited_total,
        "fraction_visited": round(float(n_visited_total) / n_states, 4),
        "method": config.method,
        "species": list(species_masks),
        "n_timesteps": n_timesteps,
        "tau": tau,
        "step": config.step,
        "dt": config.dt,
        "start_index": config.start_index,
        "particle_diameter": config.particle_diameter,
        "inhomogeneous": True,
    }

    return results, stats


def save_inhomogeneous_results(
    config: ExperimentConfig,
    partitioner: part.BasePartitioner,
    results: dict[str, Any],
    stats: dict[str, Any],
    image_data: dict[str, bytes] | None = None,
    folder_name: str | None = None,
    data_source: DataSource | None = None,
) -> None:
    """Save an inhomogeneous experiment (``P_blocks`` per species).

    Sibling of :func:`save_results` — stores ``P_blocks_{species}.npy``
    (3-D arrays) instead of ``transitionmatrix_{species}.npy`` and adds
    ``inhomogeneous_metadata.json``.

    Args:
        config: Experiment configuration.
        partitioner: The fitted partitioner.
        results: Output of :func:`run_inhomogeneous_experiment`.
        stats: Statistics dictionary.
        image_data: Optional images to store.
        folder_name: Optional folder name (computed from ``config`` when
            ``None``).
        data_source: Optional destination source (Hugging Face bucket by
            default).
    """
    if folder_name is None:
        folder_name = config.output_folder()

    # Species data: P_blocks instead of transitionmatrix.
    species_data: dict[str, np.ndarray] = {}
    if "matrix" in results:
        species_data["states_matrix"] = results["matrix"]
        print(f"   📦 states_matrix to save: {results['matrix'].shape}")

    for species, data in results.items():
        if species == "matrix":
            continue
        species_data[f"P_blocks_{species}"] = data["P_blocks"]  # (n_blocks, S, S)
        species_data[f"S_matrix_{species}"] = data["S_matrix"]
        species_data[f"times_{species}"] = data["times"]

    # Inhomogeneous metadata.
    inhomogeneous_metadata = {
        "n_blocks": stats["n_blocks"],
        "n_pairs_per_block": stats["n_pairs_per_block"],
        "species_list": [k for k in results if k != "matrix"],
        "block_start_indices": [
            config.start_index + i * (config.step + config.tau)
            for i in range(stats["n_blocks"])
        ],
    }

    stats_with_species = {
        **stats,
        "species_list": [k for k in results if k != "matrix"],
    }

    if data_source is not None:
        data_source.write_experiment(
            folder_name=folder_name,
            stats=stats_with_species,
            config=asdict(config),
            species_data=species_data,
            partitioner_data=_collect_partitioner_data(partitioner),
            image_data=image_data,
            inhomogeneous_metadata=inhomogeneous_metadata,
        )
    else:
        save_experiment_to_bucket(
            folder_name=folder_name,
            species_data=species_data,
            stats=stats_with_species,
            config=asdict(config),
            partitioner_data=_collect_partitioner_data(partitioner),
            image_data=image_data,
            particle_diameter=config.particle_diameter,
            inhomogeneous_metadata=inhomogeneous_metadata,
        )

    bucket_name = (
        "BIG"
        if config.particle_diameter == 0.008
        else "SMALL"
        if config.particle_diameter == 0.004
        else "Experiments"
    )
    print(
        f"   💾 Bucket: {bucket_name}/{folder_name}/ "
        f"(inhomogeneous, {stats['n_blocks']} blocks, "
        f"species={[k for k in results if k != 'matrix']})"
    )


def run_inhomogeneous_markov_sweep(
    method: str,
    configs: list[ExperimentConfig] | None = None,
    particle_diameter: float | None = None,
    base_dir: str = BASE_OUTPUT_DIR,
    data_source: DataSource | None = None,
) -> list[dict[str, Any]]:
    """Run the inhomogeneous sweep — sibling of :func:`run_markov_sweep`.

    Args:
        method: Partitioning method, or ``"all"``.
        configs: Optional explicit list of configurations.
        particle_diameter: Optional diameter filter.
        base_dir: Local output directory (legacy parameter).
        data_source: Optional data source; the Hugging Face bucket is used
            by default.

    Returns:
        List of ``{"config", "stats", "success"[, "error"]}`` dictionaries.
    """
    del base_dir  # kept for API compatibility

    print("=" * 70)
    print(f"  INHOMOGENEOUS SWEEP — method: {method.upper()}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    print("\n📦 Loading the parquet file...")
    timestep_dict = _load_timestep_dict(data_source)

    print("\n🔍 Sampling the coordinates for the fit...")
    sample_coords, s_velocities, _all_diameters = sample_coordinates(timestep_dict)
    print(f"   {len(sample_coords)} points sampled")

    methods = list(REGISTRY) if method == "all" else [method]

    if configs is None:
        all_configs: list[ExperimentConfig] = []
        for m in methods:
            for c in get_configs(m, particle_diameter=particle_diameter):
                c.inhomogeneous = True  # force the inhomogeneous mode
                all_configs.append(c)
    else:
        all_configs = configs

    print(f"\n📋 {len(all_configs)} inhomogeneous experiments to run:")
    print("-" * 70)

    results_summary: list[dict[str, Any]] = []
    permanent_start = PERMANENT_START * N_PARTICLES_PER_TIMESTEP

    for i, config in enumerate(all_configs):
        try:
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            print("   🔧 Fitting the partitioner...")

            _fit_partitioner_for_sweep(
                partitioner, config, sample_coords, s_velocities, permanent_start
            )

            if config.method in [
                "adaptive",
                "multizone",
                *sorted(_METHODS_FITTED_WITH_VELOCITY),
            ]:
                folder_name = config.output_folder(sample_coords=sample_coords)
            else:
                folder_name = config.output_folder()
            print(f"\n[{i + 1}/{len(all_configs)}] {folder_name}")

            results, stats = run_inhomogeneous_experiment(
                config, partitioner, timestep_dict, str(device)
            )

            save_inhomogeneous_results(
                config=config,
                partitioner=partitioner,
                results=results,
                stats=stats,
                image_data=None,
                folder_name=folder_name,
                data_source=data_source,
            )

            results_summary.append(
                {"config": asdict(config), "stats": stats, "success": True}
            )
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} states | "
                f"{stats['n_blocks']} blocks | "
                f"species={stats['species']}"
            )
        except Exception as exc:
            print(f"   ❌ Error: {exc}")
            traceback.print_exc()
            results_summary.append(
                {
                    "config": asdict(config),
                    "stats": None,
                    "success": False,
                    "error": str(exc),
                }
            )

    print("\n" + "=" * 70)
    print("SUMMARY (inhomogeneous)")
    print("=" * 70)
    ok = [r for r in results_summary if r["success"]]
    ko = [r for r in results_summary if not r["success"]]
    print(f"\n✅ Success: {len(ok)}/{len(results_summary)}")
    if ko:
        print(f"❌ Failed: {len(ko)}")
        for r in ko:
            print(f"   - {r['config']['method']}: {r.get('error', '?')}")

    summary_data = {
        "method": method,
        "total": len(results_summary),
        "success": len(ok),
        "failed": len(ko),
        "results": results_summary,
        "inhomogeneous": True,
    }
    try:
        if data_source is not None:
            data_source.write_experiment(
                folder_name=f"_summary_inhomogeneous_{method}",
                stats=summary_data,
                config={"type": "summary", "method": method, "inhomogeneous": True},
                inhomogeneous_metadata=None,
            )
        else:
            save_experiment_to_bucket(
                folder_name=f"_summary_inhomogeneous_{method}",
                species_data={},
                stats=summary_data,
                config={"type": "summary", "method": method, "inhomogeneous": True},
            )
        print(f"\n💾 Summary saved: _summary_inhomogeneous_{method}/")
    except Exception as exc:
        print(f"\n⚠️  Could not save the summary: {exc}")

    print("✨ Done!")
    return results_summary


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Command-line entry point of the sweep pipelines."""
    parser = argparse.ArgumentParser(
        description="Markovian sweep, multi-partitioning (homogeneous or inhomogeneous)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cartesian",
        choices=[*list(REGISTRY), "all"],
        help="Partitioning type (default: cartesian)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=BASE_OUTPUT_DIR,
        help=f"Output directory (default: {BASE_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--diameter",
        type=float,
        choices=[0.004, 0.008],
        default=None,
        help="Particle-diameter filter: 0.004 (SMALL), 0.008 (BIG), or None (all)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the configurations without running the computations",
    )
    parser.add_argument(
        "--inhomogeneous",
        action="store_true",
        help="Enable the inhomogeneous mode (one P matrix per NLT block)",
    )
    parser.add_argument(
        "--no-species",
        action="store_true",
        help="Disable the species masks: one P matrix for every particle "
        "(saved under the nospecies_simulations/ bucket folder)",
    )
    args = parser.parse_args()

    if args.list:
        diameter = None if args.no_species else args.diameter
        prefix = "nospecies_" if args.no_species else ""
        if args.method == "all":
            for m in REGISTRY:
                configs = get_configs(m, particle_diameter=diameter)
                print(f"\n{m.upper()} ({len(configs)} configs):")
                for c in configs:
                    p = create_partitioner(c.method, **c.method_kwargs)
                    print(
                        f"  {prefix}{p.label} NLT={c.nlt} step={c.step} dt={c.dt} "
                        f"diameter={c.particle_diameter}"
                    )
        else:
            configs = get_configs(args.method, particle_diameter=diameter)
            print(f"{args.method.upper()} ({len(configs)} configs):")
            for c in configs:
                p = create_partitioner(c.method, **c.method_kwargs)
                print(
                    f"  {prefix}{p.label} NLT={c.nlt} step={c.step} dt={c.dt} "
                    f"diameter={c.particle_diameter}"
                )
        return

    if args.no_species:
        run_no_species_sweep(args.method, base_dir=args.output)
    elif args.inhomogeneous:
        run_inhomogeneous_markov_sweep(
            args.method, particle_diameter=args.diameter, base_dir=args.output
        )
    else:
        run_markov_sweep(
            args.method, particle_diameter=args.diameter, base_dir=args.output
        )


if __name__ == "__main__":
    main()
