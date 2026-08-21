"""Core Markov chain model for DEM/MCM coupling.

:class:`Markov` orchestrates one partitioning configuration:

1. creation of a partitioner (see :mod:`dem_mcm_coupling.partitioners`);
2. loading of DEM particle data from any
   :class:`~dem_mcm_coupling.data.base.DataSource` (Hugging Face Hub, local
   directory, in-memory);
3. construction of the initial state vector ``phi(0)``;
4. propagation of the state through a transition matrix;
5. optional 3-D visualisation (PyVista/Streamlit — optional extras).

This class manages a **single** configuration (builder pattern); comparing
several configurations is the role of
:class:`~dem_mcm_coupling.analyze_results.MarkovAnalyzer`.

Example:
    >>> from dem_mcm_coupling.markov_core import Markov
    >>> mk = Markov(method="voronoi", method_kwargs={"n_cells": 125})
    >>> # mk.load_dem_data(particle_diameter=0.004)  # requires HF access
    >>> # coords = mk.get_coords([250, 300, 350])
    >>> # mk.fit_partitioner(coords)
    >>> # initial_state = mk.build_initial_state_vector(250)
    >>> # trajectory = mk.propagate_markov(initial_state.phi, M, n_steps=100)
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from dem_mcm_coupling._config import (
    TIMESTEP_TO_SECONDS,
    MarkovTrajectory,
    ParticleDiameter,
    PartitioningMethod,
    StateVectorData,
    validate_partitioning_method,
)
from dem_mcm_coupling.partitioners import BasePartitioner, create_partitioner
from dem_mcm_coupling.utils import apply_species_mask, load_parquet_as_timestep_dict

if TYPE_CHECKING:
    from dem_mcm_coupling.data.base import DataSource

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ============================================================================
# MAIN CLASS
# ============================================================================


class Markov:
    """Manager of one Markovian partitioning configuration.

    Attributes:
        method: Identifier of the partitioning method.
        partitioner: Fitted :class:`BasePartitioner` instance.
        datas: DEM data frames per timestep, ``{timestep_index: DataFrame}``.
        coords: Particle coordinates, shape ``(n_particles, 3)``.
        velocities: Particle velocities, shape ``(n_particles, 3)``.
        states: Assigned partition states, shape ``(n_particles,)``.
        initial_state: Initial state vector ``phi(0)`` (if built).
        transition_matrix: Transition matrix (if set).

    Typical workflow:
        1. ``mk = Markov(method="voronoi", method_kwargs={"n_cells": 125})``
        2. ``mk.load_dem_data(particle_diameter=0.004)``
        3. ``coords = mk.get_coords([250, 300, 350])``
        4. ``mk.fit_partitioner(coords)``
        5. ``state_0 = mk.build_initial_state_vector(250)``
        6. ``traj = mk.propagate_markov(state_0.phi, M, 100)``
    """

    def __init__(
        self,
        method: str | PartitioningMethod = "cartesian",
        method_kwargs: dict[str, Any] | None = None,
        data_source: DataSource | None = None,
    ) -> None:
        """Initialise a :class:`Markov` instance.

        Args:
            method: Partitioning method (e.g. ``"voronoi"``, ``"cartesian"``);
                see :data:`dem_mcm_coupling.partitioners.REGISTRY`.
            method_kwargs: Keyword arguments forwarded to the partitioner
                constructor (e.g. ``{"n_cells": 125}`` for Voronoi).
            data_source: Optional data source used by :meth:`load_dem_data`.
                When ``None``, a default Hugging Face source is used lazily.

        Raises:
            ValueError: If ``method`` is not a known partitioning method.
        """
        self.method: PartitioningMethod = validate_partitioning_method(method)
        self.data_source: DataSource | None = data_source
        logger.info("Initialising Markov with method=%r", self.method)

        self.partitioner: BasePartitioner = create_partitioner(
            self.method, **(method_kwargs or {})
        )

        # DEM data containers.
        self.datas: dict[int, pd.DataFrame] = {}
        self.coords: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.velocities: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.states: np.ndarray = np.array([], dtype=np.int64)

        # Markovian state.
        self.initial_state: StateVectorData | None = None
        self.transition_matrix: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_dem_data(
        self,
        particle_diameter: ParticleDiameter | None = None,
        data_source: DataSource | None = None,
    ) -> dict[int, pd.DataFrame]:
        """Load DEM particle data into memory.

        When no explicit ``data_source`` is given, a Hugging Face source is
        used and the parquet file is streamed from the Hub (cached for one
        hour).

        Args:
            particle_diameter: Diameter filter (``0.004``, ``0.008`` or
                ``None`` for every particle). Only used by the Hugging Face
                source.
            data_source: Optional source overriding ``self.data_source``.

        Returns:
            Mapping ``timestep_index -> DataFrame``.

        Raises:
            DataSourceError: If the data cannot be read.
        """
        source = data_source or self.data_source
        if source is not None:
            self.datas = source.read_timesteps()
        else:
            from dem_mcm_coupling._config import BUCKET_ID, get_bucket_prefix

            prefix = get_bucket_prefix(particle_diameter)
            parquet_path = (
                f"hf://buckets/{BUCKET_ID}/{prefix}/simulation_complete.parquet"
            )
            logger.info(
                "Loading DEM data from %s (diameter=%s)", prefix, particle_diameter
            )
            from dem_mcm_coupling.bucket_io import get_fs

            self.datas = _load_parquet_cached(parquet_path, id(get_fs()))

        logger.info(
            "Loaded %d timesteps (index %d → %d)",
            len(self.datas),
            min(self.datas),
            max(self.datas),
        )
        return self.datas

    def get_coords(self, timestep_indices: list[int] | int = 250) -> np.ndarray:
        """Return particle coordinates for one or several timesteps.

        When several timesteps are requested the coordinates are stacked
        (used to fit the partitioner over a representative window, e.g.
        ``[250, 300, 350]``).

        Args:
            timestep_indices: Timestep index or list of indices.

        Returns:
            Coordinates array of shape ``(n_particles, 3)`` (metres).

        Raises:
            KeyError: If a requested timestep is unavailable.
        """
        if isinstance(timestep_indices, int):
            timestep_indices = [timestep_indices]

        if not self.datas:
            logger.warning("Empty `datas`, loading DEM data automatically...")
            self.load_dem_data()

        coords_list: list[np.ndarray] = []
        for idx in timestep_indices:
            if idx not in self.datas:
                raise KeyError(
                    f"Timestep {idx} unavailable. Available: {list(self.datas)[:10]}..."
                )
            df = self.datas[idx]
            coords_list.append(
                df[["coordinates:0", "coordinates:1", "coordinates:2"]].to_numpy(
                    dtype=np.float32
                )
            )

        self.coords = np.vstack(coords_list) if len(coords_list) > 1 else coords_list[0]
        logger.info("Coordinates shape: %s", self.coords.shape)
        return self.coords

    def get_velocities(self, timestep_indices: list[int] | int = 250) -> np.ndarray:
        """Return particle velocities for one or several timesteps.

        Args:
            timestep_indices: Timestep index or list of indices.

        Returns:
            Velocity array of shape ``(n_particles, 3)`` (m/s).

        Raises:
            KeyError: If a requested timestep is unavailable.
        """
        if isinstance(timestep_indices, int):
            timestep_indices = [timestep_indices]

        if not self.datas:
            self.load_dem_data()

        vel_list: list[np.ndarray] = []
        for idx in timestep_indices:
            if idx not in self.datas:
                raise KeyError(f"Timestep {idx} unavailable.")
            df = self.datas[idx]
            vel_list.append(
                df[["Velocity:0", "Velocity:1", "Velocity:2"]].to_numpy(
                    dtype=np.float32
                )
            )

        self.velocities = np.vstack(vel_list) if len(vel_list) > 1 else vel_list[0]
        logger.info("Velocities shape: %s", self.velocities.shape)
        return self.velocities

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------

    def fit_partitioner(self, coordinates: np.ndarray) -> BasePartitioner:
        """Fit the partitioner on particle coordinates.

        Args:
            coordinates: Array of shape ``(n_samples, 3)``.

        Returns:
            The fitted partitioner (``self.partitioner``).
        """
        logger.info("Fitting partitioner on %d points...", coordinates.shape[0])
        self.partitioner.fit(coordinates)
        logger.info("Partitioner fitted: %d states", self.partitioner.n_cells)
        return self.partitioner

    def compute_states(self, coordinates: np.ndarray | None = None) -> np.ndarray:
        """Assign each particle to its partition state.

        Args:
            coordinates: Array of shape ``(n_particles, 3)``. When ``None``,
                ``self.coords`` is used.

        Returns:
            State index per particle, shape ``(n_particles,)``.

        Raises:
            ValueError: If the coordinates array is empty.
        """
        if coordinates is None:
            coordinates = self.coords
        if coordinates.size == 0:
            raise ValueError("Empty `coordinates`")

        self.states = self.partitioner.compute_states(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        )
        return self.states

    # ------------------------------------------------------------------
    # Markov state
    # ------------------------------------------------------------------

    def build_initial_state_vector(
        self,
        timestep_index: int,
        species_mask: np.ndarray | None = None,
        normalize: bool = True,
    ) -> StateVectorData:
        """Build the initial state vector ``phi(0)`` at a given timestep.

        ``phi_i(0)`` = number of particles in partition ``i``. An optional
        boolean species mask restricts the count to a particle species.

        Args:
            timestep_index: Timestep used to build ``phi(0)``.
            species_mask: Optional boolean mask, shape ``(n_particles,)``.
            normalize: When ``True`` (default), validate the normalisation.

        Returns:
            The initial :class:`StateVectorData`.

        Raises:
            RuntimeError: If the partitioner has not been fitted yet.
            ValueError: If the normalisation check fails and ``normalize``
                is ``True``.
            KeyError: If the timestep is unavailable.
        """
        if self.partitioner.n_cells == 0:
            raise RuntimeError("Partitioner not fitted. Call fit_partitioner() first.")

        coords = self.get_coords([timestep_index])
        states = self.compute_states(coords)

        if species_mask is not None:
            states = apply_species_mask(states, species_mask)
            total = int(np.asarray(species_mask).sum())
        else:
            total = len(states)

        phi = np.bincount(states, minlength=self.partitioner.n_cells).astype(np.float32)

        state = StateVectorData(
            phi=phi,
            timestamp=timestep_index,
            total_particles=total,
            description=f"Initial state at t={timestep_index}",
        )

        if normalize and not state.validate_normalization():
            raise ValueError(
                f"State not normalised: sum(phi)={phi.sum()}, expected={total}"
            )

        self.initial_state = state
        logger.info(
            "Initial state built: sum(phi)=%g, n_states=%d",
            phi.sum(),
            self.partitioner.n_cells,
        )
        return state

    def propagate_markov(
        self,
        initial_state: np.ndarray,
        transition_matrix: np.ndarray,
        n_steps: int,
        validate_normalization: bool = True,
    ) -> MarkovTrajectory:
        """Propagate a state vector through a transition matrix.

        Computes ``phi(t + 1) = phi(t) @ M``. The total particle count is
        invariant at every step when ``M`` is row-stochastic.

        Args:
            initial_state: Initial vector, shape ``(n_states,)``.
            transition_matrix: Row-stochastic matrix ``M`` of shape
                ``(n_states, n_states)`` (``M.sum(axis=1) == 1``).
            n_steps: Number of iterations.
            validate_normalization: When ``True``, warn when the total count
                drifts by more than 0.1% at any step.

        Returns:
            The full :class:`MarkovTrajectory`.

        Raises:
            ValueError: If the dimensions are incompatible.
        """
        n_states = transition_matrix.shape[0]

        if initial_state.shape[0] != n_states:
            raise ValueError(
                f"Dimension mismatch: initial_state={initial_state.shape[0]}, "
                f"M={n_states}x{n_states}"
            )
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("Transition matrix must be square")

        traj = np.zeros((n_steps + 1, n_states), dtype=np.float32)
        traj[0] = initial_state

        expected = float(initial_state.sum())
        logger.info("Markov propagation: %d steps...", n_steps)
        for t in range(1, n_steps + 1):
            traj[t] = traj[t - 1] @ transition_matrix
            if validate_normalization:
                relative_error = abs(float(traj[t].sum()) - expected) / max(
                    expected, 1.0
                )
                if relative_error > 1e-3:
                    logger.warning(
                        "Step %d: normalisation drift = %.4f%%",
                        t,
                        relative_error * 100,
                    )
        logger.info("Propagation finished")

        times = np.arange(n_steps + 1)
        return MarkovTrajectory(
            states=traj,
            times=times,
            times_seconds=times * TIMESTEP_TO_SECONDS,
            method=self.method,
            description=f"Markov propagation, {n_steps} steps",
        )

    # ------------------------------------------------------------------
    # Visualisation (optional dependencies)
    # ------------------------------------------------------------------

    def build_vtp(self, timestep_indices: list[int] | int = 250) -> Any:
        """Build a PyVista ``PolyData`` of the particles at given timesteps.

        Point scalars: partition state, diameter, particle id, residence time
        and velocity (when available).

        Args:
            timestep_indices: Timestep index or list of indices.

        Returns:
            The ``pyvista.PolyData`` object.

        Raises:
            ImportError: If ``pyvista`` is not installed
                (``pip install dem-mcm-coupling[viz]``).
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "build_vtp() requires pyvista: pip install dem-mcm-coupling[viz]"
            ) from exc

        if isinstance(timestep_indices, int):
            timestep_indices = [timestep_indices]

        coords = self.get_coords(timestep_indices)
        states = self.compute_states(coords)

        vtp = pv.PolyData(coords)
        vtp.point_data["partition"] = states

        if len(timestep_indices) == 1:
            df = self.datas[timestep_indices[0]]
            for col in ("Diameter", "Particle_ID", "Residence_Time"):
                if col in df.columns:
                    vtp.point_data[col] = df[col].to_numpy()
            vel_cols = ["Velocity:0", "Velocity:1", "Velocity:2"]
            if all(col in df.columns for col in vel_cols):
                vtp.point_data["Velocity"] = df[vel_cols].to_numpy()

        return vtp

    def visualize(self) -> None:
        """Render the partitioning in a Streamlit + PyVista viewer.

        Requires the optional extras ``streamlit``, ``pyvista`` and
        ``stpyvista`` (``pip install dem-mcm-coupling[app]``).

        Raises:
            ImportError: If one of the optional dependencies is missing.
        """
        try:
            import pyvista as pv
            import streamlit as st
            from stpyvista import stpyvista
        except ImportError as exc:
            raise ImportError(
                "visualize() requires streamlit, pyvista and stpyvista: "
                "pip install dem-mcm-coupling[app]"
            ) from exc

        st.subheader("🎨 Partitioner Visualization")

        vtp = self.build_vtp()

        # Older pyvista releases expose start_xvfb(); newer ones do not need
        # it, hence the defensive getattr.
        getattr(pv, "start_xvfb", lambda: None)()
        pv.OFF_SCREEN = True
        plotter = pv.Plotter(window_size=[600, 600], notebook=False)

        sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
        glyphs = vtp.glyph(geom=sphere, scale="Diameter", factor=1.0)
        plotter.add_mesh(
            glyphs, scalars="partition", cmap="tab10", show_scalar_bar=True
        )

        if st.checkbox("Activer plan de coupe"):
            direction = st.selectbox(
                "Direction coupe:", options=["xy", "yz", "xz", "oblique"]
            )
            normal_map = {
                "xy": (0, 0, 0.1),
                "yz": (0.1, 0, 0),
                "xz": (0, 0.1, 0),
                "oblique": (0.1, 0.1, 0),
            }
            clipped = glyphs.clip(normal=normal_map[direction], crinkle=True)
            plotter.clear()
            plotter.add_mesh(clipped, scalars="partition", cmap="tab10")

        plotter.camera_position = [
            (0.24, 0.32, 0.7),
            (0.02, 0.03, -0.02),
            (-0.12, 0.93, -0.34),
        ]
        stpyvista(plotter)

    def __repr__(self) -> str:
        """String representation of the instance."""
        return (
            f"Markov("
            f"method={self.method!r}, "
            f"n_cells={self.partitioner.n_cells}, "
            f"datas={len(self.datas)}, "
            f"coords_shape={self.coords.shape}"
            f")"
        )


@lru_cache(maxsize=4)
def _load_parquet_cached(parquet_path: str, fs_key: int) -> dict[int, pd.DataFrame]:
    """Stream the DEM parquet file once per hour (simple TTL-free cache)."""
    from dem_mcm_coupling.bucket_io import get_fs

    return load_parquet_as_timestep_dict(parquet_path=parquet_path, fs=get_fs())
