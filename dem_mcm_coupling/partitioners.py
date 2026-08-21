"""Spatial partitioning methods for Markov chain models.

This module provides the abstract base class and every concrete
implementation of the spatial partitioning of a granular mixer domain. Each
method subdivides the 3-D space into discrete states (cells) used to build
Markov transition matrices.

Common interface::

    partitioner = create_partitioner("voronoi", n_cells=125)
    partitioner.fit(coordinates)                 # (N, 3) numpy array
    states = partitioner.compute_states(x, y, z) # int64 state indices
    partitioner.save("output/")
    partitioner.load("output/")

Available methods:

``cartesian``
    Regular ``(x, y, z)`` grid.
``cylindrical``
    Cylindrical ``(r, theta, z)`` grid.
``voronoi``
    K-means clustering / Voronoi cells (the reference MCM method).
``quantile``
    Quantile-bounded grid (equi-population).
``octree``
    Adaptive octree (5 splitting strategies, axial or oblique).
``physics``
    K-means on position + velocity features.
``physics_full_vel``
    K-means on position + full velocity vector.
``spectral``
    Spectral clustering (graph topology).
``spectral_biclustering``
    Spectral biclustering of the position x velocity kinetics.
``gmm``
    Gaussian mixture model (ellipsoidal cells).
``adaptive``
    Adaptive zoning (fine bottom, coarse top).
``multizone``
    Two or more independent zones with different methods.
``single``
    Single cell (1-state baseline).
``dbscan``
    Density-based spatial clustering.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import numpy as np
from scipy.spatial import Voronoi, cKDTree
from sklearn.cluster import (
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
    SpectralBiclustering,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.svm import LinearSVC

logger = logging.getLogger(__name__)

__all__ = [
    "REGISTRY",
    "AdaptivePartitioner",
    "BasePartitioner",
    "CartesianPartitioner",
    "CylindricalPartitioner",
    "DBSCANPartitioner",
    "FullVectorVelocityKMeansPartitioner",
    "GaussianMixturePartitioner",
    "MultiZonePartitioner",
    "OctreePartitioner",
    "PhysicsAwarePartitioner",
    "QuantileGridPartitioner",
    "SingleCellPartitioner",
    "SpectralBiclusteringPartitioner",
    "SpectralClusteringPartitioner",
    "VoronoiPartitioner",
    "create_partitioner",
]


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================


class BasePartitioner(ABC):
    """Abstract base class for every spatial partitioner.

    Defines the common interface:

    * ``fit(coordinates)`` — learn cell boundaries from particle data;
    * ``compute_states(x, y, z[, vx, vy, vz])`` — assign each particle to a
      cell;
    * ``save(path)`` / ``load(path)`` — persist/restore the partitioner;
    * ``diagnostics(coordinates[, velocities])`` — per-cell population
      statistics.

    Subclasses must implement the ``n_cells`` and ``label`` properties and
    the ``fit`` and ``compute_states`` methods.
    """

    def __init__(self) -> None:
        #: Last computed state assignment, shape ``(n_particles,)``.
        self.states: np.ndarray = np.array([], dtype=np.int64)
        #: Whether the partitioner exploits velocity features.
        self.use_velocity: bool = False
        #: Optional DEM velocities ``(N, 3)`` used by physics-aware fits.
        self.dem_velocities: np.ndarray | None = None
        #: Name of the splitting strategy (folder/label naming).
        self.splitting_method: str | None = None

    # ── Abstract properties ────────────────────────────────────────────────

    @property
    @abstractmethod
    def n_cells(self) -> int:
        """Total number of partition cells (Markov states)."""

    @property
    @abstractmethod
    def label(self) -> str:
        """Unique identifier string (used for folder naming)."""

    # ── Abstract methods ───────────────────────────────────────────────────

    @abstractmethod
    def fit(self, coordinates: np.ndarray) -> BasePartitioner:
        """Learn the partition boundaries from particle coordinates.

        Args:
            coordinates: Array of shape ``(N, 3)`` with ``(x, y, z)``
                positions.

        Returns:
            ``self`` (fitted in place).
        """

    @abstractmethod
    def compute_states(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray | None = None,
        vy: np.ndarray | None = None,
        vz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assign each particle to its partition state (cell index).

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Optional X velocities (physics-aware methods only).
            vy: Optional Y velocities (physics-aware methods only).
            vz: Optional Z velocities (physics-aware methods only).

        Returns:
            Int64 state indices, shape ``(n_particles,)``.
        """

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the partitioner state to a directory.

        Args:
            path: Destination directory (created if needed).
        """
        os.makedirs(path, exist_ok=True)
        meta = {
            "type": type(self).__name__,
            "label": self.label,
            "n_cells": self.n_cells,
        }
        with open(os.path.join(path, "partitioner_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        self._save_data(str(path))

    def _save_data(self, path: str) -> None:
        """Subclass-specific serialisation hook."""

    def load(self, path: str | os.PathLike[str]) -> BasePartitioner:
        """Restore the partitioner state from a directory.

        Args:
            path: Source directory.

        Returns:
            ``self`` (loaded in place).
        """
        self._load_data(str(path))
        return self

    def _load_data(self, path: str) -> None:
        """Subclass-specific deserialisation hook."""

    # ── Diagnostics ────────────────────────────────────────────────────────

    def diagnostics(
        self,
        coordinates: np.ndarray,
        velocities: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        """Compute per-cell population statistics.

        Args:
            coordinates: Array of shape ``(N, 3)``.
            velocities: Optional array of shape ``(N, 3)`` forwarded to
                physics-aware partitioners.

        Returns:
            Dictionary with population minimum/maximum/mean/std, empty-cell
            count, visited-cell count and visited fraction.
        """
        coordinates = np.asarray(coordinates)
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        if velocities is not None:
            velocities = np.asarray(velocities)
            vx, vy, vz = velocities[:, 0], velocities[:, 1], velocities[:, 2]
            states = self.compute_states(x, y, z, vx, vy, vz)
        else:
            states = self.compute_states(x, y, z)

        counts = np.bincount(states, minlength=self.n_cells)
        visited = counts > 0
        return {
            "pop_min": int(counts[visited].min()) if visited.any() else 0,
            "pop_max": int(counts.max()),
            "pop_mean": float(counts.mean()),
            "pop_std": float(counts.std()),
            "n_empty": int((~visited).sum()),
            "n_visited": int(visited.sum()),
            "fraction_visited": float(visited.sum() / self.n_cells),
        }


# =============================================================================
# 1. CARTESIAN GRID
# =============================================================================


class CartesianPartitioner(BasePartitioner):
    """Regular Cartesian grid.

    Subdivides the domain into ``nx x ny x nz`` cells of equal size. Simple
    but unsuited to cylindrical geometries (empty corners).
    """

    def __init__(self, nx: int = 5, ny: int = 5, nz: int = 5) -> None:
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self._bounds: tuple[float, float, float, float, float, float] | None = None
        self.splitting_method = "cartesian"

    @property
    def n_cells(self) -> int:
        """Total number of cells (``nx * ny * nz``)."""
        return self.nx * self.ny * self.nz

    @property
    def label(self) -> str:
        """Grid label, e.g. ``cartesian_nx5_ny5_nz5``."""
        return f"cartesian_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self, coordinates: np.ndarray) -> CartesianPartitioner:
        """Compute the bounding box (with an ``eps`` margin) of the data.

        Args:
            coordinates: Array of shape ``(N, 3)``.

        Returns:
            ``self``.
        """
        coordinates = np.asarray(coordinates)
        eps = 0.001
        mins = coordinates.min(axis=0) - eps
        maxs = coordinates.max(axis=0) + eps
        self._bounds = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
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
        """Assign each particle to the cell containing its position.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Unused (kept for interface compatibility).
            vy: Unused (kept for interface compatibility).
            vz: Unused (kept for interface compatibility).

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if self._bounds is None:
            raise RuntimeError("CartesianPartitioner must be fitted before use")
        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        # Normalise each coordinate inside its cell range, then clip.
        ix = (
            np.clip(((x - xmin) * self.nx / (xmax - xmin)), 0, self.nx - 1)
            .round()
            .astype(int)
        )
        iy = (
            np.clip(((y - ymin) * self.ny / (ymax - ymin)), 0, self.ny - 1)
            .round()
            .astype(int)
        )
        iz = (
            np.clip(((z - zmin) * self.nz / (zmax - zmin)), 0, self.nz - 1)
            .round()
            .astype(int)
        )

        self.states = (ix + iy * self.nx + iz * self.nx * self.ny).astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _load_data(self, path: str) -> None:
        self._bounds = tuple(np.load(os.path.join(path, "bounds.npy")))


# =============================================================================
# 2. CYLINDRICAL GRID
# =============================================================================


class CylindricalPartitioner(BasePartitioner):
    """Cylindrical ``(r, theta, z)`` grid.

    Corrections applied over the naive grid:

    * angular edges carry an epsilon margin so that particles exactly on
      ``theta = 0`` or ``theta = 2*pi`` do not produce empty cells;
    * state numbering: ``state = ir + itheta*nr + iz*nr*ntheta``;
    * ``r_min_limit`` defaults to 0 but is detected automatically when the
      data do not cover ``r = 0`` (empty axis);
    * ``mask_in_zone`` is exposed for diagnostics.

    Two radial modes:

    * ``"equal_dr"`` — constant radial spacing;
    * ``"equal_area"`` — constant cross-section area (recommended).
    """

    def __init__(
        self,
        nr: int = 5,
        ntheta: int = 8,
        nz: int = 5,
        radial_mode: str = "equal_area",
        theta_min: float | None = None,
        theta_max: float | None = None,
        z_min_limit: float | None = None,
        z_max_limit: float | None = None,
        r_min_limit: float | None = None,
        r_max_limit: float | None = None,
    ) -> None:
        """Initialise a cylindrical grid.

        Args:
            nr: Number of radial cells.
            ntheta: Number of angular cells.
            nz: Number of axial cells.
            radial_mode: ``"equal_dr"`` or ``"equal_area"``.
            theta_min: Optional lower angular bound (radians). Defaults to 0.
            theta_max: Optional upper angular bound (radians). Defaults to
                ``2*pi``.
            z_min_limit: Optional lower axial bound (data if ``None``).
            z_max_limit: Optional upper axial bound (data if ``None``).
            r_min_limit: Optional lower radial bound (0 if ``None``).
            r_max_limit: Optional upper radial bound (data if ``None``).
        """
        super().__init__()
        self.nr = nr
        self.ntheta = ntheta
        self.nz = nz
        self.radial_mode = radial_mode

        # User-provided parameters.
        self.theta_min_input = theta_min
        self.theta_max_input = theta_max
        self.z_min_limit_input = z_min_limit
        self.z_max_limit_input = z_max_limit
        self.r_min_limit_input = r_min_limit
        self.r_max_limit_input = r_max_limit

        # Effective limits (computed at fit time).
        self.theta_min: float | None = None
        self.theta_max: float | None = None
        self.z_min_limit: float | None = None
        self.z_max_limit: float | None = None
        self.r_min_limit: float | None = None
        self.r_max_limit: float | None = None

        self._x_center: float | None = None
        self._y_center: float | None = None
        self._r_edges: np.ndarray | None = None
        self._z_edges: np.ndarray | None = None
        self._theta_edges: np.ndarray | None = None

        #: Boolean mask of the particles inside the active zone.
        self.mask_in_zone: np.ndarray | None = None
        self.splitting_method = "cylindrical"

    @property
    def n_cells(self) -> int:
        """Total number of cells (``nr * ntheta * nz``)."""
        return self.nr * self.ntheta * self.nz

    @property
    def label(self) -> str:
        """Grid label including radial mode and angular bounds."""
        tmin = f"{self.theta_min:.2f}" if self.theta_min is not None else "auto"
        tmax = f"{self.theta_max:.2f}" if self.theta_max is not None else "auto"
        return (
            f"cylindrical_nr{self.nr}_nth{self.ntheta}"
            f"_nz{self.nz}_{self.radial_mode}"
            f"_theta{tmin}_{tmax}"
        )

    def fit(self, coordinates: np.ndarray) -> CylindricalPartitioner:
        """Compute the radial, angular and axial cell edges from the data.

        Args:
            coordinates: Array of shape ``(N, 3)``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``radial_mode`` is unknown.
        """
        coordinates = np.asarray(coordinates)
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

        # Axis centred on the origin.
        self._x_center = 0.0
        self._y_center = 0.0

        dx = x - self._x_center
        dy = y - self._y_center
        r_all = np.sqrt(dx**2 + dy**2)

        self.r_min_limit = (
            self.r_min_limit_input if self.r_min_limit_input is not None else 0.0
        )
        self.r_max_limit = (
            self.r_max_limit_input
            if self.r_max_limit_input is not None
            else float(r_all.max())
        )

        self.theta_min = (
            self.theta_min_input if self.theta_min_input is not None else 0.0
        )
        self.theta_max = (
            self.theta_max_input if self.theta_max_input is not None else 2 * np.pi
        )

        self.z_min_limit = (
            self.z_min_limit_input
            if self.z_min_limit_input is not None
            else float(z.min())
        )
        self.z_max_limit = (
            self.z_max_limit_input
            if self.z_max_limit_input is not None
            else float(z.max())
        )

        # Radial edges.
        if self.radial_mode == "equal_area":
            r2_min = self.r_min_limit**2
            r2_max = self.r_max_limit**2
            self._r_edges = np.sqrt(np.linspace(r2_min, r2_max, self.nr + 1))
        elif self.radial_mode == "equal_dr":
            self._r_edges = np.linspace(self.r_min_limit, self.r_max_limit, self.nr + 1)
        else:
            raise ValueError(f"Unknown radial_mode: {self.radial_mode}")

        # Force the extreme edges to capture every particle.
        self._r_edges[0] = self.r_min_limit
        self._r_edges[-1] = self.r_max_limit * (1 + 1e-9)  # inclusive upper bound

        # Angular edges (slightly beyond 2*pi so theta ≈ 2*pi is included).
        self._theta_edges = np.linspace(self.theta_min, self.theta_max, self.ntheta + 1)
        self._theta_edges[-1] += 1e-9

        # Axial edges.
        self._z_edges = np.linspace(self.z_min_limit, self.z_max_limit, self.nz + 1)
        self._z_edges[-1] *= (1 + 1e-9) if self._z_edges[-1] > 0 else 1.0
        self._z_edges[-1] += 1e-9  # covers z_max <= 0 as well

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
        """Assign each particle to its ``(r, theta, z)`` cell.

        Particles outside the active zone receive the state ``-1`` (the
        pipeline convention for "out of zone").

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Unused (kept for interface compatibility).
            vy: Unused (kept for interface compatibility).
            vz: Unused (kept for interface compatibility).

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if (
            self._r_edges is None
            or self._theta_edges is None
            or self._z_edges is None
            or self._x_center is None
            or self._y_center is None
            or self.r_min_limit is None
            or self.r_max_limit is None
            or self.theta_min is None
            or self.theta_max is None
            or self.z_min_limit is None
            or self.z_max_limit is None
        ):
            raise RuntimeError("CylindricalPartitioner must be fitted before use")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self._x_center
        dy = y - self._y_center
        r = np.sqrt(dx**2 + dy**2)
        theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)

        mask_r = (r >= self.r_min_limit) & (r < self._r_edges[-1])
        mask_theta = (theta >= self.theta_min) & (theta < self._theta_edges[-1])
        mask_z = (z >= self.z_min_limit) & (z < self._z_edges[-1])
        mask = mask_r & mask_theta & mask_z

        # Out-of-zone particles receive state -1.
        states = np.full(len(x), -1, dtype=np.int64)

        if np.any(mask):
            r_v = r[mask]
            theta_v = theta[mask]
            z_v = z[mask]

            # searchsorted on precomputed edges guarantees a perfect
            # particle ↔ cell bijection.
            ir_v = np.clip(
                np.searchsorted(self._r_edges, r_v, side="right") - 1,
                0,
                self.nr - 1,
            )
            itheta_v = np.clip(
                np.searchsorted(self._theta_edges, theta_v, side="right") - 1,
                0,
                self.ntheta - 1,
            )
            iz_v = np.clip(
                np.searchsorted(self._z_edges, z_v, side="right") - 1,
                0,
                self.nz - 1,
            )

            # Numbering: state = ir + itheta*nr + iz*nr*ntheta.
            states[mask] = ir_v + itheta_v * self.nr + iz_v * self.nr * self.ntheta

        self.states = states
        self.mask_in_zone = mask
        return self.states

    def _save_data(self, path: str) -> None:
        data = {
            "nr": self.nr,
            "ntheta": self.ntheta,
            "nz": self.nz,
            "radial_mode": self.radial_mode,
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "z_min_limit": self.z_min_limit,
            "z_max_limit": self.z_max_limit,
            "r_min_limit": self.r_min_limit,
            "r_max_limit": self.r_max_limit,
            "x_center": self._x_center,
            "y_center": self._y_center,
            "r_edges": self._r_edges.tolist() if self._r_edges is not None else [],
            "theta_edges": self._theta_edges.tolist()
            if self._theta_edges is not None
            else [],
            "z_edges": self._z_edges.tolist() if self._z_edges is not None else [],
        }
        with open(os.path.join(path, "cylindrical_data.json"), "w") as fh:
            json.dump(data, fh, indent=2)

    def _load_data(self, path: str) -> None:
        with open(os.path.join(path, "cylindrical_data.json")) as fh:
            data = json.load(fh)
        self.nr = data["nr"]
        self.ntheta = data["ntheta"]
        self.nz = data["nz"]
        self.radial_mode = data["radial_mode"]
        self.theta_min = data["theta_min"]
        self.theta_max = data["theta_max"]
        self.z_min_limit = data["z_min_limit"]
        self.z_max_limit = data["z_max_limit"]
        self.r_min_limit = data["r_min_limit"]
        self.r_max_limit = data["r_max_limit"]
        self._x_center = data["x_center"]
        self._y_center = data["y_center"]
        self._r_edges = np.array(data["r_edges"])
        self._theta_edges = np.array(data["theta_edges"])
        self._z_edges = np.array(data["z_edges"])


# =============================================================================
# 3. VORONOI (K-MEANS)
# =============================================================================


class VoronoiPartitioner(BasePartitioner):
    """Voronoi partitioning via K-means clustering.

    Each cell is the basin of attraction of the closest centroid. The method
    adapts naturally to the particle density and is the reference MCM
    partitioning (Fan et al., Doucet et al.).
    """

    def __init__(self, n_cells: int = 125, random_state: int = 42) -> None:
        super().__init__()
        self._n_cells = n_cells
        self.random_state = random_state
        self.centroids: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self._voronoi_3d: Voronoi | None = None
        self._data_bounds_3d: tuple[float, float, float, float, float, float] | None = (
            None
        )
        self.splitting_method = "voronoi"

    @property
    def n_cells(self) -> int:
        """Number of K-means centroids."""
        return self._n_cells

    @property
    def label(self) -> str:
        """Label, e.g. ``voronoi_125cells``."""
        return f"voronoi_{self._n_cells}cells"

    def fit(self, coordinates: np.ndarray) -> VoronoiPartitioner:
        """Cluster the particle positions into ``n_cells`` centroids.

        Large inputs (> 500 000 points) are subsampled for the fit.

        Args:
            coordinates: Array of shape ``(N, 3)``.

        Returns:
            ``self``.
        """
        coordinates = np.asarray(coordinates)

        rng = np.random.RandomState(self.random_state)
        if len(coordinates) > 500_000:
            idx = rng.choice(len(coordinates), 500_000, replace=False)
            fit_data = coordinates[idx]
        else:
            fit_data = coordinates

        kmeans = KMeans(
            n_clusters=self._n_cells,
            random_state=self.random_state,
            init="k-means++",
            n_init=10,
        )
        kmeans.fit(fit_data)

        self.centroids = kmeans.cluster_centers_
        self._tree = cKDTree(self.centroids)
        self._voronoi_3d = Voronoi(self.centroids)
        self._data_bounds_3d = (
            float(coordinates[:, 0].min()),
            float(coordinates[:, 0].max()),
            float(coordinates[:, 1].min()),
            float(coordinates[:, 1].max()),
            float(coordinates[:, 2].min()),
            float(coordinates[:, 2].max()),
        )
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
        """Assign each particle to its nearest centroid.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Unused (kept for interface compatibility).
            vy: Unused (kept for interface compatibility).
            vz: Unused (kept for interface compatibility).

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if self._tree is None:
            raise RuntimeError("VoronoiPartitioner must be fitted before use")
        coords = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
        _, indices = self._tree.query(coords)
        self.states = indices.astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        if self.centroids is None:
            raise RuntimeError("VoronoiPartitioner must be fitted before save()")
        np.save(os.path.join(path, "centroids.npy"), self.centroids)

    def _load_data(self, path: str) -> None:
        self.centroids = np.load(os.path.join(path, "centroids.npy"))
        self._tree = cKDTree(self.centroids)
        self._n_cells = len(self.centroids)
        self._voronoi_3d = Voronoi(self.centroids)


# =============================================================================
# 4. QUANTILE GRID (EQUI-POPULATION)
# =============================================================================


class QuantileGridPartitioner(BasePartitioner):
    """Grid whose edges are data quantiles.

    The denser the particle concentration at a location, the finer the grid
    there: each cell contains approximately the same number of particles
    (marginal equi-population on each axis). Statistically more homogeneous
    than the regular Cartesian grid.
    """

    def __init__(self, nx: int = 5, ny: int = 5, nz: int = 5) -> None:
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self._x_edges: np.ndarray | None = None
        self._y_edges: np.ndarray | None = None
        self._z_edges: np.ndarray | None = None
        self.splitting_method = "quantile"

    @property
    def n_cells(self) -> int:
        """Total number of cells (``nx * ny * nz``)."""
        return self.nx * self.ny * self.nz

    @property
    def label(self) -> str:
        """Grid label, e.g. ``quantile_nx5_ny5_nz5``."""
        return f"quantile_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self, coordinates: np.ndarray) -> QuantileGridPartitioner:
        """Compute the quantile edges along each axis.

        Args:
            coordinates: Array of shape ``(N, 3)``.

        Returns:
            ``self``.
        """
        coordinates = np.asarray(coordinates)
        eps = 0.001
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

        # Each edge vector has nx+1 (ny+1, nz+1) entries: edge i is the
        # coordinate value of the i-th quantile along the axis.
        self._x_edges = np.quantile(x, np.linspace(0, 1, self.nx + 1))
        self._y_edges = np.quantile(y, np.linspace(0, 1, self.ny + 1))
        self._z_edges = np.quantile(z, np.linspace(0, 1, self.nz + 1))

        # Widen the extreme edges to capture every particle.
        self._x_edges[0] -= eps
        self._x_edges[-1] += eps
        self._y_edges[0] -= eps
        self._y_edges[-1] += eps
        self._z_edges[0] -= eps
        self._z_edges[-1] += eps
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
        """Assign each particle to its quantile cell.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Unused (kept for interface compatibility).
            vy: Unused (kept for interface compatibility).
            vz: Unused (kept for interface compatibility).

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if self._x_edges is None or self._y_edges is None or self._z_edges is None:
            raise RuntimeError("QuantileGridPartitioner must be fitted before use")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        ix = np.clip(
            np.searchsorted(self._x_edges, x, side="right") - 1, 0, self.nx - 1
        )
        iy = np.clip(
            np.searchsorted(self._y_edges, y, side="right") - 1, 0, self.ny - 1
        )
        iz = np.clip(
            np.searchsorted(self._z_edges, z, side="right") - 1, 0, self.nz - 1
        )

        self.states = (ix + iy * self.nx + iz * self.nx * self.ny).astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        if self._x_edges is None or self._y_edges is None or self._z_edges is None:
            raise RuntimeError("QuantileGridPartitioner must be fitted before save()")
        np.savez(
            os.path.join(path, "edges.npz"),
            x=self._x_edges,
            y=self._y_edges,
            z=self._z_edges,
        )

    def _load_data(self, path: str) -> None:
        data = np.load(os.path.join(path, "edges.npz"))
        self._x_edges = data["x"]
        self._y_edges = data["y"]
        self._z_edges = data["z"]


# =============================================================================
# 5. ADAPTIVE OCTREE
# =============================================================================


class OctreePartitioner(BasePartitioner):
    """Adaptive octree with axial or oblique splitting planes.

    The space is subdivided recursively: every cell holding more than
    ``max_particles`` particles is cut in two (or eight) until ``max_depth``
    levels are reached. Dense regions are therefore refined automatically.

    Two operating modes:

    1. **Axial** (``oblique_method`` is ``None`` or ``"axis"``) — classic
       8-octant subdivision along the coordinate medians (8-ary tree).
    2. **Oblique** (``oblique_method`` in ``"pca"``, ``"kmeans2"``,
       ``"2medians"``, ``"random"``, ``"svm"``) — binary subdivision with a
       plane oriented along the local particle geometry.

    Oblique splitting methods:

    * ``"pca"`` — plane orthogonal to the direction of maximal variance;
    * ``"kmeans2"`` — mediator plane between 2 k-means centroids;
    * ``"2medians"`` — mediator plane between 2 cluster medians (robust);
    * ``"random"`` — random plane direction, cut at the projective median;
    * ``"svm"`` — maximum-margin plane from a linear SVM.

    Advantage: dense zones are refined automatically. Drawback: the cell
    count is not controlled a priori.
    """

    #: Supported oblique splitting methods.
    OBLIQUE_METHODS: ClassVar[list[str]] = [
        "axis",
        "pca",
        "kmeans2",
        "2medians",
        "random",
        "svm",
    ]

    def __init__(
        self,
        max_particles: int = 100,
        max_depth: int = 5,
        transform_type: int | str = 0,
        oblique_method: str | None = None,
    ) -> None:
        """Initialise an adaptive octree.

        Args:
            max_particles: Filling threshold: a cell with fewer particles is
                not subdivided any further (becomes a leaf).
            max_depth: Maximum recursion depth, limiting the total number of
                subdivisions even in very dense zones.
            transform_type: Normalisation applied before splitting; only
                ``"normalize"`` (scale to ``[0, 1]``) is implemented.
            oblique_method: Splitting-plane strategy. ``None`` or ``"axis"``
                selects the axial octree; otherwise one of the oblique
                methods listed in :data:`OBLIQUE_METHODS`.
        """
        super().__init__()
        self.max_particles = max_particles
        self.max_depth = max_depth
        self.transform_type = transform_type
        self.oblique_method = oblique_method

        #: Axial leaves: list of (xmin, xmax, ymin, ymax, zmin, zmax).
        self._leaves: list[tuple[float, float, float, float, float, float]] = []
        #: Global bounding box (xmin, xmax, ymin, ymax, zmin, zmax).
        self._bounds: tuple[float, float, float, float, float, float] | None = None
        #: Statistics (min, max) used by the normalisation transform.
        self._stats: dict[str, np.ndarray] = {}
        #: Root of the binary oblique tree (dict-based node structure).
        self._oblique_root: dict[str, Any] | None = None

        self.splitting_method = f"octree_{oblique_method or 'axis'}"

    @property
    def n_cells(self) -> int:
        """Number of leaves (cells) of the oblique or axial tree."""
        if self._oblique_root is not None:
            return self._count_tree_leaves(self._oblique_root)
        return len(self._leaves)

    @property
    def label(self) -> str:
        """Label including the hyperparameters and the splitting method."""
        om = self.oblique_method or "axis"
        return f"octree_mp{self.max_particles}_md{self.max_depth}_{om}"

    # ── Oblique-tree helpers ────────────────────────────────────────────

    def _count_tree_leaves(self, node: dict[str, Any]) -> int:
        """Count the leaves of the binary oblique tree recursively."""
        if node["type"] == "leaf":
            return 1
        return self._count_tree_leaves(node["left"]) + self._count_tree_leaves(
            node["right"]
        )

    def _flatten_tree(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        """Flatten the binary oblique tree into a list of leaves."""
        if node["type"] == "leaf":
            return [node]
        return self._flatten_tree(node["left"]) + self._flatten_tree(node["right"])

    # ── Transform ───────────────────────────────────────────────────────

    def _apply_transform(self, coords: np.ndarray) -> np.ndarray:
        """Apply the optional normalisation transform to the coordinates."""
        if self.transform_type == "normalize" and self._stats:
            return (coords - self._stats["min"]) / (
                self._stats["max"] - self._stats["min"]
            )
        return coords

    # ── Fit ─────────────────────────────────────────────────────────────

    def fit(self, coordinates: np.ndarray) -> OctreePartitioner:
        """Build the partitioning tree from 3-D coordinates.

        Steps:
          1. compute the (min, max) statistics used by the normalisation;
          2. transform the coordinates when requested;
          3. define the global bounding box (with an ``eps`` margin);
          4. build a binary oblique tree or a classic 8-ary octree.

        Args:
            coordinates: Array of shape ``(N, 3)``.

        Returns:
            ``self``.
        """
        coordinates = np.asarray(coordinates)
        eps = 0.001
        self._stats["min"] = coordinates.min(axis=0) - eps
        self._stats["max"] = coordinates.max(axis=0) + eps
        transformed_coords = self._apply_transform(coordinates)

        # Global bounding box (xmin, xmax, ymin, ymax, zmin, zmax).
        self._bounds = (
            float(transformed_coords[:, 0].min()) - eps,
            float(transformed_coords[:, 0].max()) + eps,
            float(transformed_coords[:, 1].min()) - eps,
            float(transformed_coords[:, 1].max()) + eps,
            float(transformed_coords[:, 2].min()) - eps,
            float(transformed_coords[:, 2].max()) + eps,
        )

        self._leaves = []
        if self.oblique_method not in (None, "axis"):
            # Oblique mode: binary tree with oriented cutting planes.
            self._oblique_root = self._build_oblique_tree(
                transformed_coords, self._bounds, depth=0, halfspaces_sofar=[]
            )
        else:
            # Axial mode: classic octree subdivision on x, y, z medians.
            self._oblique_root = None
            self._subdivide(transformed_coords, self._bounds, depth=0)
        return self

    # ── Axial subdivision (classic octree) ───────────────────────────────

    def _subdivide(
        self,
        coords: np.ndarray,
        bounds: tuple[float, float, float, float, float, float],
        depth: int,
    ) -> None:
        """Subdivide a cell into 8 octants along the coordinate medians.

        Stopping conditions: the cell holds at most ``max_particles``
        particles or the maximum depth is reached — it is then stored as a
        leaf. Otherwise the median of x, y and z is computed, each particle
        receives a binary octant code (bit 0 = x, bit 1 = y, bit 2 = z) and
        every non-empty octant is subdivided recursively.

        Args:
            coords: Particles of the current cell, shape ``(N, 3)``.
            bounds: Bounding box ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of
                the current cell.
            depth: Current tree depth.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        n_in = len(coords)

        if n_in <= self.max_particles or depth >= self.max_depth:
            self._leaves.append(bounds)
            return

        xmid = np.median(coords[:, 0])
        ymid = np.median(coords[:, 1])
        zmid = np.median(coords[:, 2])

        # Binary octant code:
        #   bit 0 (weight 1) = x side (0: left, 1: right)
        #   bit 1 (weight 2) = y side (0: down, 1: up)
        #   bit 2 (weight 4) = z side (0: back, 1: front)
        octant = (
            (coords[:, 0] >= xmid).astype(np.int64)
            + (coords[:, 1] >= ymid).astype(np.int64) * 2
            + (coords[:, 2] >= zmid).astype(np.int64) * 4
        )

        for idx in range(8):
            # Decode the bits into the child bounding box coordinates.
            ix, iy, iz = idx % 2, (idx // 2) % 2, idx // 4
            child_bounds = (
                xmid if ix else xmin,
                xmax if ix else xmid,
                ymid if iy else ymin,
                ymax if iy else ymid,
                zmid if iz else zmin,
                zmax if iz else zmid,
            )
            child_mask = octant == idx
            self._subdivide(coords[child_mask], child_bounds, depth + 1)

    # ── Oblique splitting plane ──────────────────────────────────────────

    def _find_splitting_plane(
        self, coords: np.ndarray, method: str
    ) -> tuple[np.ndarray, float]:
        """Compute a cutting plane ``normal · x = offset`` for the particles.

        Points with ``normal · x <= offset`` go to the left child, the others
        to the right child.

        Args:
            coords: Particles of the current cell, shape ``(N, 3)``.
            method: Splitting strategy (``"pca"``, ``"kmeans2"``,
                ``"2medians"``, ``"random"`` or ``"svm"``).

        Returns:
            Tuple ``(normal, offset)`` where ``normal`` is a unit vector of
            shape ``(3,)``.

        Raises:
            ValueError: If ``method`` is unknown.
        """
        if len(coords) < 2:
            # Fewer than 2 points: default plane (x = 0).
            return np.array([1.0, 0.0, 0.0]), 0.0

        if method == "pca":
            # PCA: plane orthogonal to the direction of maximal variance,
            # cut at the median of the projections onto that direction.
            cov = np.cov(coords, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, np.argmax(eigvals)]
            proj = coords @ normal
            offset = np.median(proj)

        elif method == "kmeans2":
            # 2-cluster k-means: mediator plane of the centroids.
            # Subsampled to 10 000 points max to limit the cost.
            n = min(len(coords), 10_000)
            kmeans = KMeans(n_clusters=2, n_init=3, random_state=42).fit(coords[:n])
            c1, c2 = kmeans.cluster_centers_
            normal = c2 - c1
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
            offset = ((c1 + c2) / 2) @ normal

        elif method == "2medians":
            # Robust k-means variant: cluster centres are L1 medians instead
            # of L2 means (less sensitive to outliers).
            rng = np.random.RandomState(42)
            idx = rng.choice(len(coords), min(2, len(coords)), replace=False)
            c1, c2 = coords[idx].copy()
            for _ in range(5):
                d1 = np.linalg.norm(coords - c1, axis=1)
                d2 = np.linalg.norm(coords - c2, axis=1)
                labels = (d1 <= d2).astype(int)
                if (labels == 0).any():
                    c1 = np.median(coords[labels == 0], axis=0)
                if (labels == 1).any():
                    c2 = np.median(coords[labels == 1], axis=0)
            normal = c2 - c1
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
            offset = ((c1 + c2) / 2) @ normal

        elif method == "random":
            # Random plane: uniform direction on the unit sphere, cut at the
            # median of the projections (two equally sized halves).
            rng = np.random.RandomState(4)
            normal = rng.randn(3)
            normal /= np.linalg.norm(normal)
            proj = coords @ normal
            offset = np.median(proj)

        elif method == "svm":
            # Linear SVM maximum-margin plane. Binary labels are generated by
            # projecting on the PCA direction of maximal variance (class 1
            # above the median, class 0 below).
            cov = np.cov(coords, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            pca_normal = eigvecs[:, np.argmax(eigvals)]
            proj_pca = coords @ pca_normal
            labels = (proj_pca > np.median(proj_pca)).astype(int)
            if len(np.unique(labels)) < 2:
                return np.array([1.0, 0.0, 0.0]), 0.0
            svm = LinearSVC(max_iter=1000, C=1.0, dual="auto", random_state=42)
            svm.fit(coords, labels)
            w = svm.coef_[0].astype(np.float64)
            wn = np.linalg.norm(w)
            if wn < 1e-12:
                normal, offset = np.array([1.0, 0.0, 0.0]), 0.0
            else:
                normal = w / wn
                offset = -svm.intercept_[0].item() / wn

        else:
            raise ValueError(f"Unknown oblique method: {method}")

        return normal, offset

    # ── Oblique tree construction ────────────────────────────────────────

    def _build_oblique_tree(
        self,
        coords: np.ndarray,
        bounds: tuple[float, float, float, float, float, float],
        depth: int,
        halfspaces_sofar: list[tuple[np.ndarray, float, str]],
    ) -> dict[str, Any]:
        """Build the binary oblique tree recursively.

        At each node a cutting plane is computed with
        :meth:`_find_splitting_plane`; particles are split into ``proj <=
        offset`` (left) and ``proj > offset`` (right), and each side is
        subdivided recursively.

        Internal node structure::

            {"type": "internal", "normal": vec3, "offset": float,
             "left": node, "right": node}

        Leaf structure::

            {"type": "leaf", "bounds": (...), "centroid": vec3,
             "halfspaces": [(normal, offset, "le"|"gt"), ...]}

        ``halfspaces_sofar`` accumulates the cutting planes from the root to
        the current node; each entry is ``(normal, offset, side)`` where side
        is ``"le"`` (<=) or ``"gt"`` (>). A particle is assigned to a leaf by
        evaluating every plane along the path.

        Args:
            coords: Particles of the current cell, shape ``(N, 3)``.
            bounds: Bounding box of the current cell.
            depth: Current tree depth.
            halfspaces_sofar: Halfspaces already crossed from the root.

        Returns:
            The node dictionary.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        n_in = len(coords)

        def _make_leaf() -> dict[str, Any]:
            center = coords.mean(axis=0) if n_in > 0 else np.zeros(3)
            return {
                "type": "leaf",
                "bounds": (xmin, xmax, ymin, ymax, zmin, zmax),
                "centroid": center,
                "halfspaces": list(halfspaces_sofar),
            }

        # Stopping condition: leaf.
        if n_in <= self.max_particles or depth >= self.max_depth:
            return _make_leaf()

        # Cutting plane.
        assert self.oblique_method is not None  # guarded by fit()
        normal, offset = self._find_splitting_plane(coords, self.oblique_method)
        proj = coords @ normal
        left_mask = proj <= offset
        right_mask = proj > offset
        left_coords = coords[left_mask]
        right_coords = coords[right_mask]

        # Safety: if the plane does not separate anything (all points on the
        # same side), force a leaf to avoid an infinite loop.
        if len(left_coords) == 0 or len(right_coords) == 0:
            return _make_leaf()

        hs_left = [*halfspaces_sofar, (normal, offset, "le")]
        hs_right = [*halfspaces_sofar, (normal, offset, "gt")]

        return {
            "type": "internal",
            "normal": normal,
            "offset": offset,
            "left": self._build_oblique_tree(left_coords, bounds, depth + 1, hs_left),
            "right": self._build_oblique_tree(
                right_coords, bounds, depth + 1, hs_right
            ),
        }

    # ── State assignment ─────────────────────────────────────────────────

    def _assign_state_by_halfspaces(
        self, coords: np.ndarray, leaves: list[dict[str, Any]]
    ) -> np.ndarray:
        """Assign each particle to a leaf by evaluating the halfspaces.

        For each leaf, all the accumulated tests ``(normal·x <= offset)`` or
        ``(normal·x > offset)`` are applied sequentially to determine which
        particles belong to that cell. Unassigned particles (pathological
        numerical case) receive the closest leaf via cKDTree.

        Args:
            coords: Particles, shape ``(N, 3)``.
            leaves: Flattened list of leaf nodes.

        Returns:
            State index per particle, shape ``(N,)``.
        """
        n = len(coords)
        states = np.full(n, -1, dtype=np.int64)
        for cell_id, leaf in enumerate(leaves):
            mask = np.ones(n, dtype=bool)
            for normal, offset, side in leaf["halfspaces"]:
                proj = coords @ normal
                mask &= proj <= offset if side == "le" else proj > offset
            states[mask] = cell_id

        unassigned = states == -1
        if unassigned.any():
            centers = np.array([leaf["centroid"] for leaf in leaves])
            tree = cKDTree(centers)
            _, idx = tree.query(coords[unassigned])
            states[unassigned] = idx
        return states

    def _assign_state_by_boxes(self, coords: np.ndarray) -> np.ndarray:
        """Assign each particle to an axial leaf via bounding-box tests.

        Unassigned particles receive the closest box centre via cKDTree.

        Args:
            coords: Particles, shape ``(N, 3)``.

        Returns:
            State index per particle, shape ``(N,)``.
        """
        n = len(coords)
        states = np.full(n, -1, dtype=np.int64)
        for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
            mask = (
                (coords[:, 0] >= xmin)
                & (coords[:, 0] < xmax)
                & (coords[:, 1] >= ymin)
                & (coords[:, 1] < ymax)
                & (coords[:, 2] >= zmin)
                & (coords[:, 2] < zmax)
            )
            states[mask] = cell_id

        unassigned = states == -1
        if unassigned.any():
            centers = np.array(
                [
                    ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
                    for b in self._leaves
                ]
            )
            tree = cKDTree(centers)
            _, idx = tree.query(coords[unassigned])
            states[unassigned] = idx
        return states

    def compute_states(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray | None = None,
        vy: np.ndarray | None = None,
        vz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Assign each particle to its octree cell.

        Oblique mode: the tree is flattened into leaves and the halfspaces of
        each leaf are evaluated for every particle (brute force,
        ``O(N x n_leaves x depth)``). Axial mode: bounding-box tests.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Unused (kept for interface compatibility).
            vy: Unused (kept for interface compatibility).
            vz: Unused (kept for interface compatibility).

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        coords = np.column_stack(
            [
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                np.asarray(z, dtype=np.float64),
            ]
        )

        if self._oblique_root is not None:
            leaves = self._flatten_tree(self._oblique_root)
            states = self._assign_state_by_halfspaces(coords, leaves)
        else:
            states = self._assign_state_by_boxes(coords)

        self.states = states
        return self.states

    # ── Save / Load ──────────────────────────────────────────────────────

    def _save_data(self, path: str) -> None:
        """Save the tree (oblique or axial) to disk.

        Oblique mode: pickle of the binary tree + a text file holding the
        method name. Axial mode: numpy array of the leaves ``(N, 6)``. The
        global bounding box is always saved.
        """
        if self._oblique_root is not None:
            with open(os.path.join(path, "oblique_tree.pkl"), "wb") as fh:
                pickle.dump(self._oblique_root, fh, protocol=pickle.HIGHEST_PROTOCOL)
            with open(os.path.join(path, "octree_mode.txt"), "w") as fh:
                fh.write("oblique\n")
                fh.write(f"{self.oblique_method or 'axis'}\n")
        else:
            leaves_arr = np.array(self._leaves)
            np.save(os.path.join(path, "leaves.npy"), leaves_arr)
            with open(os.path.join(path, "octree_mode.txt"), "w") as fh:
                fh.write("axis\n")

        if self._bounds is not None:
            np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _load_data(self, path: str) -> None:
        """Load a previously saved tree.

        The mode (oblique or axial) is auto-detected from
        ``octree_mode.txt``; in oblique mode the splitting method name is
        restored as well.
        """
        mode_path = os.path.join(path, "octree_mode.txt")
        if os.path.exists(mode_path):
            with open(mode_path) as fh:
                mode = fh.readline().strip()
            if mode == "oblique":
                with open(os.path.join(path, "oblique_tree.pkl"), "rb") as fh:
                    self._oblique_root = pickle.load(fh)
                with open(mode_path) as fh:
                    lines = fh.readlines()
                if len(lines) > 1:
                    self.oblique_method = lines[1].strip()
                self._leaves = []
            else:
                leaves_arr = np.load(os.path.join(path, "leaves.npy"))
                self._leaves = [tuple(row) for row in leaves_arr]
                self._oblique_root = None
        else:
            leaves_arr = np.load(os.path.join(path, "leaves.npy"))
            self._leaves = [tuple(row) for row in leaves_arr]
            self._oblique_root = None

        bounds_path = os.path.join(path, "bounds.npy")
        if os.path.exists(bounds_path):
            self._bounds = tuple(np.load(bounds_path))


# =============================================================================
# 6. PHYSICS-AWARE (POSITION + VELOCITY)
# =============================================================================


class PhysicsAwarePartitioner(BasePartitioner):
    """K-means on physical features (position + optional velocity).

    With ``use_velocities=False`` the clustering is equivalent to a Voronoi
    partitioning on normalised positions. With ``use_velocities=True`` the
    (relative) velocity is added, following ``velocity_mode``:

    * ``"norm"`` — 4-D features ``(x, y, z, |v_rel|)``;
    * ``"components"`` — 6-D features ``(x, y, z, vx_rel, vy_rel, vz_rel)``.

    The relative velocity is the fluctuation in the rotating frame, obtained
    by subtracting the solid-body rotation around z (omega = 4 rad/s).

    Usage::

        part = PhysicsAwarePartitioner(n_cells=125, velocity_weight=0.5,
                                       velocity_mode="norm")
        part.dem_velocities = velocities
        part.fit(positions, use_velocities=True)
    """

    #: Angular velocity of the mixer (rad/s).
    OMEGA: ClassVar[float] = 4.0

    def __init__(
        self,
        n_cells: int = 125,
        velocity_weight: float = 0.5,
        random_state: int = 42,
        velocity_mode: str = "norm",
    ) -> None:
        super().__init__()
        self._n_cells = n_cells
        self.velocity_weight = velocity_weight
        self.random_state = random_state
        self.velocity_mode = velocity_mode
        self._centroids: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._n_features = 3
        self.features: np.ndarray | None = None
        self.use_velocity = False
        self.splitting_method = "physics"

    @property
    def n_cells(self) -> int:
        """Number of clusters."""
        return self._n_cells

    @property
    def label(self) -> str:
        """Label encoding the velocity mode and weight."""
        if not self.use_velocity or self._n_features == 3:
            suffix = "pos"
        elif self.velocity_mode == "norm":
            suffix = "normvel"
        else:
            suffix = "compvel"
        return f"physics_{self._n_cells}cells_{suffix}_vw{self.velocity_weight}"

    def _compute_relative_velocity(
        self, coordinates: np.ndarray, vel: np.ndarray
    ) -> np.ndarray:
        """Compute the relative velocity in the rotating frame.

        Subtracts the solid-body rotation around z (omega = 4 rad/s) from the
        absolute velocity.

        Args:
            coordinates: Positions, shape ``(N, 3)``.
            vel: Absolute velocities, shape ``(N, 3)``.

        Returns:
            Relative velocities, shape ``(N, 3)``.
        """
        x = coordinates[:, 0]
        y = coordinates[:, 1]

        # Entrainment velocity (solid rotation around z).
        vx_entr = -self.OMEGA * y
        vy_entr = self.OMEGA * x

        return np.column_stack(
            [
                vel[:, 0] - vx_entr,
                vel[:, 1] - vy_entr,
                vel[:, 2],  # axial component unchanged
            ]
        )

    def fit(
        self,
        coordinates: np.ndarray,
        use_velocities: bool | None = None,
    ) -> PhysicsAwarePartitioner:
        """Fit the partitioner on positions, optionally with velocities.

        When ``use_velocities`` is true and :attr:`dem_velocities` is set,
        the features are built according to :attr:`velocity_mode`; otherwise
        the fit runs on positions only. The velocity is the *relative*
        velocity (fluctuation in the rotating frame).

        Args:
            coordinates: Positions, shape ``(N, 3)``.
            use_velocities: Overrides :attr:`use_velocity` when given.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``velocity_mode`` is unknown.
        """
        coordinates = np.asarray(coordinates)
        use_velocities = self.use_velocity if use_velocities is None else use_velocities

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) != len(coordinates):
                logger.warning(
                    "Velocity/coordinate mismatch (%d vs %d), falling back "
                    "to positions only",
                    len(vel),
                    len(coordinates),
                )
            else:
                v_rel = self._compute_relative_velocity(coordinates, vel)

                if self.velocity_mode == "norm":
                    # 4-D: (x, y, z, |v_rel|).
                    speed = np.linalg.norm(v_rel, axis=1, keepdims=True)
                    self.features = np.hstack([coordinates, speed])
                    self._n_features = 4
                    return self._fit_internal(self.features, n_pos=3)

                if self.velocity_mode == "components":
                    # 6-D: (x, y, z, vx_rel, vy_rel, vz_rel).
                    self.features = np.hstack([coordinates, v_rel])
                    self._n_features = 6
                    return self._fit_internal(self.features, n_pos=3)

                raise ValueError(
                    f"Unknown velocity_mode: {self.velocity_mode!r}. "
                    "Choose 'norm' or 'components'."
                )

        # Fallback: positions only (3-D).
        self.features = coordinates
        self._n_features = 3
        return self._fit_internal(coordinates, n_pos=3)

    def _fit_internal(
        self, features: np.ndarray, n_pos: int = 3
    ) -> PhysicsAwarePartitioner:
        """Normalise, apply the feature weights, then run a K-means.

        Args:
            features: Feature matrix, shape ``(N, n_features)``.
            n_pos: Number of leading position columns; the following columns
                receive the ``velocity_weight`` factor.

        Returns:
            ``self``.
        """
        self._n_features = features.shape[1]

        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        x_scaled = (features - self._mean) / self._std

        # Explicit weight for the velocity dimensions.
        if x_scaled.shape[1] > n_pos:
            weights = np.ones(x_scaled.shape[1])
            weights[n_pos:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        # Subsample large inputs.
        rng = np.random.RandomState(self.random_state)
        if len(x_scaled) > 500_000:
            idx = rng.choice(len(x_scaled), 500_000, replace=False)
            x_fit = x_scaled[idx]
        else:
            x_fit = x_scaled

        kmeans = KMeans(
            n_clusters=self._n_cells,
            random_state=self.random_state,
            init="k-means++",
            n_init=10,
        )
        kmeans.fit(x_fit)
        self._centroids = kmeans.cluster_centers_
        self._tree = cKDTree(self._centroids)
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
        """Assign each particle to the closest cluster.

        When the model was trained with velocity features, the velocities
        must be provided (zero padding otherwise). The relative velocity is
        used, consistently with :meth:`fit`.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Optional X velocities.
            vy: Optional Y velocities.
            vz: Optional Z velocities.

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if self._tree is None or self._mean is None or self._std is None:
            raise RuntimeError("PhysicsAwarePartitioner must be fitted before use")

        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
        has_velocity = vx is not None and vy is not None and vz is not None

        if self._n_features == 4:
            # "norm" mode: need |v_rel|.
            if has_velocity:
                vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
                v_rel = self._compute_relative_velocity(pos, vel)
                speed = np.linalg.norm(v_rel, axis=1, keepdims=True)
            else:
                speed = np.zeros((len(pos), 1))
            features = np.hstack([pos, speed])

        elif self._n_features == 6:
            # "components" mode: need (vx_rel, vy_rel, vz_rel).
            if has_velocity:
                vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
                v_rel = self._compute_relative_velocity(pos, vel)
                features = np.hstack([pos, v_rel])
            else:
                logger.warning(
                    "velocity_mode='components' but velocities absent → zero padding"
                )
                features = np.hstack([pos, np.zeros((len(pos), 3))])

        else:
            # Positions only (3-D).
            features = pos

        x_scaled = (features - self._mean) / self._std
        if x_scaled.shape[1] > 3:
            weights = np.ones(x_scaled.shape[1])
            weights[3:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        _, indices = self._tree.query(x_scaled)
        self.states = indices.astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        if self._centroids is None or self._mean is None or self._std is None:
            raise RuntimeError("PhysicsAwarePartitioner must be fitted before save()")
        np.save(os.path.join(path, "centroids.npy"), self._centroids)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "physics_params.json"), "w") as fh:
            json.dump(
                {
                    "n_features": self._n_features,
                    "velocity_mode": self.velocity_mode,
                },
                fh,
            )

    def _load_data(self, path: str) -> None:
        self._centroids = np.load(os.path.join(path, "centroids.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._centroids)
        self._n_cells = len(self._centroids)
        with open(os.path.join(path, "physics_params.json")) as fh:
            data = json.load(fh)
            self._n_features = data["n_features"]
            # Backward compatibility with older saves.
            self.velocity_mode = data.get("velocity_mode", "norm")


# =============================================================================
# 7. FULL-VECTOR-VELOCITY K-MEANS
# =============================================================================


class FullVectorVelocityKMeansPartitioner(BasePartitioner):
    """K-means on the full velocity vector ``(vx, vy, vz)``.

    Unlike :class:`PhysicsAwarePartitioner` (which uses the velocity norm),
    this variant captures the directionality of the flow (related to the
    streamlines analysis of Doucet 2008).
    """

    def __init__(
        self,
        n_cells: int = 125,
        velocity_weight: float = 0.5,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self._n_cells = n_cells
        self.velocity_weight = velocity_weight
        self.random_state = random_state
        self._centroids: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._n_features = 6  # x, y, z, vx, vy, vz
        self.features: np.ndarray | None = None
        self.use_velocity = True
        self.splitting_method = "fullvel_kmeans"

    @property
    def n_cells(self) -> int:
        """Number of clusters."""
        return self._n_cells

    @property
    def label(self) -> str:
        """Label encoding the velocity weight."""
        return f"fullvel_kmeans_{self._n_cells}cells_vw{self.velocity_weight}"

    def fit(
        self,
        coordinates: np.ndarray,
        use_velocities: bool | None = None,
    ) -> FullVectorVelocityKMeansPartitioner:
        """Fit on positions, optionally concatenated with the full velocity.

        Args:
            coordinates: Positions, shape ``(N, 3)``.
            use_velocities: Overrides :attr:`use_velocity` when given.

        Returns:
            ``self``.
        """
        use_velocities = self.use_velocity if use_velocities is None else use_velocities
        coordinates = np.asarray(coordinates)

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) == len(coordinates):
                # Full velocity vector: (N, 6) features.
                self.features = np.hstack([coordinates, vel])
                self._n_features = 6
                return self._fit_internal(self.features, n_pos=3)
            logger.warning(
                "Velocity/coordinate mismatch (%d vs %d), falling back to "
                "positions only",
                len(vel),
                len(coordinates),
            )

        self.features = coordinates
        self._n_features = 3
        return self._fit_internal(coordinates, n_pos=3)

    def _fit_internal(
        self, features: np.ndarray, n_pos: int = 3
    ) -> FullVectorVelocityKMeansPartitioner:
        """Normalise, weight, then run a MiniBatchKMeans."""
        self._n_features = features.shape[1]

        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        x_scaled = (features - self._mean) / self._std

        if x_scaled.shape[1] > n_pos:
            weights = np.ones(x_scaled.shape[1])
            weights[n_pos:] = self.velocity_weight  # weight on vx, vy, vz
            x_scaled = x_scaled * weights[np.newaxis, :]

        rng = np.random.RandomState(self.random_state)
        if len(x_scaled) > 500_000:
            idx = rng.choice(len(x_scaled), 500_000, replace=False)
            x_fit = x_scaled[idx]
        else:
            x_fit = x_scaled

        kmeans = MiniBatchKMeans(
            n_clusters=self._n_cells,
            random_state=self.random_state,
            batch_size=min(10_000, len(x_fit)),
            n_init=10,
        )
        kmeans.fit(x_fit)
        self._centroids = kmeans.cluster_centers_
        self._tree = cKDTree(self._centroids)
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
        """Assign each particle to the closest cluster.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Optional X velocities.
            vy: Optional Y velocities.
            vz: Optional Z velocities.

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if self._tree is None or self._mean is None or self._std is None:
            raise RuntimeError(
                "FullVectorVelocityKMeansPartitioner must be fitted before use"
            )
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])

        if (
            self._n_features == 6
            and vx is not None
            and vy is not None
            and vz is not None
        ):
            vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
            features = np.hstack([pos, vel])
        elif self._n_features == 6:
            # Trained with velocity but none provided → zero padding.
            features = np.hstack([pos, np.zeros((len(pos), 3))])
        else:
            features = pos

        x_scaled = (features - self._mean) / self._std
        if x_scaled.shape[1] > 3:
            weights = np.ones(x_scaled.shape[1])
            weights[3:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        _, indices = self._tree.query(x_scaled)
        self.states = indices.astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        if self._centroids is None or self._mean is None or self._std is None:
            raise RuntimeError(
                "FullVectorVelocityKMeansPartitioner must be fitted before save()"
            )
        np.save(os.path.join(path, "centroids.npy"), self._centroids)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "params.json"), "w") as fh:
            json.dump(
                {
                    "n_features": self._n_features,
                    "n_cells": self._n_cells,
                    "vw": self.velocity_weight,
                },
                fh,
            )

    def _load_data(self, path: str) -> None:
        self._centroids = np.load(os.path.join(path, "centroids.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._centroids)
        with open(os.path.join(path, "params.json")) as fh:
            params = json.load(fh)
            self._n_features = params["n_features"]


# =============================================================================
# 8. SPECTRAL CLUSTERING
# =============================================================================


class SpectralClusteringPartitioner(BasePartitioner):
    """Spectral clustering capturing the topological structures of the flow.

    Related to the collective-modes/SVD analysis of Tjakra 2013. The fit is
    performed on a subsample (spectral clustering is O(N²)/O(N³)); inference
    is a 1-nearest-neighbour search against the support points.
    """

    def __init__(
        self,
        n_cells: int = 125,
        velocity_weight: float = 1.0,
        n_neighbors: int = 15,
        max_samples: int = 5000,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self._n_cells = n_cells
        self.velocity_weight = velocity_weight
        self.n_neighbors = n_neighbors
        self.max_samples = max_samples
        self.random_state = random_state
        self._support_data: np.ndarray | None = None
        self._support_labels: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._n_features = 4
        self.features: np.ndarray | None = None
        self.use_velocity = True
        self.splitting_method = "spectral"

    @property
    def n_cells(self) -> int:
        """Number of clusters."""
        return self._n_cells

    @property
    def label(self) -> str:
        """Label encoding the velocity weight and neighbourhood size."""
        return (
            f"spectral_{self._n_cells}cells_vw{self.velocity_weight}"
            f"_k{self.n_neighbors}_{self.use_velocity}_{self._n_features}"
        )

    def _compute_relative_velocity(
        self, coordinates: np.ndarray, vel: np.ndarray
    ) -> np.ndarray:
        """Compute the relative velocity in the rotating frame (omega=4)."""
        omega = 4.0
        x = coordinates[:, 0]
        y = coordinates[:, 1]
        vx_entr = -omega * y
        vy_entr = omega * x
        return np.column_stack([vel[:, 0] - vx_entr, vel[:, 1] - vy_entr, vel[:, 2]])

    def fit(
        self,
        coordinates: np.ndarray,
        use_velocities: bool | None = None,
    ) -> SpectralClusteringPartitioner:
        """Fit the spectral clustering.

        With velocities, the feature vector is
        ``(x, y, z, |v_rel|)`` (relative velocity norm, rotating frame).

        Args:
            coordinates: Positions, shape ``(N, 3)``.
            use_velocities: Overrides :attr:`use_velocity` when given.

        Returns:
            ``self``.
        """
        use_velocities = self.use_velocity if use_velocities is None else use_velocities
        coordinates = np.asarray(coordinates)

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) == len(coordinates):
                v_rel = self._compute_relative_velocity(coordinates, vel)
                v_rel_norm = np.linalg.norm(v_rel, axis=1)
                self.features = np.column_stack([coordinates, v_rel_norm])
                self._n_features = 4
                logger.info("Spectral clustering with velocity features")
                return self._fit_internal(self.features, n_pos=3)
            logger.warning(
                "Velocity/coordinate mismatch (%d vs %d), using positions only",
                len(vel),
                len(coordinates),
            )

        self.features = coordinates
        self._n_features = 3
        logger.info("Spectral clustering without velocity features")
        return self._fit_internal(self.features, n_pos=3)

    def _fit_internal(
        self, features: np.ndarray, n_pos: int = 3
    ) -> SpectralClusteringPartitioner:
        """Normalise, weight, subsample, then run the spectral clustering."""
        self._n_features = features.shape[1]

        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        x_scaled = (features - self._mean) / self._std

        if x_scaled.shape[1] > n_pos:
            weights = np.ones(x_scaled.shape[1])
            weights[n_pos:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        # Subsample the fit (spectral clustering is O(N²)/O(N³)).
        rng = np.random.RandomState(self.random_state)
        n_samples = min(self.max_samples, len(x_scaled))
        idx = rng.choice(len(x_scaled), n_samples, replace=False)
        x_sub = x_scaled[idx]

        spectral = SpectralClustering(
            n_clusters=self._n_cells,
            affinity="rbf",
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            assign_labels="discretize",
        )
        labels_sub = spectral.fit_predict(x_sub)

        # Keep the support points for 1-NN inference.
        self._support_data = x_sub
        self._support_labels = labels_sub
        self._tree = cKDTree(self._support_data)
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
        """Assign each particle by proximity to the spectral support points.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Unused (kept for interface compatibility).
            vy: Unused (kept for interface compatibility).
            vz: Unused (kept for interface compatibility).

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if (
            self._tree is None
            or self._mean is None
            or self._std is None
            or self._support_labels is None
        ):
            raise RuntimeError(
                "SpectralClusteringPartitioner must be fitted before use"
            )
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
        x_scaled = (pos - self._mean) / self._std
        if x_scaled.shape[1] > 3:
            weights = np.ones(x_scaled.shape[1])
            weights[3:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        _, indices = self._tree.query(x_scaled)
        self.states = self._support_labels[indices].astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        if (
            self._support_data is None
            or self._support_labels is None
            or self._mean is None
            or self._std is None
        ):
            raise RuntimeError(
                "SpectralClusteringPartitioner must be fitted before save()"
            )
        np.save(os.path.join(path, "support_data.npy"), self._support_data)
        np.save(os.path.join(path, "support_labels.npy"), self._support_labels)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "params.json"), "w") as fh:
            json.dump(
                {
                    "n_features": self._n_features,
                    "n_cells": self._n_cells,
                    "vw": self.velocity_weight,
                    "k": self.n_neighbors,
                },
                fh,
            )

    def _load_data(self, path: str) -> None:
        self._support_data = np.load(os.path.join(path, "support_data.npy"))
        self._support_labels = np.load(os.path.join(path, "support_labels.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._support_data)
        with open(os.path.join(path, "params.json")) as fh:
            params = json.load(fh)
            self._n_features = params["n_features"]


# =============================================================================
# 9. DBSCAN
# =============================================================================


class DBSCANPartitioner(BasePartitioner):
    """Density-based partitioning (DBSCAN).

    Captures irregular / non-convex mixing zones without imposing a cell
    count a priori (unlike spectral clustering or GMM). Low-density particles
    are flagged as noise (``-1``), consistent with the "out of zone"
    convention of the pipeline.
    """

    def __init__(
        self,
        eps: float = 0.1,
        min_samples: int = 10,
        velocity_weight: float = 0.5,
        max_samples: int = 5000,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.velocity_weight = velocity_weight
        self.max_samples = max_samples
        self.random_state = random_state
        self._support_data: np.ndarray | None = None
        self._support_labels: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._n_features = 6
        self._n_cells = 0  # determined dynamically after fit
        self.features: np.ndarray | None = None
        self.use_velocity = False
        self.splitting_method = "dbscan"

    @property
    def n_cells(self) -> int:
        """Number of clusters found by DBSCAN (noise excluded)."""
        return self._n_cells

    @property
    def label(self) -> str:
        """Label encoding eps, min_samples and the velocity weight."""
        return f"dbscan_eps{self.eps}_min{self.min_samples}_vw{self.velocity_weight}"

    def fit(
        self,
        coordinates: np.ndarray,
        use_velocities: bool | None = None,
    ) -> DBSCANPartitioner:
        """Fit DBSCAN on positions, optionally with velocities.

        Args:
            coordinates: Positions, shape ``(N, 3)``.
            use_velocities: Overrides :attr:`use_velocity` when given.

        Returns:
            ``self``.
        """
        use_velocities = self.use_velocity if use_velocities is None else use_velocities
        coordinates = np.asarray(coordinates)

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) == len(coordinates):
                self.features = np.hstack([coordinates, vel])
                self._n_features = 6
                return self._fit_internal(self.features, n_pos=3)
            logger.warning(
                "Velocity/coordinate mismatch (%d vs %d), using positions only",
                len(vel),
                len(coordinates),
            )

        self.features = coordinates
        self._n_features = 3
        return self._fit_internal(coordinates, n_pos=3)

    def _fit_internal(self, features: np.ndarray, n_pos: int = 3) -> DBSCANPartitioner:
        """Normalise, weight, subsample, then run DBSCAN."""
        self._n_features = features.shape[1]

        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        x_scaled = (features - self._mean) / self._std

        if x_scaled.shape[1] > n_pos:
            weights = np.ones(x_scaled.shape[1])
            weights[n_pos:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        # Subsample the fit (DBSCAN is O(N log N) to O(N²)).
        rng = np.random.RandomState(self.random_state)
        n_samples = min(self.max_samples, len(x_scaled))
        idx = rng.choice(len(x_scaled), n_samples, replace=False)
        x_sub = x_scaled[idx]

        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric="euclidean",
            n_jobs=-1,
        )
        labels_sub = dbscan.fit_predict(x_sub)  # -1 = noise

        n_found = len(set(labels_sub) - {-1})
        if n_found == 0:
            raise ValueError(
                f"DBSCAN found no cluster (everything is noise). "
                f"Increase eps (current={self.eps}) or decrease "
                f"min_samples (current={self.min_samples})."
            )
        self._n_cells = n_found

        # Keep the support points for 1-NN inference (noise is preserved: a
        # new point close to a noise point inherits -1, as intended).
        self._support_data = x_sub
        self._support_labels = labels_sub
        self._tree = cKDTree(self._support_data)
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
        """Assign each particle by proximity to the DBSCAN support points.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Optional X velocities.
            vy: Optional Y velocities.
            vz: Optional Z velocities.

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if (
            self._tree is None
            or self._mean is None
            or self._std is None
            or self._support_labels is None
        ):
            raise RuntimeError("DBSCANPartitioner must be fitted before use")
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])

        if (
            self._n_features == 6
            and vx is not None
            and vy is not None
            and vz is not None
        ):
            vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
            features = np.hstack([pos, vel])
        elif self._n_features == 6:
            features = np.hstack([pos, np.zeros((len(pos), 3))])
        else:
            features = pos

        x_scaled = (features - self._mean) / self._std
        if x_scaled.shape[1] > 3:
            weights = np.ones(x_scaled.shape[1])
            weights[3:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        _, indices = self._tree.query(x_scaled)
        self.states = self._support_labels[indices].astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        if (
            self._support_data is None
            or self._support_labels is None
            or self._mean is None
            or self._std is None
        ):
            raise RuntimeError("DBSCANPartitioner must be fitted before save()")
        np.save(os.path.join(path, "support_data.npy"), self._support_data)
        np.save(os.path.join(path, "support_labels.npy"), self._support_labels)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "params.json"), "w") as fh:
            json.dump(
                {
                    "n_features": self._n_features,
                    "n_cells": self._n_cells,
                    "eps": self.eps,
                    "min_samples": self.min_samples,
                    "vw": self.velocity_weight,
                },
                fh,
            )

    def _load_data(self, path: str) -> None:
        self._support_data = np.load(os.path.join(path, "support_data.npy"))
        self._support_labels = np.load(os.path.join(path, "support_labels.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._support_data)
        with open(os.path.join(path, "params.json")) as fh:
            params = json.load(fh)
            self._n_features = params["n_features"]
            self._n_cells = params["n_cells"]
            self.eps = params["eps"]
            self.min_samples = params["min_samples"]
            self.velocity_weight = params.get("vw", self.velocity_weight)


# =============================================================================
# 10. GAUSSIAN MIXTURE MODEL
# =============================================================================


class GaussianMixturePartitioner(BasePartitioner):
    """Gaussian mixture model (full covariance).

    The fit is subsampled to ``max_fit_samples`` points to keep the cost
    reasonable on DEM-scale datasets.
    """

    def __init__(
        self,
        n_cells: int = 125,
        velocity_weight: float = 0.5,
        random_state: int = 42,
        max_fit_samples: int = 50_000,
    ) -> None:
        super().__init__()
        self._n_cells = n_cells
        self.velocity_weight = velocity_weight
        self.random_state = random_state
        self.max_fit_samples = max_fit_samples
        self._gmm: GaussianMixture | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._centroids: np.ndarray | None = None
        self._n_features = 6
        self.features: np.ndarray | None = None
        self.use_velocity = True
        self.splitting_method = "gmm_full"

    @property
    def n_cells(self) -> int:
        """Number of Gaussian components."""
        return self._n_cells

    @property
    def label(self) -> str:
        """Label encoding the velocity weight."""
        return f"gmm_full_{self._n_cells}cells_vw{self.velocity_weight}"

    def fit(
        self,
        coordinates: np.ndarray,
        use_velocities: bool | None = None,
    ) -> GaussianMixturePartitioner:
        """Fit the Gaussian mixture, optionally with velocity features.

        Args:
            coordinates: Positions, shape ``(N, 3)``.
            use_velocities: Overrides :attr:`use_velocity` when given.

        Returns:
            ``self``.
        """
        use_velocities = self.use_velocity if use_velocities is None else use_velocities
        coordinates = np.asarray(coordinates)

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) == len(coordinates):
                self.features = np.hstack([coordinates, vel])
                self._n_features = 6
                return self._fit_internal(self.features, n_pos=3)
            logger.warning(
                "Velocity/coordinate mismatch (%d vs %d), using positions only",
                len(vel),
                len(coordinates),
            )

        self.features = coordinates
        self._n_features = 3
        return self._fit_internal(coordinates, n_pos=3)

    def _fit_internal(
        self, features: np.ndarray, n_pos: int = 3
    ) -> GaussianMixturePartitioner:
        """Normalise, weight, subsample, then fit the GMM."""
        self._n_features = features.shape[1]

        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        x_scaled = (features - self._mean) / self._std

        if x_scaled.shape[1] > n_pos:
            weights = np.ones(x_scaled.shape[1])
            weights[n_pos:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        # Subsampling is crucial for GMM on DEM-scale data.
        rng = np.random.RandomState(self.random_state)
        if len(x_scaled) > self.max_fit_samples:
            logger.info(
                "Subsampling GMM fit: %d → %d points",
                len(x_scaled),
                self.max_fit_samples,
            )
            idx = rng.choice(len(x_scaled), self.max_fit_samples, replace=False)
            x_scaled = x_scaled[idx]

        # Speed-tuned parameters (single init, kmeans initialisation).
        self._gmm = GaussianMixture(
            n_components=self._n_cells,
            covariance_type="full",
            random_state=self.random_state,
            n_init=1,
            max_iter=50,
            tol=1e-3,
            init_params="kmeans",
        )
        logger.info("Fitting GMM on %d points...", x_scaled.shape[0])
        self._gmm.fit(x_scaled)
        self._centroids = self._gmm.means_
        logger.info("GMM fit done")
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
        """Assign each particle to its most probable Gaussian component.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Optional X velocities.
            vy: Optional Y velocities.
            vz: Optional Z velocities.

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if self._gmm is None or self._mean is None or self._std is None:
            raise RuntimeError("GaussianMixturePartitioner must be fitted before use")
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])

        if (
            self._n_features == 6
            and vx is not None
            and vy is not None
            and vz is not None
        ):
            vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
            features = np.hstack([pos, vel])
        elif self._n_features == 6:
            features = np.hstack([pos, np.zeros((len(pos), 3))])
        else:
            features = pos

        x_scaled = (features - self._mean) / self._std
        if x_scaled.shape[1] > 3:
            weights = np.ones(x_scaled.shape[1])
            weights[3:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        self.states = self._gmm.predict(x_scaled).astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        if self._gmm is None or self._mean is None or self._std is None:
            raise RuntimeError(
                "GaussianMixturePartitioner must be fitted before save()"
            )
        with open(os.path.join(path, "gmm_model.pkl"), "wb") as fh:
            pickle.dump(self._gmm, fh)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)

    def _load_data(self, path: str) -> None:
        with open(os.path.join(path, "gmm_model.pkl"), "rb") as fh:
            self._gmm = pickle.load(fh)
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._centroids = self._gmm.means_


# =============================================================================
# 11. SPECTRAL BICLUSTERING
# =============================================================================


class SpectralBiclusteringPartitioner(BasePartitioner):
    """Spectral biclustering of the coupled position x velocity kinetics.

    Requires :attr:`dem_velocities` before :meth:`fit`. Features are
    ``(r_cyl, theta, z, |v|, vx, vy, vz)`` (or positions only when no
    velocity is available); inference is 1-NN against the support points.
    """

    def __init__(
        self,
        n_cells: int = 30,
        n_col_clusters: int = 3,
        method: str = "log",
        velocity_weight: float = 0.6,
        n_components: int = 10,
        n_best: int = 5,
        svd_method: str = "randomized",
        max_samples: int = 4000,
        random_state: int = 42,
    ) -> None:
        super().__init__()
        self._n_cells = n_cells
        self.n_col_clusters = n_col_clusters
        self.method = method
        self.velocity_weight = velocity_weight
        self.n_components = n_components
        self.n_best = n_best
        self.svd_method = svd_method
        self.max_samples = max_samples
        self.random_state = random_state
        self._support_data: np.ndarray | None = None
        self._support_labels: np.ndarray | None = None
        self._tree: cKDTree | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._log_shift: np.ndarray | None = None
        self._n_features = 7
        self._col_labels: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.use_velocity = True
        self.splitting_method = "spectral_biclustering"

    @property
    def n_cells(self) -> int:
        """Number of row clusters."""
        return self._n_cells

    @property
    def label(self) -> str:
        """Label encoding the biclustering hyperparameters."""
        return (
            f"spectral_biclustering_{self._n_cells}cells"
            f"_nc{self.n_col_clusters}_m{self.method}"
            f"_vw{self.velocity_weight}"
        )

    def _build_features(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        vx: np.ndarray | None = None,
        vy: np.ndarray | None = None,
        vz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build ``(r_cyl, theta, z[, |v|, vx, vy, vz])`` features."""
        r_cyl = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        if vx is not None and vy is not None and vz is not None:
            v_norm = np.linalg.norm(np.column_stack([vx, vy, vz]), axis=1)
            return np.column_stack([r_cyl, theta, z, v_norm, vx, vy, vz])
        return np.column_stack([r_cyl, theta, z])

    def fit(
        self,
        coordinates: np.ndarray,
        use_velocities: bool | None = None,
    ) -> SpectralBiclusteringPartitioner:
        """Fit the biclustering.

        Args:
            coordinates: Positions, shape ``(N, 3)``.
            use_velocities: Overrides :attr:`use_velocity` when given.

        Returns:
            ``self``.
        """
        coordinates = np.asarray(coordinates)
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

        if self.dem_velocities is not None and len(self.dem_velocities) == len(
            coordinates
        ):
            vel = np.asarray(self.dem_velocities)
            vx, vy, vz = vel[:, 0], vel[:, 1], vel[:, 2]
            self.features = self._build_features(x, y, z, vx, vy, vz)
            logger.info(
                "Biclustering features with velocities: %s", self.features.shape
            )
        else:
            logger.warning(
                "dem_velocities absent — fitting on coordinates only (3 features)"
            )
            self.features = self._build_features(x, y, z)

        self._n_features = self.features.shape[1]
        return self._fit_internal(self.features)

    def _fit_internal(self, features: np.ndarray) -> SpectralBiclusteringPartitioner:
        """Normalise, weight, log-shift, subsample, then fit the model."""
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        x_scaled = (features - self._mean) / self._std

        if x_scaled.shape[1] > 3:
            weights = np.ones(x_scaled.shape[1])
            weights[3:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        rng = np.random.RandomState(self.random_state)
        n_samples = min(self.max_samples, len(x_scaled))
        idx = rng.choice(len(x_scaled), n_samples, replace=False)
        x_sub = x_scaled[idx]

        # Store the log shift once and for all.
        if self.method == "log":
            self._log_shift = x_sub.min(axis=0)
            x_sub = x_sub - self._log_shift + 1e-6

        model = SpectralBiclustering(
            n_clusters=(self._n_cells, self.n_col_clusters),
            method=self.method,
            n_components=max(self.n_components, self._n_cells + self.n_col_clusters),
            n_best=self.n_best,
            svd_method=self.svd_method,
            random_state=self.random_state,
        )
        model.fit(x_sub)

        n_distinct = len(np.unique(model.row_labels_))
        if n_distinct < self._n_cells:
            logger.warning(
                "%d distinct clusters / %d requested → increase max_samples "
                "or decrease n_cells",
                n_distinct,
                self._n_cells,
            )

        self._support_data = x_sub
        self._support_labels = model.row_labels_.astype(np.int64)
        self._col_labels = model.column_labels_
        self._tree = cKDTree(self._support_data)
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
        """Assign each particle to the closest biclustering support point.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Optional X velocities.
            vy: Optional Y velocities.
            vz: Optional Z velocities.

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if (
            self._tree is None
            or self._mean is None
            or self._std is None
            or self._support_labels is None
        ):
            raise RuntimeError(
                "SpectralBiclusteringPartitioner must be fitted before use"
            )

        features = self._build_features(
            np.asarray(x),
            np.asarray(y),
            np.asarray(z),
            np.asarray(vx) if vx is not None else None,
            np.asarray(vy) if vy is not None else None,
            np.asarray(vz) if vz is not None else None,
        )

        # Same mean/std as the fit.
        x_scaled = (features - self._mean) / self._std
        if x_scaled.shape[1] > 3:
            weights = np.ones(x_scaled.shape[1])
            weights[3:] = self.velocity_weight
            x_scaled = x_scaled * weights[np.newaxis, :]

        # Log shift consistent with the fit.
        if self.method == "log" and self._log_shift is not None:
            x_scaled = x_scaled - self._log_shift + 1e-6

        _, indices = self._tree.query(x_scaled)
        self.states = self._support_labels[indices].astype(np.int64)
        return self.states

    def diagnostics(
        self,
        coordinates: np.ndarray,
        velocities: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        """Population statistics; velocities are zero-padded when absent.

        Overridden so that the 7-feature normalisation stays coherent.
        """
        del velocities  # zero padding is applied below
        coords = coordinates
        coords = np.asarray(coords)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

        n = len(x)
        states = self.compute_states(x, y, z, np.zeros(n), np.zeros(n), np.zeros(n))
        counts = np.bincount(states, minlength=self._n_cells)
        visited = counts > 0
        return {
            "n_visited": int(visited.sum()),
            "n_empty": int((~visited).sum()),
            "pop_min": int(counts[visited].min()) if visited.any() else 0,
            "pop_max": int(counts.max()),
            "pop_mean": float(counts[visited].mean()) if visited.any() else 0.0,
            "pop_std": float(counts[visited].std()) if visited.any() else 0.0,
        }

    def _save_data(self, path: str) -> None:
        if (
            self._support_data is None
            or self._support_labels is None
            or self._mean is None
            or self._std is None
            or self._col_labels is None
        ):
            raise RuntimeError(
                "SpectralBiclusteringPartitioner must be fitted before save()"
            )
        np.save(os.path.join(path, "support_data.npy"), self._support_data)
        np.save(os.path.join(path, "support_labels.npy"), self._support_labels)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        np.save(os.path.join(path, "col_labels.npy"), self._col_labels)
        if self._log_shift is not None:
            np.save(os.path.join(path, "log_shift.npy"), self._log_shift)
        with open(os.path.join(path, "params.json"), "w") as fh:
            json.dump(
                {
                    "n_features": self._n_features,
                    "n_cells": self._n_cells,
                    "n_col_clusters": self.n_col_clusters,
                    "method": self.method,
                    "velocity_weight": self.velocity_weight,
                    "n_components": self.n_components,
                    "n_best": self.n_best,
                },
                fh,
                indent=2,
            )

    def _load_data(self, path: str) -> None:
        self._support_data = np.load(os.path.join(path, "support_data.npy"))
        self._support_labels = np.load(os.path.join(path, "support_labels.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._col_labels = np.load(os.path.join(path, "col_labels.npy"))
        self._tree = cKDTree(self._support_data)
        log_shift_path = os.path.join(path, "log_shift.npy")
        if os.path.exists(log_shift_path):
            self._log_shift = np.load(log_shift_path)
        with open(os.path.join(path, "params.json")) as fh:
            params = json.load(fh)
            self._n_features = params["n_features"]
            self._n_cells = params["n_cells"]
            self.n_col_clusters = params["n_col_clusters"]
            self.method = params["method"]


# =============================================================================
# 12. ADAPTIVE TOP/BOTTOM PARTITIONING
# =============================================================================


class AdaptivePartitioner(BasePartitioner):
    """Adaptive partitioning along ``y``.

    Splits the domain into two zones:

    * top zone (``y > y_split``) — few cells (coarse);
    * bottom zone (``y <= y_split``) — fine partitioning.

    Args:
        y_split: Separation coordinate (or quantile when
            ``y_split_mode="quantile"``).
        y_split_mode: ``"absolute"`` or ``"quantile"``.
        n_cells_top: Number of cells requested for the top zone.
        top_method: Partitioning method of the top zone (``"single"`` gives a
            single cell).
        top_kwargs: Arguments of the top partitioner.
        bottom_method: Partitioning method of the bottom zone.
        bottom_kwargs: Arguments of the bottom partitioner.
    """

    def __init__(
        self,
        y_split: float | None = None,
        y_split_mode: str = "quantile",
        n_cells_top: int = 1,
        top_method: str = "single",
        top_kwargs: dict[str, Any] | None = None,
        bottom_method: str = "cylindrical",
        bottom_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.splitting_method = "adaptive"
        self.y_split_input = y_split
        self.y_split_mode = y_split_mode
        self.top_method = top_method
        self.top_kwargs = top_kwargs or {}
        self.bottom_method = bottom_method
        self.bottom_kwargs = bottom_kwargs or {}

        # Attributes computed at fit time.
        self.y_split: float | None = None
        self._y_min: float | None = None
        self._y_max: float | None = None
        self._top_partitioner: BasePartitioner | None = None
        self._bottom_partitioner: BasePartitioner | None = None
        self._n_cells_top: int | None = None
        self._n_cells_bottom: int | None = None

    @property
    def n_cells(self) -> int:
        """Total cells (0 before fitting)."""
        if self._n_cells_top is None or self._n_cells_bottom is None:
            return 0
        return self._n_cells_top + self._n_cells_bottom

    @property
    def label(self) -> str:
        """Label encoding the zone methods and the split."""
        return (
            f"adaptive_y_{self.bottom_method}"
            f"_top{self._n_cells_top}_bot{self._n_cells_bottom}"
            f"_split{self.y_split_input}_mode{self.y_split_mode}"
        )

    def fit(self, coordinates: np.ndarray) -> AdaptivePartitioner:
        """Fit the bottom and top partitioners on their own particles.

        Args:
            coordinates: Array of shape ``(N, 3)``.

        Returns:
            ``self``.

        Raises:
            ValueError: If ``y_split_mode`` is unknown.
        """
        coordinates = np.asarray(coordinates)
        y = coordinates[:, 1]

        self._y_min = float(y.min())
        self._y_max = float(y.max())

        # Determine the split position.
        if self.y_split_mode == "quantile":
            quantile = self.y_split_input if self.y_split_input else 0.9
            self.y_split = float(np.quantile(y, quantile))
        elif self.y_split_mode == "absolute":
            self.y_split = (
                (self._y_min + self._y_max) / 2
                if self.y_split_input is None
                else self.y_split_input
            )
        else:
            raise ValueError(f"Unknown y_split_mode: {self.y_split_mode}")

        # Split the data.
        mask_bottom = y <= self.y_split
        mask_top = y > self.y_split
        coords_bottom = coordinates[mask_bottom]
        coords_top = coordinates[mask_top]

        # Bottom zone.
        self._bottom_partitioner = create_partitioner(
            self.bottom_method, **self.bottom_kwargs
        )
        if len(coords_bottom) > 0:
            self._bottom_partitioner.fit(coords_bottom)
        self._n_cells_bottom = self._bottom_partitioner.n_cells

        # Top zone.
        if self.top_method == "single":
            self._top_partitioner = None
            self._n_cells_top = 1
        else:
            self._top_partitioner = create_partitioner(
                self.top_method, **self.top_kwargs
            )
            if len(coords_top) > 0:
                self._top_partitioner.fit(coords_top)
            self._n_cells_top = self._top_partitioner.n_cells
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
        """Assign each particle to the cell of its zone.

        Bottom-zone states are ``0 … n_cells_bottom - 1``; top-zone states
        are ``n_cells_bottom … n_cells - 1``.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Unused (kept for interface compatibility).
            vy: Unused (kept for interface compatibility).
            vz: Unused (kept for interface compatibility).

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        if (
            self.y_split is None
            or self._bottom_partitioner is None
            or self._n_cells_bottom is None
        ):
            raise RuntimeError("AdaptivePartitioner must be fitted before use")

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        n = len(x)
        states = np.zeros(n, dtype=np.int64)

        mask_bottom = y <= self.y_split
        mask_top = ~mask_bottom

        # Bottom zone: states 0 … n_cells_bottom - 1.
        if mask_bottom.any():
            states[mask_bottom] = self._bottom_partitioner.compute_states(
                x[mask_bottom], y[mask_bottom], z[mask_bottom]
            )

        # Top zone: states n_cells_bottom … n_cells - 1.
        if mask_top.any():
            if self._top_partitioner is None:
                states[mask_top] = self._n_cells_bottom
            else:
                top_states = self._top_partitioner.compute_states(
                    x[mask_top], y[mask_top], z[mask_top]
                )
                states[mask_top] = top_states + self._n_cells_bottom

        # Hybrid methods (adaptive/multizone) delegate the out-of-zone
        # handling to the sub-partitioners they instantiate.
        self.states = states
        return self.states


# =============================================================================
# 13. MULTI-ZONE PARTITIONING (generalisation)
# =============================================================================


class MultiZonePartitioner(BasePartitioner):
    """Generalised multi-zone partitioning along ``y``.

    Several zones, each with its own partitioning method::

        zones = [
            {"y_min": -np.inf, "y_max": 0.5, "method": "cylindrical",
             "kwargs": {...}},
            {"y_min": 0.5, "y_max": 0.8, "method": "voronoi",
             "kwargs": {"n_cells": 50}},
            {"y_min": 0.8, "y_max": np.inf, "method": "single", "kwargs": {}},
        ]

    Args:
        zones: List of zone dictionaries (see example above).
        y_mode: ``"absolute"`` or ``"quantile"`` bounds interpretation.
    """

    def __init__(self, zones: list[dict[str, Any]], y_mode: str = "absolute") -> None:
        super().__init__()
        self.splitting_method = "multizone"
        self.zones_config = zones
        self.y_mode = y_mode
        self._zones: list[tuple[float, float, BasePartitioner]] = []
        self._cell_offsets: list[int] = []
        self._total_cells = 0

    @property
    def n_cells(self) -> int:
        """Total cells across every zone."""
        return self._total_cells

    @property
    def label(self) -> str:
        """Label listing the per-zone methods."""
        methods = "_".join(str(z["method"]) for z in self.zones_config)
        return f"multizone_{len(self.zones_config)}zones_{methods}"

    def fit(self, coordinates: np.ndarray) -> MultiZonePartitioner:
        """Fit one partitioner per zone on the zone particles.

        Args:
            coordinates: Array of shape ``(N, 3)``.

        Returns:
            ``self``.
        """
        coordinates = np.asarray(coordinates)
        y = coordinates[:, 1]

        self._zones = []
        self._cell_offsets = [0]

        for i, zone_cfg in enumerate(self.zones_config):
            # Convert the bounds in quantile mode.
            if self.y_mode == "quantile":
                y_min = float(np.quantile(y, zone_cfg.get("y_min", 0)))
                y_max = float(np.quantile(y, zone_cfg.get("y_max", 1)))
            else:
                y_min = float(zone_cfg.get("y_min", y.min()))
                y_max = float(zone_cfg.get("y_max", y.max()))

            # Select the particles of this zone (last zone includes its max).
            if i == len(self.zones_config) - 1:
                mask = (y >= y_min) & (y <= y_max)
            else:
                mask = (y >= y_min) & (y < y_max)
            coords_zone = coordinates[mask]

            method = zone_cfg.get("method", "single")
            kwargs = dict(zone_cfg.get("kwargs", {}))
            if method == "single":
                partitioner: BasePartitioner = SingleCellPartitioner()
            else:
                partitioner = create_partitioner(method, **kwargs)

            if len(coords_zone) > 0:
                partitioner.fit(coords_zone)

            self._zones.append((y_min, y_max, partitioner))
            self._cell_offsets.append(self._cell_offsets[-1] + partitioner.n_cells)

            logger.info(
                "Zone %d: y ∈ [%.3f, %.3f], %d cells, %d particles",
                i,
                y_min,
                y_max,
                partitioner.n_cells,
                len(coords_zone),
            )

        self._total_cells = self._cell_offsets[-1]
        logger.info("Total: %d cells", self._total_cells)
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
        """Assign each particle to the cell of its zone.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Y coordinates, shape ``(n_particles,)``.
            z: Z coordinates, shape ``(n_particles,)``.
            vx: Unused (kept for interface compatibility).
            vy: Unused (kept for interface compatibility).
            vz: Unused (kept for interface compatibility).

        Returns:
            State index per particle, shape ``(n_particles,)``.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        n = len(x)
        states = np.zeros(n, dtype=np.int64)
        assigned = np.zeros(n, dtype=bool)

        for i, (y_min, y_max, partitioner) in enumerate(self._zones):
            if i == len(self._zones) - 1:
                mask = (y >= y_min) & (y <= y_max) & ~assigned
            else:
                mask = (y >= y_min) & (y < y_max) & ~assigned

            if mask.any():
                zone_states = partitioner.compute_states(x[mask], y[mask], z[mask])
                states[mask] = zone_states + self._cell_offsets[i]
                assigned[mask] = True

        self.states = states
        return self.states

    def _save_data(self, path: str) -> None:
        config = {
            "zones_config": self.zones_config,
            "y_mode": self.y_mode,
            "cell_offsets": self._cell_offsets,
            "zones_bounds": [(y_min, y_max) for y_min, y_max, _ in self._zones],
        }
        with open(os.path.join(path, "multizone_config.json"), "w") as fh:
            json.dump(config, fh, indent=2)

        for i, (_, _, partitioner) in enumerate(self._zones):
            partitioner.save(os.path.join(path, f"zone_{i}"))

    def _load_data(self, path: str) -> None:
        with open(os.path.join(path, "multizone_config.json")) as fh:
            config = json.load(fh)

        self.zones_config = config["zones_config"]
        self.y_mode = config["y_mode"]
        self._cell_offsets = config["cell_offsets"]
        self._total_cells = self._cell_offsets[-1]

        self._zones = []
        for i, (y_min, y_max) in enumerate(config["zones_bounds"]):
            zone_cfg = self.zones_config[i]
            method = zone_cfg.get("method", "single")
            kwargs = zone_cfg.get("kwargs", {})
            if method == "single":
                partitioner: BasePartitioner = SingleCellPartitioner()
            else:
                partitioner = create_partitioner(method, **kwargs)
            partitioner.load(os.path.join(path, f"zone_{i}"))
            self._zones.append((y_min, y_max, partitioner))


# =============================================================================
# 14. SINGLE CELL
# =============================================================================


class SingleCellPartitioner(BasePartitioner):
    """Degenerate partitioning: a single cell for the whole domain."""

    def __init__(self) -> None:
        super().__init__()
        self.splitting_method = "single"

    @property
    def n_cells(self) -> int:
        """Always 1."""
        return 1

    @property
    def label(self) -> str:
        """Always ``"single_cell"``."""
        return "single_cell"

    def fit(self, coordinates: np.ndarray) -> SingleCellPartitioner:
        """No-op: nothing to learn for a single cell.

        Args:
            coordinates: Ignored.

        Returns:
            ``self``.
        """
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
        """Assign every particle to state 0.

        Args:
            x: X coordinates, shape ``(n_particles,)``.
            y: Ignored.
            z: Ignored.
            vx: Ignored.
            vy: Ignored.
            vz: Ignored.

        Returns:
            Zeros of shape ``(n_particles,)``.
        """
        self.states = np.zeros(len(np.asarray(x)), dtype=np.int64)
        return self.states


# =============================================================================
# REGISTRY
# =============================================================================

#: Mapping ``method identifier -> partitioner class``.
REGISTRY: dict[str, type[BasePartitioner]] = {
    # Basic geometric methods.
    "cartesian": CartesianPartitioner,
    "cylindrical": CylindricalPartitioner,
    "voronoi": VoronoiPartitioner,
    "quantile": QuantileGridPartitioner,
    "octree": OctreePartitioner,
    # Physics-based methods (Doucet, Tjakra, Zhou).
    "physics": PhysicsAwarePartitioner,  # K-means with the velocity norm |v|
    "physics_full_vel": FullVectorVelocityKMeansPartitioner,  # full (vx, vy, vz)
    "spectral": SpectralClusteringPartitioner,  # graph topology
    "gmm": GaussianMixturePartitioner,  # ellipsoidal cells
    "spectral_biclustering": SpectralBiclusteringPartitioner,
    # Other advanced methods.
    "adaptive": AdaptivePartitioner,
    "multizone": MultiZonePartitioner,
    "single": SingleCellPartitioner,
    "dbscan": DBSCANPartitioner,
}


# =============================================================================
# FACTORY
# =============================================================================


def create_partitioner(method: str, **kwargs: Any) -> BasePartitioner:
    """Instantiate a partitioner from the registry.

    Args:
        method: Method identifier (see :data:`REGISTRY`).
        **kwargs: Arguments forwarded to the partitioner constructor.

    Returns:
        The partitioner instance.

    Raises:
        ValueError: If ``method`` is unknown.

    Example:
        >>> p = create_partitioner("voronoi", n_cells=125)
        >>> p = create_partitioner("cylindrical", nr=5, ntheta=8, nz=5)
    """
    if method not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown method: {method!r}. Available: {available}")
    return REGISTRY[method](**kwargs)
