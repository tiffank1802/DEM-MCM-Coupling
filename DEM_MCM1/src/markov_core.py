"""
markov_core.py.
==============
Classe principale Markov pour gestion d'un schéma de partitionnement.

Cette classe orchestre:
1. Création/chargement d'un partitioner
2. Chargement de données DEM depuis HuggingFace
3. Construction du vecteur d'état initial φ(0)
4. Propagation de l'état via matrice transition
5. Visualisation 3D (PyVista + Streamlit)

IMPORTANT - Ce que Markov NE FAIT PAS:
- Ne compare PAS plusieurs configs (→ MarkovAnalyzer)
- Ne gère PAS les analyses RSD détaillées (→ MarkovAnalyzer)
- Ne charge PAS depuis HF (→ MarkovAnalyzer)

Architecture:
    Markov = gestion d'UNE config (builder pattern)
    MarkovAnalyzer = orchestration PLUSIEURS configs (comparator pattern)

Examples:
    >>> # Cas 1: Créer un nouveau découpage
    >>> mk = Markov(method="voronoi", method_kwargs={"n_cells": 125})
    >>> mk.load_dem_data(particle_diameter=0.004)
    >>> coords = mk.get_coords([250, 300, 350])  # Multi-timesteps
    >>> mk.fit_partitioner(coords)
    >>> initial_state = mk.build_initial_state_vector(250)
    >>> print(f"φ(0) normalisé: {initial_state.phi.sum()} particles")
    >>>
    >>> # Cas 2: Propager
    >>> M = np.random.rand(125, 125)
    >>> M /= M.sum(axis=1, keepdims=True)
    >>> trajectory = mk.propagate_markov(
    ...     initial_state=initial_state.phi,
    ...     transition_matrix=M,
    ...     n_steps=100
    ... )
    >>> print(f"Trajectoire shape: {trajectory['states'].shape}")  # (101, 125)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import pyvista as pv
import streamlit as st

# Internal imports
from ._config import (
    BUCKET_ID,
    TIMESTEP_TO_SECONDS,
    ParticleDiameter,
    PartitioningMethod,
    StateTrajectory,
    StateVector,
    get_bucket_prefix,
    validate_partitioning_method,
)
from .bucket_io import get_fs
from .partitioners import BasePartitioner, create_partitioner
from .utils import apply_species_mask, load_parquet_as_timestep_dict

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ============================================================================
# MAIN CLASS
# ============================================================================


class Markov:
    """
    Gestionnaire d'un schéma de partitionnement Markovien.

    Cette classe représente UNE configuration de découpage et gère:
    - Le partitioner (création ou chargement)
    - Les données DEM (chargement en mémoire via cache)
    - Les vecteurs d'état et leur propagation
    - La visualisation 3D

    Attributes:
        method (PartitioningMethod): Type de partitionnement
        partitioner (BasePartitioner): Instance du partitioner
        datas (Dict[int, pd.DataFrame]): DEM data par timestep
        coords (np.ndarray): Coordonnées particules (n_particles, 3)
        velocities (np.ndarray): Vitesses particules (n_particles, 3)
        states (np.ndarray): États assignés (n_particles,)
        initial_state (StateVector | None): Vecteur d'état φ(0)
        transition_matrix (np.ndarray | None): Matrice transition M
        vtp_states (pv.PolyData): Données pour visualisation PyVista

    Workflow typique:
        1. mk = Markov(method="voronoi", method_kwargs={"n_cells": 125})
        2. mk.load_dem_data(particle_diameter=0.004)
        3. coords = mk.get_coords([250, 300, 350])
        4. mk.fit_partitioner(coords)
        5. state_0 = mk.build_initial_state_vector(250)
        6. traj = mk.propagate_markov(state_0.phi, M, 100)

    Examples:
        >>> # Initialiser
        >>> mk = Markov(
        ...     method="cylindrical",
        ...     method_kwargs={"nr": 5, "ntheta": 8, "nz": 6}
        ... )
        >>>
        >>> # Charger et fit
        >>> mk.load_dem_data(particle_diameter=0.004)
        >>> coords = mk.get_coords([250])
        >>> mk.fit_partitioner(coords)
        >>>
        >>> # Construire état initial
        >>> state_0 = mk.build_initial_state_vector(250)
        >>> print(f"État initial: {state_0}")
    """

    def __init__(
        self,
        method: str | PartitioningMethod = "cartesian",
        method_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialiser une instance Markov.

        Args:
            method: Type de partitionnement (ex: "voronoi", "cartesian")
                    Voir AVAILABLE_METHODS dans _config.py
            method_kwargs: Arguments pour le constructeur du partitioner
                          Ex: {"n_cells": 125} pour voronoi

        Raises:
            ValueError: Si method n'est pas reconnue
            Exception: Si création partitioner échoue

        Examples:
            >>> # Partitionneur simple
            >>> mk1 = Markov("cartesian")
            >>>
            >>> # Avec paramètres
            >>> mk2 = Markov(
            ...     method="voronoi",
            ...     method_kwargs={"n_cells": 125}
            ... )
            >>>
            >>> # Cylindrique
            >>> mk3 = Markov(
            ...     method="cylindrical",
            ...     method_kwargs={"nr": 5, "ntheta": 8, "nz": 6}
            ... )
        """
        # ---- Valider et stocker method ----
        self.method: PartitioningMethod = validate_partitioning_method(method)
        logger.info(f"Initialisation Markov avec method='{self.method}'")

        # ---- Créer le partitioner ----
        if method_kwargs is None:
            method_kwargs = {}

        try:
            self.partitioner: BasePartitioner = create_partitioner(
                self.method, **method_kwargs
            )
            logger.info(
                f"✅ Partitioner créé: {self.partitioner.__class__.__name__} "
                f"avec {method_kwargs}"
            )
        except Exception as e:
            logger.error(f"❌ Erreur création partitioner: {e}")
            raise

        # ---- Initialiser data containers ----
        self.datas: dict[int, pd.DataFrame] = {}
        self.coords: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.velocities: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.states: np.ndarray = np.array([], dtype=np.int32)

        # ---- État Markovien ----
        self.initial_state: StateVector | None = None
        self.transition_matrix: np.ndarray | None = None

        # ---- Visualisation ----
        self.vtp_states: pv.PolyData = pv.PolyData()

        logger.debug(f"Instance Markov initialisée: {self}")

    def load_dem_data(
        self,
        particle_diameter: ParticleDiameter | None = None,
    ) -> dict[int, pd.DataFrame]:
        """
        Charger les données DEM depuis HuggingFace en mémoire.

        ⚠️  CACHED via @st.cache_data (voir _load_dem_data_cached)

        Utilise load_parquet_as_timestep_dict pour charger directement
        depuis le parquet HF sans télécharger localement.

        Args:
            particle_diameter: Diamètre filtré (0.004, 0.008, None)
                              None = toutes les particules

        Returns:
            Dict[timestep_index: DataFrame] avec colonnes particules

        Raises:
            FileNotFoundError: Si bucket HF non accessible
            ValueError: Si parquet mal formé

        Examples:
            >>> mk = Markov("voronoi")
            >>>
            >>> # Charger toutes particules
            >>> datas_all = mk.load_dem_data()
            >>> print(f"Timesteps: {list(datas_all.keys())[:5]}")
            >>>
            >>> # Charger particules 0.004 m seulement
            >>> datas_small = mk.load_dem_data(particle_diameter=0.004)
            >>> print(f"Nombre timesteps: {len(datas_small)}")
        """
        prefix = get_bucket_prefix(particle_diameter)
        bucket_base = f"hf://buckets/{BUCKET_ID}/{prefix}"
        parquet_path = f"{bucket_base}/simulation_complete.parquet"

        logger.info(f"Chargement DEM depuis {prefix} (diameter={particle_diameter})...")

        try:
            # Essayer d'utiliser Streamlit cache si disponible
            try:

                @st.cache_data(show_spinner="📦 Loading DEM data...", ttl=3600)
                def _load_cached() -> Dict[int, pd.DataFrame]:
                    return load_parquet_as_timestep_dict(
                        parquet_path=parquet_path, fs=get_fs()
                    )

                self.datas = _load_cached()
            except:
                # Fallback: chargement direct (pas de cache)
                self.datas = load_parquet_as_timestep_dict(
                    parquet_path=parquet_path, fs=get_fs()
                )

            logger.info(
                f"✅ {len(self.datas)} timesteps chargés "
                f"(index {min(self.datas.keys())} → {max(self.datas.keys())})"
            )
            return self.datas

        except Exception as e:
            logger.error(f"❌ Erreur chargement DEM: {e}")
            raise

    def get_coords(
        self,
        timestep_indices: list[int] | int = 250,
    ) -> np.ndarray:
        """
        Récupérer les coordonnées des particules.

        Peut agréger coordonnées de plusieurs timesteps pour fit du
        partitioner (ex: fit sur 250, 300, 350 pour capture diversité).

        Si plusieurs timesteps, les coordonnées sont CONCATÉNÉES.

        Args:
            timestep_indices: Index(es) timestep(s) à charger

        Returns:
            Array (n_particles, 3) avec coordonnées en mètres
            Si plusieurs timesteps: (n_timesteps * n_particles, 3)

        Raises:
            KeyError: Si timestep non disponible
            ValueError: Si datas vides

        Examples:
            >>> mk = Markov("voronoi")
            >>> mk.load_dem_data()
            >>>
            >>> # Un seul timestep
            >>> coords_t250 = mk.get_coords([250])
            >>> print(f"Shape: {coords_t250.shape}")  # (N, 3)
            >>>
            >>> # Plusieurs timesteps (concaténés)
            >>> coords_multi = mk.get_coords([250, 300, 350])
            >>> print(f"Shape: {coords_multi.shape}")  # (3*N, 3)
        """
        if not isinstance(timestep_indices, list):
            timestep_indices = [timestep_indices]

        if not self.datas:
            logger.warning("datas vides, chargement automatique...")
            self.load_dem_data()

        coords_list = []
        for idx in timestep_indices:
            if idx not in self.datas:
                raise KeyError(
                    f"Timestep {idx} non disponible. "
                    f"Disponibles: {list(self.datas.keys())[:10]}..."
                )

            df = self.datas[idx]
            coords = df[["coordinates:0", "coordinates:1", "coordinates:2"]].to_numpy(
                dtype=np.float32
            )
            coords_list.append(coords)
            logger.debug(f"Chargé timestep {idx}: {coords.shape[0]} particules")

        # Concaténer si plusieurs timesteps
        self.coords = np.vstack(coords_list) if len(coords_list) > 1 else coords_list[0]
        logger.info(f"Coordonnées shape: {self.coords.shape}")

        return self.coords

    def get_velocities(
        self,
        timestep_indices: list[int] | int = 250,
    ) -> np.ndarray:
        """
        Récupérer les vitesses des particules.

        Args:
            timestep_indices: Index(es) à charger

        Returns:
            Array (n_particles, 3) avec vitesses [m/s]

        Examples:
            >>> velocities = mk.get_velocities([250])
            >>> print(f"Vitesse moyenne: {velocities.mean(axis=0)}")
        """
        if not isinstance(timestep_indices, list):
            timestep_indices = [timestep_indices]

        if not self.datas:
            self.load_dem_data()

        vel_list = []
        for idx in timestep_indices:
            df = self.datas[idx]
            vel = df[["Velocity:0", "Velocity:1", "Velocity:2"]].to_numpy(
                dtype=np.float32
            )
            vel_list.append(vel)

        self.velocities = np.vstack(vel_list) if len(vel_list) > 1 else vel_list[0]
        logger.info(f"Vitesses shape: {self.velocities.shape}")

        return self.velocities

    def fit_partitioner(
        self,
        coordinates: np.ndarray,
    ) -> BasePartitioner:
        """
        Entraîner le partitioner sur les coordonnées données.

        Cette étape détermine les frontières des partitions
        (ex: centroïdes Voronoi, grid Cartesian, etc.).

        Args:
            coordinates: Array (n_samples, 3) de coordonnées

        Returns:
            Le partitioner entraîné (self.partitioner)

        Examples:
            >>> mk = Markov("voronoi")
            >>> mk.load_dem_data()
            >>> coords = mk.get_coords([250, 300, 350])
            >>> mk.fit_partitioner(coords)
            >>> print(f"Partitioner fitted: {mk.partitioner.n_cells} cells")
        """
        logger.info(f"Fit partitioner sur {coordinates.shape[0]} points...")

        self.partitioner.fit(coordinates)

        logger.info(f"✅ Partitioner fitted: {self.partitioner.n_cells} états")
        return self.partitioner

    def compute_states(
        self,
        coordinates: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Assigner chaque particule à son état (partition).

        Args:
            coordinates: Array (n_particles, 3). Si None, utilise self.coords

        Returns:
            Array (n_particles,) avec state_id pour chaque particule

        Examples:
            >>> coords = mk.get_coords([250])
            >>> states = mk.compute_states(coords)
            >>> print(f"États trouvés: {np.unique(states)}")
            >>> print(f"Occupation: {np.bincount(states)}")
        """
        if coordinates is None:
            coordinates = self.coords

        if coordinates.size == 0:
            raise ValueError("coordinates vides")

        self.states = self.partitioner.compute_states(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        )
        logger.debug(f"États: {self.states.shape}")

        return self.states

    def build_initial_state_vector(
        self,
        timestep_index: int,
        species_mask: np.ndarray | None = None,
        normalize: bool = True,
    ) -> StateVector:
        """
        Construire le vecteur d'état initial φ(0) à un timestep donné.

        φᵢ(0) = nombre de particules dans partition i

        Optionnellement filtrer par espèce (ex: seulement particules Type-A).

        Args:
            timestep_index: Timestep pour φ(0)
            species_mask: Masque bool (n_particles,) pour espèce
                         Ex: (particle_phase_id == 1)
            normalize: Valider que ∑φ = N (raise si False et invalide)

        Returns:
            StateVector avec φ(0) et metadata

        Raises:
            ValueError: Si normalisation échoue et normalize=True
            KeyError: Si timestep non disponible
            RuntimeError: Si partitioner non entraîné

        Examples:
            >>> mk = Markov("voronoi")
            >>> mk.load_dem_data()
            >>> coords = mk.get_coords([250])
            >>> mk.fit_partitioner(coords)
            >>>
            >>> # Toutes les particules
            >>> init_state = mk.build_initial_state_vector(250)
            >>> print(f"φ(0): {init_state.phi}")
            >>> print(f"Total: {init_state.total_particles}")
            >>>
            >>> # Seulement espèce A
            >>> mask_A = (mk.datas[250]['Particle_Phase_ID'] == 1).to_numpy()
            >>> init_A = mk.build_initial_state_vector(
            ...     250,
            ...     species_mask=mask_A
            ... )
        """
        # ---- Vérifier que partitioner est fit ----
        if self.partitioner.n_cells == 0:
            raise RuntimeError(
                "Partitioner non entraîné. Appelez fit_partitioner() d'abord."
            )

        # ---- Obtenir coordonnées au timestep ----
        coords = self.get_coords([timestep_index])
        states = self.compute_states(coords)

        # ---- Construire vecteur brut ----
        phi_raw = np.bincount(states, minlength=self.partitioner.n_cells).astype(
            np.float32
        )

        # ---- Appliquer masque espèce si fourni ----
        if species_mask is not None:
            states_filtered = apply_species_mask(states, species_mask)
            phi = np.bincount(
                states_filtered, minlength=self.partitioner.n_cells
            ).astype(np.float32)
            total = species_mask.sum()
        else:
            phi = phi_raw
            total = len(states)

        # ---- Créer StateVector ----
        state = StateVector(
            phi=phi,
            timestamp=timestep_index,
            total_particles=int(total),
            description=f"Initial state at t={timestep_index}",
        )

        # ---- Valider normalisation ----
        if normalize:
            is_valid = state.validate_normalization()
            if not is_valid:
                logger.warning(
                    f"⚠️  Normalisation échouée: ∑φ={phi.sum()}, expected={total}"
                )
                raise ValueError("État non normalisé!")

        self.initial_state = state
        logger.info(
            f"✅ État initial construit: "
            f"φ(0) sum={phi.sum()}, n_states={self.partitioner.n_cells}"
        )

        return state

    def propagate_markov(
        self,
        initial_state: np.ndarray,
        transition_matrix: np.ndarray,
        n_steps: int,
        validate_normalization: bool = True,
    ) -> StateTrajectory:
        """
        Propager le vecteur d'état via la matrice transition.

        Calcule: φ(t+1) = φ(t) @ M

        Invariant: ∑φ(t) = ∑φ(0) pour tous les t (si M normalisée)

        Args:
            initial_state: Vecteur (n_states,) initial
            transition_matrix: Matrice M (n_states, n_states)
                              DOIT être line-stochastic: M[i,:].sum() = 1
            n_steps: Nombre d'itérations
            validate_normalization: Vérifier ∑φ(t) = N à chaque pas

        Returns:
            StateTrajectory avec historique complet

        Raises:
            ValueError: Si dimensions incompatibles
            ValueError: Si normalisation échoue

        Examples:
            >>> mk = Markov("voronoi")
            >>> # ... setup ...
            >>> initial_state = mk.build_initial_state_vector(250).phi
            >>> M = np.random.rand(125, 125)
            >>> M /= M.sum(axis=1, keepdims=True)  # Normalize rows
            >>>
            >>> trajectory = mk.propagate_markov(
            ...     initial_state=initial_state,
            ...     transition_matrix=M,
            ...     n_steps=100
            ... )
            >>>
            >>> print(f"Shape: {trajectory['states'].shape}")  # (101, 125)
            >>> print(f"Conservation: {trajectory['states'].sum(axis=1)}")
        """
        n_states = transition_matrix.shape[0]

        # ---- Validation dimensions ----
        if initial_state.shape[0] != n_states:
            raise ValueError(
                f"Dimension mismatch: initial_state={initial_state.shape[0]}, "
                f"M={n_states}×{n_states}"
            )

        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("Matrice transition doit être carrée")

        # ---- Initialiser trajectoire ----
        traj = np.zeros((n_steps + 1, n_states), dtype=np.float32)
        traj[0] = initial_state

        # ---- Propager ----
        logger.info(f"Propagation Markov: {n_steps} pas...")
        for t in range(1, n_steps + 1):
            traj[t] = traj[t - 1] @ transition_matrix

            # Validation normalisation
            if validate_normalization:
                total = traj[t].sum()
                expected = initial_state.sum()
                relative_error = abs(total - expected) / max(expected, 1.0)

                if relative_error > 1e-3:
                    logger.warning(
                        f"⚠️  Pas {t}: écart normalisation = {relative_error * 100:.4f}%"
                    )

        logger.info("✅ Propagation terminée")

        # ---- Créer dictionnaire résultat ----
        times = np.arange(n_steps + 1)
        times_seconds = times * TIMESTEP_TO_SECONDS

        result: StateTrajectory = {
            "states": traj,
            "times": times,
            "times_seconds": times_seconds,
            "method": self.method,
            "description": f"Markov propagation {n_steps} steps",
        }

        return result

    def build_vtp(
        self,
        timestep_indices: list[int] | int = 250,
    ) -> pv.PolyData:
        """
        Construire objet PyVista PolyData pour visualisation.

        Attache les scalars (état, vitesse, diamètre, etc.)
        sur les points particules.

        Args:
            timestep_indices: Index(es) pour les particules

        Returns:
            pv.PolyData prêt pour stpyvista

        Examples:
            >>> mk = Markov("voronoi")
            >>> # ... setup ...
            >>> vtp = mk.build_vtp([250])
            >>> print(f"Points: {vtp.n_points}")
            >>> print(f"Scalars: {list(vtp.point_data.keys())}")
        """
        if not isinstance(timestep_indices, list):
            timestep_indices = [timestep_indices]

        # Obtenir coordonnées
        coords = self.get_coords(timestep_indices)
        states = self.compute_states(coords)

        # Créer PolyData
        vtp = pv.PolyData(coords)
        vtp.point_data["partition"] = states

        # Ajouter propriétés particules
        if len(timestep_indices) == 1:
            idx = timestep_indices[0]
            df = self.datas[idx]

            for col in ["Diameter", "Particle_ID", "Residence_Time"]:
                if col in df.columns:
                    vtp.point_data[col] = df[col].to_numpy()

            vel_cols = ["Velocity:0", "Velocity:1", "Velocity:2"]
            if all(c in df.columns for c in vel_cols):
                vtp.point_data["Velocity"] = df[vel_cols].to_numpy()

        self.vtp_states = vtp
        logger.info(f"VTP créé: {vtp.n_points} points")

        return vtp

    def visualize(self) -> None:
        """
        Visualiser le partitioning et les particules via Streamlit+PyVista.

        Affiche grille 3D avec:
        - Particules coloriées par partition
        - Option coupe (xy, yz, xz, oblique)
        - Scaling par diamètre

        Requires: Streamlit + PyVista + stpyvista

        Examples:
            >>> mk = Markov("cylindrical", method_kwargs={"nr": 5, "ntheta": 8, "nz": 6})
            >>> # ... setup ...
            >>> mk.visualize()  # Ouvre dans Streamlit
        """
        try:
            from stpyvista import stpyvista
        except ImportError:
            st.error("⚠️  stpyvista not installed. Install with: pip install stpyvista")
            return

        st.subheader("🎨 Partitioner Visualization")

        if self.vtp_states.is_empty:
            self.build_vtp()

        # Créer plotter
        pv.start_xvfb()
        pv.OFF_SCREEN = True
        pl = pv.Plotter(window_size=[600, 600], notebook=False)

        # Glyphs (sphères par particule)
        sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
        glyphs = self.vtp_states.glyph(
            geom=sphere,
            scale="Diameter",
            factor=1.0,
        )

        pl.add_mesh(glyphs, scalars="partition", cmap="tab10", show_scalar_bar=True)

        # Option coupe
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
            pl.clear()
            pl.add_mesh(clipped, scalars="partition", cmap="tab10")

        pl.camera_position = [
            (0.24, 0.32, 0.7),
            (0.02, 0.03, -0.02),
            (-0.12, 0.93, -0.34),
        ]

        stpyvista(pl)

    def __repr__(self) -> str:
        """Représentation string de l'instance."""
        return (
            f"Markov("
            f"method={self.method!r}, "
            f"n_cells={self.partitioner.n_cells}, "
            f"datas={len(self.datas)}, "
            f"coords_shape={self.coords.shape}"
            f")"
        )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Exemples d'utilisation de la classe Markov.

    Exécuter avec:
        python -m src.Markov.markov_core
    """

    logger.info("=" * 70)
    logger.info("MARKOV CLASS - EXAMPLES")
    logger.info("=" * 70)

    # Exemple 1: Initialisation simple
    logger.info("\n[Example 1] Initialisation simple")
    mk = Markov("voronoi", method_kwargs={"n_cells": 125})
    print(mk)

    # Exemple 2 (requiert HF access):
    logger.info("\n[Example 2] Full workflow (requires HF access)")
    logger.info("""
    # Uncomment to run (requires HuggingFace token):

    mk.load_dem_data(particle_diameter=0.004)
    coords = mk.get_coords([250, 300, 350])
    mk.fit_partitioner(coords)
    state0 = mk.build_initial_state_vector(250)
    print(f"φ(0) sum: {state0.phi.sum()}")

    # Propagation
    M = np.random.rand(125, 125)
    M /= M.sum(axis=1, keepdims=True)
    traj = mk.propagate_markov(state0.phi, M, 50)
    print(f"Trajectory shape: {traj['states'].shape}")
    """)
