"""
_config.py
==========
Configuration types et constantes globales pour le module DEM_MCM Markov.

Ce fichier centralise:
- TypedDict pour structures de données complexes
- Enums pour énumérations
- Dataclasses pour objets immuables
- Constantes globales
- Types personnalisés pour type-checking strict

Ce module est la FONDATION du type-checking de toute l'app.
Tous les autres modules dépendent de ces types.

Examples:
    >>> from src.Markov._config import PartitioningMethod, LoadedModel, AppContext
    >>> method: PartitioningMethod = "cartesian"
    >>> model = LoadedModel(
    ...     folder_name="voronoi_exp001",
    ...     method="voronoi",
    ...     n_states=125,
    ... )
"""

from __future__ import annotations
from typing import (
    TypedDict, Literal, Optional, Dict, Any, Final, Union, Sequence
)
from dataclasses import dataclass, field, asdict
from datetime import datetime
import time
import numpy as np
import json

# ============================================================================
# LITERAL TYPES (Énumérations de type)
# ============================================================================

PartitioningMethod = Literal[
    "cartesian",
    "cylindrical",
    "voronoi",
    "quantile",
    "octree",
    "physics",
    "adaptive",
    "multizone",
    "single",
]
"""Méthodes de partitionnement disponibles."""

ParticleDiameter = Literal[0.004, 0.008, None]
"""Diamètres de particules disponibles (None = toutes)."""

ViewMode = Literal["Grid", "Toggle", "Transparent"]
"""Modes d'affichage 3D."""


# ============================================================================
# CONSTANTES GLOBALES
# ============================================================================

AVAILABLE_METHODS: Final[tuple] = (
    "cartesian",
    "cylindrical",
    "voronoi",
    "quantile",
    "octree",
    "physics",
    "adaptive",
    "multizone",
    "single",
)
"""Tuple des méthodes disponibles."""

AVAILABLE_DIAMETERS: Final[tuple] = (None, 0.004, 0.008)
"""Tuple des diamètres disponibles."""

BUCKET_ID: Final[str] = "ktongue/DEM_MCM"
"""ID du bucket HuggingFace."""

DEM_START_TIMESTEP: Final[int] = 250
"""Timestep de départ des simulations DEM."""

TIMESTEP_TO_SECONDS: Final[float] = 0.01
"""Conversion timestep → secondes (1 timestep = 0.01 s)."""

MIN_FRACTION_VISITED: Final[float] = 1.0
"""Fraction d'états visités minimale acceptable."""

SESSION_CACHE_DIR: Final[str] = ".streamlit_cache"
"""Répertoire cache pour persistance session."""

SESSION_CONFIG_FILE: Final[str] = "session_state.json"
"""Nom fichier config session."""


# ============================================================================
# TYPED DICTS (Structures simples)
# ============================================================================

class PartitionerConfig(TypedDict, total=False):
    """
    Configuration pour créer un partitioner.
    
    Attributes:
        method: Type de partitionnement (ex: "voronoi")
        **kwargs: Paramètres spécifiques à la méthode
        
    Examples:
        >>> config: PartitionerConfig = {
        ...     "method": "voronoi",
        ...     "n_cells": 125,
        ... }
    """
    method: PartitioningMethod


class HFExperimentMetadata(TypedDict, total=False):
    """
    Métadonnées d'une expérience sauvegardée sur HuggingFace.
    
    Chargée typiquement depuis stats.json ou config.json
    sur le bucket HF.
    
    Attributes:
        folder: Nom du dossier expérience
        method: Type de partitionnement
        n_states: Nombre d'états
        nlt: Nombre de timesteps (Lagrangian trajectories)
        tau: Pas de temps Markov
        particle_diameter: Diamètre filtré
        fraction_visited: Fraction d'états visités (0.0-1.0)
        n_particles: Nombre total particules
    """
    folder: str
    method: PartitioningMethod
    n_states: int
    nlt: Optional[int]
    tau: Optional[int]
    particle_diameter: ParticleDiameter
    fraction_visited: float
    n_particles: int


class StateTrajectory(TypedDict, total=False):
    """
    Trajectoire complète de l'état au cours du temps.
    
    Représente l'évolution φ(t) pour tous les timesteps.
    
    Attributes:
        states: Array (n_timesteps, n_states) - historique
        times: Array (n_timesteps,) - indices timestep
        times_seconds: Array (n_timesteps,) - en secondes
        method: Méthode de partitionnement
        description: Label optionnel
    """
    states: np.ndarray
    times: np.ndarray
    times_seconds: Optional[np.ndarray]
    method: PartitioningMethod
    description: Optional[str]


# ============================================================================
# DATACLASSES (Objets immuables)
# ============================================================================

@dataclass(frozen=True)
class StateVector:
    """
    Vecteur d'état Markovien immutable.
    
    Représente la distribution de particules dans chaque partition
    à un instant donné.
    
    Invariants:
        - φ doit être 1D
        - φ.dtype doit être float
        - ∑φ = total_particles (optionnellement validé)
    
    Attributes:
        phi (np.ndarray): Vecteur d'état φ (shape: n_states,)
        timestamp (int): Indice temporel correspondant
        total_particles (int): Nombre total particules
        description (str): Label optionnel
        
    Methods:
        validate_normalization: Vérifier ∑φ = N_particles
        
    Examples:
        >>> phi = np.array([10, 15, 8, 12], dtype=np.float32)
        >>> state = StateVector(
        ...     phi=phi,
        ...     timestamp=250,
        ...     total_particles=45,
        ...     description="Initial state at t=250"
        ... )
        >>> assert state.validate_normalization()  # True
        >>> print(state.phi.sum())  # 45.0
    """
    phi: np.ndarray  # shape (n_states,), dtype float
    timestamp: int
    total_particles: int
    description: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Valider que phi a les bonnes dimensions au moment création."""
        if self.phi.ndim != 1:
            raise ValueError(
                f"phi doit être 1D, reçu shape {self.phi.shape}"
            )
        if self.phi.dtype not in [np.float32, np.float64, int]:
            raise TypeError(
                f"phi doit être float ou int, reçu {self.phi.dtype}"
            )
    
    def validate_normalization(self, tolerance: float = 1e-6) -> bool:
        """
        Vérifier que ∑φ = total_particles (conservation).
        
        Args:
            tolerance: Tolérance relative admissible
            
        Returns:
            bool: True si normalisé, False sinon
            
        Examples:
            >>> state.validate_normalization()  # True
            >>> state.validate_normalization(tolerance=1e-10)  # True ou False
        """
        total = self.phi.sum()
        relative_error = abs(total - self.total_particles) / max(
            self.total_particles, 1.0
        )
        return relative_error <= tolerance


@dataclass(frozen=True)
class LoadedModel:
    """
    Modèle chargé et prêt pour analyse.
    
    Immutable (frozen=True) pour éviter modifications accidentelles.
    Représente les métadonnées + données de UNE configuration.
    
    Attributes:
        folder_name (str): Clé unique dans HF bucket
        method (PartitioningMethod): Partitioning method utilisée
        particle_diameter (ParticleDiameter): Diamètre filtré
        n_states (int): Nombre de partitions
        n_particles (int): Nombre total particules
        nlt (int): Nombre Lagrangian trajectories
        tau (int): Pas de temps Markov
        fraction_visited (float): Fraction états visités
        description (str): Label optionnel
        
        # Cache (rempli lazy):
        transition_matrix (np.ndarray): M chargée ou None
        config_dict (Dict): Config JSON complet ou None
        
    Examples:
        >>> model = LoadedModel(
        ...     folder_name="voronoi_0.004_001",
        ...     method="voronoi",
        ...     particle_diameter=0.004,
        ...     n_states=125,
        ...     n_particles=5000,
        ...     nlt=100,
        ...     tau=50,
        ... )
        >>> print(model)  # Nice repr
        >>> model.is_data_loaded()  # False (pas encore chargée)
    """
    folder_name: str
    method: PartitioningMethod
    particle_diameter: ParticleDiameter
    n_states: int
    n_particles: int
    
    # Metadata
    nlt: Optional[int] = None
    tau: Optional[int] = None
    fraction_visited: float = 1.0
    description: Optional[str] = None
    
    # Cache (défaut None)
    transition_matrix: Optional[np.ndarray] = field(
        default=None,
        repr=False,
        compare=False
    )
    config_dict: Optional[Dict[str, Any]] = field(
        default=None,
        repr=False,
        compare=False
    )
    
    def __repr__(self) -> str:
        """Représentation courte pour affichage."""
        return (
            f"Model({self.method} | "
            f"d={self.particle_diameter} | "
            f"n={self.n_states} | "
            f"τ={self.tau})"
        )
    
    def is_data_loaded(self) -> bool:
        """Vérifier si données matrices/config sont chargées."""
        return (
            self.transition_matrix is not None
            or self.config_dict is not None
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialiser en dict (pour persistance, excluant cache)."""
        return {
            'folder_name': self.folder_name,
            'method': self.method,
            'particle_diameter': self.particle_diameter,
            'n_states': self.n_states,
            'n_particles': self.n_particles,
            'nlt': self.nlt,
            'tau': self.tau,
            'fraction_visited': self.fraction_visited,
            'description': self.description,
        }


@dataclass
class AppContext:
    """
    Contexte global de l'app - SYNCHRONISÉ via st.session_state.
    
    C'est l'unique "source de vérité" pour l'app.
    Toutes les pages lisent/écrivent UNIQUEMENT via cet objet.
    
    Invariants:
        - selected_models: liste des LoadedModel actuels
        - active_model_index: index valide ou 0
        - last_modified: timestamp UTC
    
    Attributes:
        selected_models (list[LoadedModel]): Configs actuellement actives
        last_modified (float): Timestamp UTC dernier changement
        active_model_index (int): Index modèle "focus"
        compare_mode (bool): True si comparaison multi-modèles
        current_filters (Dict): Filtres actuels (pour detect changements)
        version (int): Compteur incrémenté à chaque changement
        
    Examples:
        >>> ctx = AppContext()
        >>> model1 = LoadedModel(folder_name="exp1", ...)
        >>> ctx.add_model(model1)
        >>> print(ctx)  # Nice summary
        >>> ctx.remove_model("exp1")
        >>> ctx.clear_models()
    """
    selected_models: list[LoadedModel] = field(default_factory=list)
    last_modified: float = field(default_factory=time.time)
    active_model_index: int = 0
    compare_mode: bool = False
    
    # Filtres actuels (pour détection changements)
    current_filters: Dict[str, Any] = field(default_factory=dict)
    
    # Version (incrémentée à chaque changement)
    version: int = 0
    
    def add_model(self, model: LoadedModel) -> None:
        """
        Ajouter un modèle et marquer changement.
        
        Args:
            model: LoadedModel à ajouter
            
        Examples:
            >>> ctx.add_model(model)
            >>> assert model in ctx.selected_models
        """
        if model not in self.selected_models:
            self.selected_models.append(model)
            self._mark_changed()
    
    def remove_model(self, folder_name: str) -> None:
        """
        Retirer un modèle par folder_name.
        
        Args:
            folder_name: Clé du modèle à retirer
        """
        self.selected_models = [
            m for m in self.selected_models
            if m.folder_name != folder_name
        ]
        self._mark_changed()
    
    def clear_models(self) -> None:
        """Vider tous les modèles chargés."""
        self.selected_models.clear()
        self.active_model_index = 0
        self._mark_changed()
    
    def get_active_model(self) -> Optional[LoadedModel]:
        """
        Récupérer le modèle actuellement actif (focus).
        
        Returns:
            LoadedModel ou None si aucun modèle
        """
        if 0 <= self.active_model_index < len(self.selected_models):
            return self.selected_models[self.active_model_index]
        return None
    
    def set_active_model(self, index: int) -> None:
        """
        Définir le modèle actif.
        
        Args:
            index: Index dans selected_models
        """
        if 0 <= index < len(self.selected_models):
            self.active_model_index = index
            self._mark_changed()
    
    def _mark_changed(self) -> None:
        """Marquer contexte comme modifié (appelé en interne)."""
        self.last_modified = time.time()
        self.version += 1
    
    def summary(self) -> str:
        """
        Résumé pour affichage.
        
        Returns:
            str: Texte descriptif
        """
        return (
            f"Context: {len(self.selected_models)} modèles, "
            f"compare={self.compare_mode}, "
            f"active={self.active_model_index}, "
            f"version={self.version}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Sérialiser en dict (pour persistance).
        
        Returns:
            Dict sérialisable en JSON
        """
        return {
            'selected_models': [m.to_dict() for m in self.selected_models],
            'active_model_index': self.active_model_index,
            'compare_mode': self.compare_mode,
            'current_filters': self.current_filters,
            'last_modified': self.last_modified,
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> AppContext:
        """
        Restaurer depuis dict (pour persistance).
        
        Args:
            data: Dict provenant to_dict()
            
        Returns:
            AppContext restauré
        """
        ctx = AppContext()
        # Note: Reconstruction sans les données matrices (lazy-loaded)
        ctx.active_model_index = data.get('active_model_index', 0)
        ctx.compare_mode = data.get('compare_mode', False)
        ctx.current_filters = data.get('current_filters', {})
        ctx.last_modified = data.get('last_modified', time.time())
        
        # Réconstruire modèles (métadata seulement)
        for m_dict in data.get('selected_models', []):
            model = LoadedModel(
                folder_name=m_dict['folder_name'],
                method=m_dict['method'],
                particle_diameter=m_dict.get('particle_diameter'),
                n_states=m_dict.get('n_states', 0),
                n_particles=m_dict.get('n_particles', 0),
                nlt=m_dict.get('nlt'),
                tau=m_dict.get('tau'),
                fraction_visited=m_dict.get('fraction_visited', 1.0),
                description=m_dict.get('description'),
            )
            ctx.selected_models.append(model)
        
        return ctx


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bucket_prefix(diameter: ParticleDiameter) -> str:
    """
    Retourner le prefix HF selon diamètre particule.
    
    Args:
        diameter: Diamètre filtré (0.004, 0.008, None)
        
    Returns:
        str: Prefix bucket ("BIG", "SMALL", "Experiments")
        
    Examples:
        >>> get_bucket_prefix(0.008)
        'BIG'
        >>> get_bucket_prefix(None)
        'Experiments'
    """
    if diameter == 0.008:
        return "BIG"
    elif diameter == 0.004:
        return "SMALL"
    else:
        return "Experiments"


def validate_partitioning_method(method: str) -> PartitioningMethod:
    """
    Valider et typer la méthode de partitionnement.
    
    Args:
        method: Chaîne à valider
        
    Returns:
        PartitioningMethod typée
        
    Raises:
        ValueError: Si method non reconnue
        
    Examples:
        >>> validate_partitioning_method("voronoi")
        'voronoi'
        >>> validate_partitioning_method("invalid")
        ValueError: ...
    """
    if method not in AVAILABLE_METHODS:
        raise ValueError(
            f"Méthode inconnue: '{method}'. "
            f"Disponibles: {', '.join(AVAILABLE_METHODS)}"
        )
    return method  # type: ignore
