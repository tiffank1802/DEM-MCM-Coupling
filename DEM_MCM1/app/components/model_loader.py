"""
components/model_loader.py
===========================
Lazy model loading and caching for HuggingFace DEM_MCM experiments.

Provides utilities to:
- Load HF config.json and build Markov instances
- Cache matrices and partitioner data
- Build LoadedModel dataclass instances
- Handle HF authentication
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

from src.Markov._config import (
    LoadedModel,
    PartitioningMethod,
)
from src.Markov.markov_core import Markov
from src.Markov.bucket_io import get_fs

logger = logging.getLogger(__name__)


class ModelLoaderCache:
    """
    Cache for lazy-loaded HF models.
    
    Usage:
        >>> cache = ModelLoaderCache()
        >>> loaded = cache.get_or_load("voronoi_125_run1")
        >>> print(loaded.transition_matrix.shape)
    """
    
    def __init__(self):
        self._models_cache: dict[str, LoadedModel] = {}
        self._matrices_cache: dict[str, np.ndarray] = {}
    
    def get_or_load(
        self,
        folder_name: str,
        bucket_base: str = "hf://buckets/ktongue/DEM_MCM/SMALL",
    ) -> LoadedModel:
        """
        Get LoadedModel from cache or load from HF.
        
        Args:
            folder_name: Name of experiment folder
            bucket_base: HF bucket path prefix
            
        Returns:
            LoadedModel instance with lazy-loaded matrices
            
        Raises:
            FileNotFoundError: If model not found on HF
            ValueError: If config.json invalid
        """
        # Check cache first
        if folder_name in self._models_cache:
            logger.debug(f"✅ Cache hit: {folder_name}")
            return self._models_cache[folder_name]
        
        logger.info(f"📦 Loading model: {folder_name}")
        
        try:
            fs = get_fs()
            prefix = f"{bucket_base}/{folder_name}"
            
            # Load config.json
            config_path = f"{prefix}/config.json"
            with fs.open(config_path, "r") as f:
                import json
                config = json.load(f)
            
            # Load stats.json for metadata
            stats_path = f"{prefix}/stats.json"
            try:
                with fs.open(stats_path, "r") as f:
                    import json
                    stats = json.load(f)
            except:
                stats = {}
            
            # Extract key metadata
            method: PartitioningMethod = config.get("method", "unknown")
            n_states = config.get("n_states", stats.get("n_states", 1))
            n_particles = stats.get("n_particles", 100)
            particle_diameter = stats.get("particle_diameter")
            
            # Create LoadedModel with lazy-loaded matrices
            loaded_model = LoadedModel(
                folder_name=folder_name,
                method=method,
                n_states=n_states,
                n_particles=n_particles,
                particle_diameter=particle_diameter,
                nlt=stats.get("n_timesteps_used"),
                tau=config.get("tau"),
                fraction_visited=stats.get("fraction_visited", 1.0),
                transition_matrix=None,  # Will load on-demand
                config_dict=config,
            )
            
            # Cache it
            self._models_cache[folder_name] = loaded_model
            
            logger.info(
                f"✅ Loaded: {folder_name} "
                f"({method}, {n_states} states, "
                f"d={particle_diameter})"
            )
            
            return loaded_model
            
        except Exception as e:
            logger.error(f"❌ Failed to load {folder_name}: {e}")
            raise
    
    def load_matrix(
        self,
        folder_name: str,
        bucket_base: str = "hf://buckets/ktongue/DEM_MCM/SMALL",
    ) -> np.ndarray:
        """
        Load transition matrix from HF (cached).
        
        Args:
            folder_name: Name of experiment folder
            bucket_base: HF bucket path prefix
            
        Returns:
            Transition matrix, shape (n_states, n_states)
            
        Raises:
            FileNotFoundError: If matrix.npy not found
        """
        # Check cache
        if folder_name in self._matrices_cache:
            logger.debug(f"✅ Matrix cache hit: {folder_name}")
            return self._matrices_cache[folder_name]
        
        logger.info(f"📦 Loading matrix: {folder_name}")
        
        try:
            fs = get_fs()
            matrix_path = f"{bucket_base}/{folder_name}/transitionmatrix.npy"
            
            with fs.open(matrix_path, "rb") as f:
                import io
                matrix = np.load(io.BytesIO(f.read()))
            
            # Cache it
            self._matrices_cache[folder_name] = matrix
            
            logger.info(f"✅ Matrix loaded: shape {matrix.shape}")
            
            return matrix
            
        except Exception as e:
            logger.error(f"❌ Failed to load matrix {folder_name}: {e}")
            raise
    
    def build_markov(
        self,
        folder_name: str,
        bucket_base: str = "hf://buckets/ktongue/DEM_MCM/SMALL",
    ) -> Markov:
        """
        Build a Markov instance from cached model + config.
        
        Args:
            folder_name: Name of experiment folder
            bucket_base: HF bucket path prefix
            
        Returns:
            Configured Markov instance (ready for propagation)
            
        Example:
            >>> loader = ModelLoaderCache()
            >>> mk = loader.build_markov("voronoi_125_run1")
            >>> trajectory = mk.propagate_markov(
            ...     initial_state=mk.build_initial_state_vector(250).phi,
            ...     transition_matrix=loader.load_matrix("voronoi_125_run1"),
            ...     n_steps=100
            ... )
        """
        # Load model metadata
        loaded = self.get_or_load(folder_name, bucket_base)
        config = loaded.config_dict
        
        # Extract Markov config
        method = loaded.method
        method_kwargs = config.get("method_kwargs", {})
        
        # Create Markov instance
        mk = Markov(method=method, method_kwargs=method_kwargs)
        
        logger.info(
            f"✅ Markov instance created: {method} "
            f"with kwargs={method_kwargs}"
        )
        
        return mk
    
    def clear_cache(self):
        """Clear all cached models and matrices."""
        self._models_cache.clear()
        self._matrices_cache.clear()
        logger.info("✅ Cache cleared")
    
    def cache_info(self) -> dict:
        """Get cache statistics."""
        return {
            "models_cached": len(self._models_cache),
            "matrices_cached": len(self._matrices_cache),
            "model_names": list(self._models_cache.keys()),
        }


# Global singleton instance
_default_loader: Optional[ModelLoaderCache] = None


def get_model_loader() -> ModelLoaderCache:
    """
    Get or create default ModelLoaderCache instance.
    
    Returns:
        Singleton ModelLoaderCache
    """
    global _default_loader
    if _default_loader is None:
        _default_loader = ModelLoaderCache()
    return _default_loader


def load_and_cache_models(
    folder_names: list[str],
    bucket_base: str = "hf://buckets/ktongue/DEM_MCM/SMALL",
) -> list[LoadedModel]:
    """
    Batch load multiple models into cache.
    
    Args:
        folder_names: List of experiment folder names
        bucket_base: HF bucket path prefix
        
    Returns:
        List of LoadedModel instances
        
    Example:
        >>> models = load_and_cache_models([
        ...     "voronoi_125_run1",
        ...     "voronoi_125_run2",
        ...     "cartesian_100_run1",
        ... ])
        >>> print(f"Cached {len(models)} models")
    """
    loader = get_model_loader()
    loaded_models = []
    
    for folder_name in folder_names:
        try:
            model = loader.get_or_load(folder_name, bucket_base)
            loaded_models.append(model)
        except Exception as e:
            logger.warning(f"⚠️  Failed to load {folder_name}: {e}")
            continue
    
    logger.info(f"✅ Batch loaded {len(loaded_models)} models")
    return loaded_models
