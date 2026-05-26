"""
components/visualizer.py
========================
PyVista 3D visualization utilities for DEM_MCM partitionings.

Provides functions to:
- Load DEM coordinates and fit partitioners
- Build PyVista scenes with particles colored by partition
- Render multiple view modes (grid, toggle, overlay)
- Export to images
"""

import numpy as np
import logging
from typing import Optional, Dict, List, Tuple
import tempfile
import os

try:
    import pyvista as pv
    from pyvista import Plotter
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    logging.warning("⚠️  PyVista not installed, 3D visualization disabled")

from src.Markov._config import LoadedModel, PartitioningMethod
from src.Markov.markov_core import Markov
from app.components.model_loader import get_model_loader

logger = logging.getLogger(__name__)


class PartitioningVisualizer:
    """
    Visualize DEM data with partitioning colormap.
    
    Usage:
        >>> viz = PartitioningVisualizer()
        >>> img = viz.render_single(
        ...     model=my_model,
        ...     coords=coordinates,
        ...     height=400
        ... )
        >>> # img is a numpy array (RGB image)
    """
    
    def __init__(self):
        """Initialize visualizer."""
        if not HAS_PYVISTA:
            raise RuntimeError(
                "❌ PyVista not installed. "
                "Install with: pip install pyvista"
            )
        
        self.logger = logging.getLogger(__name__)
        self.cmap = "tab20"  # Colormap for partitions
    
    def fit_partitioner(
        self,
        model: LoadedModel,
        coords: np.ndarray,
    ):
        """
        Fit a Markov partitioner to coordinates.
        
        Args:
            model: LoadedModel with method and config
            coords: Coordinates array (N, 3)
            
        Returns:
            Fitted partitioner object
            
        Raises:
            ValueError: If model config invalid
        """
        from app.components.model_loader import get_model_loader
        
        loader = get_model_loader()
        
        try:
            mk = loader.build_markov(model.folder_name)
        except Exception as e:
            logger.error(f"❌ Could not build Markov for {model.folder_name}: {e}")
            raise
        
        # Fit partitioner to coordinates
        try:
            mk.fit_partitioner(coords)
            return mk.partitioner
        except Exception as e:
            logger.error(f"❌ Could not fit partitioner: {e}")
            raise
    
    def compute_partition_states(
        self,
        coords: np.ndarray,
        partitioner,
    ) -> np.ndarray:
        """
        Compute partition state for each coordinate.
        
        Args:
            coords: Coordinates array (N, 3)
            partitioner: Fitted partitioner object
            
        Returns:
            State array (N,) with partition indices
        """
        try:
            states = partitioner.compute_states(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2]
            )
            return states
        except Exception as e:
            logger.error(f"❌ Could not compute states: {e}")
            raise
    
    def render_single(
        self,
        model: LoadedModel,
        coords: np.ndarray,
        height: int = 600,
        width: int = 800,
        background: str = "white",
        point_size: float = 5.0,
        show_grid: bool = True,
        camera_position: str = "isometric",
    ) -> np.ndarray:
        """
        Render a single partitioning visualization.
        
        Args:
            model: LoadedModel with method and config
            coords: Coordinates array (N, 3)
            height: Image height in pixels
            width: Image width in pixels
            background: Background color ("white", "black", etc)
            point_size: Size of rendered points
            show_grid: Whether to show grid axes
            camera_position: Camera angle ("isometric", "xy", "xz", "yz", etc)
            
        Returns:
            RGB image array, shape (height, width, 3), dtype uint8
            
        Raises:
            ValueError: If model config invalid or coords wrong shape
        """
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
        
        logger.info(
            f"🎨 Rendering {model}: "
            f"{model.n_states} states, {coords.shape[0]} particles"
        )
        
        try:
            # Fit partitioner
            partitioner = self.fit_partitioner(model, coords)
            
            # Compute states
            states = self.compute_partition_states(coords, partitioner)
            
            # Create PyVista mesh
            cloud = pv.PolyData(coords)
            cloud["partition"] = states
            
            # Create plotter
            plotter = Plotter(
                off_screen=True,
                window_size=(width, height),
            )
            
            # Add mesh with color mapping
            plotter.add_mesh(
                cloud,
                scalars="partition",
                point_size=point_size,
                render_points_as_spheres=True,
                cmap=self.cmap,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "Partition",
                    "vertical": True,
                },
            )
            
            # Configure view
            plotter.background_color = background
            if show_grid:
                plotter.show_grid(
                    xtitle="X", ytitle="Y", ztitle="Z",
                    show_xaxis=True, show_yaxis=True, show_zaxis=True
                )
            
            # Set camera position
            if camera_position == "isometric":
                plotter.view_isometric()
            elif camera_position == "xy":
                plotter.view_xy()
            elif camera_position == "xz":
                plotter.view_xz()
            elif camera_position == "yz":
                plotter.view_yz()
            else:
                plotter.view_isometric()
            
            # Render to image
            plotter.camera.zoom(0.9)  # Small zoom out for better view
            image = plotter.screenshot()
            plotter.close()
            
            logger.info(f"✅ Rendered: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Rendering failed: {e}")
            raise
    
    def render_grid(
        self,
        models: List[LoadedModel],
        coords: np.ndarray,
        n_cols: int = 2,
        height: int = 400,
        width: int = 500,
        **render_kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Render multiple models in a grid layout.
        
        Returns separate images (not composited into grid).
        
        Args:
            models: List of LoadedModel instances
            coords: Coordinates array (N, 3)
            n_cols: Number of columns (for reference only)
            height: Height per image
            width: Width per image
            **render_kwargs: Additional arguments to render_single
            
        Returns:
            Dict mapping model.folder_name → image array
            
        Example:
            >>> images = viz.render_grid([model1, model2])
            >>> for name, img in images.items():
            ...     print(f"{name}: {img.shape}")
        """
        images = {}
        
        for i, model in enumerate(models):
            try:
                img = self.render_single(
                    model=model,
                    coords=coords,
                    height=height,
                    width=width,
                    **render_kwargs
                )
                images[model.folder_name] = img
            except Exception as e:
                logger.warning(f"⚠️  Failed to render {model.folder_name}: {e}")
                images[model.folder_name] = None
        
        return images
    
    def render_overlay(
        self,
        models: List[LoadedModel],
        coords: np.ndarray,
        opacities: Optional[Dict[str, float]] = None,
        height: int = 600,
        width: int = 800,
        background: str = "white",
        show_grid: bool = True,
    ) -> np.ndarray:
        """
        Render multiple models overlaid in a single scene.
        
        Args:
            models: List of LoadedModel instances
            coords: Coordinates array (N, 3)
            opacities: Dict mapping model.folder_name → opacity (0-1).
                      If None, uses 0.7 for all.
            height: Image height
            width: Image width
            background: Background color
            show_grid: Whether to show grid
            
        Returns:
            RGB image array with all models overlaid
            
        Note:
            - Partitions colored per model with different colormaps
            - Useful for comparing partitioning schemes
        """
        if opacities is None:
            opacities = {m.folder_name: 0.7 for m in models}
        
        logger.info(f"🎨 Rendering overlay of {len(models)} models")
        
        try:
            plotter = Plotter(
                off_screen=True,
                window_size=(width, height),
            )
            
            # Color map for models (to distinguish them)
            model_colors = [
                "#FF6B6B",  # Red
                "#4ECDC4",  # Teal
                "#45B7D1",  # Blue
                "#FFA07A",  # Salmon
                "#98D8C8",  # Mint
            ]
            
            for i, model in enumerate(models):
                try:
                    # Fit and compute states
                    partitioner = self.fit_partitioner(model, coords)
                    states = self.compute_partition_states(coords, partitioner)
                    
                    # Create mesh
                    cloud = pv.PolyData(coords)
                    cloud["partition"] = states
                    
                    # Add with opacity
                    color = model_colors[i % len(model_colors)]
                    opacity = opacities.get(model.folder_name, 0.7)
                    
                    plotter.add_mesh(
                        cloud,
                        color=color,
                        point_size=5.0,
                        render_points_as_spheres=True,
                        opacity=opacity,
                        label=f"{model.method} (d={model.particle_diameter})",
                    )
                    
                except Exception as e:
                    logger.warning(
                        f"⚠️  Could not render {model.folder_name} in overlay: {e}"
                    )
                    continue
            
            # Configure view
            plotter.background_color = background
            if show_grid:
                plotter.show_grid(
                    xtitle="X", ytitle="Y", ztitle="Z",
                    show_xaxis=True, show_yaxis=True, show_zaxis=True
                )
            
            plotter.add_legend()
            plotter.view_isometric()
            plotter.camera.zoom(0.9)
            
            # Render
            image = plotter.screenshot()
            plotter.close()
            
            logger.info(f"✅ Overlay rendered: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Overlay rendering failed: {e}")
            raise
    
    def render_with_clipping(
        self,
        model: LoadedModel,
        coords: np.ndarray,
        clip_plane: str = "z",
        clip_value: float = 0.5,
        height: int = 600,
        width: int = 800,
        **render_kwargs
    ) -> np.ndarray:
        """
        Render with a clipping plane to cut away part of the domain.
        
        Args:
            model: LoadedModel
            coords: Coordinates array (N, 3)
            clip_plane: "x", "y", or "z" - axis perpendicular to clipping plane
            clip_value: Position along clip_plane axis (0-1 normalized, or actual value)
            height: Image height
            width: Image width
            **render_kwargs: Additional render arguments
            
        Returns:
            RGB image array with clipping applied
            
        Example:
            >>> img = viz.render_with_clipping(
            ...     model, coords,
            ...     clip_plane="z",
            ...     clip_value=0.5  # Cut at middle of z-axis
            ... )
        """
        logger.info(
            f"🎨 Rendering with clipping: plane={clip_plane}, value={clip_value}"
        )
        
        try:
            # Get bounds
            bounds = [coords[:, i].min() for i in range(3)] + \
                     [coords[:, i].max() for i in range(3)]
            
            # Fit partitioner
            partitioner = self.fit_partitioner(model, coords)
            states = self.compute_partition_states(coords, partitioner)
            
            # Create mesh
            cloud = pv.PolyData(coords)
            cloud["partition"] = states
            
            # Apply clipping
            plane_axis = {"x": 0, "y": 1, "z": 2}[clip_plane]
            if 0 <= clip_value <= 1:
                # Normalize
                clip_pos = bounds[plane_axis] + \
                           clip_value * (bounds[3 + plane_axis] - bounds[plane_axis])
            else:
                clip_pos = clip_value
            
            # Clip mesh
            normal = [0, 0, 0]
            normal[plane_axis] = 1
            clipped = cloud.clip(normal=normal, origin=[clip_pos if i == plane_axis else 0 for i in range(3)])
            
            # Create plotter
            plotter = Plotter(
                off_screen=True,
                window_size=(width, height),
            )
            
            # Add clipped mesh
            plotter.add_mesh(
                clipped,
                scalars="partition",
                point_size=5.0,
                render_points_as_spheres=True,
                cmap=self.cmap,
                show_scalar_bar=True,
            )
            
            # Add clipping plane visualization
            plane_center = [clip_pos if i == plane_axis else 0 for i in range(3)]
            plane = pv.Plane(center=plane_center, i=normal)
            plotter.add_mesh(plane, color="red", opacity=0.2, style="wireframe")
            
            plotter.background_color = render_kwargs.get("background", "white")
            if render_kwargs.get("show_grid", True):
                plotter.show_grid()
            
            plotter.view_isometric()
            plotter.camera.zoom(0.9)
            
            image = plotter.screenshot()
            plotter.close()
            
            logger.info(f"✅ Clipped view rendered: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"❌ Clipping rendering failed: {e}")
            raise


def save_image(image: np.ndarray, filename: str) -> str:
    """
    Save image array to file.
    
    Args:
        image: RGB image array
        filename: Output filename (can include path)
        
    Returns:
        Absolute path to saved file
    """
    try:
        from PIL import Image
        img = Image.fromarray(image)
        img.save(filename)
        logger.info(f"✅ Image saved: {filename}")
        return os.path.abspath(filename)
    except Exception as e:
        logger.error(f"❌ Failed to save image: {e}")
        raise
