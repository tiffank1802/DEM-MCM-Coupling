"""
===================================================================================
 PARTITIONERS — Méthodes de partitionnement spatial pour chaînes de Markov
===================================================================================

 Interface commune:
     partitioner = create_partitioner("voronoi", n_cells=125)
     partitioner.fit(coordinates)                    # (N, 3) numpy array
     states = partitioner.compute_states(x, y, z)    # → indices int64
     partitioner.save("output/")
     partitioner.load("output/")
 
 Méthodes disponibles:
     cartesian    — grille régulière (x, y, z)
     cylindrical  — grille cylindrique (r, θ, z)
     voronoi      — clustering K-means / cellules de Voronoï
     quantile     — grille avec bords par quantiles (équi-population)
     octree       — octree adaptatif à la densité
     physics      — K-means sur position + champs physiques
===================================================================================
"""
from __future__ import annotations
import numpy as np
import os
import io
import pyvista as pv
import json
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits
from scipy.spatial import ConvexHull, Voronoi
import matplotlib.animation as animation
import streamlit as st

# Imports relatifs (notebooks) vs absolus (script direct)
try:
    from . import analyze_results as ar
except ImportError:
    import analyze_results as ar

__all__ = [
    "BasePartitioner",
    "CartesianPartitioner",
    "CylindricalPartitioner",
    "VoronoiPartitioner",
    "QuantileGridPartitioner",
    "OctreePartitioner",
    "PhysicsAwarePartitioner",
    "FullVectorVelocityKMeansPartitioner",
    "SpectralClusteringPartitioner",
    "GaussianMixturePartitioner",
    "AdaptivePartitioner",
    "MultiZonePartitioner",
    "SingleCellPartitioner",
    "create_partitioner",
    "REGISTRY",
]

# =============================================================================
# CLASSE DE BASE
# =============================================================================


class BasePartitioner(ABC,ar.MarkovAnalyzer):
    """Interface commune pour tous les partitionneurs."""
    def __init__(self:BasePartitioner)->None:
        self._y_split:float=0
        self.y_seuil:float=None # type:ignore
        self._splitting_method: str = None #type: ignore
        super().__init__()
        self.PARTICLE_NUMBER:int=1030
        self.particle_diameters:np.ndarray=None #type:ignore
        self.states=np.array([])
        
    # analyzer=ar.MarkovAnalyzer()

    @property
    @abstractmethod
    def n_cells(self:BasePartitioner)-> int:
        """Nombre total d'états."""
        ...

    @property
    @abstractmethod
    def label(self:BasePartitioner)-> str:
        """Identifiant unique (utilisé pour le nom de dossier)."""
        ...

    @abstractmethod
    @st.cache_data(
        hash_funcs={"__main__.BasePartitioner":lambda x:x.label}
    )
    def fit(_self:BasePartitioner, coordinates: np.ndarray)->BasePartitioner:
        """
        Apprend le partitionnement sur des données représentatives.

        Args:
            coordinates: np.ndarray shape (N, 3)
        Returns:
            self
        """
        ...

    @abstractmethod
    @st.cache_data(
        hash_funcs={"__main__.BasePartitioner":lambda x:x.label}
    )
    def compute_states(self:BasePartitioner, x:np.ndarray, y:np.ndarray, z:np.ndarray,vx: np.ndarray=None,vy: np.ndarray=None,vz: np.ndarray=None)->np.ndarray: #type: ignore   
        """
        Assigne un indice d'état à chaque particule.

        Args:
            x, y, z: arrays ou Polars Series
        Returns:
            np.ndarray dtype int64
        """
        ...

    def save(self: BasePartitioner, path: str)->None:
        """Sauvegarde le partitionneur dans un dossier."""
        os.makedirs(path, # The above code is not valid Python code. It appears to be a comment with
        # the text "ex" followed by multiple pound symbols.
        exist_ok=True)
        meta = {
            "type": type(self).__name__,
            "label": self.label,
            "n_cells": self.n_cells,
        }
        with open(os.path.join(path, "partitioner_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._save_data(path)

    def _save_data(self: BasePartitioner, path: str)->None:
        pass

    def load(self: BasePartitioner, path: str)->BasePartitioner:
        """Charge le partitionneur depuis un dossier."""
        self._load_data(path)
        return self

    def _load_data(self: BasePartitioner, path: str)->None:
        pass


    def visualize_scientific(
    self,
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    vx: np.ndarray = None, vy: np.ndarray = None, vz: np.ndarray = None,
    particle_diameters: np.ndarray = None,
    time_index: int = 0,
    n_particles_per_step: int = 1030,
    plot_types: list = None,
    save_prefix: str = "partition_scientific",
    slice_thickness: float = 0.02,
    show_boundaries: bool = True,
    show_diameter: bool = True,
    dpi: int = 300,
    figsize_2d=(10, 8),
    figsize_3d=(12, 10),
    cmap_states="tab20",
    font_family="sans-serif",
) -> dict:
        """
        Génère des figures scientifiques propres pour un instant donné.
        
        Args:
            x, y, z           : coordonnées de TOUTES les particules sur TOUS les instants (shape: n_steps * n_particles)
            vx, vy, vz        : vitesses de TOUTES les particules sur TOUS les instants (optionnelles)
            particle_diameters: diamètres de TOUTES les particules (shape: n_steps * n_particles ou n_particles)
            time_index        : index de l'instant à visualiser (défaut: 0)
            n_particles_per_step: nombre de particules par instant (défaut: 1030)
            plot_types        : liste parmi ["projection_xy","projection_yz","slice_xy","slice_yz","slice_xz","3d"]
            save_prefix       : préfixe des fichiers sauvegardés
            slice_thickness   : épaisseur de la coupe (m)
            show_boundaries   : dessiner l'enveloppe convexe de chaque partition
            show_diameter     : adapter la taille des markers au diamètre
            dpi               : résolution de sortie
            figsize_2d        : taille des figures 2D
            figsize_3d        : taille des figures 3D
            cmap_states       : colormap discrète pour les états
            font_family       : famille de police
        
        Returns:
            dict : { "projection_xy.png": bytes, "slice_xy.png": bytes, ... }
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.patches import Polygon
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from scipy.spatial import ConvexHull
        import io
        
        # Valeurs par défaut
        if plot_types is None:
            plot_types = ["projection_xy", "projection_yz", "3d"]
        
        # Configuration matplotlib
        plt.rcParams.update({
            "font.family": font_family,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
        })
        
        # ============================================================
        #  EXTRACTION DES DONNÉES POUR L'INSTANT SPÉCIFIÉ
        # ============================================================
        n_steps = len(x) // n_particles_per_step
        
        if time_index < 0 or time_index >= n_steps:
            raise ValueError(f"time_index={time_index} hors limites [0, {n_steps-1}]")
        
        # Indices pour extraire les données de l'instant souhaité
        start_idx = time_index * n_particles_per_step
        end_idx = start_idx + n_particles_per_step
        
        # Extraction des coordonnées pour cet instant
        x_instant = np.asarray(x[start_idx:end_idx])
        y_instant = np.asarray(y[start_idx:end_idx])
        z_instant = np.asarray(z[start_idx:end_idx])
        
        # Extraction des états pour cet instant (self.states contient déjà tous les états)
        if not hasattr(self, 'states') or self.states is None:
            raise AttributeError("self.states n'est pas défini. Appelez fit() ou compute_states() au préalable.")
        
        states_instant = self.states[start_idx:end_idx]
        
        # Extraction des vitesses si disponibles
        vx_instant = vy_instant = vz_instant = None
        if vx is not None and len(vx) > 0:
            vx = np.asarray(vx)
            vy = np.asarray(vy) if vy is not None else None
            vz = np.asarray(vz) if vz is not None else None
            if len(vx) == len(x):  # vx contient toutes les vitesses
                vx_instant = vx[start_idx:end_idx]
                vy_instant = vy[start_idx:end_idx] if vy is not None else None
                vz_instant = vz[start_idx:end_idx] if vz is not None else None
            else:  # vx contient déjà uniquement l'instant souhaité
                vx_instant = vx
                vy_instant = vy
                vz_instant = vz
        
        # Extraction des diamètres si disponibles
        diameters_instant = None
        if show_diameter:
            if particle_diameters is not None:
                particle_diameters = np.asarray(particle_diameters)
                if len(particle_diameters) == len(x):  # Tous les diamètres
                    diameters_instant = particle_diameters[start_idx:end_idx]
                else:  # Déjà filtré pour l'instant
                    diameters_instant = particle_diameters
            elif hasattr(self, 'dem_diameters') and self.dem_diameters is not None:
                if len(self.dem_diameters) == len(x):
                    diameters_instant = np.asarray(self.dem_diameters[start_idx:end_idx])
        
        # ============================================================
        #  PRÉPARATION DES DONNÉES
        # ============================================================
        # Borne des axes
        bounds = {
            "x": (x_instant.min(), x_instant.max()),
            "y": (y_instant.min(), y_instant.max()),
            "z": (z_instant.min(), z_instant.max()),
        }
        margin_factor = 0.05
        for k, (lo, hi) in bounds.items():
            m = (hi - lo) * margin_factor
            bounds[k] = (lo - m, hi + m)
        
        # Colormap discrète
        cmap = cm.get_cmap(cmap_states)
        unique_states = np.unique(states_instant)
        n_states = len(unique_states)
        
        def state_color(s):
            if n_states <= 1:
                return cmap(0.0)
            return cmap(s / (n_states - 1))
        
        # Tailles de markers
        if diameters_instant is not None and diameters_instant.max() > 0:
            sizes = 10 + (diameters_instant / diameters_instant.max()) ** 2 * 110
        else:
            sizes = 18 * np.ones(len(x_instant))
        
        image_data = {}
        
        # ============================================================
        #  UTILITAIRES INTERNES
        # ============================================================
        def _draw_boundaries_2d(ax, coords_2d, states, unique_states, color_func):
            if not show_boundaries:
                return
            for s in unique_states:
                mask = states == s
                pts = coords_2d[mask]
                if len(pts) < 3:
                    continue
                try:
                    hull = ConvexHull(pts)
                    poly = Polygon(pts[hull.vertices], closed=True,
                                facecolor=color_func(s), alpha=0.15,
                                edgecolor=color_func(s), linewidth=1.2,
                                linestyle="--")
                    ax.add_patch(poly)
                except Exception:
                    pass
        
        def _draw_boundaries_3d(ax, coords_3d, states, unique_states, color_func):
            if not show_boundaries:
                return
            for s in unique_states:
                mask = states == s
                pts = coords_3d[mask]
                if len(pts) < 4:
                    continue
                try:
                    hull = ConvexHull(pts)
                    faces = [pts[simplex] for simplex in hull.simplices]
                    collection = Poly3DCollection(
                        faces, alpha=0.08,
                        facecolor=color_func(s),
                        edgecolor=color_func(s),
                        linewidth=0.8, linestyle="--")
                    ax.add_collection3d(collection)
                except Exception:
                    pass
        
        def _style_axis(ax, xlabel, ylabel, title, xlim=None, ylim=None):
            ax.set_xlabel(xlabel, fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_title(title, fontweight="bold", fontsize=13)
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
            ax.set_aspect("equal", adjustable="box")
        
        def _to_bytes(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            buf.seek(0)
            return buf.getvalue()
        
        # ============================================================
        #  1. PROJECTION 2D
        # ============================================================
        if "projection_xy" in plot_types:
            fig, ax = plt.subplots(figsize=figsize_2d, facecolor="white")
            sc = ax.scatter(x_instant, y_instant, s=sizes, c=states_instant, cmap=cmap,
                            alpha=0.85, edgecolors="black", linewidth=0.3, zorder=3)
            _draw_boundaries_2d(ax, np.column_stack([x_instant, y_instant]), 
                                states_instant, unique_states, state_color)
            _style_axis(ax, "X (m)", "Y (m)",
                        f"Projection XY — t={time_index} — {self.label}")
            cb = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
            cb.set_label("État (ID partition)", fontweight="bold")
            plt.tight_layout()
            image_data[f"{save_prefix}_projection_xy_t{time_index}.png"] = _to_bytes(fig)
            plt.close(fig)
        
        if "projection_yz" in plot_types:
            fig, ax = plt.subplots(figsize=figsize_2d, facecolor="white")
            sc = ax.scatter(y_instant, z_instant, s=sizes, c=states_instant, cmap=cmap,
                            alpha=0.85, edgecolors="black", linewidth=0.3, zorder=3)
            _draw_boundaries_2d(ax, np.column_stack([y_instant, z_instant]),
                                states_instant, unique_states, state_color)
            _style_axis(ax, "Y (m)", "Z (m)",
                        f"Projection YZ — t={time_index} — {self.label}")
            cb = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
            cb.set_label("État (ID partition)", fontweight="bold")
            plt.tight_layout()
            image_data[f"{save_prefix}_projection_yz_t{time_index}.png"] = _to_bytes(fig)
            plt.close(fig)
        
        if "projection_xz" in plot_types:
            fig, ax = plt.subplots(figsize=figsize_2d, facecolor="white")
            sc = ax.scatter(x_instant, z_instant, s=sizes, c=states_instant, cmap=cmap,
                            alpha=0.85, edgecolors="black", linewidth=0.3, zorder=3)
            _draw_boundaries_2d(ax, np.column_stack([x_instant, z_instant]),
                                states_instant, unique_states, state_color)
            _style_axis(ax, "X (m)", "Z (m)",
                        f"Projection XZ — t={time_index} — {self.label}")
            cb = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
            cb.set_label("État (ID partition)", fontweight="bold")
            plt.tight_layout()
            image_data[f"{save_prefix}_projection_xz_t{time_index}.png"] = _to_bytes(fig)
            plt.close(fig)
        
        # ============================================================
        #  2. COUPES 2D
        # ============================================================
        if "slice_xy" in plot_types:
            z_center = (bounds["z"][0] + bounds["z"][1]) / 2
            mask = np.abs(z_instant - z_center) <= slice_thickness / 2
            fig, ax = plt.subplots(figsize=figsize_2d, facecolor="white")
            if mask.sum() > 0:
                sc = ax.scatter(x_instant[mask], y_instant[mask], s=sizes[mask],
                                c=states_instant[mask], cmap=cmap,
                                alpha=0.9, edgecolors="black", linewidth=0.3, zorder=3)
                _draw_boundaries_2d(ax, np.column_stack([x_instant[mask], y_instant[mask]]),
                                    states_instant[mask], unique_states, state_color)
            _style_axis(ax, "X (m)", "Y (m)",
                        f"Coupe XY @ z={z_center:.3f} m — t={time_index} — {self.label}",
                        xlim=bounds["x"], ylim=bounds["y"])
            ax.text(0.02, 0.02,
                    f"épaisseur = {slice_thickness*1000:.1f} mm\nn_particules = {mask.sum()}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
            if mask.sum() > 0:
                cb = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
                cb.set_label("État (ID partition)", fontweight="bold")
            plt.tight_layout()
            image_data[f"{save_prefix}_slice_xy_t{time_index}.png"] = _to_bytes(fig)
            plt.close(fig)
        
        if "slice_yz" in plot_types:
            x_center = (bounds["x"][0] + bounds["x"][1]) / 2
            mask = np.abs(x_instant - x_center) <= slice_thickness / 2
            fig, ax = plt.subplots(figsize=figsize_2d, facecolor="white")
            if mask.sum() > 0:
                sc = ax.scatter(y_instant[mask], z_instant[mask], s=sizes[mask],
                                c=states_instant[mask], cmap=cmap,
                                alpha=0.9, edgecolors="black", linewidth=0.3, zorder=3)
                _draw_boundaries_2d(ax, np.column_stack([y_instant[mask], z_instant[mask]]),
                                    states_instant[mask], unique_states, state_color)
            _style_axis(ax, "Y (m)", "Z (m)",
                        f"Coupe YZ @ x={x_center:.3f} m — t={time_index} — {self.label}",
                        xlim=bounds["y"], ylim=bounds["z"])
            ax.text(0.02, 0.02,
                    f"épaisseur = {slice_thickness*1000:.1f} mm\nn_particules = {mask.sum()}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
            if mask.sum() > 0:
                cb = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
                cb.set_label("État (ID partition)", fontweight="bold")
            plt.tight_layout()
            image_data[f"{save_prefix}_slice_yz_t{time_index}.png"] = _to_bytes(fig)
            plt.close(fig)
        
        if "slice_xz" in plot_types:
            y_center = (bounds["y"][0] + bounds["y"][1]) / 2
            mask = np.abs(y_instant - y_center) <= slice_thickness / 2
            fig, ax = plt.subplots(figsize=figsize_2d, facecolor="white")
            if mask.sum() > 0:
                sc = ax.scatter(x_instant[mask], z_instant[mask], s=sizes[mask],
                                c=states_instant[mask], cmap=cmap,
                                alpha=0.9, edgecolors="black", linewidth=0.3, zorder=3)
                _draw_boundaries_2d(ax, np.column_stack([x_instant[mask], z_instant[mask]]),
                                    states_instant[mask], unique_states, state_color)
            _style_axis(ax, "X (m)", "Z (m)",
                        f"Coupe XZ @ y={y_center:.3f} m — t={time_index} — {self.label}",
                        xlim=bounds["x"], ylim=bounds["z"])
            ax.text(0.02, 0.02,
                    f"épaisseur = {slice_thickness*1000:.1f} mm\nn_particules = {mask.sum()}",
                    transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
            if mask.sum() > 0:
                cb = fig.colorbar(sc, ax=ax, shrink=0.8, aspect=20, pad=0.02)
                cb.set_label("État (ID partition)", fontweight="bold")
            plt.tight_layout()
            image_data[f"{save_prefix}_slice_xz_t{time_index}.png"] = _to_bytes(fig)
            plt.close(fig)
        
        # ============================================================
        #  3. VUE 3D
        # ============================================================
        if "3d" in plot_types:
            fig = plt.figure(figsize=figsize_3d, facecolor="white")
            ax = fig.add_subplot(111, projection="3d")
            
            sc = ax.scatter(x_instant, y_instant, z_instant, s=sizes, c=states_instant, cmap=cmap,
                            alpha=0.85, edgecolors="black", linewidth=0.25, depthshade=True, zorder=3)
            
            _draw_boundaries_3d(ax, np.column_stack([x_instant, y_instant, z_instant]),
                                states_instant, unique_states, state_color)
            
            ax.set_xlabel("X (m)", fontweight="bold")
            ax.set_ylabel("Y (m)", fontweight="bold")
            ax.set_zlabel("Z (m)", fontweight="bold")
            ax.set_title(f"Vue 3D — t={time_index} — {self.label}", fontweight="bold", fontsize=13)
            ax.set_xlim(*bounds["x"])
            ax.set_ylim(*bounds["y"])
            ax.set_zlim(*bounds["z"])
            
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor("lightgray")
            ax.yaxis.pane.set_edgecolor("lightgray")
            ax.zaxis.pane.set_edgecolor("lightgray")
            ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.8)
            ax.view_init(elev=22, azim=-60)
            
            cb = fig.colorbar(sc, ax=ax, shrink=0.6, aspect=15, pad=0.08)
            cb.set_label("État (ID partition)", fontweight="bold")
            
            fig.text(0.5, 0.01,
                    f"Méthode: {self._splitting_method} | N_cells: {self.n_cells} | "
                    f"t={time_index} | N_particules: {len(x_instant)}",
                    ha="center", fontsize=8, color="dimgray", style="italic")
            
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            image_data[f"{save_prefix}_3d_t{time_index}.png"] = _to_bytes(fig)
            plt.close(fig)
        
        return image_data


    def diagnostics(self, coordinates,velocities=None):
            """
            Statistiques de population par cellule pour le partitionneur adaptatif.
            """
            coordinates = np.asarray(coordinates)
            x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
            if velocities is not None:
                vx,vy,vz=velocities[:,0],velocities[:,1],velocities[:,2]
                states = self.compute_states(x, y, z,vx,vy,vz) #type: ignore
            else: 
                states=self.compute_states(x,y,z)
            counts = np.bincount(states, minlength=self.n_cells)
            return {
                "pop_min": int(counts.min()),
                "pop_max": int(counts.max()),
                "pop_mean": float(counts.mean()),
                "pop_std": float(counts.std()),
                "n_empty": int((counts == 0).sum()),
                "n_visited": int((counts > 0).sum()),
                "fraction_visited": float((counts > 0).sum() / self.n_cells),
            }

    


# =============================================================================
# 1. CARTÉSIEN
# =============================================================================


class CartesianPartitioner(BasePartitioner):
    """
    Grille cartésienne régulière.

    Découpe le domaine en nx × ny × nz cellules de taille égale.
    Simple mais inadapté aux géométries cylindriques (coins vides).
    """

    def __init__(self:CartesianPartitioner, nx: int=5, ny: int=5, nz: int=5)->None:
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self._bounds:tuple = None #type:ignore
        self._splitting_method: str = "cartesian"

    @property
    def n_cells(self:CartesianPartitioner)-> int:
        return self.nx * self.ny * self.nz

    @property
    def label(self:CartesianPartitioner)-> str:
        return f"cartesian_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self:CartesianPartitioner, coordinates:np.ndarray)->CartesianPartitioner:
        eps = 0.001
        coordinates=np.asarray(coordinates) # contient les coordonnées [x,y,z] de toutes les particules
        mins = coordinates.min(axis=0) - eps # contient le minimum de [x,y,z]
        maxs = coordinates.max(axis=0) + eps # contient le maximum de [x,y,z]
        self._bounds = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]) # (min_x,max_x,min_y,max_y,min_z,max_z)
        return self

    def compute_states(self:CartesianPartitioner, x:np.ndarray, y:np.ndarray, z:np.ndarray,vx:np.ndarray=None,vy:np.ndarray=None,vz:np.ndarray=None)-> np.ndarray: #type:ignore
        """"Cette fonction permet de determiner l'état de la particule: la partition dans laquelle la particule reside."""
        # convertion des coordonnées en tableaux numpy
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds

        ix = np.clip(
            ((x - xmin) * self.nx / (xmax - xmin)).astype(np.int64), 0, self.nx - 1 # attribut une partition suivant l'axe des abcisses à chacune des particules
            # la fonction clip permet de normaliser la position de la particule dans l'ensemble des partitions
        )
        iy = np.clip(
            ((y - ymin) * self.ny / (ymax - ymin)).astype(np.int64), 0, self.ny - 1
        )
        iz = np.clip(
            ((z - zmin) * self.nz / (zmax - zmin)).astype(np.int64), 0, self.nz - 1
        )
        #n=int(len(x)/self.PARTICLE_NUMBER)
        self.states=ix + iy * self.nx + iz * self.nx * self.ny
        return self.states#[np.tile(self.species_labels,n)]

    def _save_data(self:CartesianPartitioner, path: str)->None:
        np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _load_data(self:CartesianPartitioner, path: str)->None:
        self._bounds = tuple(np.load(os.path.join(path, "bounds.npy")))

    def _get_cell_polygons_2d(self:CartesianPartitioner, view: str='xy')-> list:
        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds
        dx = (xmax - xmin) / self.nx
        dy = (ymax - ymin) / self.ny
        dz = (zmax - zmin) / self.nz
        results = []

        if view == 'xy':
            for iz in range(self.nz):
                for iy in range(self.ny):
                    for ix in range(self.nx):
                        state_id = ix + iy * self.nx + iz * self.nx * self.ny
                        x0 = xmin + ix * dx
                        y0 = ymin + iy * dy
                        x1 = x0 + dx
                        y1 = y0 + dy
                        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                        results.append((state_id, pts))
        elif view == 'yz':
            for iz in range(self.nz):
                for iy in range(self.ny):
                    for ix in range(self.nx):
                        state_id = ix + iy * self.nx + iz * self.nx * self.ny
                        y0 = ymin + iy * dy
                        z0 = zmin + iz * dz
                        y1 = y0 + dy
                        z1 = z0 + dz
                        pts = np.array([[y0, z0], [y1, z0], [y1, z1], [y0, z1]])
                        results.append((state_id, pts))
        return results

    def _get_cell_polyhedra_3d(self:CartesianPartitioner)-> list:
        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds
        dx = (xmax - xmin) / self.nx
        dy = (ymax - ymin) / self.ny
        dz = (zmax - zmin) / self.nz
        results = []

        face_indices = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5],  # right
        ]

        for iz in range(self.nz):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    state_id = ix + iy * self.nx + iz * self.nx * self.ny
                    x0 = xmin + ix * dx
                    y0 = ymin + iy * dy
                    z0 = zmin + iz * dz
                    x1 = x0 + dx
                    y1 = y0 + dy
                    z1 = z0 + dz
                    vertices = np.array([
                        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
                    ])
                    faces = face_indices
                    results.append((state_id, vertices, faces))
        return results


# =============================================================================
# 2. CYLINDRIQUE
# =============================================================================


class CylindricalPartitioner(BasePartitioner):
    """
    Grille cylindrique (r, θ, z).

    Idéal pour les mélangeurs à symétrie axiale.
    Deux modes radiaux:
      - "equal_dr"  : Δr constant
      - "equal_area": aire de section constante (recommandé)

    Avec ntheta=1 → partitionnement purement axisymétrique.
    """

    def __init__(self:CylindricalPartitioner, nr:int=5, ntheta:int=8, nz:int=5, radial_mode:str="equal_area")->None:
        super().__init__()
        self.nr = nr
        self.ntheta = ntheta
        self.nz = nz
        self.radial_mode = radial_mode
        self._x_center:float = None#type: ignore
        self._y_center:float = None#type: ignore
        self._r_max:float = None#type: ignore
        self._z_min:float = None#type: ignore
        self._z_max:float = None#type: ignore
        self._r_edges:np.ndarray = None#type: ignore
        self._splitting_method:str = "cylindrical"

    @property
    def n_cells(self:CylindricalPartitioner)-> int:
        return self.nr * self.ntheta * self.nz

    @property
    def label(self:CylindricalPartitioner)-> str:
        return (
            f"cylindrical_nr{self.nr}_nth{self.ntheta}"
            f"_nz{self.nz}_{self.radial_mode}"
        )

    def fit(self:CylindricalPartitioner, coordinates: np.ndarray)->CylindricalPartitioner:
        eps = 0.00
        coordinates=np.asarray(coordinates)
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

        # self._x_center = (x.min() + x.max()) / 2  # est un scalaire       # position en x du centre de la distribution des particules 
        # self._y_center = (y.min() + y.max()) / 2  # est un scalaire       # position en y du centre de la distribution des particules
        self._x_center = 0 # est un scalaire       # position en x du centre de la distribution des particules 
        self._y_center = 0 # est un scalaire       # position en y du centre de la distribution des particules

        r = np.sqrt((x - self._x_center) ** 2 + (y - self._y_center) ** 2) # rayon issue des positions recentrées des particules
        self._r_max = r.max() + eps
        self._z_min = z.min() - eps
        self._z_max = z.max() + eps

        if self.radial_mode == "equal_area":
            # aire π(r_{i+1}² - r_i²) = constante → r_i = R√(i/nr)
            self._r_edges = self._r_max * np.sqrt(np.linspace(0, 1, self.nr + 1)) # construction de la liste des Rayons pour respecter le fait que les surfaces soient identiques
        elif self.radial_mode == "equal_dr":
            self._r_edges = np.linspace(0, self._r_max, self.nr + 1)
        else:
            raise ValueError(f"radial_mode inconnu: {self.radial_mode}")

        return self

    def compute_states(self:CylindricalPartitioner, x:np.ndarray, y:np.ndarray, z:np.ndarray,vx:np.ndarray=None,vy:np.ndarray=None,vz:np.ndarray=None): #type: ignore
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        # analyzer=ar.MarkovAnalyzer()

        dx = x - self._x_center
        dy = y - self._y_center
        # convertit la position des particules du système de coordonnées cartésiens vers le système de coordonnées cylindriques
        r = np.sqrt(dx**2 + dy**2) 
        theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)  # [0, 2π]

        ir = np.clip(
            np.searchsorted(self._r_edges, r, side="right") - 1, # renvoit la liste d'indices  de la liste des partitions(selon le rayon) dans laquelle les rayons des particules ont été insérés 
            # le vecteur que renvoir la fonction  searchsorted est de dimension de r (nombres de particules)
            0, self.nr - 1  # les particules sont raménées dans l'intervalle des partitions suivant le rayon
        )
        itheta = np.clip(
            (theta * self.ntheta / (2 * np.pi)).astype(np.int64), 0, self.ntheta - 1 # le cylindre est partionné sur toute la circonference de sa base
            # chaque particule est placée dans une partition en fonction de son angle theta
        )
        dz = (self._z_max - self._z_min) / self.nz
        iz = np.clip(
            ((z - self._z_min) / dz).astype(np.int64), 0, self.nz - 1
        )
        #n=int(len(x)/self.PARTICLE_NUMBER)
        self.states=ir + itheta * self.nr + iz * self.nr * self.ntheta
        return  self.states#[np.tile(self.species_labels,n)] # la numérotation des partitons se fait partant des rayons, puis les angles et enfin les hauteurs z

    def _save_data(self:CylindricalPartitioner, path: str)->None:
        params = {
            "x_center": self._x_center,
            "y_center": self._y_center,
            "r_max": self._r_max,
            "z_min": self._z_min,
            "z_max": self._z_max,
        }
        with open(os.path.join(path, "cylindrical_params.json"), "w") as f:
            json.dump(params, f, indent=2)
        np.save(os.path.join(path, "r_edges.npy"), self._r_edges)

    def _load_data(self:CylindricalPartitioner, path: str)->None:
        with open(os.path.join(path, "cylindrical_params.json")) as f:
            p = json.load(f)
        self._x_center = p["x_center"]
        self._y_center = p["y_center"]
        self._r_max = p["r_max"]
        self._z_min = p["z_min"]
        self._z_max = p["z_max"]
        self._r_edges = np.load(os.path.join(path, "r_edges.npy"))

    def _arc_points(self:CylindricalPartitioner, r:float, theta_start:float, theta_end:float, n_segments:int=20)->np.ndarray:
        theta_vals = np.linspace(theta_start, theta_end, n_segments)
        return np.column_stack([r * np.cos(theta_vals), r * np.sin(theta_vals)])

    def _get_cell_polygons_2d(self:CylindricalPartitioner, view: str='xy')-> list:
        results = []
        if view == 'xy':
            for iz in range(self.nz):
                for itheta in range(self.ntheta):
                    for ir in range(self.nr):
                        state_id = ir + itheta * self.nr + iz * self.nr * self.ntheta
                        r0 = self._r_edges[ir]
                        r1 = self._r_edges[ir + 1]
                        t0 = itheta * 2 * np.pi / self.ntheta
                        t1 = (itheta + 1) * 2 * np.pi / self.ntheta
                        pts_inner = self._arc_points(r0, t1, t0, 10)
                        pts_outer = self._arc_points(r1, t0, t1, 10)
                        pts = np.vstack([pts_outer, pts_inner])
                        results.append((state_id, pts))
        elif view == 'yz':
            for iz in range(self.nz):
                for ir in range(self.nr):
                    state_id_base = ir + iz * self.nr * self.ntheta
                    r0 = self._r_edges[ir]
                    r1 = self._r_edges[ir + 1]
                    z0 = self._z_min + iz * (self._z_max - self._z_min) / self.nz
                    z1 = z0 + (self._z_max - self._z_min) / self.nz
                    for itheta in range(self.ntheta):
                        state_id = state_id_base + itheta * self.nr
                        y0 = -r1 if itheta >= self.ntheta // 2 else r0
                        y1 = r1
                        pts = np.array([[y0, z0], [y1, z0], [y1, z1], [y0, z1]])
                        results.append((state_id, pts))
        return results

    def _get_cell_polyhedra_3d(self:CylindricalPartitioner)-> list:
        results = []
        face_bottom = [0, 1, 2, 3]
        face_top = [4, 5, 6, 7]
        face_inner = [0, 3, 7, 4]
        face_outer = [1, 5, 6, 2]
        face_left = [0, 4, 5, 1]
        face_right = [3, 2, 6, 7]
        face_indices = [face_bottom, face_top, face_inner, face_outer, face_left, face_right]

        for iz in range(self.nz):
            z0 = self._z_min + iz * (self._z_max - self._z_min) / self.nz
            z1 = z0 + (self._z_max - self._z_min) / self.nz
            for itheta in range(self.ntheta):
                t0 = itheta * 2 * np.pi / self.ntheta
                t1 = (itheta + 1) * 2 * np.pi / self.ntheta
                for ir in range(self.nr):
                    state_id = ir + itheta * self.nr + iz * self.nr * self.ntheta
                    r0 = self._r_edges[ir]
                    r1 = self._r_edges[ir + 1]
                    cos_t0, sin_t0 = np.cos(t0), np.sin(t0)
                    cos_t1, sin_t1 = np.cos(t1), np.sin(t1)
                    vertices = np.array([
                        [r0*cos_t0, r0*sin_t0, z0],
                        [r1*cos_t0, r1*sin_t0, z0],
                        [r1*cos_t1, r1*sin_t1, z0],
                        [r0*cos_t1, r0*sin_t1, z0],
                        [r0*cos_t0, r0*sin_t0, z1],
                        [r1*cos_t0, r1*sin_t0, z1],
                        [r1*cos_t1, r1*sin_t1, z1],
                        [r0*cos_t1, r0*sin_t1, z1],
                    ])
                    results.append((state_id, vertices, face_indices))
        return results


# =============================================================================
# 3. VORONOÏ (K-MEANS)
# =============================================================================


class VoronoiPartitioner(BasePartitioner):
    """
    Partitionnement Voronoï par K-means.

    Chaque cellule = le bassin d'attraction du centroïde le plus proche.
    S'adapte naturellement à la densité de particules.

    C'est la méthode de référence en MCM (Fan et al., Doucet et al.).
    """

    def __init__(self:VoronoiPartitioner, n_cells:int=125, random_state: int=42)->None:
        super().__init__()
        self._n_cells = n_cells
        self.random_state = random_state
        self.centroids:np.ndarray = None #type: ignore
        self._tree:scipy.spacial.cKDTree = None #type: ignore
        self._splitting_method:str = "voronoi"

    @property
    def n_cells(self: VoronoiPartitioner)-> int:
        return self._n_cells

    @property
    def label(self: VoronoiPartitioner)-> str:
        return f"voronoi_{self._n_cells}cells"

    def fit(self: VoronoiPartitioner, coordinates: np.ndarray)->VoronoiPartitioner:
        coordinates=np.asarray(coordinates)
        from sklearn.cluster import MiniBatchKMeans
        from scipy.spatial import cKDTree # type:ignore

        rng = np.random.RandomState(self.random_state)
        if len(coordinates) > 500_000:
            idx = rng.choice(len(coordinates), 500_000, replace=False)
            fit_data = coordinates[idx]
        else:
            fit_data = coordinates
        kmeans = MiniBatchKMeans(
            n_clusters=self._n_cells,
            random_state=self.random_state, # me rassure que je commence avec les points au initiaux identiques
            batch_size=min(10_000, len(fit_data)),
            n_init=10, # j'initialise 10 fois pour être sur que les centres tombent bien au même endroit après l'exécution de l'algorithme ou du moins de façon proche
        )
        kmeans.fit(fit_data)
        self.centroids = kmeans.cluster_centers_
        self._tree = cKDTree(self.centroids)
        self._voronoi_3d = Voronoi(self.centroids)
        self._data_bounds_3d = (
            coordinates[:, 0].min(), coordinates[:, 0].max(),
            coordinates[:, 1].min(), coordinates[:, 1].max(),
            coordinates[:, 2].min(), coordinates[:, 2].max(),
        )
        return self

    def compute_states(self: VoronoiPartitioner, x: np.ndarray, y: np.ndarray, z: np.ndarray,vx: np.ndarray=None,vy: np.ndarray=None,vz: np.ndarray=None)->np.ndarray:#type: ignore
        coords = np.column_stack(
            [np.asarray(x), np.asarray(y), np.asarray(z)]
        )
        #n=int(len(x)/self.PARTICLE_NUMBER)

        _, indices = self._tree.query(coords)
        self.states=indices.astype(np.int64)
        return self.states#[np.tile(self.species_labels,n)]

    def _save_data(self: VoronoiPartitioner, path: str)->None:
        np.save(os.path.join(path, "centroids.npy"), self.centroids)

    def _load_data(self: VoronoiPartitioner, path: str)->None:
        from scipy.spatial import cKDTree # type:ignore

        self.centroids = np.load(os.path.join(path, "centroids.npy"))
        self._tree = cKDTree(self.centroids)
        self._n_cells = len(self.centroids)
        self._voronoi_3d = Voronoi(self.centroids)

    def _get_cell_polygons_2d(self: VoronoiPartitioner, view: str='xy')-> list:
        if view == 'xy':
            pts_2d = self.centroids[:, :2]
            x_idx, y_idx = 0, 1
        elif view == 'yz':
            pts_2d = self.centroids[:, 1:]
            x_idx, y_idx = 1, 2
        else:
            return []

        vor = Voronoi(pts_2d)
        results = []
        for state_id in range(self._n_cells):
            region_idx = vor.point_region[state_id]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            polygon_pts = vor.vertices[region]
            results.append((state_id, polygon_pts))
        return results

    def _get_cell_polyhedra_3d(self: VoronoiPartitioner)-> list:
        vor = self._voronoi_3d
        results = []
        for state_id in range(self._n_cells):
            region_idx = vor.point_region[state_id]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            vertices = vor.vertices[region]
            if len(vertices) < 4:
                continue
            hull = ConvexHull(vertices)
            faces = hull.simplices.tolist()
            results.append((state_id, vertices, faces))
        return results

    def diagnostics(self: VoronoiPartitioner, coordinates: np.ndarray,velocities: np.ndarray=None)-> dict: #type: ignore
            """
            Statistiques de population par cellule pour le partitionneur adaptatif.
            """
            coordinates = np.asarray(coordinates)
            x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
            states = self.compute_states(x, y, z)
            counts = np.bincount(states, minlength=self.n_cells)
            return {
                "pop_min": int(counts.min()),
                "pop_max": int(counts.max()),
                "pop_mean": float(counts.mean()),
                "pop_std": float(counts.std()),
                "n_empty": int((counts == 0).sum()),
                "n_visited": int((counts > 0).sum()),
                "fraction_visited": float((counts > 0).sum() / self.n_cells),
            }


# =============================================================================
# 4. GRILLE PAR QUANTILES (ÉQU-POPULATION)
# =============================================================================


class QuantileGridPartitioner(BasePartitioner):
    """
    Grille dont les bords sont des quantiles des données.
    plus il y aura une concentration de points en un endroit et plus la grille sera grande à cet endroit.

    Chaque cellule contient approximativement le même nombre de particules
    (équi-population marginale sur chaque axe).

    Meilleure homogénéité statistique que la grille cartésienne régulière.
    """

    def __init__(self: QuantileGridPartitioner, nx:int=5, ny:int=5, nz:int=5)->None:
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self._x_edges:np.ndarray = None # type: ignore
        self._y_edges:np.ndarray = None # type: ignore
        self._z_edges:np.ndarray = None # type: ignore
        self._splitting_method:str = "quantile"

    @property
    def n_cells(self: QuantileGridPartitioner)-> int:
        return self.nx * self.ny * self.nz

    @property
    def label(self: QuantileGridPartitioner)-> str:
        return f"quantile_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self: QuantileGridPartitioner, coordinates: np.ndarray)->QuantileGridPartitioner:
        coordinates=np.asarray(coordinates)
        eps = 0.001
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        # chaque edge est un vecteur de taille self.nx+1 ou self.ny+1 ou self.nz+1 dont chaque indice correspond à la valeur de x correspondant au quantile donné
        self._x_edges = np.quantile(x, np.linspace(0, 1, self.nx + 1))
        self._y_edges = np.quantile(y, np.linspace(0, 1, self.ny + 1))
        self._z_edges = np.quantile(z, np.linspace(0, 1, self.nz + 1))

        # Élargir les bords extrêmes
        self._x_edges[0] -= eps
        self._x_edges[-1] += eps
        self._y_edges[0] -= eps 
        self._y_edges[-1] += eps
        self._z_edges[0] -= eps
        self._z_edges[-1] += eps

        return self

    def compute_states(self:QuantileGridPartitioner, x: np.ndarray, y: np.ndarray, z: np.ndarray,vx: np.ndarray=None,vy: np.ndarray=None,vz: np.ndarray=None)->np.ndarray: # type: ignore
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        ix = np.clip(
            np.searchsorted(self._x_edges, x, side="right") - 1, 0, self.nx - 1 #type: ignore
        )
        iy = np.clip(
            np.searchsorted(self._y_edges, y, side="right") - 1, 0, self.ny - 1 #type: ignore
        )
        iz = np.clip(
            np.searchsorted(self._z_edges, z, side="right") - 1, 0, self.nz - 1 #type: ignore
        )
        #n=int(len(x)/self.PARTICLE_NUMBER)
        self.states= ix + iy * self.nx + iz * self.nx * self.ny
        return self.states#[np.tile(self.species_labels,n)]

    def _save_data(self: QuantileGridPartitioner, path: str)->None:
        np.savez(
            os.path.join(path, "edges.npz"),
            x=self._x_edges,
            y=self._y_edges,
            z=self._z_edges,
        )

    def _load_data(self: QuantileGridPartitioner, path: str)->None:
        data = np.load(os.path.join(path, "edges.npz"))
        self._x_edges = data["x"]
        self._y_edges = data["y"]
        self._z_edges = data["z"]

    def _get_cell_polygons_2d(self: QuantileGridPartitioner, view:str='xy')-> list:
        results = []
        if view == 'xy':
            for iz in range(self.nz):
                for iy in range(self.ny):
                    for ix in range(self.nx):
                        state_id = ix + iy * self.nx + iz * self.nx * self.ny
                        x0, x1 = self._x_edges[ix], self._x_edges[ix + 1]
                        y0, y1 = self._y_edges[iy], self._y_edges[iy + 1]
                        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
                        results.append((state_id, pts))
        elif view == 'yz':
            for iz in range(self.nz):
                for iy in range(self.ny):
                    for ix in range(self.nx):
                        state_id = ix + iy * self.nx + iz * self.nx * self.ny
                        y0, y1 = self._y_edges[iy], self._y_edges[iy + 1]
                        z0, z1 = self._z_edges[iz], self._z_edges[iz + 1]
                        pts = np.array([[y0, z0], [y1, z0], [y1, z1], [y0, z1]])
                        results.append((state_id, pts))
        return results

    def _get_cell_polyhedra_3d(self: QuantileGridPartitioner)-> list:
        results = []
        face_indices = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        for iz in range(self.nz):
            for iy in range(self.ny):
                for ix in range(self.nx):
                    state_id = ix + iy * self.nx + iz * self.nx * self.ny
                    x0, x1 = self._x_edges[ix], self._x_edges[ix + 1]
                    y0, y1 = self._y_edges[iy], self._y_edges[iy + 1]
                    z0, z1 = self._z_edges[iz], self._z_edges[iz + 1]
                    vertices = np.array([
                        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
                    ])
                    results.append((state_id, vertices, face_indices))
        return results


# =============================================================================
# 5. OCTREE ADAPTATIF
# =============================================================================


class OctreePartitioner(BasePartitioner):
    """
    Octree adaptatif avec coupes axiales ou obliques.

    Principe général :
      Subdivise récursivement l'espace 3D en cellules de plus en plus fines
      dans les zones denses. Chaque cellule contenant plus de `max_particles`
      particules est coupée en deux par un plan (axial ou oblique), jusqu'à
      `max_depth` niveaux de profondeur.

    Deux modes de fonctionnement :
      1. **Axial** (`oblique_method=None` ou `"axis"`) :
         Découpage classique en 8 octants alignés sur les axes x, y, z
         selon les médianes des coordonnées. Construit un arbre 8-aire.

      2. **Oblique** (`oblique_method` parmi "pca", "kmeans2", "2medians",
         "random", "svm") :
         Découpage binaire avec un plan de coupe orienté selon la géométrie
         locale des particules. Construit un arbre binaire.

    Méthodes de coupe oblique :
      - "pca"      : Plan orthogonal à la direction de variance maximale
                     (analyse en composantes principales).
      - "kmeans2"  : Plan médiateur entre 2 centroïdes obtenus par k-means.
      - "2medians" : Plan médiateur entre 2 médianes de cluster
                     (variante robuste du k-means, insensible aux outliers).
      - "random"   : Plan de direction aléatoire (uniforme sur la sphère),
                     coupure à la médiane projective.
      - "svm"      : Plan de marge maximale calculé par SVM linéaire,
                     avec étiquettes binaires issues de PCA.

    Avantage : raffine automatiquement les zones denses.
    Inconvénient : nombre de cellules non contrôlé a priori.
    """

    def __init__(self: OctreePartitioner, max_particles=100, max_depth=5, transform_type=0, oblique_method: str=None)->None: #type: ignore
        """
        Parameters
        ----------
        max_particles : int
            Seuil de remplissage : une cellule avec moins de `max_particles`
            particules n'est plus subdivisée (devient une feuille).
        max_depth : int
            Profondeur maximale de récursion. Limite le nombre total de
            subdivisions, même dans les zones très denses.
        transform_type : int ou str
            Type de normalisation appliqué aux coordonnées avant découpage.
            Par exemple "normalize" pour mettre à l'échelle [0,1].
        oblique_method : str ou None
            Méthode de calcul du plan de coupe. None ou "axis" → coupes
            axiales (octree classique). Sinon, choisir parmi "pca",
            "kmeans2", "2medians", "random", "svm".
        """
        super().__init__()
        self.max_particles = max_particles
        self.max_depth = max_depth
        self.transform_type = transform_type
        self.oblique_method = oblique_method
        self._leaves = []          # feuilles axiales : liste de tuples (xmin,xmax,ymin,ymax,zmin,zmax)
        self._bounds = None        # bounding box globale (xmin,xmax,ymin,ymax,zmin,zmax)
        self._stats = {}           # statistiques (min, max) pour la normalisation
        self._oblique_root = None  # racine de l'arbre binaire oblique (dict)
        om = oblique_method or "axis"
        self._splitting_method = f"octree_{om}"

    @property
    def n_cells(self: OctreePartitioner)-> int:
        """Nombre de feuilles (cellules) dans l'arbre oblique ou axial."""
        if self._oblique_root is not None:
            return self._count_tree_leaves(self._oblique_root)
        return len(self._leaves) if self._leaves else 0

    @property
    def label(self:OctreePartitioner)-> str:
        """Étiquette descriptive incluant les hyperparamètres et la méthode de coupe."""
        om = self.oblique_method or "axis"
        return f"octree_mp{self.max_particles}_md{self.max_depth}_{om}"

    # ── Helpers arbre oblique ──────────────────────────────────────────

    def _count_tree_leaves(self: OctreePartitioner, node: dict)-> int:
        """Parcourt récursivement l'arbre binaire oblique et compte les feuilles."""
        if node["type"] == "leaf":
            return 1
        return self._count_tree_leaves(node["left"]) + self._count_tree_leaves(node["right"])

    def _flatten_tree(self: OctreePartitioner, node: dict)->list:
        """Concatène récursivement toutes les feuilles de l'arbre oblique en une liste plate."""
        if node["type"] == "leaf":
            return [node]
        return self._flatten_tree(node["left"]) + self._flatten_tree(node["right"])

    # ── Transformation ─────────────────────────────────────────────────

    def _apply_transform(self: OctreePartitioner, coords:np.ndarray)->np.ndarray:
        """Normalisation éventuelle des coordonnées avant découpage."""
        if self.transform_type == 'normalize':
            return (coords - self._stats["min"]) / (self._stats["max"] - self._stats["min"])
        return coords

    # ── Fit ────────────────────────────────────────────────────────────

    def fit(self: OctreePartitioner, coordinates: np.ndarray)-> OctreePartitioner:
        """
        Construit l'arbre de partitionnement à partir des coordonnées 3D.

        Étapes :
          1. Calcule les stats (min, max) pour la normalisation éventuelle.
          2. Transforme les coordonnées si demandé.
          3. Définit la bounding box globale (avec une marge eps).
          4. Selon le mode :
             - Oblique : construit un arbre binaire via _build_oblique_tree.
             - Axial   : construit un octree 8-aire via _subdivide.
        """
        coordinates = np.asarray(coordinates)
        eps = 0.001
        self._stats["min"] = coordinates.min(axis=0) - eps
        self._stats["max"] = coordinates.max(axis=0) + eps
        if self.transform_type is not None:
            transformed_coords = self._apply_transform(coordinates)
        else:
            transformed_coords = coordinates
        # Bounding box globale (xmin, xmax, ymin, ymax, zmin, zmax)
        self._bounds = (
            transformed_coords[:, 0].min() - eps,
            transformed_coords[:, 0].max() + eps,
            transformed_coords[:, 1].min() - eps,
            transformed_coords[:, 1].max() + eps,
            transformed_coords[:, 2].min() - eps,
            transformed_coords[:, 2].max() + eps,
        )

        if self.oblique_method not in (None, "axis"):
            # Mode oblique : arbre binaire avec plans de coupe orientés
            self._oblique_root = self._build_oblique_tree(
                transformed_coords, self._bounds, depth=0,
                halfspaces_sofar=[]
            )
            self._leaves = []
        else:
            # Mode axial : octree classique par médianes x, y, z
            self._leaves = []
            self._oblique_root = None
            self._subdivide(transformed_coords, self._bounds, depth=0)
        return self

    # ── Subdivision axiale (octree classique) ──────────────────────────

    def _subdivide(self: OctreePartitioner, coords: np.ndarray, bounds:tuple, depth:int)->None:
        """
        Subdivision axiale récursive : coupe chaque cellule en 8 octants
        selon les médianes des coordonnées x, y, z.

        Principe :
          1. Si le nombre de particules <= max_particles ou profondeur max
             atteinte → stocker comme feuille et arrêter.
          2. Sinon, calculer la médiane de x, y, z.
          3. Créer un code d'octant binaire (bit 0=x, bit 1=y, bit 2=z).
          4. Pour chacun des 8 octants, calculer sa bounding box et
             subdiviser récursivement les particules qu'il contient.

        Paramètres
        ----------
        coords : ndarray (N, 3)
            Coordonnées des particules dans cette cellule.
        bounds : tuple (xmin, xmax, ymin, ymax, zmin, zmax)
            Bounding box de la cellule courante.
        depth : int
            Profondeur actuelle dans l'arbre.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        n_in = len(coords)

        # Condition d'arrêt : assez peu de particules ou profondeur max
        if n_in <= self.max_particles or depth >= self.max_depth:
            self._leaves.append(bounds)
            return

        # Médianes selon chaque axe
        xmid = np.median(coords[:, 0])
        ymid = np.median(coords[:, 1])
        zmid = np.median(coords[:, 2])

        # Encodage binaire de l'octant :
        #   bit 0 (poids 1) = côté x (0: gauche, 1: droite)
        #   bit 1 (poids 2) = côté y (0: bas,   1: haut)
        #   bit 2 (poids 4) = côté z (0: avant, 1: arrière)
        octant = (
            (coords[:, 0] >= xmid).astype(np.int64)
            + (coords[:, 1] >= ymid).astype(np.int64) * 2
            + (coords[:, 2] >= zmid).astype(np.int64) * 4
        )

        for idx in range(8):
            # Décode les bits en coordonnées de la bounding box enfant
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

    # ── Plan de coupe oblique ──────────────────────────────────────────

    def _find_splitting_plane(self: OctreePartitioner, coords: np.ndarray, method: str)-> tuple:
        """
        Calcule un plan de coupe (normale, offset) pour séparer les
        particules en deux groupes, selon la méthode spécifiée.

        Le plan est défini par :  normale · x = offset
        - Les points avec normale · x <= offset vont à gauche.
        - Les points avec normale · x >  offset vont à droite.

        Paramètres
        ----------
        coords : ndarray (N, 3)
            Positions des particules dans la cellule courante.
        method : str
            Méthode de coupe ("pca", "kmeans2", "2medians", "random", "svm").

        Retour
        ------
        normal : ndarray (3,)
            Vecteur normal unitaire au plan de coupe.
        offset : float
            Décalage (seuil) du plan.
        """
        if len(coords) < 2:
            # Moins de 2 points : plan par défaut (x=0)
            return np.array([1.0, 0.0, 0.0]), 0.0

        if method == "pca":
            # PCA : plan orthogonal à la direction de variance maximale.
            # 1. Matrice de covariance 3×3 des positions.
            # 2. Décomposition en valeurs/vecteurs propres (eigh = hermitien).
            # 3. Le vecteur propre de plus grande valeur propre = direction
            #    de plus grande dispersion.
            # 4. On coupe à la médiane des projections sur cette normale.
            cov = np.cov(coords, rowvar=False) # car coords est une matrice (N,3) 3: represente les variables donc les colonnes
            eigvals, eigvecs = np.linalg.eigh(cov)
            normal = eigvecs[:, np.argmax(eigvals)] # la normale est la direction de valeur propre maximale
            proj = coords @ normal # produit scalaire des coordonnées sur la direction normale du plan de projection
            offset = np.median(proj)#le offset est la médiane des projections des particules sur le plan 

        elif method == "kmeans2":
            # k-means à 2 clusters : plan médiateur des centroïdes.
            # 1. Clustering k-means des positions en 2 groupes.
            # 2. La normale du plan = vecteur reliant les 2 centroïdes.
            # 3. L'offset = projection du point milieu sur cette normale.
            # Échantillonnage à 10k points max pour limiter le coût.
            from sklearn.cluster import KMeans
            n = min(len(coords), 10000)
            kmeans = KMeans(n_clusters=2, n_init=3, random_state=42).fit(coords[:n])
            c1, c2 = kmeans.cluster_centers_
            normal = c2 - c1
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
            offset = ((c1 + c2) / 2) @ normal

        elif method == "2medians":
            # 2-médianes : variante robuste du k-means.
            # Au lieu des moyennes (L2), on utilise les médianes (L1)
            # comme centres, ce qui est moins sensible aux outliers.
            # 1. Initialisation : 2 points aléatoires distincts.
            # 2. 5 itérations d'affectation-recalcul :
            #    a. Distance L2 aux 2 centres → labels.
            #    b. Chaque centre = médiane des points de son cluster.
            # 3. Plan médiateur des 2 centres finaux (comme kmeans2).
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
            # Plan aléatoire : direction uniforme sur la sphère unité.
            # 1. Tirer un vecteur gaussien 3D et le normaliser.
            # 2. Projeter les points sur cette direction.
            # 3. Offset = médiane des projections (sépare en 2 parts égales).
            rng = np.random.RandomState(4)
            normal = rng.randn(3)
            normal /= np.linalg.norm(normal)
            proj = coords @ normal
            offset = np.median(proj)

        elif method == "svm":
            # SVM linéaire : plan de marge maximale.
            # 1. PCA pour générer des étiquettes binaires :
            #    - Projeter sur la direction de variance max.
            #    - Classe 1 si au-dessus de la médiane PCA, classe 0 sinon.
            # 2. Entraîner un SVM linéaire (LinearSVC) à séparer ces 2 classes.
            # 3. La normale = vecteur des coefficients SVM normalisé.
            # 4. L'offset = -intercepte / ||w||.
            # Note : les étiquettes PCA biaisent le SVM vers la direction
            # de variance max, mais le SVM optimise la marge localement.
            from sklearn.svm import LinearSVC
            cov = np.cov(coords, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            pca_normal = eigvecs[:, np.argmax(eigvals)]
            proj_pca = coords @ pca_normal
            labels = (proj_pca > np.median(proj_pca)).astype(int)
            if len(np.unique(labels)) < 2:
                return np.array([1.0, 0.0, 0.0]), 0.0
            svm = LinearSVC(max_iter=1000, C=1.0, dual='auto', random_state=42) #type:ignore
            svm.fit(coords, labels)
            w = svm.coef_[0].astype(np.float64)
            wn = np.linalg.norm(w)
            if wn < 1e-12:
                normal, offset = np.array([1.0, 0.0, 0.0]), 0.0
            else:
                normal = w / wn
                offset = -svm.intercept_[0].item() / wn

        else:
            raise ValueError(f"Méthode oblique inconnue: {method}")

        return normal, offset

    # ── Construction arbre oblique ─────────────────────────────────────

    def _build_oblique_tree(self, coords, bounds, depth, halfspaces_sofar):
        """
        Construit récursivement un arbre binaire oblique.

        Principe :
          À chaque nœud, on calcule un plan de coupe via
        `_find_splitting_plane`, on sépare les particules en deux groupes
        (gauche : proj <= offset, droite : proj > offset), et on
        subdivise récursivement chaque côté.

        Structure d'un nœud interne :
          {"type": "internal", "normal": vec3, "offset": float,
           "left": node, "right": node}

        Structure d'une feuille :
          {"type": "leaf", "bounds": (xmin,...,zmax),
           "centroid": vec3, "halfspaces": list}

        La liste `halfspaces_sofar` accumule l'historique des
        demi-espaces depuis la racine jusqu'à la feuille. Chaque
        entrée est un tuple (normal, offset, sens) où sens est "le"
        (<=) ou "gt" (>). Cela permet d'assigner un état à une
        particule en évaluant tous les plans de coupe le long du chemin.

        Paramètres
        ----------
        coords : ndarray (N, 3)
            Particules dans la cellule courante.
        bounds : tuple (xmin, xmax, ymin, ymax, zmin, zmax)
            Bounding box (utile pour la visualisation).
        depth : int
            Profondeur actuelle.
        halfspaces_sofar : list
            Demi-espaces déjà traversés depuis la racine.
        """
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        n_in = len(coords)

        # Condition d'arrêt : feuille
        if n_in <= self.max_particles or depth >= self.max_depth:
            center = coords.mean(axis=0) if len(coords) > 0 else np.zeros(3)
            return {
                "type": "leaf",
                "bounds": (xmin, xmax, ymin, ymax, zmin, zmax),
                "centroid": center,
                "halfspaces": list(halfspaces_sofar),
            }

        # Calcul du plan de coupe
        normal, offset = self._find_splitting_plane(coords, self.oblique_method)
        proj = coords @ normal
        left_mask = (proj <= offset)
        right_mask =  (proj > offset)
        left_coords = coords[left_mask]
        right_coords = coords[right_mask]

        # Sécurité : si le plan ne sépare rien (tous les points du même
        # côté), on force une feuille pour éviter une boucle infinie.
        if len(left_coords) == 0 or len(right_coords) == 0:
            center = coords.mean(axis=0) if len(coords) > 0 else np.zeros(3)
            return {
                "type": "leaf",
                "bounds": (xmin, xmax, ymin, ymax, zmin, zmax),
                "centroid": center,
                "halfspaces": list(halfspaces_sofar),
            }

        # Propagation des demi-espaces dans chaque branche
        hs_left = list(halfspaces_sofar) + [(normal, offset, "le")]
        hs_right = list(halfspaces_sofar) + [(normal, offset, "gt")]

        return {
            "type": "internal",
            "normal": normal,
            "offset": offset,
            "left": self._build_oblique_tree(left_coords, bounds, depth + 1, hs_left),
            "right": self._build_oblique_tree(right_coords, bounds, depth + 1, hs_right),
        }

    # ── Compute States ─────────────────────────────────────────────────

    def _assign_state_by_halfspaces(self, coords, leaves):
        """
        Assigne un état (cell_id) à chaque particule en évaluant les
        demi-espaces accumulés le long du chemin dans l'arbre oblique.

        Pour chaque feuille, on applique séquentiellement tous les
        tests (normal·x <= offset) ou (normal·x > offset) pour
        déterminer quelles particules appartiennent à cette cellule.

        Fallback : si une particule n'est dans aucune feuille (cas
        pathologique dû à des erreurs numériques), on lui assigne
        la feuille la plus proche via cKDTree.
        """
        n = len(coords)
        states = np.full(n, -1, dtype=np.int64)
        for cell_id, leaf in enumerate(leaves):
            mask = np.ones(n, dtype=bool)
            for normal, offset, side in leaf["halfspaces"]:
                proj = coords @ normal
                if side == "le":
                    mask &= proj <= offset
                else:
                    mask &= proj > offset
            states[mask] = cell_id
        unassigned = states == -1
        if unassigned.any():
            centers = np.array([l["centroid"] for l in leaves])
            from scipy.spatial import cKDTree # type:ignore
            tree = cKDTree(centers)
            _, idx = tree.query(coords[unassigned])
            states[unassigned] = idx
        return states

    def _compute_states_oblique_inlined(self: OctreePartitioner, coords: np.ndarray)->np.ndarray:
        """
        Version inline de l'assignation oblique (sans appeler
        _assign_state_by_halfspaces). Même logique.
        Conservée comme alternative — actuellement non utilisée
        car la version avec `compute_states` ci-dessous prime.
        """
        leaves = self._flatten_tree(self._oblique_root) #type: ignore
        n = len(coords)
        states = np.full(n, -1, dtype=np.int64)
        for cell_id, leaf in enumerate(leaves):
            mask = np.ones(n, dtype=bool)
            for normal, offset, side in leaf["halfspaces"]:
                proj = coords @ normal
                if side == "le":
                    mask &= proj <= offset
                else:
                    mask &= proj > offset
            states[mask] = cell_id
        unassigned = states == -1
        if unassigned.any():
            centers = np.array([l["centroid"] for l in leaves])
            from scipy.spatial import cKDTree #type: ignore # type:ignore
            tree = cKDTree(centers)
            _, idx = tree.query(coords[unassigned])
            states[unassigned] = idx
        return states

    # ─── ATTENTION : Duplication de compute_states ─────────────────────
    # Les deux définitions ci-dessous sont en conflit : la seconde
    # (ligne ~1616) écrase la première (ligne ~1552).
    # La seconde version utilise une logique inline au lieu d'appeler
    # _assign_state_by_halfspaces. À nettoyer : garder une seule version.
    # ───────────────────────────────────────────────────────────────────

    def compute_states(self: OctreePartitioner, x: np.ndarray, y: np.ndarray, z: np.ndarray)->np.ndarray: #type: ignore
        """
        Assigne un état (indice de cellule) à chaque particule.

        Mode oblique : on aplatit l'arbre binaire en feuilles, puis
        on évalue les demi-espaces de chaque feuille pour chaque
        particule (approche "force brute" : O(N × n_leaves × depth)).

        Mode axial : pour chaque cellule feuille, on teste si la
        particule est dans sa bounding box (xmin ≤ x < xmax, etc.).
        Fallback cKDTree pour les particules non assignées.

        NOTE : Cette méthode est immédiatement écrasée par la seconde
        définition de compute_states plus bas !
        """
        coords = np.column_stack([
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
        ])
        n = len(coords)

        if self._oblique_root is not None:
            leaves = self._flatten_tree(self._oblique_root)
            states = self._assign_state_by_halfspaces(coords, leaves)
        else:
            states = np.full(n, -1, dtype=np.int64)
            for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
                mask = (
                    (coords[:, 0] >= xmin) & (coords[:, 0] < xmax)
                    & (coords[:, 1] >= ymin) & (coords[:, 1] < ymax)
                    & (coords[:, 2] >= zmin) & (coords[:, 2] < zmax)
                )
                states[mask] = cell_id
            unassigned = states == -1
            if unassigned.any():
                from scipy.spatial import cKDTree #type: ignore # type:ignore
                centers = np.array([
                    ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
                    for b in self._leaves
                ])
                tree = cKDTree(centers)
                _, idx = tree.query(coords[unassigned])
                states[unassigned] = idx

        self.states = states
        return self.states

    # ── Méthodes obsolètes / à nettoyer ────────────────────────────────
    # Les méthodes _traverse_tree et _assign_cell_ids ci-dessous sont
    # des tentatives de parcours plus efficaces (traverser l'arbre au
    # lieu d'évaluer toutes les feuilles) mais ne sont pas utilisées
    # actuellement. _assign_cell_ids est clairement inachevée.

    def _traverse_tree(self: OctreePartitioner, coords: np.ndarray, node: dict, states: np.ndarray, mask:np.ndarray)->None:
        """
        Parcourt récursivement l'arbre oblique en suivant le plan de
        coupe à chaque nœud. Évite d'évaluer toutes les feuilles.
        [NON UTILISÉ — approche plus efficace mais non intégrée]
        """
        if node["type"] == "leaf":
            return
        proj = coords @ node["normal"]
        left_mask = mask & (proj <= node["offset"])
        right_mask = mask & (proj > node["offset"])
        if left_mask.any():
            self._traverse_tree(coords, node["left"], states, left_mask)
        if right_mask.any():
            self._traverse_tree(coords, node["right"], states, right_mask)

    def _assign_cell_ids(self, states):
        """
        [NON UTILISÉ — code mort, inachevé]
        Tentative d'assigner les cell_ids aux feuilles après
        une première traversée.
        """
        leaves = self._flatten_tree(self._oblique_root) #type: ignore
        for cell_id, leaf in enumerate(leaves):
            leaf["cell_id"] = cell_id

    # ── Deuxième définition (écrase la première) ───────────────────────
    # Même logique que la première compute_states, mais la partie
    # oblique est inlinée (répète le code de _assign_state_by_halfspaces).
    # C'est cette version qui est effective à l'exécution.
    # TODO: Supprimer la redondance et garder une seule méthode propre.

    def compute_states(self: OctreePartitioner, x: np.ndarray, y: np.ndarray, z: np.ndarray,vx:np.ndarray=None,vy:np.ndarray=None,vz:np.ndarray=None)->np.ndarray: #type: ignore
        coords = np.column_stack([
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
        ])
        n = len(coords)

        if self._oblique_root is not None:
            leaves = self._flatten_tree(self._oblique_root)
            states = np.full(n, -1, dtype=np.int64)
            for cell_id, leaf in enumerate(leaves):
                mask = np.ones(n, dtype=bool)
                for normal, offset, side in leaf["halfspaces"]:
                    proj = coords @ normal
                    if side == "le":
                        mask &= proj <= offset
                    else:
                        mask &= proj > offset
                states[mask] = cell_id
            unassigned = states == -1
            if unassigned.any():
                centers = np.array([l["centroid"] for l in leaves])
                from scipy.spatial import cKDTree # type:ignore
                tree = cKDTree(centers)
                _, idx = tree.query(coords[unassigned])
                states[unassigned] = idx
        else:
            states = np.full(n, -1, dtype=np.int64)
            for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
                mask = (
                    (coords[:, 0] >= xmin) & (coords[:, 0] < xmax)
                    & (coords[:, 1] >= ymin) & (coords[:, 1] < ymax)
                    & (coords[:, 2] >= zmin) & (coords[:, 2] < zmax)
                )
                states[mask] = cell_id
            unassigned = states == -1
            if unassigned.any():
                from scipy.spatial import cKDTree # type:ignore
                centers = np.array([
                    ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
                    for b in self._leaves
                ])
                tree = cKDTree(centers)
                _, idx = tree.query(coords[unassigned])
                states[unassigned] = idx

        self.states = states
        return self.states

    # ── Save / Load ────────────────────────────────────────────────────

    def _save_data(self: OctreePartitioner, path: str)->None:
        """Sauvegarde l'arbre (oblique ou axial) sur disque.

        Mode oblique : pickle de l'arbre binaire + fichier texte
        indiquant le nom de la méthode.
        Mode axial : tableau numpy des feuilles (N×6).
        Toujours sauvegarder la bounding box globale.
        """
        if self._oblique_root is not None:
            self._save_oblique_tree(path, self._oblique_root)
            with open(os.path.join(path, "octree_mode.txt"), "w") as f:
                f.write("oblique\n")
                f.write(f"{self.oblique_method or 'axis'}\n")
        else:
            leaves_arr = np.array(self._leaves)
            np.save(os.path.join(path, "leaves.npy"), leaves_arr)
            with open(os.path.join(path, "octree_mode.txt"), "w") as f:
                f.write("axis\n")
        if self._bounds:
            np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _save_oblique_tree(self, path, node):
        """Sauvegarde l'arbre oblique complet via pickle."""
        import pickle as pk
        with open(os.path.join(path, "oblique_tree.pkl"), "wb") as f:
            pk.dump(self._oblique_root, f, protocol=pk.HIGHEST_PROTOCOL)

    def _load_data(self: OctreePartitioner, path: str)->None:
        """Charge un arbre préalablement sauvegardé.

        Détecte automatiquement le mode (oblique ou axial) via le
        fichier octree_mode.txt. En mode oblique, restore aussi
        le nom de la méthode (pca, svm, etc.).
        """
        mode_path = os.path.join(path, "octree_mode.txt")
        if os.path.exists(mode_path):
            with open(mode_path) as f:
                mode = f.readline().strip()
            if mode == "oblique":
                import pickle as pk
                with open(os.path.join(path, "oblique_tree.pkl"), "rb") as f:
                    self._oblique_root = pk.load(f)
                om_path = os.path.join(path, "octree_mode.txt")
                with open(om_path) as f:
                    lines = f.readlines()
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

    # ── Polygones 2D (pour visualisation) ──────────────────────────────

    def _get_cell_polygons_2d(self: OctreePartitioner, view: str='xy')-> list:
        """
        Retourne les polygones 2D des feuilles pour la visualisation.

        Pour chaque feuille (oblique ou axiale), on projette sa
        bounding box sur le plan demandé ('xy' ou 'yz') et on
        retourne un rectangle (4 sommets).

        Paramètres
        ----------
        view : str
            Plan de projection : 'xy' (défaut) ou 'yz'.

        Retour
        ------
        list of (cell_id, pts)
        """
        results = []
        if self._oblique_root is not None:
            leaves = self._flatten_tree(self._oblique_root)
            for cell_id, leaf in enumerate(leaves):
                xmin, xmax, ymin, ymax, zmin, zmax = leaf["bounds"]
                if view == 'xy':
                    pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                elif view == 'yz':
                    pts = np.array([[ymin, zmin], [ymax, zmin], [ymax, zmax], [ymin, zmax]])
                else:
                    continue
                results.append((cell_id, pts))
        else:
            for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
                if view == 'xy':
                    pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
                elif view == 'yz':
                    pts = np.array([[ymin, zmin], [ymax, zmin], [ymax, zmax], [ymin, zmax]])
                else:
                    continue
                results.append((cell_id, pts))
        return results

    # ── Polyèdres 3D (pour visualisation) ──────────────────────────────

    def _get_cell_polyhedra_3d(self: OctreePartitioner)-> list:
        """
        Retourne les polyèdres 3D des feuilles pour visualisation.

        Chaque feuille est représentée par un parallélépipède
        (8 sommets, 6 faces quadrilatères) défini par sa bounding
        box. Utile pour matplotlib 3D ou plotly.

        Retour
        ------
        list of (cell_id, vertices, face_indices)
          vertices : ndarray (8, 3) — les 8 coins du parallélépipède
          face_indices : list of 6 lists de 4 indices chacune
        """
        results = []
        face_indices = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        if self._oblique_root is not None:
            leaves = self._flatten_tree(self._oblique_root)
            for cell_id, leaf in enumerate(leaves):
                xmin, xmax, ymin, ymax, zmin, zmax = leaf["bounds"]
                vertices = np.array([
                    [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
                    [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
                ])
                results.append((cell_id, vertices, face_indices))
        else:
            for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
                vertices = np.array([
                    [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
                    [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
                ])
                results.append((cell_id, vertices, face_indices))
        return results

    # ── Visualisation depuis le bucket ─────────────────────────────────

    OBLIQUE_METHODS = ["axis", "pca", "kmeans2", "2medians", "random", "svm"]
    DEFAULT_METHOD = "pca"

    @classmethod
    def visualize_from_bucket(cls, method=DEFAULT_METHOD, particle_diameter=0.004,
                               save_prefix="oblique_from_bucket", plot_types=None):
        """
        Charge les résultats pré-calculés depuis le bucket HuggingFace
        et génère les visualisations (matrice + RSD).

        Args:
            method: méthode oblique parmi "axis", "pca", "kmeans2", "2medians", "random", "svm"
            particle_diameter: 0.004 (SMALL) ou 0.008 (BIG)
            save_prefix: préfixe pour les fichiers image
            plot_types: liste de types de plot ("matrix", "rsd", "comparison")

        Returns:
            dict d'images: {"matrix.png": bytes, "rsd.png": bytes, ...}
        """
        if method not in cls.OBLIQUE_METHODS:
            print(f"⚠️  Méthode '{method}' inconnue. Utilisation de '{cls.DEFAULT_METHOD}'")
            method = cls.DEFAULT_METHOD

        from bucket_io import load_experiment_from_bucket

        diam_str = str(particle_diameter).replace(".", "")
        folder = (
            f"octree_mp100_md3_{method}"
            f"_NLT10_step10_dt2_tau50_start250_d{diam_str}"
        )

        print(f"  Chargement depuis le bucket: {folder}")
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        try:
            data = load_experiment_from_bucket(folder)
        except Exception as e:
            print(f"❌ Aucune donnée trouvée pour '{method}' (diamètre={particle_diameter})")
            print(f"   Lancez d'abord: python runs/run_oblique_comparison.py --diameter {particle_diameter}")
            print(f"   ({e})")
            return {}

        if data is None or data.get("matrix") is None or data["matrix"].size == 0:
            print(f"❌ Aucune donnée trouvée pour '{method}' (diamètre={particle_diameter})")
            print(f"   Lancez d'abord: python runs/run_oblique_comparison.py --diameter {particle_diameter}")
            return {}

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import io

        P = np.asarray(data["matrix"], dtype=np.float64)
        stats = data.get("stats", {})
        config = data.get("config", {})
        image_data = {}

        if plot_types is None:
            plot_types = ["matrix", "rsd"]

        # ── Matrice de transition ──
        if "matrix" in plot_types:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(P, aspect='auto', cmap='viridis', interpolation='nearest')
            plt.colorbar(im, ax=ax, label='P(i→j)')
            ax.set_xlabel('État précédent (i)', fontsize=12)
            ax.set_ylabel('État suivant (j)', fontsize=12)
            ax.set_title(f'Matrice P — octree_{method} ({P.shape[0]} états)',
                        fontsize=14, fontweight='bold')
            info = f"Méthode de découpage : octree_{method}"
            fig.text(0.02, 0.01, info, fontsize=9, style='italic', alpha=0.7,
                     transform=fig.transFigure)
            plt.tight_layout(rect=(0, 0.03, 1, 1))
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image_data[f"{save_prefix}_matrix.png"] = buf.getvalue()
            plt.close()

        # ── RSD simulé ──
        if "rsd" in plot_types:
            n_steps = 200
            n_states = P.shape[0]
            conc = np.zeros(n_states)
            conc[:n_states // 2] = 1.0
            conc = conc / conc.sum()
            rsd = []
            for _ in range(n_steps):
                conc = P.T @ conc
                mean_c = conc.mean()
                std_c = conc.std()
                rsd.append(std_c / mean_c if mean_c > 0 else 0)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(rsd, linewidth=2, color='#1f77b4')
            ax.set_xlabel('Pas de temps', fontsize=12)
            ax.set_ylabel('RSD', fontsize=12)
            ax.set_title(f'RSD — octree_{method}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            info = f"Méthode de découpage : octree_{method}"
            fig.text(0.02, 0.01, info, fontsize=9, style='italic', alpha=0.7,
                     transform=fig.transFigure)
            plt.tight_layout(rect=(0, 0.03, 1, 1))
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image_data[f"{save_prefix}_rsd.png"] = buf.getvalue()
            plt.close()

        # ── Comparaison RSD toutes méthodes ──
        if "comparison" in plot_types:
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
            for i, om in enumerate(cls.OBLIQUE_METHODS):
                om_diam_str = str(particle_diameter).replace(".", "")
                om_folder = (
                    f"octree_mp100_md3_{om}"
                    f"_NLT10_step10_dt2_tau50_start250_d{om_diam_str}"
                )
                om_data = load_experiment_from_bucket(om_folder)
                if om_data is None or om_data.get("matrix") is None or om_data["matrix"].size == 0:
                    continue
                om_P = np.asarray(om_data["matrix"], dtype=np.float64)
                n_states = om_P.shape[0]
                conc = np.zeros(n_states)
                conc[:n_states // 2] = 1.0
                conc = conc / conc.sum()
                om_rsd = []
                for _ in range(n_steps): #type: ignore
                    conc = om_P.T @ conc
                    mean_c = conc.mean()
                    std_c = conc.std()
                    om_rsd.append(std_c / mean_c if mean_c > 0 else 0)
                ax.plot(om_rsd, color=colors[i % len(colors)], label=f"octree_{om}",
                        linewidth=2)
            ax.set_xlabel('Pas de temps', fontsize=12)
            ax.set_ylabel('RSD', fontsize=12)
            ax.set_title('Comparaison RSD — Méthodes obliques', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image_data[f"{save_prefix}_comparison.png"] = buf.getvalue()
            plt.close()

        print(f"  ✅ {len(image_data)} images générées depuis le bucket")
        return image_data


# =============================================================================
# 6. PHYSIQUE-AWARE (POSITION + VITESSE)
# =============================================================================


class PhysicsAwarePartitioner(BasePartitioner):
    """
    K-means sur des features physiques (position + vitesse optionnelle).

    Par défaut, fonctionne sur les positions normalisées (équivalent Voronoï).
    Si use_velocities=True, le clustering utilise aussi les composantes vx, vy, vz.

    Usage:
        part = PhysicsAwarePartitioner(n_cells=125, velocity_weight=0.5)
        part.fit(positions)                    # positions seules
        part.fit_with_physics(positions, velocities)  # positions + vitesses
    """

    def __init__(self, n_cells=125, velocity_weight=0.5, random_state=42):
        super().__init__()
        self._n_cells:int = n_cells
        self.velocity_weight:float = velocity_weight
        self.random_state:int  = random_state
        self._centroids:np.ndarray = None #type:ignore
        self._tree = None
        self._mean:np.ndarray = None #type: ignore
        self._std:np.ndarray = None #type: ignore
        self._n_features:int = 3
        self._splitting_method: str = "physics"
        self.use_velocity=False
        self.features: np.ndarray = None 

    @property
    def n_cells(self:PhysicsAwarePartitioner)-> int:
        return self._n_cells

    @property
    def label(self:PhysicsAwarePartitioner)-> str:
        suffix = "pos" if self.velocity_weight == 0 else "withvel"
        return f"physics_{self._n_cells}cells_{suffix}_vw{self.velocity_weight}"

    def fit(self, coordinates: np.ndarray, use_velocities: bool = None) -> PhysicsAwarePartitioner:
        use_velocities = self.use_velocity
        coordinates = np.asarray(coordinates)

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) == len(coordinates):
                # ✅ Module du vecteur vitesse : (N,3) → (N,1)
                speed = np.linalg.norm(vel, axis=1, keepdims=True)  # ‖v‖ = sqrt(vx²+vy²+vz²)
                self.features = np.hstack([coordinates, speed])  # (N, 4)
                self._n_features = 4
                return self._fit_internal(self.features, n_pos=3)
            else:
                print(f"⚠️ Mismatch velocities ({len(vel)}) vs coordinates ({len(coordinates)}), fallback positions only")

        self.features = coordinates
        return self._fit_internal(coordinates, n_pos=3)

    

    def _fit_internal(self: PhysicsAwarePartitioner, features: np.ndarray, n_pos: int = 3) -> PhysicsAwarePartitioner:
        from sklearn.cluster import MiniBatchKMeans
        from scipy.spatial import cKDTree

        self._n_features = features.shape[1]

        # Normalisation
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0

        X = (features - self._mean) / self._std

        # ✅ Créer un vecteur de poids explicite
        if X.shape[1] > n_pos:
            # Poids = [1, 1, 1, velocity_weight] pour (x, y, z, speed)
            weights = np.ones(X.shape[1])
            weights[n_pos:] = self.velocity_weight
            
            # Appliquer les poids element-wise
            X = X * weights[np.newaxis, :]  # Broadcasting sur toutes les samples

        # Sous-échantillonner
        rng = np.random.RandomState(self.random_state)
        if len(X) > 500_000:
            idx = rng.choice(len(X), 500_000, replace=False)
            X_fit = X[idx]
        else:
            X_fit = X

        kmeans = MiniBatchKMeans(
            n_clusters=self._n_cells,
            random_state=self.random_state,
            batch_size=min(10_000, len(X_fit)),
            n_init=10,
        )
        kmeans.fit(X_fit)
        self._centroids = kmeans.cluster_centers_
        self._tree = cKDTree(self._centroids)
        return self

    def compute_states(self, x, y, z, vx=None, vy=None, vz=None) -> np.ndarray:
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])

        if self._n_features == 4 and vx is not None and vy is not None and vz is not None:
            vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
            speed = np.linalg.norm(vel, axis=1, keepdims=True)  # (N,1)
            features = np.hstack([pos, speed])
        elif self._n_features == 4:
            # Modèle entraîné avec ‖v‖ mais pas fourni → padding zéro
            features = np.hstack([pos, np.zeros((len(pos), 1))])
        else:
            features = pos

        X = (features - self._mean) / self._std
        
        # ✅ Appliquer les mêmes poids qu'au fit
        if X.shape[1] > 3:
            weights = np.ones(X.shape[1])
            weights[3:] = self.velocity_weight
            X = X * weights[np.newaxis, :]
        
        _, indices = self._tree.query(X)

        self.states = indices.astype(np.int64)
        return self.states

    def _save_data(self:PhysicsAwarePartitioner, path:str)->None:
        np.save(os.path.join(path, "centroids.npy"), self._centroids)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "physics_params.json"), "w") as f:
            json.dump({"n_features": self._n_features}, f)

    def _load_data(self:PhysicsAwarePartitioner, path: str)->None:
        from scipy.spatial import cKDTree # type: ignore # type:ignore

        self._centroids = np.load(os.path.join(path, "centroids.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._centroids)
        self._n_cells = len(self._centroids)
        with open(os.path.join(path, "physics_params.json")) as f:
            self._n_features = json.load(f)["n_features"]

    def visualize(self:PhysicsAwarePartitioner, x: np.ndarray, y: np.ndarray, z: np.ndarray,vx: np.ndarray=None,vy: np.ndarray=None,vz: np.ndarray=None, plot_types: list=["3d", "2d_xy"], save_prefix: str="partition_visualization", # type: ignore
                  particle_diameters=None, use_diameter: bool=True, **kwargs: dict)-> dict:
        """
        Génère des visualisations - NE refitte PAS si déjà fitté.
        """
        
        
        if self._centroids is None:
            raise ValueError("Partitioner not fitted! Call fit_with_physics() first.")
        
        if np.array([vx,vy,vz]).all() is not None:
            states=self.compute_states(x,y,z,vx,vy,vz)
        else:
            states = self.compute_states(x, y, z)

        diameters = None
        if use_diameter:
            if particle_diameters is not None:
                diameters = np.asarray(particle_diameters)
            elif hasattr(self, 'particle_diameters') and self.particle_diameters is not None:
                diameters = np.asarray(self.particle_diameters)
            elif hasattr(self, 'dem_diameters') and self.dem_diameters is not None:
                if len(self.dem_diameters) == len(x):
                    diameters = np.asarray(self.dem_diameters)

        image_data = {}
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        zmin, zmax = z.min(), z.max()
        self._data_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)

        if diameters is not None and diameters.max() > 0:
            sizes = (diameters / diameters.max()) * 200 + 10
        else:
            sizes = 30

        if "2d_xy" in plot_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            info = f"Méthode de découpage : {self._splitting_method}"
            fig.text(0.02, 0.01, info, fontsize=9, style='italic', alpha=0.7,
                     transform=fig.transFigure)

            sc1 = ax1.scatter(x, y, c=states, cmap='tab20', s=sizes, alpha=0.7,
                              edgecolors='black', linewidth=0.3)
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax1.set_title(f'Vue XY - {self.label}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_aspect('equal', adjustable='box')
            plt.colorbar(sc1, ax=ax1, label='État', shrink=0.8)

            sc2 = ax2.scatter(y, z, c=states, cmap='tab20', s=sizes, alpha=0.7,
                              edgecolors='black', linewidth=0.3)
            ax2.set_xlim(ymin, ymax)
            ax2.set_ylim(zmin, zmax)
            ax2.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
            ax2.set_title(f'Vue YZ - {self.label}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_aspect('equal', adjustable='box')
            plt.colorbar(sc2, ax=ax2, label='État', shrink=0.8)

            plt.tight_layout(rect=(0, 0.03, 1, 1))
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image_data[f"{save_prefix}_2d.png"] = buf.getvalue()
            plt.close()

        if "3d" in plot_types:
            fig = plt.figure(figsize=(12, 10))

            info = f"Méthode de découpage : {self._splitting_method}"
            fig.text(0.02, 0.01, info, fontsize=9, style='italic', alpha=0.7,
                     transform=fig.transFigure)

            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x, y, z, c=states, cmap='tab20', s=sizes, alpha=0.7, # type: ignore
                           edgecolors='black', linewidth=0.3)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
            ax.set_xlabel('X (m)', fontsize=10)
            ax.set_ylabel('Y (m)', fontsize=10)
            ax.set_zlabel('Z (m)', fontsize=10)
            ax.set_title(f'Vue 3D - {self.label}', fontsize=14, fontweight='bold')
            plt.colorbar(sc, ax=ax, label='État', shrink=0.6)
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            image_data[f"{save_prefix}_3d.png"] = buf.getvalue()
            plt.close()

        return image_data

    def _get_cell_polygons_2d(self:PhysicsAwarePartitioner, view: str='xy')-> list:
        pos_centroids = self._centroids[:, :3]
        if view == 'xy':
            pts_2d = pos_centroids[:, :2]
        elif view == 'yz':
            pts_2d = pos_centroids[:, 1:]
        else:
            return []
        vor = Voronoi(pts_2d)# Voronoi de scipy.spatial
        results = []
        for state_id in range(self._n_cells):
            region_idx = vor.point_region[state_id]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            polygon_pts = vor.vertices[region]
            results.append((state_id, polygon_pts))
        return results

    def _get_cell_polyhedra_3d(self:PhysicsAwarePartitioner)-> list:
        pos_centroids = self._centroids[:, :3]
        vor = Voronoi(pos_centroids)
        results = []
        for state_id in range(self._n_cells):
            region_idx = vor.point_region[state_id]
            region = vor.regions[region_idx]
            if not region or -1 in region:
                continue
            vertices = vor.vertices[region]
            if len(vertices) < 4:
                continue
            hull = ConvexHull(vertices)
            faces = hull.simplices.tolist()
            results.append((state_id, vertices, faces))
        return results

import os
import json
import pickle
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

class FullVectorVelocityKMeansPartitioner(BasePartitioner):
    """
    K-Means utilisant le vecteur vitesse complet (vx, vy, vz) au lieu de la norme.
    Capture la directionnalité de l'écoulement (lié aux streamlines de Doucet 2008).
    """
    def __init__(self, n_cells=125, velocity_weight=0.5, random_state=42):
        super().__init__()
        self._n_cells: int = n_cells
        self.velocity_weight: float = velocity_weight
        self.random_state: int = random_state
        
        self._centroids: np.ndarray = None
        self._tree = None
        self._mean: np.ndarray = None
        self._std: np.ndarray = None
        self._n_features: int = 6  # x, y, z, vx, vy, vz
        self._splitting_method: str = "fullvel_kmeans"
        self.use_velocity: bool = True
        self.features: np.ndarray = None

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def label(self) -> str:
        return f"fullvel_kmeans_{self._n_cells}cells_vw{self.velocity_weight}"

    def fit(self, coordinates: np.ndarray, use_velocities: bool = None) -> 'FullVectorVelocityKMeansPartitioner':
        use_velocities = self.use_velocity if use_velocities is None else use_velocities
        coordinates = np.asarray(coordinates)

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) == len(coordinates):
                # ✅ Vecteur vitesse complet : (N,3)
                self.features = np.hstack([coordinates, vel])  # (N, 6)
                self._n_features = 6
                return self._fit_internal(self.features, n_pos=3)
            else:
                print(f"⚠️ Mismatch velocities ({len(vel)}) vs coordinates ({len(coordinates)}), fallback positions only")

        self.features = coordinates
        self._n_features = 3
        return self._fit_internal(coordinates, n_pos=3)

    def _fit_internal(self, features: np.ndarray, n_pos: int = 3) -> 'FullVectorVelocityKMeansPartitioner':
        self._n_features = features.shape[1]

        # Normalisation
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        X = (features - self._mean) / self._std

        # ✅ Création et application du vecteur de poids explicite
        if X.shape[1] > n_pos:
            weights = np.ones(X.shape[1])
            weights[n_pos:] = self.velocity_weight  # Poids sur vx, vy, vz
            X = X * weights[np.newaxis, :]

        # Sous-échantillonner
        rng = np.random.RandomState(self.random_state)
        X_fit = X[rng.choice(len(X), 500_000, replace=False)] if len(X) > 500_000 else X

        kmeans = MiniBatchKMeans(
            n_clusters=self._n_cells, random_state=self.random_state, 
            batch_size=min(10_000, len(X_fit)), n_init=10
        )
        kmeans.fit(X_fit)
        self._centroids = kmeans.cluster_centers_
        self._tree = cKDTree(self._centroids)
        return self

    def compute_states(self, x, y, z, vx=None, vy=None, vz=None) -> np.ndarray:
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])

        if self._n_features == 6 and vx is not None and vy is not None and vz is not None:
            vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
            features = np.hstack([pos, vel])
        elif self._n_features == 6:
            # Modèle entraîné avec vélocité mais pas fournie → padding zéro
            features = np.hstack([pos, np.zeros((len(pos), 3))])
        else:
            features = pos

        X = (features - self._mean) / self._std
        
        if X.shape[1] > 3:
            weights = np.ones(X.shape[1])
            weights[3:] = self.velocity_weight
            X = X * weights[np.newaxis, :]
        
        _, indices = self._tree.query(X)
        self.states = indices.astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "centroids.npy"), self._centroids)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump({"n_features": self._n_features, "n_cells": self._n_cells, "vw": self.velocity_weight}, f)

    def _load_data(self, path: str) -> None:
        self._centroids = np.load(os.path.join(path, "centroids.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._centroids)
        with open(os.path.join(path, "params.json")) as f:
            params = json.load(f)
            self._n_features = params["n_features"]


from sklearn.cluster import SpectralClustering

class SpectralClusteringPartitioner(BasePartitioner):
    """
    Spectral Clustering pour capturer les structures topologiques / connectivité de l'écoulement.
    Lié à l'analyse des modes collectifs et SVD de Tjakra 2013.
    """
    def __init__(self, n_cells=125, velocity_weight=0.5, n_neighbors=15, max_samples=5000, random_state=42):
        super().__init__()
        self._n_cells: int = n_cells
        self.velocity_weight: float = velocity_weight
        self.n_neighbors: int = n_neighbors
        self.max_samples: int = max_samples
        self.random_state: int = random_state
        
        self._support_data: np.ndarray = None
        self._support_labels: np.ndarray = None
        self._tree = None  # Sera un KDTree sur les points de support
        self._mean: np.ndarray = None
        self._std: np.ndarray = None
        self._n_features: int = 6
        self._splitting_method: str = "spectral"
        self.use_velocity: bool = True
        self.features: np.ndarray = None

    @property
    def n_cells(self) -> int:
        return self._n_cells

    @property
    def label(self) -> str:
        return f"spectral_{self._n_cells}cells_vw{self.velocity_weight}_k{self.n_neighbors}"

    def fit(self, coordinates: np.ndarray, use_velocities: bool = None) -> 'SpectralClusteringPartitioner':
        use_velocities = self.use_velocity if use_velocities is None else use_velocities
        coordinates = np.asarray(coordinates)

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) == len(coordinates):
                self.features = np.hstack([coordinates, vel])
                self._n_features = 6
                return self._fit_internal(self.features, n_pos=3)
        
        self.features = coordinates
        self._n_features = 3
        return self._fit_internal(coordinates, n_pos=3)

    def _fit_internal(self, features: np.ndarray, n_pos: int = 3) -> 'SpectralClusteringPartitioner':
        self._n_features = features.shape[1]

        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        X = (features - self._mean) / self._std

        if X.shape[1] > n_pos:
            weights = np.ones(X.shape[1])
            weights[n_pos:] = self.velocity_weight
            X = X * weights[np.newaxis, :]

        # Sous-échantillonnage pour le fit (Spectral est O(N^2) ou O(N^3))
        rng = np.random.RandomState(self.random_state)
        n_samples = min(self.max_samples, len(X))
        idx = rng.choice(len(X), n_samples, replace=False)
        X_sub = X[idx]

        spectral = SpectralClustering(
            n_clusters=self._n_cells, affinity='nearest_neighbors', 
            n_neighbors=self.n_neighbors, random_state=self.random_state, assign_labels='kmeans'
        )
        labels_sub = spectral.fit_predict(X_sub)

        # Sauvegarde des points de support pour l'inférence 1-NN
        self._support_data = X_sub
        self._support_labels = labels_sub
        self._tree = cKDTree(self._support_data)
        return self

    def compute_states(self, x, y, z, vx=None, vy=None, vz=None) -> np.ndarray:
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])

        if self._n_features == 6 and vx is not None and vy is not None and vz is not None:
            vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
            features = np.hstack([pos, vel])
        elif self._n_features == 6:
            features = np.hstack([pos, np.zeros((len(pos), 3))])
        else:
            features = pos

        X = (features - self._mean) / self._std
        if X.shape[1] > 3:
            weights = np.ones(X.shape[1])
            weights[3:] = self.velocity_weight
            X = X * weights[np.newaxis, :]
        
        # Inférence par proximité aux points du graphe spectral
        _, indices = self._tree.query(X)
        self.states = self._support_labels[indices].astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "support_data.npy"), self._support_data)
        np.save(os.path.join(path, "support_labels.npy"), self._support_labels)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "params.json"), "w") as f:
            json.dump({"n_features": self._n_features, "n_cells": self._n_cells, "vw": self.velocity_weight, "k": self.n_neighbors}, f)

    def _load_data(self, path: str) -> None:
        self._support_data = np.load(os.path.join(path, "support_data.npy"))
        self._support_labels = np.load(os.path.join(path, "support_labels.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._support_data)
        with open(os.path.join(path, "params.json")) as f:
            params = json.load(f)
            self._n_features = params["n_features"]


from sklearn.mixture import GaussianMixture
import numpy as np
import pickle
import os

class GaussianMixturePartitioner(BasePartitioner):
    """
    Gaussian Mixture Model (covariance_type='full').
    Optimisé avec sous-échantillonnage pour la vitesse.
    """
    def __init__(self, n_cells=125, velocity_weight=0.5, random_state=42, max_fit_samples=500_000):
        super().__init__()
        self._n_cells: int = n_cells
        self.velocity_weight: float = velocity_weight
        self.random_state: int = random_state
        self.max_fit_samples: int = max_fit_samples #  NOUVEAU : Limite les points pour le fit
        
        self._gmm = None
        self._mean: np.ndarray = None
        self._std: np.ndarray = None
        self._n_features: int = 6
        self._splitting_method: str = "gmm_full"
        self.use_velocity: bool = True
        self.features: np.ndarray = None

    @property
    def n_cells(self) -> int: return self._n_cells

    @property
    def label(self) -> str:
        return f"gmm_full_{self._n_cells}cells_vw{self.velocity_weight}"

    def fit(self, coordinates: np.ndarray, use_velocities: bool = None) -> 'GaussianMixturePartitioner':
        use_velocities = self.use_velocity if use_velocities is None else use_velocities
        coordinates = np.asarray(coordinates)

        if use_velocities and self.dem_velocities is not None:
            vel = np.asarray(self.dem_velocities)
            if len(vel) == len(coordinates):
                self.features = np.hstack([coordinates, vel])
                self._n_features = 6
                return self._fit_internal(self.features, n_pos=3)
        
        self.features = coordinates
        self._n_features = 3
        return self._fit_internal(coordinates, n_pos=3)

    def _fit_internal(self, features: np.ndarray, n_pos: int = 3) -> 'GaussianMixturePartitioner':
        self._n_features = features.shape[1]

        # Normalisation
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0
        X = (features - self._mean) / self._std

        # Application des poids
        if X.shape[1] > n_pos:
            weights = np.ones(X.shape[1])
            weights[n_pos:] = self.velocity_weight
            X = X * weights[np.newaxis, :]

        # 🚀 SOUS-ÉCHANTILLONNAGE CRUCIAL POUR GMM
        rng = np.random.RandomState(self.random_state)
        if len(X) > self.max_fit_samples:
            print(f"   ⚡ Sous-échantillonnage GMM : {len(X)} -> {self.max_fit_samples} points")
            idx = rng.choice(len(X), self.max_fit_samples, replace=False)
            X = X[idx]

        # Paramètres optimisés pour la vitesse sans trop perdre en précision
        self._gmm = GaussianMixture(
            n_components=self._n_cells, 
            covariance_type='full', 
            random_state=self.random_state, 
            n_init=1,          # Réduit de 5 à 1 (suffisant pour un sweep)
            max_iter=100,      # Réduit de 200 à 100
            tol=1e-3,          # Tolérance légèrement assouplie
            init_params='kmeans' # Initialisation par KMeans (beaucoup plus rapide que random)
        )
        
        print(f"    Fit GMM en cours sur {X.shape[0]} points...")
        self._gmm.fit(X)
        
        self._centroids = self._gmm.means_ 
        print(f"   ✅ Fit GMM terminé.")
        return self

    def compute_states(self, x, y, z, vx=None, vy=None, vz=None) -> np.ndarray:
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])

        if self._n_features == 6 and vx is not None and vy is not None and vz is not None:
            vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
            features = np.hstack([pos, vel])
        elif self._n_features == 6:
            features = np.hstack([pos, np.zeros((len(pos), 3))])
        else:
            features = pos

        X = (features - self._mean) / self._std
        if X.shape[1] > 3:
            weights = np.ones(X.shape[1])
            weights[3:] = self.velocity_weight
            X = X * weights[np.newaxis, :]
        
        self.states = self._gmm.predict(X).astype(np.int64)
        return self.states

    def _save_data(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "gmm_model.pkl"), "wb") as f:
            pickle.dump(self._gmm, f)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)

    def _load_data(self, path: str) -> None:
        with open(os.path.join(path, "gmm_model.pkl"), "rb") as f:
            self._gmm = pickle.load(f)
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._centroids = self._gmm.means_
# =============================================================================
# 7. PARTITIONNEMENT ADAPTATIF HAUT/BAS
# =============================================================================


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

class AdaptivePartitioner(BasePartitioner):
    """
    Partitionnement adaptatif en y.
    
    Divise le domaine en deux zones:
      - Zone haute (y > y_split): peu de cellules (grossier)
      - Zone basse (y ≤ y_split): partitionnement fin
    
    Args:
        y_split: coordonnée de séparation (ou quantile si y_split_mode="quantile")
        y_split_mode: "absolute" ou "quantile"
        n_cells_top: nombre de cellules pour la zone haute
        bottom_method: méthode de partitionnement pour la zone basse
        bottom_kwargs: arguments pour le partitionneur du bas
    """
    
    def __init__(
        self,
        y_split: float = None,#type: ignore
        y_split_mode: str = "quantile",
        n_cells_top: int = 1,
        top_method: str = "single",
        top_kwargs: dict = None,#type: ignore
        bottom_method: str = "cylindrical",
        bottom_kwargs: dict = None,#type: ignore
    ):
        self._splitting_method = "adaptive"
        self.y_split_input = y_split
        self.y_split_mode = y_split_mode
        self.n_cells_top_target = n_cells_top
        self.top_method = top_method
        self.top_kwargs = top_kwargs or {}
        self.bottom_method = bottom_method
        self.bottom_kwargs = bottom_kwargs or {}
        
        # Calculés au fit
        self._y_split:float = None #type: ignore
        self.y_seuil:float = None #type: ignore
        self._y_min:float = None #type: ignore
        self._y_max:float = None #type: ignore
        self._top_partitioner:BasePartitioner = None #type: ignore
        self._bottom_partitioner:BasePartitioner = None #type: ignore
        self._n_cells_top:int = None #type: ignore
        self._n_cells_bottom:int = None #type: ignore
    
    @property
    def n_cells(self:AdaptivePartitioner)-> int:
        if self._n_cells_top is None or self._n_cells_bottom is None:
            return 0
        return self._n_cells_top + self._n_cells_bottom
    
    @property
    def label(self:AdaptivePartitioner)-> str:
        """Propriété manquante nécessaire à l'instanciation"""
        return (
            f"adaptive_y_{self.bottom_method}"
            f"_top{self._n_cells_top}_bot{self._n_cells_bottom}"
            f"_split{self.y_split_input}_mode{self.y_split_mode}"
        )
    
    def fit(self:AdaptivePartitioner, coordinates: np.ndarray)->AdaptivePartitioner:
        coordinates = np.asarray(coordinates)
        y = coordinates[:, 1]  # Utilisation de la coordonnée y
        
        self._y_min = y.min()
        self._y_max = y.max()
        
        # ── Déterminer y_split ──
        if self.y_split_mode == "quantile":
            quantile = self.y_split_input if self.y_split_input else 0.7
            self._y_split = np.quantile(y, quantile) #type:ignore
        elif self.y_split_mode == "absolute":
            if self.y_split_input is None:
                self._y_split = (self._y_min + self._y_max) / 2
            else:
                self._y_split = self.y_split_input
        else:
            raise ValueError(f"y_split_mode inconnu: {self.y_split_mode}")
        
        self.y_seuil = self._y_split
        
        # ── Séparer les données ──
        mask_bottom = y <= self._y_split
        mask_top = y > self._y_split
        
        coords_bottom = coordinates[mask_bottom]
        coords_top = coordinates[mask_top]
        
        # ── Fit zone basse ──
        self._bottom_partitioner = create_partitioner(
            self.bottom_method, **self.bottom_kwargs
        )
        if len(coords_bottom) > 0:
            self._bottom_partitioner.fit(coords_bottom)
        self._n_cells_bottom = self._bottom_partitioner.n_cells
        
        # ── Fit zone haute ──
        if self.top_method == "single":
            self._top_partitioner = None #type:ignore
            self._n_cells_top = 1
        else:
            self._top_partitioner = create_partitioner(
                self.top_method, **self.top_kwargs
            )
            if len(coords_top) > 0:
                self._top_partitioner.fit(coords_top)
            self._n_cells_top = self._top_partitioner.n_cells
        return self
    
    def compute_states(self:AdaptivePartitioner, x: np.ndarray, y: np.ndarray, z: np.ndarray,vx:np.ndarray=None,vy:np.ndarray=None,vz:np.ndarray=None)->np.ndarray: #type:ignore
        # ── Convertir en numpy arrays pour éviter les erreurs de masquage booléen ──
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        
        n = len(x)
        states = np.zeros(n, dtype=np.int64)
        
        mask_bottom = y <= self._y_split
        mask_top = ~mask_bottom
        
        # ── Zone basse : états 0 à n_cells_bottom-1 ──
        if mask_bottom.any():
            states[mask_bottom] = self._bottom_partitioner.compute_states( # type: ignore
                x[mask_bottom], y[mask_bottom], z[mask_bottom]
            )
        
        # ── Zone haute : états n_cells_bottom à n_cells-1 ──
        if mask_top.any():
            if self._top_partitioner is None:
                states[mask_top] = self._n_cells_bottom
            else:
                top_states = self._top_partitioner.compute_states(
                    x[mask_top], y[mask_top], z[mask_top]
                )
                states[mask_top] = top_states + self._n_cells_bottom
        # #n=int(len(x)/self.PARTICLE_NUMBER)
        self.states= states # Les méthode de découpage hybrides comme le adaptive et le multizone ne necessite pas l'application de masque car
        # elles appellent déjà d'autres méthode de computestate des classes de découpage qu'elle instancient.
        return self.states

    def _get_cell_polygons_2d(self:AdaptivePartitioner, view: str='xy')-> list:
        results = []
        if self._bottom_partitioner is not None:
            for state_id, pts in self._bottom_partitioner._get_cell_polygons_2d(view):
                results.append((state_id, pts))
        if self._top_partitioner is not None and self.top_method != "single":
            offset = self._n_cells_bottom
            for state_id, pts in self._top_partitioner._get_cell_polygons_2d(view):
                results.append((state_id + offset, pts))
        elif self._top_partitioner is None:
            offset = self._n_cells_bottom
            results.append((offset, np.array([[0, 0], [1, 0], [1, 1], [0, 1]])))
        return results

    def _get_cell_polyhedra_3d(self:AdaptivePartitioner)-> list:
        results = []
        if self._bottom_partitioner is not None:
            for state_id, vertices, faces in self._bottom_partitioner._get_cell_polyhedra_3d():
                results.append((state_id, vertices, faces))
        if self._top_partitioner is not None and self.top_method != "single":
            offset = self._n_cells_bottom
            for state_id, vertices, faces in self._top_partitioner._get_cell_polyhedra_3d():
                results.append((state_id + offset, vertices, faces))
        return results


# =============================================================================
# 8. PARTITIONNEMENT MULTI-ZONES (généralisation)
# =============================================================================

class MultiZonePartitioner(BasePartitioner):
    """
    Partitionnement multi-zones généralisé (basé sur l'axe Y).
    
    Permet de définir plusieurs zones avec des partitionnements différents.
    Plus flexible que AdaptiveYPartitioner.
    
    Args:
        zones: liste de dicts définissant chaque zone
            [
                {"y_min": -inf, "y_max": 0.5, "method": "cylindrical", "kwargs": {...}},
                {"y_min": 0.5, "y_max": 0.8, "method": "voronoi", "kwargs": {"n_cells": 50}},
                {"y_min": 0.8, "y_max": inf, "method": "single", "kwargs": {}},
            ]
        y_mode: "absolute" ou "quantile"
    """
    
    def __init__(
        self:MultiZonePartitioner,
        zones: list,
        y_mode: str = "absolute"
    ):
        self._splitting_method:str = "multizone"
        self.zones_config = zones
        self.y_mode = y_mode
        self._zones:list = []  # [(y_min, y_max, partitioner), ...]
        self._cell_offsets:list = []
        self._total_cells:int = 0
    
    @property
    def n_cells(self:MultiZonePartitioner)->int:
        return self._total_cells
    
    @property
    def label(self:MultiZonePartitioner)-> str:
        methods = "_".join(z["method"] for z in self.zones_config)
        return f"multizone_{len(self.zones_config)}zones_{methods}"
    
    def fit(self:MultiZonePartitioner, coordinates: np.ndarray)->MultiZonePartitioner:
        coordinates = np.asarray(coordinates)
        y = coordinates[:, 1]  # Utilisation de l'axe Y (index 1)
        
        self._zones = []
        self._cell_offsets = [0]
        
        for i, zone_cfg in enumerate(self.zones_config):
            # Convertir les bornes si mode quantile
            if self.y_mode == "quantile":
                y_min = np.quantile(y, zone_cfg.get("y_min", 0))
                y_max = np.quantile(y, zone_cfg.get("y_max", 1))
            else:
                y_min = zone_cfg.get("y_min", y.min())
                y_max = zone_cfg.get("y_max", y.max())
            
            # Sélectionner les particules de cette zone
            if i == len(self.zones_config) - 1:
                mask = (y >= y_min) & (y <= y_max)  # Inclure le max pour la dernière zone
            else:
                mask = (y >= y_min) & (y < y_max)
                
            coords_zone = coordinates[mask]
            
            method = zone_cfg.get("method", "single")
            kwargs = zone_cfg.get("kwargs", {})
            
            if method == "single":
                partitioner = SingleCellPartitioner()
            else:
                partitioner = create_partitioner(method, **kwargs)
            
            if len(coords_zone) > 0:
                partitioner.fit(coords_zone)
            
            self._zones.append((y_min, y_max, partitioner))
            self._cell_offsets.append(
                self._cell_offsets[-1] + partitioner.n_cells
            )
            
            print(f"   Zone {i}: y ∈ [{y_min:.3f}, {y_max:.3f}], "
                  f"{partitioner.n_cells} cellules, {len(coords_zone)} particules")
        
        self._total_cells = self._cell_offsets[-1]
        print(f"   Total: {self._total_cells} cellules")
        
        return self
    
    def compute_states(self:MultiZonePartitioner, x: np.ndarray, y: np.ndarray, z: np.ndarray,vx:np.ndarray=None,vy:np.ndarray=None,vz:np.ndarray=None)->np.ndarray: #type:ignore
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
                zone_states = partitioner.compute_states(
                    x[mask], y[mask], z[mask]
                )
                states[mask] = zone_states + self._cell_offsets[i]
                assigned[mask] = True
        # #n=int(len(x)/self.PARTICLE_NUMBER)
        self.states= states
        return self.states
    
    def _save_data(self:MultiZonePartitioner, path:str)->None:
        config = {
            "zones_config": self.zones_config,
            "y_mode": self.y_mode,
            "cell_offsets": self._cell_offsets,
            "zones_bounds": [(y_min, y_max) for y_min, y_max, _ in self._zones],
        }
        with open(os.path.join(path, "multizone_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        for i, (_, _, partitioner) in enumerate(self._zones):
            zone_path = os.path.join(path, f"zone_{i}")
            partitioner.save(zone_path)
    
    def _load_data(self:MultiZonePartitioner, path: str)->None:
        with open(os.path.join(path, "multizone_config.json")) as f:
            config = json.load(f)
        
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
                partitioner = SingleCellPartitioner()
            else:
                partitioner = create_partitioner(method, **kwargs)
            
            zone_path = os.path.join(path, f"zone_{i}")
            partitioner.load(zone_path)
            
            self._zones.append((y_min, y_max, partitioner))

    def _get_cell_polygons_2d(self:MultiZonePartitioner, view:str='xy')-> list:
        results = []
        for zone_idx, (y_min, y_max, partitioner) in enumerate(self._zones):
            offset = self._cell_offsets[zone_idx]
            for state_id, pts in partitioner._get_cell_polygons_2d(view):
                results.append((state_id + offset, pts))
        return results

    def _get_cell_polyhedra_3d(self:MultiZonePartitioner)-> list:
        results = []
        for zone_idx, (y_min, y_max, partitioner) in enumerate(self._zones):
            offset = self._cell_offsets[zone_idx]
            for state_id, vertices, faces in partitioner._get_cell_polyhedra_3d():
                results.append((state_id + offset, vertices, faces))
        return results


# =============================================================================
# SINGLE CELL
# =============================================================================

class SingleCellPartitioner(BasePartitioner):
    """Une seule cellule pour tout le domaine."""

    def __init__(self:SingleCellPartitioner, **kwargs)->None:
        super().__init__(**kwargs)
        self._splitting_method = "single"

    @property
    def n_cells(self:SingleCellPartitioner)-> int:
        return 1

    @property
    def label(self:SingleCellPartitioner)-> str:
        return "single_cell"

    def fit(self:SingleCellPartitioner, coordinates:np.ndarray)->SingleCellPartitioner:
        return self

    def compute_states(self:SingleCellPartitioner, x: np.ndarray, y: np.ndarray, z: np.ndarray,vx:np.ndarray=None,vy:np.ndarray=None,vz:np.ndarray=None)->np.ndarray: #type:ignore
        self.states= np.zeros(len(np.asarray(x)), dtype=np.int64)
        return self.states

    def _get_cell_polygons_2d(self:SingleCellPartitioner, view: str='xy')-> list:
        if hasattr(self, '_data_bounds') and self._data_bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self._data_bounds
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = -1, 1, -1, 1, -1, 1
        if view == 'xy':
            pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        elif view == 'yz':
            pts = np.array([[ymin, zmin], [ymax, zmin], [ymax, zmax], [ymin, zmax]])
        else:
            return []
        return [(0, pts)]

    def _get_cell_polyhedra_3d(self:SingleCellPartitioner)-> list:
        if hasattr(self, '_data_bounds') and self._data_bounds is not None:
            xmin, xmax, ymin, ymax, zmin, zmax = self._data_bounds
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = -1, 1, -1, 1, -1, 1
        vertices = np.array([
            [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
            [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
        ])
        faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        return [(0, vertices, faces)]


# =============================================================================
# MISE À JOUR DU REGISTRY
# =============================================================================

REGISTRY = {
    # Méthodes géométriques de base
    "cartesian": CartesianPartitioner,
    "cylindrical": CylindricalPartitioner,
    "voronoi": VoronoiPartitioner,
    "quantile": QuantileGridPartitioner,
    "octree": OctreePartitioner,
    
    # Méthodes basées sur la physique (Doucet, Tjakra, Zhou)
    "physics": PhysicsAwarePartitioner,               # K-Means avec la norme de la vitesse (|v|)
    "physics_full_vel": FullVectorVelocityKMeansPartitioner, # K-Means avec le vecteur vitesse complet (vx, vy, vz)
    "spectral": SpectralClusteringPartitioner,        # Spectral Clustering (topologie/connexion du graphe)
    "gmm": GaussianMixturePartitioner,                # Gaussian Mixture Model (cellules ellipsoïdales)
    
    # Autres méthodes avancées
    "adaptive": AdaptivePartitioner,      
    "multizone": MultiZonePartitioner,     
    "single": SingleCellPartitioner,       
}

# =============================================================================
# FACTORY
# =============================================================================




def create_partitioner(method:str, **kwargs)-> BasePartitioner:
    """
    Crée un partitionneur.

    Args:
        method: "cartesian", "cylindrical", "voronoi", "quantile",
                "octree", "physics"
        **kwargs: arguments passés au constructeur

    Returns:
        instance de BasePartitioner

    Exemple:
        p = create_partitioner("voronoi", n_cells=125)
        p = create_partitioner("cylindrical", nr=5, ntheta=8, nz=5)
    """
    if method not in REGISTRY:
        available = ", ".join(REGISTRY.keys())
        raise ValueError(f"Méthode inconnue: '{method}'. Disponibles: {available}")
    return REGISTRY[method](**kwargs) # crée une instance de la classe de partitionnement souhaité