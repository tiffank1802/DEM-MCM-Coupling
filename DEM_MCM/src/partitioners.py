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

import numpy as np
import os
import io
import json
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull, Voronoi

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
    "adaptive",   
    "multizone", 
    "single",   
    "create_partitioner",
    "REGISTRY",
]


# =============================================================================
# CLASSE DE BASE
# =============================================================================


class BasePartitioner(ABC,ar.MarkovAnalyzer):
    """Interface commune pour tous les partitionneurs."""
    def __init__(self):
        self._y_split=0
        super().__init__()
        # self.load_dem_snapshots(file_indices=[250])
        # self.label_species()
        self.PARTICLE_NUMBER=1030
        
    # analyzer=ar.MarkovAnalyzer()

    @property
    @abstractmethod
    def n_cells(self)-> int:
        """Nombre total d'états."""
        ...

    @property
    @abstractmethod
    def label(self)-> str:
        """Identifiant unique (utilisé pour le nom de dossier)."""
        ...

    @abstractmethod
    def fit(self, coordinates: np.ndarray)->object:
        """
        Apprend le partitionnement sur des données représentatives.

        Args:
            coordinates: np.ndarray shape (N, 3)
        Returns:
            self
        """
        ...

    @abstractmethod
    def compute_states(self, x:np.ndarray, y:np.ndarray, z:np.ndarray)->np.ndarray:
        """
        Assigne un indice d'état à chaque particule.

        Args:
            x, y, z: arrays ou Polars Series
        Returns:
            np.ndarray dtype int64
        """
        ...

    def save(self, path: str):
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

    def _save_data(self, path):
        pass

    def load(self, path):
        """Charge le partitionneur depuis un dossier."""
        self._load_data(path)
        return self

    def _load_data(self, path):
        pass

    def visualize(self, x, y, z, plot_types=["3d", "2d_xy"], save_prefix="partition_visualization", particle_diameters=None, use_diameter=True):
        """
        Génère 4 images: particules avec diamètres + limites réelles des partitions.
        
        Args:
            particle_diameters: array de diamètres pour représenter chaque particule avec sa taille
            use_diameter: si True (défaut), utilise les diamètres si disponibles (DEM ou explicites)
            
        Returns:
            dict: {
                "{prefix}_particles_2d.png": bytes,
                "{prefix}_particles_3d.png": bytes,
                "{prefix}_boundaries_2d.png": bytes,
                "{prefix}_boundaries_3d.png": bytes,
            }
        """
        self.fit(np.column_stack([x,y,z]))
        states = self.compute_states(x, y, z)
        
        image_data = {}
        
        if "2d_xy" not in plot_types and "3d" not in plot_types:
            return image_data
        
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        zmin, zmax = z.min(), z.max()
        self._data_bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # ════════════ IMAGES LIMITES DE PARTITIONS (toujours) ════════════
        try:
            boundary_data = self._visualize_cell_boundaries(x, y, z, states, plot_types, save_prefix)
            image_data.update(boundary_data)
        except Exception as e:
            print(f"⚠️  Visualisation des limites échouée: {e}")
        
        # ════════════ IMAGES PARTICULES (si diamètres dispos) ════════════
        diameters = None
        if use_diameter:
            if particle_diameters is not None:
                diameters = particle_diameters
            elif hasattr(self, 'particle_diameters') and self.particle_diameters is not None:
                diameters = self.particle_diameters
            elif hasattr(self, 'dem_diameters') and self.dem_diameters is not None:
                if len(self.dem_diameters) == len(x):
                    diameters = self.dem_diameters
                else:
                    print(f"⚠️  dem_diameters ({len(self.dem_diameters)}) != nombre de particules ({len(x)})")
        
        if diameters is not None:
            try:
                particle_data = self._visualize_particles_with_diameter(
                    x, y, z, states, diameters, plot_types, save_prefix
                )
                image_data.update(particle_data)
            except Exception as e:
                print(f"⚠️  Visualisation des particules échouée: {e}")
        
        return image_data

    def _visualize_cell_boundaries(self, x, y, z, states, plot_types, save_prefix):
        """
        Visualise les limites réelles des partitions (sans particules).
        Utilise _get_cell_polygons_2d() et _get_cell_polyhedra_3d() implémentés par chaque sous-classe.
        """
        import matplotlib.cm as cm
        import matplotlib.patches as patches
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        image_data = {}
        cmap = cm.get_cmap('tab20')
        unique_states = np.unique(states)
        n_states = len(unique_states)
        
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        zmin, zmax = z.min(), z.max()
        
        if "2d_xy" in plot_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # ════════ Vue XY ════════
            try:
                polygons_xy = self._get_cell_polygons_2d(view='xy')
                for state_id, polygon_pts in polygons_xy:
                    if len(polygon_pts) < 3:
                        continue
                    color = cmap(state_id / max(n_states - 1, 1))
                    poly = patches.Polygon(polygon_pts, closed=True,
                                          facecolor=color, alpha=0.6,
                                          edgecolor='black', linewidth=1.5, zorder=1)
                    ax1.add_patch(poly)
            except Exception as e:
                print(f"⚠️  Limites XY non disponibles: {e}")
                ax1.text(0.5, 0.5, 'Limites non disponibles', ha='center', va='center',
                        transform=ax1.transAxes, fontsize=14, color='gray')
            
            ax1.set_xlim(xmin, xmax)
            ax1.set_ylim(ymin, ymax)
            ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax1.set_title('Vue XY - Limites des partitions', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.set_aspect('equal', adjustable='box')
            
            # ════════ Vue YZ ════════
            try:
                polygons_yz = self._get_cell_polygons_2d(view='yz')
                for state_id, polygon_pts in polygons_yz:
                    if len(polygon_pts) < 3:
                        continue
                    color = cmap(state_id / max(n_states - 1, 1))
                    poly = patches.Polygon(polygon_pts, closed=True,
                                          facecolor=color, alpha=0.6,
                                          edgecolor='black', linewidth=1.5, zorder=1)
                    ax2.add_patch(poly)
            except Exception as e:
                print(f"⚠️  Limites YZ non disponibles: {e}")
                ax2.text(0.5, 0.5, 'Limites non disponibles', ha='center', va='center',
                        transform=ax2.transAxes, fontsize=14, color='gray')
            
            ax2.set_xlim(ymin, ymax)
            ax2.set_ylim(zmin, zmax)
            ax2.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
            ax2.set_title('Vue YZ - Limites des partitions', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.set_aspect('equal', adjustable='box')
            
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            image_data[f"{save_prefix}_boundaries_2d.png"] = img_buffer.getvalue()
            plt.close()
        
        if "3d" in plot_types:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            try:
                polyhedra = self._get_cell_polyhedra_3d()
                for state_id, vertices, faces in polyhedra:
                    if len(vertices) < 4 or len(faces) == 0:
                        continue
                    color = cmap(state_id / max(n_states - 1, 1))
                    face_verts = [vertices[f] for f in faces]
                    collection = Poly3DCollection(face_verts, alpha=0.5,
                                                 facecolor=color, edgecolor='black',
                                                 linewidth=0.8, zorder=1)
                    ax.add_collection3d(collection)
            except Exception as e:
                print(f"⚠️  Limites 3D non disponibles: {e}")
            
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_zlim(zmin, zmax)
            ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
            ax.set_title(f'Limites des partitions 3D - {self.label}',
                        fontsize=14, fontweight='bold')
            
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            image_data[f"{save_prefix}_boundaries_3d.png"] = img_buffer.getvalue()
            plt.close()
        
        return image_data

    def _get_cell_polygons_2d(self, view='xy'):
        """Retourne une liste de (state_id, polygon_pts_2d). À implémenter par chaque sous-classe."""
        return []

    def _get_cell_polyhedra_3d(self):
        """Retourne une liste de (state_id, vertices_3d, faces_3d). À implémenter par chaque sous-classe."""
        return []

    def visualize_enhanced(self, x, y, z, plot_types=["3d", "2d_xy"], save_prefix="partition_visualization", 
                          particle_diameters=None, show_filled_partitions=True):
        """
        ✅ **NOUVELLE MÉTHODE** - Génère 2 représentations (4 images total):
        
        Représentation 1: Particules avec diamètre (bidisperses)
        Représentation 2: Partitions remplies (contours + remplissage)
        
        Args:
            x, y, z: coordonnées des particules
            plot_types: ["3d", "2d_xy"] ou sous-ensemble
            particle_diameters: array de diamètres (None → pas de particules)
            show_filled_partitions: True → afficher aussi les partitions remplies
        
        Returns:
            dict: {
                "particles_2d.png": bytes,
                "particles_3d.png": bytes,
                "partitions_2d.png": bytes,
                "partitions_3d.png": bytes,
            }
        """
        self.fit(np.column_stack([x, y, z]))
        states = self.compute_states(x, y, z)
        
        image_data = {}
        
        # ========== REPRÉSENTATION 1: Particules avec diamètre ==========
        if particle_diameters is not None:
            img_parts = self._visualize_particles_with_diameter(
                x, y, z, states, particle_diameters, plot_types, save_prefix
            )
            image_data.update(img_parts)
        
        # ========== REPRÉSENTATION 2: Partitions remplies ==========
        if show_filled_partitions:
            img_parts_filled = self._visualize_partitions_filled(
                x, y, z, states, plot_types, save_prefix
            )
            image_data.update(img_parts_filled)
        
        return image_data
    
    def _visualize_particles_with_diameter(self, x, y, z, states, diameters, 
                                          plot_types, save_prefix):
        """
        Visualise les particules bidisperses avec diamètre proportionnel.
        Deux images: 2D (XY + YZ) et 3D
        """
        import matplotlib.cm as cm
        
        image_data = {}
        cmap = cm.get_cmap('tab20')
        
        # Normaliser les diamètres pour la taille des points
        diameters_norm = np.asarray(diameters)
        size_scale = (diameters_norm / diameters_norm.max()) * 200  # [0, 200]
        
        # ════════════ 2D (XY + YZ) ════════════
        if "2d_xy" in plot_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # Vue XY
            scatter1 = ax1.scatter(x, y, s=size_scale, c=states, cmap='tab20', 
                                   alpha=0.6, edgecolors='black', linewidth=0.3)
            ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax1.set_title('Vue XY - Particules avec diamètre', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter1, ax=ax1, label='ID Partition')
            
            # Vue YZ
            scatter2 = ax2.scatter(y, z, s=size_scale, c=states, cmap='tab20', 
                                   alpha=0.6, edgecolors='black', linewidth=0.3)
            ax2.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
            ax2.set_title('Vue YZ - Particules avec diamètre', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter2, ax=ax2, label='ID Partition')
            
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            image_data[f"{save_prefix}_particles_2d.png"] = img_buffer.getvalue()
            plt.close()
        
        # ════════════ 3D ════════════
        if "3d" in plot_types:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(x, y, z, s=size_scale, c=states, cmap='tab20', 
                                alpha=0.6, edgecolors='black', linewidth=0.3)
            
            ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
            ax.set_title(f'Particules bidisperses 3D - {self.label}', 
                        fontsize=14, fontweight='bold')
            
            plt.colorbar(scatter, ax=ax, label='ID Partition', shrink=0.6)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            image_data[f"{save_prefix}_particles_3d.png"] = img_buffer.getvalue()
            plt.close()
        
        return image_data
    
    def _visualize_partitions_filled(self, x, y, z, states, plot_types, save_prefix):
        """
        Visualise les partitions remplies avec contours nets.
        """
        from scipy.spatial import ConvexHull
        import matplotlib.patches as patches
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        image_data = {}
        cmap = cm.get_cmap('tab20')
        unique_states = np.unique(states)
        
        # ════════════ 2D (XY + YZ) ════════════
        if "2d_xy" in plot_types:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            
            # ────── Vue XY ──────
            coords_xy = np.column_stack([x, y])
            for state_id in unique_states:
                mask = states == state_id
                points = coords_xy[mask]
                
                if len(points) < 3:
                    continue
                
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    
                    # Remplissage
                    color = cmap(state_id / max(unique_states.max(), 1))
                    polygon = patches.Polygon(hull_points, closed=True, 
                                            facecolor=color, alpha=0.5, 
                                            edgecolor='black', linewidth=2, zorder=1)
                    ax1.add_patch(polygon)
                except:
                    pass
            
            ax1.set_xlim(x.min(), x.max())
            ax1.set_ylim(y.min(), y.max())
            ax1.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax1.set_title('Vue XY - Partitions remplies', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            # ────── Vue YZ ──────
            coords_yz = np.column_stack([y, z])
            for state_id in unique_states:
                mask = states == state_id
                points = coords_yz[mask]
                
                if len(points) < 3:
                    continue
                
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    
                    # Remplissage
                    color = cmap(state_id / max(unique_states.max(), 1))
                    polygon = patches.Polygon(hull_points, closed=True, 
                                            facecolor=color, alpha=0.5, 
                                            edgecolor='black', linewidth=2, zorder=1)
                    ax2.add_patch(polygon)
                except:
                    pass
            
            ax2.set_xlim(y.min(), y.max())
            ax2.set_ylim(z.min(), z.max())
            ax2.set_xlabel('Y (m)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Z (m)', fontsize=12, fontweight='bold')
            ax2.set_title('Vue YZ - Partitions remplies', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            image_data[f"{save_prefix}_partitions_2d.png"] = img_buffer.getvalue()
            plt.close()
        
        # ════════════ 3D ════════════
        if "3d" in plot_types:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Remplissage 3D (surfaces)
            for state_id in unique_states:
                mask = states == state_id
                points_3d = np.column_stack([x[mask], y[mask], z[mask]])
                
                if len(points_3d) < 4:
                    continue
                
                try:
                    hull = ConvexHull(points_3d)
                    faces = []
                    for simplex in hull.simplices:
                        faces.append(points_3d[simplex])
                    
                    color = cmap(state_id / max(unique_states.max(), 1))
                    collection = Poly3DCollection(faces, alpha=0.5, 
                                                 facecolor=color, edgecolor='black',
                                                 linewidth=1, zorder=1)
                    ax.add_collection3d(collection)
                except:
                    pass
            
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max())
            ax.set_zlim(z.min(), z.max())
            ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
            ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
            ax.set_title(f'Partitions remplies 3D - {self.label}', 
                        fontsize=14, fontweight='bold')
            
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            image_data[f"{save_prefix}_partitions_3d.png"] = img_buffer.getvalue()
            plt.close()
        
        return image_data


    def diagnostics(self, coordinates):
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
# 1. CARTÉSIEN
# =============================================================================


class CartesianPartitioner(BasePartitioner):
    """
    Grille cartésienne régulière.

    Découpe le domaine en nx × ny × nz cellules de taille égale.
    Simple mais inadapté aux géométries cylindriques (coins vides).
    """

    def __init__(self, nx=5, ny=5, nz=5):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self._bounds = None

    @property
    def n_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def label(self)-> str:
        return f"cartesian_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self, coordinates:np.ndarray):
        eps = 0.001
        coordinates=np.asarray(coordinates) # contient les coordonnées [x,y,z] de toutes les particules
        mins = coordinates.min(axis=0) - eps # contient le minimum de [x,y,z]
        maxs = coordinates.max(axis=0) + eps # contient le maximum de [x,y,z]
        self._bounds = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]) # (min_x,max_x,min_y,max_y,min_z,max_z)
        return self

    def compute_states(self, x:np.ndarray, y:np.ndarray, z:np.ndarray)-> int:
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
        n=int(len(x)/self.PARTICLE_NUMBER)
        self.states=ix + iy * self.nx + iz * self.nx * self.ny
        return self.states#[np.tile(self.species_labels,n)]

    def _save_data(self, path):
        np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _load_data(self, path):
        self._bounds = tuple(np.load(os.path.join(path, "bounds.npy")))

    def _get_cell_polygons_2d(self, view='xy'):
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

    def _get_cell_polyhedra_3d(self):
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

    def __init__(self, nr=5, ntheta=8, nz=5, radial_mode="equal_area"):
        super().__init__()
        self.nr = nr
        self.ntheta = ntheta
        self.nz = nz
        self.radial_mode = radial_mode
        self._x_center = None
        self._y_center = None
        self._r_max = None
        self._z_min = None
        self._z_max = None
        self._r_edges = None
        
        # self.species_labels=self.label_species()

    @property
    def n_cells(self):
        return self.nr * self.ntheta * self.nz

    @property
    def label(self):
        return (
            f"cylindrical_nr{self.nr}_nth{self.ntheta}"
            f"_nz{self.nz}_{self.radial_mode}"
        )

    def fit(self, coordinates):
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

    def compute_states(self, x, y, z):
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
        n=int(len(x)/self.PARTICLE_NUMBER)
        self.states=ir + itheta * self.nr + iz * self.nr * self.ntheta
        return  self.states#[np.tile(self.species_labels,n)] # la numérotation des partitons se fait partant des rayons, puis les angles et enfin les hauteurs z

    def _save_data(self, path):
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

    def _load_data(self, path):
        with open(os.path.join(path, "cylindrical_params.json")) as f:
            p = json.load(f)
        self._x_center = p["x_center"]
        self._y_center = p["y_center"]
        self._r_max = p["r_max"]
        self._z_min = p["z_min"]
        self._z_max = p["z_max"]
        self._r_edges = np.load(os.path.join(path, "r_edges.npy"))

    def _arc_points(self, r, theta_start, theta_end, n_segments=20):
        theta_vals = np.linspace(theta_start, theta_end, n_segments)
        return np.column_stack([r * np.cos(theta_vals), r * np.sin(theta_vals)])

    def _get_cell_polygons_2d(self, view='xy'):
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

    def _get_cell_polyhedra_3d(self):
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

    def __init__(self, n_cells=125, random_state=42):
        super().__init__()
        self._n_cells = n_cells
        self.random_state = random_state
        self.centroids = None
        self._tree = None

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def label(self):
        return f"voronoi_{self._n_cells}cells"

    def fit(self, coordinates):
        coordinates=np.asarray(coordinates)
        from sklearn.cluster import MiniBatchKMeans
        from scipy.spatial import cKDTree

        rng = np.random.RandomState(self.random_state)
        if len(coordinates) > 500_000:
            idx = rng.choice(len(coordinates), 500_000, replace=False)
            fit_data = coordinates[idx]
        else:
            fit_data = coordinates
        kmeans = MiniBatchKMeans(
            n_clusters=self._n_cells,
            random_state=self.random_state,
            batch_size=min(10_000, len(fit_data)),
            n_init=10,
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

    def compute_states(self, x, y, z):
        coords = np.column_stack(
            [np.asarray(x), np.asarray(y), np.asarray(z)]
        )
        n=int(len(x)/self.PARTICLE_NUMBER)

        _, indices = self._tree.query(coords)
        self.states=indices.astype(np.int64)
        return self.states#[np.tile(self.species_labels,n)]

    def _save_data(self, path):
        np.save(os.path.join(path, "centroids.npy"), self.centroids)

    def _load_data(self, path):
        from scipy.spatial import cKDTree

        self.centroids = np.load(os.path.join(path, "centroids.npy"))
        self._tree = cKDTree(self.centroids)
        self._n_cells = len(self.centroids)
        self._voronoi_3d = Voronoi(self.centroids)

    def _get_cell_polygons_2d(self, view='xy'):
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

    def _get_cell_polyhedra_3d(self):
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

    def diagnostics(self, coordinates):
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

    def __init__(self, nx=5, ny=5, nz=5):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self._x_edges = None
        self._y_edges = None
        self._z_edges = None

    @property
    def n_cells(self):
        return self.nx * self.ny * self.nz

    @property
    def label(self):
        return f"quantile_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self, coordinates):
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

    def compute_states(self, x, y, z):
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
        n=int(len(x)/self.PARTICLE_NUMBER)
        self.states= ix + iy * self.nx + iz * self.nx * self.ny
        return self.states#[np.tile(self.species_labels,n)]

    def _save_data(self, path):
        np.savez(
            os.path.join(path, "edges.npz"),
            x=self._x_edges,
            y=self._y_edges,
            z=self._z_edges,
        )

    def _load_data(self, path):
        data = np.load(os.path.join(path, "edges.npz"))
        self._x_edges = data["x"]
        self._y_edges = data["y"]
        self._z_edges = data["z"]

    def _get_cell_polygons_2d(self, view='xy'):
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

    def _get_cell_polyhedra_3d(self):
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
    Octree adaptatif.

    Subdivise récursivement les cellules contenant plus de max_particles
    particules, jusqu'à max_depth niveaux.

    Avantage : raffine automatiquement les zones denses.
    Inconvénient : nombre de cellules non contrôlé a priori.
    """

    def __init__(self, max_particles=100, max_depth=5,transform_type=0):
        super().__init__()
        self.max_particles = max_particles
        self.max_depth = max_depth
        self.transform_type=transform_type
        self._leaves = []  # liste de tuples (xmin, xmax, ymin, ymax, zmin, zmax)
        self._bounds = None
        self._stats={}

    @property
    def n_cells(self):
        return len(self._leaves) if self._leaves else 0

    @property
    def label(self):
        return f"octree_mp{self.max_particles}_md{self.max_depth}"
    
    def _apply_transform(self,coords):
        if self.transform_type=='normalize':
            return (coords-self._stats["min"])/self._stats["max"]-self._stats["min"]
        return coords 

    def fit(self, coordinates):
        coordinates=np.asarray(coordinates)
        eps = 0.001
        self._stats["min"]=coordinates.min(axis=0)-eps
        self._stats["max"]=coordinates.max(axis=0)+eps
        # 2. Application de la transformation (si définie)
        if self.transform_type is not None:
            transformed_coords = self._apply_transform(coordinates)
        else:
            # Sécurité : si pas de transform, on utilise les coordonnées brutes
            transformed_coords = coordinates
        self._bounds = (
            transformed_coords[:, 0].min() - eps,
            transformed_coords[:, 0].max() + eps,
            transformed_coords[:, 1].min() - eps,
            transformed_coords[:, 1].max() + eps,
            transformed_coords[:, 2].min() - eps,
            transformed_coords[:, 2].max() + eps,
        )
        self._leaves = []
        self._subdivide(transformed_coords, self._bounds, depth=0)
        return self

    def _subdivide(self, coords, bounds, depth):
        """Subdivision récursive."""
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        n_in = len(coords)

        # Condition d'arrêt
        if n_in <= self.max_particles or depth >= self.max_depth:
            self._leaves.append(bounds)
            return

        # Point de coupe = milieu
        # xmid = (xmin + xmax) / 2
        # ymid = (ymin + ymax) / 2
        # zmid = (zmin + zmax) / 2
        xmid = np.median(coords[:,0])
        ymid = np.median(coords[:,1])
        zmid = np.median(coords[:,2])

        # Assigner chaque particule à un octant (0-7)
        octant = (
            (coords[:, 0] >= xmid).astype(np.int64)
            + (coords[:, 1] >= ymid).astype(np.int64) * 2
            + (coords[:, 2] >= zmid).astype(np.int64) * 4
        )

        # Récursion sur les 8 enfants
        for idx in range(8):
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

    def compute_states(self, x, y, z):
        coords = np.column_stack(
            [np.asarray(x, dtype=np.float64),
             np.asarray(y, dtype=np.float64),
             np.asarray(z, dtype=np.float64)]
        )
        n = len(coords)
        states = np.full(n, -1, dtype=np.int64)

        # Assignation par bounding box
        for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
            mask = (
                (coords[:, 0] >= xmin) & (coords[:, 0] < xmax)
                & (coords[:, 1] >= ymin) & (coords[:, 1] < ymax)
                & (coords[:, 2] >= zmin) & (coords[:, 2] < zmax)
            )
            states[mask] = cell_id

        # Points non assignés → cellule la plus proche
        unassigned = states == -1
        if unassigned.any():
            from scipy.spatial import cKDTree

            centers = np.array(
                [
                    (
                        (b[0] + b[1]) / 2,
                        (b[2] + b[3]) / 2,
                        (b[4] + b[5]) / 2,
                    )
                    for b in self._leaves
                ]
            )
            tree = cKDTree(centers)
            _, idx = tree.query(coords[unassigned])
            states[unassigned] = idx
        n_n=int(len(x)/self.PARTICLE_NUMBER)

        self.states= states
        return self.states#[np.tile(self.species_labels,n_n)]

    def _save_data(self, path):
        leaves_arr = np.array(self._leaves)
        np.save(os.path.join(path, "leaves.npy"), leaves_arr)
        if self._bounds:
            np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _load_data(self, path):
        leaves_arr = np.load(os.path.join(path, "leaves.npy"))
        self._leaves = [tuple(row) for row in leaves_arr]
        bounds_path = os.path.join(path, "bounds.npy")
        if os.path.exists(bounds_path):
            self._bounds = tuple(np.load(bounds_path))

    def _get_cell_polygons_2d(self, view='xy'):
        results = []
        for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
            if view == 'xy':
                pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            elif view == 'yz':
                pts = np.array([[ymin, zmin], [ymax, zmin], [ymax, zmax], [ymin, zmax]])
            else:
                continue
            results.append((cell_id, pts))
        return results

    def _get_cell_polyhedra_3d(self):
        results = []
        face_indices = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[0,3,7,4],[1,2,6,5]]
        for cell_id, (xmin, xmax, ymin, ymax, zmin, zmax) in enumerate(self._leaves):
            vertices = np.array([
                [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymax, zmin], [xmin, ymax, zmin],
                [xmin, ymin, zmax], [xmax, ymin, zmax], [xmax, ymax, zmax], [xmin, ymax, zmax],
            ])
            results.append((cell_id, vertices, face_indices))
        return results


# =============================================================================
# 6. PHYSIQUE-AWARE (POSITION + VITESSE)
# =============================================================================


class PhysicsAwarePartitioner(BasePartitioner):
    """
    K-means sur des features physiques (position + vitesse optionnelle).

    Par défaut, fonctionne sur les positions normalisées (équivalent Voronoï).
    Si des vitesses sont fournies via fit_with_physics(), le clustering
    tient aussi compte de la norme de vitesse.

    Usage avancé:
        part = PhysicsAwarePartitioner(n_cells=125, velocity_weight=0.3)
        part.fit_with_physics(positions, velocities)
        states = part.compute_states_with_physics(x, y, z, vx, vy, vz)
    """

    def __init__(self, n_cells=125, velocity_weight=0.0, random_state=42):
        super().__init__()
        self._n_cells = n_cells
        self.velocity_weight = velocity_weight
        self.random_state = random_state
        self._centroids = None
        self._tree = None
        self._mean = None
        self._std = None
        self._n_features = 3  # 3 = position seule, 4 = position + vitesse

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def label(self):
        suffix = "withvel" if self._n_features > 3 else "pos"
        return f"physics_{self._n_cells}cells_{suffix}"

    def fit(self, coordinates):
        """Fit sur positions seules (équivalent Voronoï normalisé)."""
        coordinates=np.asarray(coordinates)
        return self._fit_internal(coordinates)

    def fit_with_physics(self, positions, velocities):
        """
        Fit sur positions + norme de vitesse.

        Args:
            positions: (N, 3)
            velocities: (N, 3)
        """
        speed = np.linalg.norm(velocities, axis=1, keepdims=True)
        features = np.hstack([positions, speed * self.velocity_weight])
        self._n_features = 4
        return self._fit_internal(features)

    def _fit_internal(self, features):
        from sklearn.cluster import MiniBatchKMeans
        from scipy.spatial import cKDTree

        self._n_features = features.shape[1]

        # Normalisation
        self._mean = features.mean(axis=0)
        self._std = features.std(axis=0)
        self._std[self._std == 0] = 1.0

        X = (features - self._mean) / self._std

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

    def compute_states(self, x, y, z):
        """Assigne les états (position seule, vitesse=0 si fitté avec)."""
        coords = np.column_stack(
            [np.asarray(x), np.asarray(y), np.asarray(z)]
        )
        if self._n_features > 3:
            padding = np.zeros((len(coords), self._n_features - 3))
            coords = np.hstack([coords, padding])

        X = (coords - self._mean) / self._std
        _, indices = self._tree.query(X)
        n=int(len(x)/self.PARTICLE_NUMBER)
        self.states=indices.astype(np.int64)
        return self.states#[np.tile(self.species_labels,n)]

    def compute_states_with_physics(self, x, y, z, vx, vy, vz):
        """Assigne les états avec vitesse."""
        pos = np.column_stack([np.asarray(x), np.asarray(y), np.asarray(z)])
        vel = np.column_stack([np.asarray(vx), np.asarray(vy), np.asarray(vz)])
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        features = np.hstack([pos, speed * self.velocity_weight])

        X = (features - self._mean) / self._std
        _, indices = self._tree.query(X)
        n=int(len(x)/self.PARTICLE_NUMBER)
        self.states=indices.astype(np.int64)
        return self.states#[np.tile(self.species_labels,n)]

    def _save_data(self, path):
        np.save(os.path.join(path, "centroids.npy"), self._centroids)
        np.save(os.path.join(path, "mean.npy"), self._mean)
        np.save(os.path.join(path, "std.npy"), self._std)
        with open(os.path.join(path, "physics_params.json"), "w") as f:
            json.dump({"n_features": self._n_features}, f)

    def _load_data(self, path):
        from scipy.spatial import cKDTree

        self._centroids = np.load(os.path.join(path, "centroids.npy"))
        self._mean = np.load(os.path.join(path, "mean.npy"))
        self._std = np.load(os.path.join(path, "std.npy"))
        self._tree = cKDTree(self._centroids)
        self._n_cells = len(self._centroids)
        with open(os.path.join(path, "physics_params.json")) as f:
            self._n_features = json.load(f)["n_features"]

    def _get_cell_polygons_2d(self, view='xy'):
        pos_centroids = self._centroids[:, :3]
        if view == 'xy':
            pts_2d = pos_centroids[:, :2]
        elif view == 'yz':
            pts_2d = pos_centroids[:, 1:]
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

    def _get_cell_polyhedra_3d(self):
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
        y_split: float = None,
        y_split_mode: str = "quantile",
        n_cells_top: int = 1,
        top_method: str = "single",
        top_kwargs: dict = None,
        bottom_method: str = "cylindrical",
        bottom_kwargs: dict = None,
    ):
        # super().__init__()
        self.y_split_input = y_split
        self.y_split_mode = y_split_mode
        self.n_cells_top_target = n_cells_top
        self.top_method = top_method
        self.top_kwargs = top_kwargs or {}
        self.bottom_method = bottom_method
        self.bottom_kwargs = bottom_kwargs or {}
        
        # Calculés au fit
        self._y_split = None
        self._y_min = None
        self._y_max = None
        self._top_partitioner = None
        self._bottom_partitioner = None
        self._n_cells_top = None
        self._n_cells_bottom = None
    
    @property
    def n_cells(self):
        if self._n_cells_top is None or self._n_cells_bottom is None:
            return 0
        return self._n_cells_top + self._n_cells_bottom
    
    @property
    def label(self):
        """Propriété manquante nécessaire à l'instanciation"""
        return (
            f"adaptive_y_{self.bottom_method}"
            f"_top{self._n_cells_top}_bot{self._n_cells_bottom}"
            f"_split{self.y_split_input}_mode{self.y_split_mode}"
        )
    
    def fit(self, coordinates: np.ndarray):
        coordinates = np.asarray(coordinates)
        y = coordinates[:, 1]  # Utilisation de la coordonnée y
        
        self._y_min = y.min()
        self._y_max = y.max()
        
        # ── Déterminer y_split ──
        if self.y_split_mode == "quantile":
            quantile = self.y_split_input if self.y_split_input else 0.7
            self._y_split = np.quantile(y, quantile)
        elif self.y_split_mode == "absolute":
            if self.y_split_input is None:
                self._y_split = (self._y_min + self._y_max) / 2
            else:
                self._y_split = self.y_split_input
        else:
            raise ValueError(f"y_split_mode inconnu: {self.y_split_mode}")
        
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
            self._top_partitioner = None
            self._n_cells_top = 1
        else:
            self._top_partitioner = create_partitioner(
                self.top_method, **self.top_kwargs
            )
            if len(coords_top) > 0:
                self._top_partitioner.fit(coords_top)
            self._n_cells_top = self._top_partitioner.n_cells
    
    def compute_states(self, x, y, z):
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
            states[mask_bottom] = self._bottom_partitioner.compute_states(
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
        # n=int(len(x)/self.PARTICLE_NUMBER)
        self.states= states # Les méthode de découpage hybrides comme le adaptive et le multizone ne necessite pas l'application de masque car
        # elles appellent déjà d'autres méthode de computestate des classes de découpage qu'elle instancient.
        return self.states

    def _get_cell_polygons_2d(self, view='xy'):
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

    def _get_cell_polyhedra_3d(self):
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
        self,
        zones: list,
        y_mode: str = "absolute"
    ):
        # super().__init__()
        self.zones_config = zones
        self.y_mode = y_mode
        self._zones = []  # [(y_min, y_max, partitioner), ...]
        self._cell_offsets = []
        self._total_cells = 0
    
    @property
    def n_cells(self):
        return self._total_cells
    
    @property
    def label(self):
        methods = "_".join(z["method"] for z in self.zones_config)
        return f"multizone_{len(self.zones_config)}zones_{methods}"
    
    def fit(self, coordinates):
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
    
    def compute_states(self, x, y, z):
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
        # n=int(len(x)/self.PARTICLE_NUMBER)
        self.states= states
        return self.states
    
    def _save_data(self, path):
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
    
    def _load_data(self, path):
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

    def _get_cell_polygons_2d(self, view='xy'):
        results = []
        for zone_idx, (y_min, y_max, partitioner) in enumerate(self._zones):
            offset = self._cell_offsets[zone_idx]
            for state_id, pts in partitioner._get_cell_polygons_2d(view):
                results.append((state_id + offset, pts))
        return results

    def _get_cell_polyhedra_3d(self):
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def n_cells(self):
        return 1

    @property
    def label(self):
        return "single_cell"

    def fit(self, coordinates):
        return self

    def compute_states(self, x, y, z):
        self.states= np.zeros(len(np.asarray(x)), dtype=np.int64)
        return self.states

    def _get_cell_polygons_2d(self, view='xy'):
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

    def _get_cell_polyhedra_3d(self):
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
    "cartesian": CartesianPartitioner,
    "cylindrical": CylindricalPartitioner,
    "voronoi": VoronoiPartitioner,
    "quantile": QuantileGridPartitioner,
    "octree": OctreePartitioner,
    "physics": PhysicsAwarePartitioner,
    "adaptive": AdaptivePartitioner,      # ← nouveau
    "multizone": MultiZonePartitioner,     # ← nouveau
    "single": SingleCellPartitioner,       # ← nouveau
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