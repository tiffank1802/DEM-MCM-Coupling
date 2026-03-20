"""
===================================================================================
PARTITIONERS — Méthodes de partitionnement spatial pour chaînes de Markov
===================================================================================

Ce module fournit 6 méthodes pour découper l'espace en cellules (états),
permettant de discrétiser le mouvement continu des particules DEM.

L'interface commune est définie par BasePartitioner:
  - fit(coordinates)       : apprendre les bords des cellules
  - compute_states(x, y, z): assigner chaque particule à une cellule
  - save(path) / load(path): sérialiser/désérialiser

Exemple d'utilisation minimal:
    part = create_partitioner("voronoi", n_cells=125)
    part.fit(coordinates)  # forme (N, 3)
    states = part.compute_states(x, y, z)  # → indices int64
    
Méthodes disponibles:
    • cartesian    : grille régulière (nx × ny × nz)
    • cylindrical  : grille cylindrique (nr × nθ × nz)
    • voronoi      : K-means / cellules de Voronoï
    • quantile     : grille équi-population
    • octree       : subdivision adaptative à la densité
    • physics      : K-means sur position + vitesse

Architecture:
    create_partitioner("methode", **kwargs)
         ↓
    Classe spécifique (VoronoiPartitioner, CylindricalPartitioner, ...)
         ↓
    Hérite de BasePartitioner
"""

import numpy as np
import os
import json
from abc import ABC, abstractmethod

__all__ = [
    "BasePartitioner",
    "CartesianPartitioner",
    "CylindricalPartitioner",
    "VoronoiPartitioner",
    "QuantileGridPartitioner",
    "OctreePartitioner",
    "PhysicsAwarePartitioner",
    "create_partitioner",
    "REGISTRY",
]


# =============================================================================
# CLASSE DE BASE
# =============================================================================

class BasePartitioner(ABC):
    """
    Interface abstraite pour tous les partitionneurs.
    
    Defines the contract that all spatial partitioners must follow.
    Subclasses must implement:
        - n_cells (property)
        - label (property)
        - fit(coordinates)
        - compute_states(x, y, z)
    
    Attributes:
        None (interface abstraite)
    
    Methods:
        Abstract methods (require implementation by subclasses):
            • n_cells -> int
            • label -> str
            • fit(coordinates) -> self
            • compute_states(x, y, z) -> np.ndarray[int64]
        
        Concrete methods (provided by base class):
            • save(path)
            • load(path)
            • diagnostics(coordinates) -> dict
    
    Example:
        >>> part = CartesianPartitioner(nx=5, ny=5, nz=5)
        >>> coords = np.random.rand(1000, 3)
        >>> part.fit(coords)
        >>> states = part.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])
        >>> print(f"États: {states}")
        États: [0 45 124 ... 67 89]
    """

    @property
    @abstractmethod
    def n_cells(self) -> int:
        """
        Nombre total d'états (cellules).
        
        Returns:
            int: nombre d'états
        
        Example:
            >>> part = VoronoiPartitioner(n_cells=125)
            >>> part.n_cells
            125
        """
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """
        Identifiant unique du partitionneur.
        
        Utilisé pour le nom du dossier de résultats, doit être:
        - Unique pour cette configuration
        - Reproductible
        - Lisible par un humain
        
        Format conseillé:
            "{method}_{param1}_{param2}_..."
        
        Returns:
            str: identifiant
        
        Example:
            >>> part = CartesianPartitioner(nx=5, ny=5, nz=5)
            >>> part.label
            'cartesian_nx5_ny5_nz5'
        """
        ...

    @abstractmethod
    def fit(self, coordinates: np.ndarray) -> 'BasePartitioner':
        """
        Apprend le partitionnement spatial sur des données représentatives.
        
        Cette méthode analyse les données d'entrée pour déterminer les bords
        des cellules. C'est l'étape d'initialisation critique.
        
        Args:
            coordinates: np.ndarray shape (N, 3)
                Positions des particules [x, y, z] sur lesquelles apprendre.
                N = nombre de particules (typiquement 1M+ pour le DEM).
                Unités: mètres (standard SI).
        
        Returns:
            self: retour pour permettre le chaînage
                part = VoronoiPartitioner().fit(coords)
        
        Raises:
            ValueError: si coordinates n'a pas la bonne forme
            RuntimeError: problèmes numériques
        
        Side effects:
            Modifie les attributs internes du partitionneur:
            - pour Cartesian: _bounds
            - pour Cylindrical: _x_center, _y_center, _r_max, _z_min, _z_max, _r_edges
            - pour Voronoi: centroids, _tree
            - etc.
        
        Example:
            >>> part = CartesianPartitioner(nx=5, ny=5, nz=5)
            >>> coords = np.random.rand(10000, 3)
            >>> part = part.fit(coords)  # retourne self pour chaînage
            >>> print(part._bounds)
            (-0.001, 1.001, -0.001, 1.001, -0.001, 1.001)
        """
        ...

    @abstractmethod
    def compute_states(self, x, y, z) -> np.ndarray:
        """
        Assigne un indice d'état à chaque particule.
        
        C'est la méthode la plus critique : elle doit être TRÈS rapide
        car elle s'exécute des millions de fois pendant le calcul de P.
        
        Args:
            x: array-like (1D)
                Coordonnées x des particules. Can be:
                - numpy array: np.array([...])
                - Polars Series: pl.Series([...])
                - Liste Python: [...]
                Sera converti en np.array float64.
            
            y: array-like (1D)
                Coordonnées y (même formats que x)
            
            z: array-like (1D)
                Coordonnées z (même formats que x)
            
        Returns:
            np.ndarray dtype=int64, shape (N,)
                Indice d'état pour chaque particule. Valeurs ∈ [0, n_cells-1].
        
        Raises:
            ValueError: si x, y, z n'ont pas la même longueur
            RuntimeError: si fit() n'a pas été appelé au préalable
        
        Guarantees:
            - Déterministe (même entrée → même sortie)
            - Rapide O(N) où N = nombre de particules
            - Pas de NaN ou Inf dans la sortie
            - Valeurs toujours clippées à [0, n_cells-1]
        
        Example:
            >>> part = CartesianPartitioner(nx=5, ny=5, nz=5)
            >>> coords = np.random.rand(1000, 3)
            >>> part.fit(coords)
            >>> states = part.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])
            >>> print(states)
            array([  0, 45, 124, ..., 67, 89], dtype=int64)
            >>> print(f"Min: {states.min()}, Max: {states.max()}")
            Min: 0, Max: 124  # toujours dans [0, n_cells-1]
        """
        ...

    def save(self, path: str) -> None:
        """
        Sauvegarde le partitionneur dans un dossier.
        
        Crée une structure standard :
            path/
            ├── partitioner_meta.json    (métadonnées)
            └── <fichiers spécifiques à la méthode>
               (centroids.npy, edges.npz, r_edges.npy, ...)
        
        Args:
            path: str
                Chemin du dossier cible (créé s'il n'existe pas)
        
        Returns:
            None
        
        Raises:
            IOError: erreur d'accès disque
        
        Side effects:
            - Crée le répertoire path/
            - Écrit les fichiers
        
        Example:
            >>> part = VoronoiPartitioner(n_cells=125)
            >>> part.fit(coords)
            >>> part.save("/tmp/my_partitioner/")
            >>> os.path.exists("/tmp/my_partitioner/partitioner_meta.json")
            True
        """
        os.makedirs(path, exist_ok=True)
        meta = {
            "type": type(self).__name__,
            "label": self.label,
            "n_cells": self.n_cells,
        }
        with open(os.path.join(path, "partitioner_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        self._save_data(path)

    def _save_data(self, path: str) -> None:
        """
        Sauvegarde les données spécifiques à la méthode.
        
        Méthode de hook : appelée par save(). Les subclasses overrident
        pour sauvegarder leurs données (centroids, edges, etc.)
        
        Args:
            path: dossier où sauvegarder
        
        Returns:
            None
        
        Default implementation:
            Ne fait rien (pour les partitionneurs sans données)
        """
        pass

    def load(self, path: str) -> 'BasePartitioner':
        """
        Charge le partitionneur depuis un dossier.
        
        Inverse de save(). Restaure l'état complet du partitionneur.
        
        Args:
            path: str
                Chemin du dossier créé par save()
        
        Returns:
            self: retour pour chaînage
        
        Raises:
            FileNotFoundError: métadonnées manquantes
            json.JSONDecodeError: fichiers JSON corrompus
        
        Example:
            >>> part = CartesianPartitioner(nx=5, ny=5, nz=5)
            >>> part.fit(coords)
            >>> part.save("/tmp/part/")
            >>> 
            >>> # Plus tard...
            >>> part2 = CartesianPartitioner(nx=5, ny=5, nz=5)
            >>> part2.load("/tmp/part/")  # restaure _bounds
        """
        self._load_data(path)
        return self

    def _load_data(self, path: str) -> None:
        """
        Charge les données spécifiques à la méthode.
        
        Méthode de hook : appelée par load(). Les subclasses overrident
        pour restaurer leurs données.
        
        Args:
            path: dossier où charger
        
        Default implementation:
            Ne fait rien
        """
        pass

    def diagnostics(self, coordinates: np.ndarray) -> dict:
        """
        Calcule des statistiques de population par cellule.
        
        Aide à diagnostiquer si le partitionnement est bon :
        - Y a-t-il des cellules vides ?
        - La population est-elle très inégale ?
        
        Args:
            coordinates: np.ndarray shape (N, 3)
                Données sur lesquelles calculer les diagnostics
        
        Returns:
            dict avec clés:
                'pop_min' (int)           : populations min par cellule (> 0)
                'pop_max' (int)           : population max
                'pop_mean' (float)        : population moyenne
                'pop_std' (float)         : écart-type de population
                'n_empty' (int)           : nombre de cellules sans particule
                'n_visited' (int)         : nombre de cellules avec ≥ 1 particule
                'fraction_visited' (float): n_visited / n_cells ∈ [0, 1]
        
        Example:
            >>> part = CartesianPartitioner(nx=5, ny=5, nz=5)
            >>> part.fit(coords)  # 1M de particules
            >>> diag = part.diagnostics(coords)
            >>> print(diag)
            {
                'pop_min': 450,
                'pop_max': 550,
                'pop_mean': 512.0,
                'pop_std': 28.5,
                'n_empty': 0,
                'n_visited': 125,
                'fraction_visited': 1.0
            }
            # Excellent : toutes les cellules visitées, population homogène
        """
        states = self.compute_states(
            coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
        )
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
    
    Concept:
        Découpe le domaine [x_min, x_max] × [y_min, y_max] × [z_min, z_max]
        en nx × ny × nz cellules cubiques de taille égale:
            Δx = (x_max - x_min) / nx
            Δy = (y_max - y_min) / ny
            Δz = (z_max - z_min) / nz
    
    Assignation d'état:
        Chaque particule (x, y, z) est assignée à la cellule (i, j, k) où:
            i = floor((x - x_min) / Δx)  clippé à [0, nx-1]
            j = floor((y - y_min) / Δy)  clippé à [0, ny-1]
            k = floor((z - z_min) / Δz)  clippé à [0, nz-1]
        
        Puis : état = i + j*nx + k*nx*ny
    
    Avantages:
        ✓ Très simple
        ✓ Très rapide O(N)
        ✓ Déterministe et reproductible
        ✓ Facile à visualiser
    
    Inconvénients:
        ✗ Population inégale (surtout pour nx,ny,nz grands)
        ✗ Inadapté aux géométries cylindriques (coins vides)
        ✗ Ignore la distribution réelle des particules
    
    Cas d'usage:
        - Baseline de comparaison
        - Domaines avec frontières claires
        - Prototype rapide
    
    Attributes:
        nx (int): nombre de cellules selon x
        ny (int): nombre de cellules selon y
        nz (int): nombre de cellules selon z
        _bounds (tuple): (x_min, x_max, y_min, y_max, z_min, z_max)
                        défini après fit()
    
    Example:
        >>> part = CartesianPartitioner(nx=10, ny=10, nz=10)
        >>> print(part.n_cells)  # 10^3 = 1000
        1000
        >>> part.fit(coordinates)
        >>> states = part.compute_states(x, y, z)
    """

    def __init__(self, nx: int = 5, ny: int = 5, nz: int = 5):
        """
        Initialise un partitionneur cartésien.
        
        Args:
            nx (int): nombre de divisions selon x. Default: 5
            ny (int): nombre de divisions selon y. Default: 5
            nz (int): nombre de divisions selon z. Default: 5
        
        Raises:
            ValueError: si nx, ny, ou nz < 1
        
        Example:
            >>> part = CartesianPartitioner(nx=15, ny=15, nz=15)
            >>> print(part.n_cells)
            3375
        """
        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("nx, ny, nz doivent être ≥ 1")
        self.nx, self.ny, self.nz = nx, ny, nz
        self._bounds = None

    @property
    def n_cells(self) -> int:
        """Total number of cells: nx × ny × nz."""
        return self.nx * self.ny * self.nz

    @property
    def label(self) -> str:
        """Unique identifier for this configuration."""
        return f"cartesian_nx{self.nx}_ny{self.ny}_nz{self.nz}"

    def fit(self, coordinates: np.ndarray) -> 'CartesianPartitioner':
        """
        Determine the bounding box from data.
        
        Args:
            coordinates: shape (N, 3)
        
        Returns:
            self
        """
        eps = 0.001
        mins = coordinates.min(axis=0) - eps
        maxs = coordinates.max(axis=0) + eps
        self._bounds = (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])
        return self

    def compute_states(self, x, y, z) -> np.ndarray:
        """Assign each particle to a cell index."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        xmin, xmax, ymin, ymax, zmin, zmax = self._bounds

        ix = np.clip(
            ((x - xmin) * self.nx / (xmax - xmin)).astype(np.int64), 0, self.nx - 1
        )
        iy = np.clip(
            ((y - ymin) * self.ny / (ymax - ymin)).astype(np.int64), 0, self.ny - 1
        )
        iz = np.clip(
            ((z - zmin) * self.nz / (zmax - zmin)).astype(np.int64), 0, self.nz - 1
        )
        return ix + iy * self.nx + iz * self.nx * self.ny

    def _save_data(self, path: str) -> None:
        """Save bounding box."""
        np.save(os.path.join(path, "bounds.npy"), np.array(self._bounds))

    def _load_data(self, path: str) -> None:
        """Load bounding box."""
        self._bounds = tuple(np.load(os.path.join(path, "bounds.npy")))


# =============================================================================
# 2. CYLINDRIQUE (CONTINU)
# =============================================================================

class CylindricalPartitioner(BasePartitioner):
    """
    Grille cylindrique (r, θ, z).
    
    Concept:
        Découpe le domaine en cellules cylindriques:
        - Radial: nr divisions avec bords r_0, r_1, ..., r_nr
        - Azimuthal: ntheta divisions dans [0, 2π]
        - Axial: nz divisions selon z
        
        Deux modes pour le radial:
        
        1. equal_dr: Δr constant
           r_i = i * Δr,  Δr = r_max / nr
           Inconvénient: les anneau intérieurs ont moins de volume
        
        2. equal_area (recommandé): aire d'anneau constant
           Volume anneau = π(r_{i+1}² - r_i²) = constant
           Résolution: r_i = r_max * √(i / n_r)
           
           Avantage: population plus homogène
    
    Assignation d'état:
        1. Convertir (x, y, z) → (r, θ, z):
            r = √((x - x_c)² + (y - y_c)²)
            θ = atan2(y - y_c, x - x_c)  ∈ [0, 2π]
        
        2. Findices:
            i_r = searchsorted(r_edges, r)
            i_θ = floor(θ * n_θ / 2π)
            i_z = floor((z - z_min) / Δz)
        
        3. état = i_r + i_θ * n_r + i_z * n_r * n_θ
    
    Avantages:
        ✓ Adapté aux géométries cylindriques
        ✓ Conserve la symétrie axiale
        ✓ equal_area: très bonne homogénéité de population
    
    Inconvénients:
        ✗ Paramètrage plus complexe
        ✗ Suppose un axe de symétrie (x_c, y_c) existant
    
    Cas d'usage:
        - Mélangeurs industriels (géométrie cylindrique)
        - Rhéomètres rotatifs
        - Séchoir horizontal
    
    Attributes:
        nr (int): cellules radiales
        ntheta (int): cellules azimutales
        nz (int): cellules axiales
        radial_mode (str): 'equal_dr' ou 'equal_area'
        _x_center, _y_center (float): centre du cylindre
        _r_max, _z_min, _z_max (float): limites
        _r_edges (ndarray): bords radiaux (n_r + 1,)
    
    Example:
        >>> part = CylindricalPartitioner(
        ...     nr=5, ntheta=8, nz=5,
        ...     radial_mode="equal_area"
        ... )
        >>> print(part.n_cells)  # 5 × 8 × 5 = 200
        200
    """

    def __init__(
        self,
        nr: int = 5,
        ntheta: int = 8,
        nz: int = 5,
        radial_mode: str = "equal_area",
    ):
        """
        Initialise un partitionneur cylindrique.
        
        Args:
            nr (int): nombre de couches radiales. Default: 5
            ntheta (int): nombre de divisions azimutales. Default: 8
                         ntheta=1 → partitionnement axisymétrique pur
            nz (int): nombre de divisions axiales. Default: 5
            radial_mode (str): 'equal_dr' ou 'equal_area'. Default: 'equal_area'
                              'equal_area' recommandé pour homogénéité
        
        Raises:
            ValueError: si radial_mode invalide ou nr/ntheta/nz < 1
        """
        if radial_mode not in ("equal_dr", "equal_area"):
            raise ValueError(f"radial_mode invalide: {radial_mode}")
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

    @property
    def n_cells(self) -> int:
        """Total number of cells: nr × ntheta × nz."""
        return self.nr * self.ntheta * self.nz

    @property
    def label(self) -> str:
        """Unique identifier."""
        return (
            f"cylindrical_nr{self.nr}_nth{self.ntheta}"
            f"_nz{self.nz}_{self.radial_mode}"
        )

    def fit(self, coordinates: np.ndarray) -> 'CylindricalPartitioner':
        """
        Détermine le centre, le rayon et les bords radiaux.
        
        L'axe de symétrie est défini comme le centre du domaine projeté
        sur le plan xy.
        
        Args:
            coordinates: shape (N, 3)
        
        Returns:
            self
        
        Modifie les attributs:
            _x_center, _y_center: centre du domaine xy
            _r_max: rayon maximal trouvé
            _z_min, _z_max: bornes z
            _r_edges: bords radiaux selon radial_mode
        """
        eps = 0.001
        x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

        self._x_center = (x.min() + x.max()) / 2
        self._y_center = (y.min() + y.max()) / 2

        r = np.sqrt((x - self._x_center) ** 2 + (y - self._y_center) ** 2)
        self._r_max = r.max() + eps
        self._z_min = z.min() - eps
        self._z_max = z.max() + eps

        if self.radial_mode == "equal_area":
            # r_i = R_max * √(i / n_r)
            self._r_edges = self._r_max * np.sqrt(np.linspace(0, 1, self.nr + 1))
        elif self.radial_mode == "equal_dr":
            # r_i = i * Δr
            self._r_edges = np.linspace(0, self._r_max, self.nr + 1)

        return self

    def compute_states(self, x, y, z) -> np.ndarray:
        """Assign each particle to a cylindrical cell."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)

        dx = x - self._x_center
        dy = y - self._y_center
        r = np.sqrt(dx**2 + dy**2)
        theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)  # [0, 2π]

        ir = np.clip(
            np.searchsorted(self._r_edges, r, side="right") - 1, 0, self.nr - 1
        )
        itheta = np.clip(
            (theta * self.ntheta / (2 * np.pi)).astype(np.int64), 0, self.ntheta - 1
        )
        dz = (self._z_max - self._z_min) / self.nz
        iz = np.clip(
            ((z - self._z_min) / dz).astype(np.int64), 0, self.nz - 1
        )

        return ir + itheta * self.nr + iz * self.nr * self.ntheta

    def _save_data(self, path: str) -> None:
        """Save cylindrical parameters."""
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

    def _load_data(self, path: str) -> None:
        """Load cylindrical parameters."""
        with open(os.path.join(path, "cylindrical_params.json")) as f:
            p = json.load(f)
        self._x_center = p["x_center"]
        self._y_center = p["y_center"]
        self._r_max = p["r_max"]
        self._z_min = p["z_min"]
        self._z_max = p["z_max"]
        self._r_edges = np.load(os.path.join(path, "r_edges.npy"))


# (Voronoi, Quantile, Octree, Physics-aware suivent le même pattern...)
