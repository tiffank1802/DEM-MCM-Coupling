"""
==========================================================================
PARTITIONERS- Méthode de partitionnement spatial pour chaines de Markov
==========================================================================

Interface commune:
    partiotioner=create_partioner("voronoi", n_cells=125)
    partitioner.fit(coordinates)                # (N ,3) numpy array
    states=partitioner.compute_states(x, y, z)  # -> indices int64
    partioner.save("outpout/")
    partitioner.load("output/")
    
    Méthodes disponibles:
        cartésian   - grille régulière (x, y, z)
        cylindrical - grille cylindrique(r, ø, z)
        voronoi     - clustering K-means / cellules de Voronoï
        quantile    -grille avec bords par quantiles (équi-population)
        octree      -octree adaptatif à la densité
        physics     - K-means sur position + champs physiques
==========================================================================
"""

from typing import override
import numpy as np 
import os 
import json
from abc import ABC, abstractmethod

__all__=[
    "BasePartitioner",
    "CartesianPartitioner"
    "CylindricalPartitioner",
    "VoronoiPartitioner",
    "QuantilePartitioner",
    "OctreePartitioner",
    "PhysicsAwarePartitioner",
    "create_partitioner",
    "REGISTER",
]

#==========================================================================
# CLASSE DE BASE
#==========================================================================
class BasePartitioner(ABC):

    """Interface commune pour tous les partitionneurs"""

    @property
    @abstractmethod
    def n_cells(self):
        """Nombre total d'états"""
        ...
    @property
    @abstractmethod
    def label(self):
        """Identifiant unique (utilisé pour le nom de dossier)."""
        ...
    @abstractmethod
    def fit(self,coordinates):
        """
        Apprend le partitionnement sur les données représentatives.
        

        Args:
            coordinates (np.ndarray): shape (N,3)
        Returns:
            self
        """
        ...
    @abstractmethod
    def save(self,path):
        """Sauvegarde le partitionnement dans un dossier."""
        os.makedirs(path,exist_ok=True)
        meta={
            "type":type(self).__name__,
            "label":self.label,
            "n_cells":self.n_cells,
            
        }
        with open(os.path.join(path, "partitioner_meta.json"),"w")as f:
            json.dump(meta,f,indent=2)
        self._save_data(path)
    def _save_data(self,path):
        pass
    def load(self,path):
        """Charge le partitionneur depuis un dossier."""
        self._load_data(path)
        return self
    def _load_data(self,path):
        pass
    def diagnostics(self,coordinates):
        """
        Statistiques de population par cellule.
        
        Args:
            coordinates: np.ndarray (N,3)
        Return:
                dict avec min, max, mean, std, n_empty
        """
        states=self.compute_states(
            coordinates[:,0],coordinates[:,1],coordinates[:,2]
        )
        counts=np.bincount(states,minlength=self.n_cells)
        return{
            "pop_min":int(counts.min()),
            "pop_max":int(counts.max()),
            "pop_mean":float(counts.mean()),
            "pop_std":float(counts.std()),
            "n_empty":int((counts==0).sum()),
            "n_visited":int((counts>0).sum()),
            "fraction_visited":float((counts>0).sum()/self.n_cells),
        }
    
    
#==========================================================================
# 1.CARTÉSIEN
#==========================================================================
class CartesianPartitioner(BasePartitioner):
    """
    Grille cartésienne régulière.
    
    Découpe le domaine en nx•ny•nz cellules de taille égale.
    Simple mais inadapté aux géométries cylindriques (couns vides).

    Args:
        BasePartitioner (_type_): _description_
    """
    def __init__(self,nx=5,ny=5,nz=5) -> None:
        super().__init__()
        self.nx=nx
        self.ny=ny
        self.nz=nz
        self._bounds=None
        
    @property
    @override
    def n_cells(self):
        return self.nx*self.ny*self.nz

    @property
    @override
    def label(self):
        return f"cartesian_nx{self.nx}_ny{self.ny}_nz{self.nz}"
    
    def fit(self,coordinates):
        eps=1e-3
        mins=coordinates.min(axis=0)-eps
        maxs=coordinates.max(axis=1)+eps
        self._bounds=(mins[0],maxs[0],mins[1],maxs[1],mins[2],maxs[2])
        return self
        
    def compute_states(self,x,y,z):
        """"Cette fonction permet de determiner l'état de la particule: la partition dans laquelle la particule reside."""
        # convertion des coordonnées en tableaux numpy
        x=np.asarray(x,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)
        z=np.asarray(z,dtype=np.float64)
        xmin,xmax,ymin,ymax,zmin,zmax=self._bounds
        ix=np.clip(
            ((x-xmin)*self.nx/(xmax-xmin)).astype(np.int64),0,self.nx-1 # attribut une partition suivant l'axe des abcisses à chacune des particules
            # la fonction clip permet de normaliser la position de la particule dans l'ensemble des partitions
        ) 
        iy=np.clip(
            ((y-ymin)*self.ny/(ymax-ymin)).astype(np.int64),0,self.ny-1
            
        )
        iz=np.clip(
            ((z- zmin)*self.nz/(zmax-zmin)).astype(np.int64),0,self.nz-1
            
        )
    
    @override
    def _save_data(self, path):
        np.save(os.path.join(path,"bounds.npy"),np.array(self._bounds))
    
    @override
    def _load_data(self, path):
        self._bounds=tuple(np.load(os.path.join(path,"bounds.npy")))

#==========================================================================
# 1.CYLINDRIQUE
#==========================================================================


class CylindricalPartitioner(BasePartitioner):
    """
    Grille cylindrique (r,œ,z).
    
    Idéal pour les mélangeurs à symétrie axiale.
    Deux modes radiaux:
        -"équal_dr" :delta_r constant
        -"equal_area:aire de section constant
    Avec ntheta=1-> partitionnement purement axisymetrique.

    """
    
    def __init__(self,nr=5,ntheta=8,radial_mode="equal_erea") -> None:
        self.nr=nr
        self.ntheta=ntheta
        self.nz=self.nz
        self.radial_mode=radial_mode
        self._x_center=None
        self._y_center=None
        self._r_max=None
        self._z_min=None
        self._z_max=None
        self._r_edges=None
    
    @property
    def n_cells(self):
        return self.nr*self.ntheta*self.nz
    
    @property
    def label(self):
        return {
            f"cylindrical_nr{self.nr}_nth{self.ntheta}"
            f"_nz{self.nz}_{self.radial_mode}"
        }
    
    def fit(self,coordinates:np.ndarray):
        eps=1e-3
        x,y,z=coordinates[:,0],coordinates[:,1],coordinates[:,2]
        # Repère les coordonnées des centres des particules
        self._x_center=(x.min()+x.max())/2
        self._y_center=(y.min()+y.max())/2
        
        r=np.sqrt((x-self._x_center)**2+(y-self._y_center)**2)
        self._r_max=r.max()+eps
        self._z_min=z.min()-eps
        self._z_max=z.max()+eps
        
        
        if self.radial_mode=="equal_area":
            
            self._r_edges=self._r_max*np.sqrt(np.linspace(0,1,self.nr +1))    
        elif self.radial_mode=="equal_dr":
            self._r_edges=np.linspace(0,self._r_max,self.nr+1)
        else:
                raise ValueError(f"radial_mode inconnu: {self.radial_mode}")
        
        return self
    
    
    def compute_states(self,x,y,z):
        x=np.asarray(x,dtype=np.float64)
        y=np.asarray(y,dtype=np.float64)
        z=np.asarray(z,dtype=np.float64)
        
        dx=x-self._x_center
        dy=y-self._y_center
        r=np.sqrt(dx**2+dy**2)
        theta=(np.arctan2(dy,dx)+2*np.pi)%(2*np.pi) # [0, 2*np.pi]
        
        ir=np.clip(
            np.searchsorted(self._r_edges,r,side="right")-1,0, self.nr-1
        )
        itheta=np.clip(
            (theta*self.ntheta/(2*np.pi))
        )
        
        
        

