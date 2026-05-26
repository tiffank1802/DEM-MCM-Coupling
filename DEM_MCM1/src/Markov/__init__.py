from __future__ import annotations
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

# Optional imports
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    pv = None

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None

try:
    from stpyvista import stpyvista
    HAS_STPYVISTA = True
except ImportError:
    HAS_STPYVISTA = False
    stpyvista = None


try:
    from .analyze_results import MarkovAnalyzer
    from .bucket_io import (
        BUCKET_BASE,BUCKET_PREFIX,BUCKET_ID,
        get_api,get_fs
    )
    from .partitioners import(
        create_partitioner,
        REGISTRY,
)
    from .run_sweep import (
        get_configs,
        ExperimentConfig,
        HF_FOLDER,
        run_experiment,
    )
 
    from .utils import load_parquet_as_timestep_dict
except ImportError:
    from analyze_results import MarkovAnalyzer
    from bucket_io import(
        BUCKET_BASE,BUCKET_PREFIX,BUCKET_ID,
        get_api,get_fs
    )
    from partitioners import (
        
        create_partitioner,
        REGISTRY,
)
    from run_sweep import (
        get_configs,
        ExperimentConfig,
        HF_FOLDER,
        run_experiment,
    )
    from utils import load_parquet_as_timestep_dict

# Only define cache function if Streamlit is available
if HAS_STREAMLIT:
    @st.cache_data(show_spinner="Chargement des données DEM...")
    def _load_dem_data_cached() -> dict[int, pd.DataFrame]:
        """Fonction standalone pour bénéficier du cache Streamlit."""
        
        return load_parquet_as_timestep_dict(parquet_path=HF_FOLDER,fs=get_fs())
else:
    def _load_dem_data_cached() -> dict[int, pd.DataFrame]:
        """Fallback when Streamlit not available."""
        return load_parquet_as_timestep_dict(parquet_path=HF_FOLDER,fs=get_fs())

class Markov:
    """
    On choisit la classe de partitionnement qu'on veut attribuer au constructeur lors de l'instanciation de l'objet Markov
    Cette classe permet de:
        - charger les données (coordonnées, vitesses,...) des particules de toute la simulation DEM et les stocker en memoire pour l'acces des variables tout au long de la simulation
        - Créér des labels des particules sur un pas de temps de la DEM
        - Transformer les coordonnées de particules au format vtp(VTK PolyData) pour la visualisation avec pyvista
        - Calculer et stocker ces labels issus de la méthode de partitionnement lors de l'appel de compute state
        - Sur un même visuel, afficher les PolyData et les labels des particules
        - 
    Utilisation:
        - Cette classe permet de créér un objet BasePartitionner lors de l'instanciation avec par défaut un CartesianPartitioner
        - Elle utilise les paramètres par défaut définis dans le fichiers run_sweep de la méthode _get_defaut_kwargs
    """
    def __init__(self,
                 method: str="cartesian"


                 ) -> None:
        """
        Défini la méthode de partitionnement à adopter par défaut est le cartésien

        """
        self.method=method
        self.default_config=get_configs(method=method) # renvoie une liste des configurations par défaut normalement un nombre dont il faut se rassurer pour le choisir
        # dans ce cas nous choisissons juste la première configuration de la liste
        # on pourraàit aussi bien choisir la dernière ou n'importe la quelle de la liste des configurations
        
        self.partitioner=create_partitioner(self.default_config[0].method,**self.default_config[0].method_kwargs)

        self.datas:dict[int,pd.DataFrame]={0:pd.DataFrame()}
        self.coords:np.ndarray=np.empty((0,3))
        self.velocities:np.ndarray=np.empty((0,3))
        self.indices:list=[]
        self.vtp_states:pv.PolyData=pv.PolyData()
        self.states:np.ndarray=np.array([])
        self.fichiers_cibles:pd.Series[bool]=pd.Series()


    
    def load_dem_data(self):
        # Délègue à la fonction cachée — le résultat est réutilisé entre les reruns
        self.datas = _load_dem_data_cached()
        return self.datas
    def get_coords(
            self,
            indices: list=[250]
            ):
        if self.datas[0].empty:
            self.datas=self.load_dem_data()
        if not self.indices:
            self.indices=indices
        self.coords=self.datas[indices[0]][['coordinates:0','coordinates:1','coordinates:2']].to_numpy()
        return self.coords
    def get_velocities(
            self,
            indices: list=[250]
            ):
        if self.datas[0].empty:
            self.datas=self.load_dem_data()
        if not self.indices:
            self.indices=indices
        self.velocities=self.datas[indices[0]][['Velocity:0', 'Velocity:1','Velocity:2']].to_numpy()
        return self.velocities

    def get_states(self,indices=[250]):
        self.indices=indices
        if self.coords.shape[0]==0:
            self.coords=self.get_coords(self.indices)
        self.partitioner.fit(self.coords)
        self.states=self.partitioner.compute_states(self.coords[:,0],self.coords[:,1],self.coords[:,2])
        return self.states
    def get_vtp(self,indices=[250]):
        self.indices=indices
        if not self.states:
            _=self.get_states(self.indices)
        self.vtp_states=pv.PolyData(self.coords)
        self.vtp_states.point_data['partitions']=self.states
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Diameter']].to_numpy(),name='Diameter')
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Velocity:0', 'Velocity:1','Velocity:2']].to_numpy(),name='Velocity')
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Angular_velocity:0', 'Angular_velocity:1','Angular_velocity:2']].to_numpy(),name='Angular_Velocity')
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Orientation:0', 'Orientation:1', 'Orientation:2']].to_numpy(),name='Orientation')
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Collision_force:0','Collision_force:1', 'Collision_force:2']].to_numpy(),name='Collision_forces')
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Particle_Rank']].to_numpy(),name='Particle_Rank')
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Particle_Phase_ID']].to_numpy(),name='Particle_Phase_ID')
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Particle_ID']].to_numpy(),name='Particle_ID')
        self.vtp_states.point_data.set_array(data=self.datas[indices[0]][['Residence_Time']].to_numpy(),name='Residence_Time')
        return self.vtp_states
        
    def visualize(self):
        import streamlit as st
        st.subheader("Visualisation des particules")
        if self.vtp_states.is_empty:
            _=self.get_vtp()
        pv.start_xvfb()
        pv.OFF_SCREEN=True
        pv.global_theme.trame.jupyter_extension_enabled = False
        pl=pv.Plotter(window_size=[400,400],notebook=False)
        # # --- Glyphs : une sphère par particule, taille = Diameter ---
        sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)  # résolution basse = perf

        self.glyphs = self.vtp_states.glyph(
            geom=sphere,
            scale="Diameter",      # colonne qui pilote la taille
            orient=False,          # pas d'orientation des sphères
            factor=1.0,            # multiplicateur global si besoin d'ajuster
        )
        pl.add_mesh(self.glyphs,
                    scalars='partitions',
                    cmap="tab10",
                    show_scalar_bar=True,
                    label='Clipped',
                    # style='wireframe',
                    # show_edges=True,
                    )
        if st.checkbox("Coupe"):
            option=st.selectbox("Direction du plan",['xy','yz','xz','oblique'])

            pl.clear()
            if option=='xy':
                normal=(0,0,.1)
            elif option=='yz':
                normal=(.1,0,0)
            elif option=='xz':
                normal=(0,.1,0)
            else:
                normal=(.1,.1,0)
            plane=pv.Plane(i_size=30,j_size=30,direction=normal)
            self.crinkled=self.glyphs.clip(normal=normal,crinkle=True)
            pl.add_mesh(self.crinkled,
                    scalars='partitions',
                    cmap="tab10",
                    show_scalar_bar=True,
                    label='Clipped',
                    # style='wireframe',
                    # show_edges=True,
                    )
            pl.add_mesh(plane.extract_feature_edges(), color='r')

        pl.camera_position = pv.CameraPosition(
            position=(0.24, 0.32, 0.7),
            focal_point=(0.02, 0.03, -0.02),
            viewup=(-0.12, 0.93, -0.34),
        )
        return stpyvista(pl)


        
    
        
    """
    On a besoin de définir un objet qui construit et analyse la matrice de transition et les vecteurs d'état
    """
    analyzer=MarkovAnalyzer()

   




        

        

        
        
