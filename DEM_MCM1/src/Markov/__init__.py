from __future__ import annotations
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import pyvista as pv
# import streamlit as st
from stpyvista import stpyvista 
import threading
import uvicorn
from fastapi import FastAPI
import subprocess

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
        HF_FOLDER,
    )
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
        HF_FOLDER,
    )


api = FastAPI()
SHARED_DATA = {"mesh": None}

@api.get("/get_mesh")
def get_mesh():
    # Streamlit viendra chercher le mesh ici en mémoire
    # On le convertit en dictionnaire VTK pour le transfert réseau rapide
    return {"points": SHARED_DATA["mesh"].points.tolist()}
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

        self.datas:pd.DataFrame=pd.DataFrame()
        self.coords:np.ndarray=np.empty((0,3))
        self.velocities:np.ndarray=np.empty((0,3))
        self.indices:list=[]
        self.vtp_states:pv.PolyData=pv.PolyData()
        self.states:np.ndarray=np.empty(1030)
        self.fichiers_cibles:pd.Series[bool]=pd.Series()



    def load_dem_data(self):
        fs=get_fs()
        with fs.open(HF_FOLDER, "rb") as fh:
            # 1. On ouvre le fichier avec PyArrow sans charger les données en mémoire
            fichier_parquet = pq.ParquetFile(fh)
            
            # 2. On récupère le nombre total de groupes de lignes (Row Groups)
            nb_groupes = fichier_parquet.num_row_groups
            
            list_dfs = []
        
            # 3. On configure tqdm sur le nombre de groupes à lire
            with tqdm(total=nb_groupes, desc='Chargement du fichier (Groupes)', unit='bloc') as bar_progression:
                for i in range(nb_groupes):
                    # Lecture du groupe i et conversion immédiate en DataFrame Pandas
                    df_groupe = fichier_parquet.read_row_group(i).to_pandas()
                    list_dfs.append(df_groupe)
                    
                    # On met à jour la barre de progression d'une unité
                    bar_progression.update(1)
                    
            # 4. On fusionne tous les morceaux en un seul DataFrame final
            self.datas = pd.concat(list_dfs, ignore_index=True)
        return self.datas
    def get_coords(
            self,
            indices: list=[250]
            ):
        if self.datas.empty:
            self.datas=self.load_dem_data()
        self.indices=indices
        self.fichiers_cibles=self.datas['Fichier_Source'].isin([f'data_{indice}.csv' for indice in self.indices])
        self.coords=self.datas.loc[self.fichiers_cibles ,['coordinates:0','coordinates:1','coordinates:2']].to_numpy()
        return self.coords
    def get_velocities(
            self,
            indices: list=[250]
            ):
        if self.datas.empty:
            self.datas=self.load_dem_data()
        self.indices=indices
        self.fichiers_cibles=self.datas['Fichier_Source'].isin([f'data_{indice}.csv' for indice in self.indices])
        self.velocities=self.datas.loc[self.fichiers_cibles ,['Velocity:0', 'Velocity:1','Velocity:2']].to_numpy()
        return self.velocities
    def get_states(self):
        if self.coords.shape[0]==0:
            self.coords=self.get_coords()
        self.partitioner.fit(self.coords)
        self.states=self.partitioner.compute_states(self.coords[:,0],self.coords[:,1],self.coords[:,2])
        return self.states
    def get_vtp(self):
        if not self.indices:
            _=self.get_states()
        self.vtp_states=pv.PolyData(self.coords)
        self.vtp_states.point_data['partitions']=self.states
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Diameter']].to_numpy(),name='Diameter')
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Velocity:0', 'Velocity:1','Velocity:2']].to_numpy(),name='Velocity')
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Angular_velocity:0', 'Angular_velocity:1','Angular_velocity:2']].to_numpy(),name='Angular_Velocity')
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Orientation:0', 'Orientation:1', 'Orientation:2']].to_numpy(),name='Orientation')
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Collision_force:0','Collision_force:1', 'Collision_force:2']].to_numpy(),name='Collision_forces')
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Particle_Rank']].to_numpy(),name='Particle_Rank')
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Particle_Phase_ID']].to_numpy(),name='Particle_Phase_ID')
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Particle_ID']].to_numpy(),name='Particle_ID')
        self.vtp_states.point_data.set_array(data=self.datas.loc[self.fichiers_cibles,['Residence_Time']].to_numpy(),name='Residence_Time')
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
        pl.add_mesh(self.vtp_states)
        return stpyvista(pl)
        
    
   




        

        

        
        
