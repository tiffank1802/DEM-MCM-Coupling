import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from directory import(
    BUCKET_BASE,
    BUCKET_ID,
    BUCKET_PREFIX,
)
from huggingface_hub import HfFileSystem
fs=HfFileSystem()
from DEM_MCM1.src.partitioners import create_partitioner
from DEM_MCM1.src.utils import load_parquet_as_timestep_dict
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import json

pv.start_xvfb()
pv.OFF_SCREEN = True
LOCAL_OUTPUT="/teamspace/studios/this_studio/MyStudio/outputs"
datas=load_parquet_as_timestep_dict(f'hf://buckets/{BUCKET_ID}/simulation_complete.parquet',fs)


# dossiers=fs.ls(f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}")
dossiers=fs.ls(f"hf://buckets/{BUCKET_ID}/Experiment")
path=""
for dossier in dossiers:
    if all(j in dossier.get('name', '') for j in ["physics_30cells_withvel_vw0.5_NLT200_step50_dt2_tau150_start250"]):
        path=dossier['name']
P=np.array([])
config={}
with fs.open(f'{path}/transitionmatrix.npy','rb') as f:
    P=np.load(f)
    print(np.round(P.sum(axis=0),4))

with fs.open(f'{path}/config.json','r') as f:
    config=json.load(f)
    
partitioner=create_partitioner(config.get("method",''),**config.get("method_kwargs",""))

start=config.get("start_index","")
tau=config.get("tau","")
def screenshot(coords, states):
    mesh = pv.PolyData(coords)
    mesh.point_data['partitions'] = states
    mesh.point_data.set_array(data=datas[start][['Diameter']].to_numpy(), name='Diameter')
    
    sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
    glyph = mesh.glyph(geom=sphere, orient=False, factor=1.0, scale="Diameter")
    
    # Configuration du Plotter
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(glyph, scalars="partitions")
    
    # --- Configuration caméra initiale ---
    pl.view_xy()
    pl.camera.zoom(2)
    
    # --- ÉTAPE 1 : Capture de l'image fixe ---
    pl.screenshot(f"{LOCAL_OUTPUT}/images/mesh.png")
    
    # --- ÉTAPE 2 : Initialisation de la vidéo ---
    pl.open_movie(f"{LOCAL_OUTPUT}/images/mesh.mp4", framerate=30)
    
    # --- ÉTAPE 3 : Animation multi-axes (plus lente) ---
    n_frames = 180  # Double la durée pour un rendu plus fluide (6 sec à 30 fps)
    
    for i in range(n_frames):
        pl.camera.azimuth += 1.0    # Rotation autour de l'axe vertical (Z)
        pl.camera.elevation += 0.5  # Inclinaison haut/bas (axe horizontal)
        # pl.camera.roll += 0.2     # Décommente pour un léger effet "banjo"
        
        pl.render()                 # Force le rendu avant la capture
        pl.write_frame()
        
    pl.close()
coords=datas[start][['coordinates:0','coordinates:1','coordinates:2']].to_numpy()
partitioner.fit(coords)
states=partitioner.compute_states(coords[:,0],coords[:,1],coords[:,2])
S=np.bincount(states)
fig,ax=plt.subplots()
fig.set_size_inches(10,10)
ax.imshow(P)
for i in range(P.shape[0]):
    for j in range(P.shape[0]):
        ax.text(j,i,np.round(P[i,j],3),ha='center',va='center',fontsize=5)

ax.set_title(f'{path.replace("buckets/ktongue/DEM_MCM/Experiments/","")}')
fig.tight_layout()
plt.savefig(f"{LOCAL_OUTPUT}/images/transition_{path.replace("buckets/ktongue/DEM_MCM/Experiment/","")}.png")
print(S)
screenshot(coords,states)
states_history=[]
states_dem_history=[]
for i in range(start,6000,tau):
    coords=datas[i][['coordinates:0','coordinates:1','coordinates:2']].to_numpy()
    partitioner.fit(coords)
    states=partitioner.compute_states(coords[:,0],coords[:,1],coords[:,2])
    states_dem_history.append(np.bincount(states))
    S=S@P
    states_history.append(S)
states_dem_history=np.array(states_dem_history)
states_history=np.array(states_history)
times=np.arange(start,6000,tau)
fig,ax=plt.subplots()
# ax.plot(times,states_history[:,0],"o-",label=f"S0")
# ax.plot(times,states_history[:,1],"o-",label=f"S1")
# ax.plot(times,states_history[:,2],"o-",label=f"S2")
# ax.plot(times,states_history[:,-1],"o-",label=f"S-1")
n=states_history.shape[0] # à vérifier c'est un vecteur ligne ou un vecteur colonne
for i in range(n):

    ax.plot(times,states_history[:,i],"o-",label=f"S {i} Markov")
    ax.plot(times,states_dem_history[:,i],"o-",label=f"S {i} DEM")
    ax.set_ylabel("Nombre de particules")
    ax.set_xlabel("Temps en centiemes de secondes")
    fig.suptitle("Comparaison Markov DEM",fontsize=7)

    ax.set_title(f'Etats au cours du temps {path.replace("buckets/ktongue/DEM_MCM/Experiment/","")}',fontsize=8)
    fig.tight_layout()
    plt.legend()
    plt.savefig(f'{LOCAL_OUTPUT}/images/etats_comparaison_{i}_{path.replace("buckets/ktongue/DEM_MCM/Experiment/","")}.png')
    ax.clear()
    np.savetxt(f'{LOCAL_OUTPUT}/fichiers/transition_{i}_{path.replace("buckets/ktongue/DEM_MCM/Experiment/","")}.txt',states_history)


fig,ax=plt.subplots()
rsd_markov=states_history.std(axis=1)/states_history.mean(axis=1)
rsd_dem=states_dem_history.std(axis=1)/states_dem_history.mean(axis=1)
fig.suptitle("Comparaison DEM vs Markov",fontsize=7)
ax.plot(times,rsd_markov,label="Markov")
ax.plot(times,rsd_dem,label="DEM")
ax.set_xlabel("Temps en centiemes de secondes")
ax.set_title(f'RSD {path.replace("buckets/ktongue/DEM_MCM/Experiment/","")}',fontsize=8)
fig.tight_layout()
fig.tight_layout()
plt.legend()
plt.savefig(f'{LOCAL_OUTPUT}/images/rsd_{path.replace("buckets/ktongue/DEM_MCM/Experiment/","")}.png')



# with fs.open(f"{BUCKET_BASE}/{BUCKET_PREFIX}/.keep" ,"w") as f:
#     f.write("")



