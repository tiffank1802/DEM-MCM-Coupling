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
import pandas as pd
pv.start_xvfb()
pv.OFF_SCREEN = True
LOCAL_OUTPUT="/teamspace/studios/this_studio/MyStudio/outputs"
dossiers=fs.ls(f"hf://buckets/{BUCKET_ID}/_Good/Experiment")

path=""
for dossier in dossiers:
    if all(j in dossier.get('name', '') for j in ["voronoi_30cells_NLT200_step50_dt2_tau100_start250"]):
        path=dossier['name']
P=np.array([])
states=np.array([])
config={}

with fs.open(f'{path}/transitionmatrix.npy','rb') as f:
    P=np.load(f)
    print(np.round(P.sum(axis=0),4))

with fs.open(f'{path}/states.npy','rb') as f:
    states=np.load(f)
    print(states.shape)

with fs.open(f'{path}/config.json','r') as f:
    config=json.load(f)

start=config.get("start_index","")
tau=config.get("tau","")
tau_dem=50

states_history=[] # vecteur de l'historique des états des particules
S=np.bincount(states[start]) # comptage du nombre de particules par partitions

for i in range(start,6000,tau):
    states_history.append(S)
    S=S@P
states_history=np.array(states_history)
# Visualisation du nombre de particules par partition au cours du temps

times_markov=np.arange(start,6000,tau)
times_dem=np.arange(start,6000,tau_dem)
fig,ax=plt.subplots()
count_states=np.bincount(states[0,:],minlength=30)
for i in range(1,states.shape[0]):
    count_states=np.vstack((count_states,np.bincount(states[i,:],minlength=30)))

n=states_history.shape[1] # à vérifier c'est un vecteur ligne ou un vecteur colonne
for i in range(n):

    ax.plot(times_markov,states_history[:,i],"o-",label=f"S {i} Markov")
    ax.plot(times_dem,count_states[start::tau_dem,i],"o-",label=f"S {i} DEM")
    ax.set_ylabel("Nombre de particules")
    ax.set_xlabel("Temps en centiemes de secondes")
    fig.suptitle("Comparaison Markov DEM",fontsize=7)

    ax.set_title(f'Etats au cours du temps {path.replace("buckets/ktongue/DEM_MCM/_Good/Experiment/","")}',fontsize=8)
    fig.tight_layout()
    plt.legend()
    plt.savefig(f'{LOCAL_OUTPUT}/images/etats/etats_comparaison_{i}_{path.replace("buckets/ktongue/DEM_MCM/_Good/Experiment/","")}.png')
    ax.clear()
    # np.savetxt(f'{LOCAL_OUTPUT}/fichiers/transitions/transition_{i}_{path.replace("buckets/ktongue/DEM_MCM/_Good/BIG/","")}.txt',states_history)

fig,ax=plt.subplots()
rsd_markov=states_history.std(axis=1)/states_history.mean(axis=1)
rsd_dem=count_states[start::tau_dem,:].std(axis=1)/count_states[start::tau_dem,:].mean(axis=1)
fig.suptitle("Comparaison DEM vs Markov",fontsize=7)
ax.plot(times_markov,rsd_markov,label="Markov")
ax.plot(times_dem,rsd_dem,label="DEM")
ax.set_xlabel("Temps en centiemes de secondes")
ax.set_title(f'RSD {path.replace("buckets/ktongue/DEM_MCM/_Good/Experiment/","")}',fontsize=8)
fig.tight_layout()
fig.tight_layout()
plt.legend()
plt.savefig(f'{LOCAL_OUTPUT}/images/rsd/rsd_{path.replace("buckets/ktongue/DEM_MCM/_Good/Experiment/","")}.png')
