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
import json
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
states_history=[S]
for _ in range(start,6000,tau):
    S=S@P
    states_history.append(S)

states_history=np.array(states_history)
fig,ax=plt.subplots()
ax.plot(states_history[:,0],label=f"S0")
ax.plot(states_history[:,1],label=f"S1")
ax.plot(states_history[:,2],label=f"S2")
ax.set_title("Etats des partitions au cours du temps")
fig.tight_layout()
plt.legend()
plt.savefig(f'{LOCAL_OUTPUT}/images/etats_{path.replace("buckets/ktongue/DEM_MCM/Experiment/","")}.png')

np.savetxt(f'{LOCAL_OUTPUT}/fichiers/transition_{path.replace("buckets/ktongue/DEM_MCM/Experiment/","")}.txt',states_history)

# with fs.open(f"{BUCKET_BASE}/{BUCKET_PREFIX}/.keep" ,"w") as f:
#     f.write("")
   
