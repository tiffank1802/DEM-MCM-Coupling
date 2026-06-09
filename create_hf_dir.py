import os
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import json
import pandas as pd

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

pv.OFF_SCREEN = True
LOCAL_OUTPUT = "/teamspace/studios/this_studio/MyStudio/outputs"

timestep_dict = load_parquet_as_timestep_dict(f'hf://buckets/{BUCKET_ID}/simulation_complete.parquet', fs)

# Find the path
dossiers = fs.ls(f"hf://buckets/{BUCKET_ID}/_Good/BIG")
path = ""
for dossier in dossiers:
    if all(j in dossier.get('name', '') for j in ["voronoi_30cells_NLT200_step50_dt2_tau100_start250_d0008"]):
        path = dossier['name']

# --- FIX: Extract clean name and setup directory helper ---
clean_name = os.path.basename(path) 
def ensure_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

# Load P and config
P = np.array([])
config = {}
with fs.open(f'{path}/transitionmatrix.npy', 'rb') as f:
    P = np.load(f)
    
with fs.open(f'{path}/config.json', 'r') as f:
    config = json.load(f)
    
partitioner = create_partitioner(config.get("method", ''), **config.get("method_kwargs", ""))

# --- FIX: Provide default integers to prevent np.arange errors ---
start = config.get("start_index", 0)
tau = config.get("tau", 50) 
tau_dem = 50

def screenshot(coords, states):
    mesh = pv.PolyData(coords)
    mesh.point_data['partitions'] = states
    mesh.point_data.set_array(data=timestep_dict[start][['Diameter']].to_numpy(), name='Diameter')
    
    sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
    glyph = mesh.glyph(geom=sphere, orient=False, factor=1.0, scale="Diameter")
    
    # --- FIX: Use clean_name and ensure_dir ---
    mesh_path = f'{LOCAL_OUTPUT}/fichiers/mesh_{clean_name}.vtp'
    ensure_dir(mesh_path)
    glyph.save(mesh_path)
    
    # ... (rest of your pyvista plotting code) ...

# --- FIX: Fit partitioner ONLY on the initial state ---
initial_coords = timestep_dict[start][['coordinates:0','coordinates:1','coordinates:2']].to_numpy()
partitioner.fit(initial_coords)
states = partitioner.compute_states(initial_coords[:,0], initial_coords[:,1], initial_coords[:,2])

# --- FIX: Use minlength to prevent array shape mismatches ---
S = np.bincount(states, minlength=P.shape[0]) 

fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax.imshow(P)
for i in range(P.shape[0]):
    for j in range(P.shape[0]):
        ax.text(j, i, np.round(P[i,j], 3), ha='center', va='center', fontsize=5)

ax.set_title(f'{clean_name}')
fig.tight_layout()

# --- FIX: Save transition matrix plot safely ---
img_path = f"{LOCAL_OUTPUT}/images/transitions/transition_{clean_name}.png"
ensure_dir(img_path)
plt.savefig(img_path)

print(S)
screenshot(initial_coords, states)

states_history = []
states_dem_history = []

# Markov evolution
for i in range(start, 6000, tau):
    states_history.append(S)
    S = P @ S

# DEM evolution
for i in range(start, 6000, tau_dem):
    # --- FIX: Use timestep_dict[i] instead of datas[i] ---
    coords = timestep_dict[i][['coordinates:0','coordinates:1','coordinates:2']].to_numpy()
    
    # --- FIX: DO NOT re-fit the partitioner here! Just compute states. ---
    # partitioner.fit(coords) 
    states = partitioner.compute_states(coords[:,0], coords[:,1], coords[:,2])
    
    # --- FIX: Use minlength ---
    states_dem_history.append(np.bincount(states, minlength=P.shape[0]))
    
states_dem_history = np.array(states_dem_history)
states_history = np.array(states_history)
times_markov = np.arange(start, 6000, tau)
times_dem = np.arange(start, 6000, tau_dem)

fig, ax = plt.subplots()
n = states_history.shape[1] 

for i in range(n):
    ax.plot(times_markov, states_history[:,i], "o-", label=f"S {i} Markov")
    ax.plot(times_dem, states_dem_history[:,i], "o-", label=f"S {i} DEM")
    ax.set_ylabel("Nombre de particules")
    ax.set_xlabel("Temps en centiemes de secondes")
    fig.suptitle("Comparaison Markov DEM", fontsize=7)
    ax.set_title(f'Etats au cours du temps {clean_name}', fontsize=8)
    fig.tight_layout()
    plt.legend()
    
    # --- FIX: Save plots and text files safely ---
    img_path = f'{LOCAL_OUTPUT}/images/etats/etats_comparaison_{i}_{clean_name}.png'
    ensure_dir(img_path)
    plt.savefig(img_path)
    
    txt_path = f'{LOCAL_OUTPUT}/fichiers/transitions/transition_{i}_{clean_name}.txt'
    ensure_dir(txt_path)
    np.savetxt(txt_path, states_history)
    
    ax.clear()

# RSD Plot
fig, ax = plt.subplots()
rsd_markov = states_history.std(axis=1) / states_history.mean(axis=1)
rsd_dem = states_dem_history.std(axis=1) / states_dem_history.mean(axis=1)

fig.suptitle("Comparaison DEM vs Markov", fontsize=7)
ax.plot(times_markov, rsd_markov, label="Markov")
ax.plot(times_dem, rsd_dem, label="DEM")
ax.set_xlabel("Temps en centiemes de secondes")
ax.set_title(f'RSD {clean_name}', fontsize=8)
fig.tight_layout()
plt.legend()

# --- FIX: Save RSD plot safely ---
rsd_path = f'{LOCAL_OUTPUT}/images/rsd/rsd_{clean_name}.png'
ensure_dir(rsd_path)
plt.savefig(rsd_path)