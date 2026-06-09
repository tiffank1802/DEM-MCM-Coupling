import os
import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from directory import (BUCKET_BASE, BUCKET_ID, BUCKET_PREFIX)
from huggingface_hub import HfFileSystem
fs = HfFileSystem()

import numpy as np
import matplotlib.pyplot as plt
import io
import json
import pyvista as pv

# Imports pour le maillage et le partitionnement
from DEM_MCM1.src.partitioners import create_partitioner
from DEM_MCM1.src.utils import load_parquet_as_timestep_dict

# pv.start_xvfb()
pv.OFF_SCREEN = True
LOCAL_OUTPUT = "/teamspace/studios/this_studio/MyStudio/outputs"

# Création des dossiers de sortie s'ils n'existent pas pour éviter les FileNotFoundError
os.makedirs(f"{LOCAL_OUTPUT}/fichiers", exist_ok=True)
os.makedirs(f"{LOCAL_OUTPUT}/images/transitions", exist_ok=True)
os.makedirs(f"{LOCAL_OUTPUT}/images/etats", exist_ok=True)
os.makedirs(f"{LOCAL_OUTPUT}/images/rsd", exist_ok=True)

# ─ 1. Trouver le dossier et charger les configurations ──────────────────────
dossiers = fs.ls(f"hf://buckets/{BUCKET_ID}/_Good/Experiment")
path = ""
for dossier in dossiers:
    if all(j in dossier.get('name', '') for j in [
        #    "adaptive_y_spectral_top1_bot150_splitNone_modequantile_NLT30_step10_dt2_tau157_start250",
        #    "gmm_full_20cells_vw0.5_NLT30_step10_dt2_tau50_start250",
           "cartesian_nx4_ny4_nz4_NLT30_step10_dt2_tau157_start250",
        #    "cartesian_nx5_ny5_nz5_NLT30_step20_dt4_tau157_start250",
        #    "cylindrical_nr3_nth3_nz3_equal_area_NLT30_step10_dt2_tau157_start250",
        #    "voronoi_20cells_NLT30_step10_dt2_tau157_start250",
           ]):
        path = dossier['name']
print(f"📂 Dossier trouvé : {path}")

# Définition du chemin court pour les sauvegardes
short_path = path.replace(f"buckets/{BUCKET_ID}/_Good/Experiment/", "")

def load_npy(fs, path, filename):
    with fs.open(f"{path}/{filename}", "rb") as f:
        return np.load(io.BytesIO(f.read()))

def load_json(fs, path, filename):
    with fs.open(f"{path}/{filename}", "r") as f:
        return json.load(f)

config  = load_json(fs, path, "stats.json")   
species_list = config.get("species_list", ["small", "large"]) # Liste des tailles de particules
exp_config   = load_json(fs, path, "config.json") 
start    = exp_config.get("start_index", 250) 
tau      = exp_config.get("tau", 50) 
print(f"   start={start}, tau={tau}, espèces={species_list}")

# ── 1.5 Calcul du maillage et enregistrement (PyVista) ──────────────────────
try:
    print("⏳ Chargement des données de simulation pour le maillage...")
    timestep_dict = load_parquet_as_timestep_dict(f'hf://buckets/{BUCKET_ID}/simulation_complete.parquet', fs)
    partitioner = create_partitioner(exp_config.get("method", ''), **exp_config.get("method_kwargs", {}))
    
    coords = timestep_dict[start][['coordinates:0', 'coordinates:1', 'coordinates:2']].to_numpy()
    partitioner.fit(coords)
    states = partitioner.compute_states(coords[:,0], coords[:,1], coords[:,2])
    
    mesh = pv.PolyData(coords)
    mesh.point_data['partitions'] = states
    
    if 'Diameter' in timestep_dict[start].columns:
        mesh.point_data.set_array(data=timestep_dict[start]['Diameter'].to_numpy().flatten(), name='Diameter')
        sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
        glyph = mesh.glyph(geom=sphere, orient=False, factor=1.0, scale="Diameter")
        mesh_to_save = glyph
    else:
        mesh_to_save = mesh
        
    mesh_path = f'{LOCAL_OUTPUT}/fichiers/meshs/mesh_{short_path}.vtp'
    mesh_to_save.save(mesh_path)
    print(f"✅ Maillage 3D enregistré : {mesh_path}")
except Exception as e:
    print(f"⚠️ Impossible de générer le maillage : {e}")

# ── 2. Post-traitement de la matrice P (Zhou et al., 2021) ───────────────────
def clean_transition_matrix(P, threshold=0.5):
    """
    Désactive les cellules vides (espace libre) et normalise la matrice.
    Suppose que P est left-stochastic (les colonnes somment à 1).
    """
    P_clean = P.copy()
    col_sums = P_clean.sum(axis=0)
    activated = col_sums >= threshold
    
    # Mettre à 0 les colonnes des cellules inactivées
    P_clean[:, ~activated] = 0.0
    
    # Normaliser les colonnes activées
    safe_sums = col_sums.copy()
    safe_sums[~activated] = 1.0
    P_clean = P_clean / safe_sums
    
    n_inactivated = (~activated).sum()
    print(f"🧹 Nettoyage de P : {n_inactivated} cellules vides désactivées sur {len(col_sums)}.")
    return P_clean, activated

# Chargement et nettoyage des données par espèce
species_data = {} 
for species in species_list:
    P        = load_npy(fs, path, f"transitionmatrix_{species}.npy")
    S_matrix = load_npy(fs, path, f"S_matrix_{species}.npy")   
    times    = load_npy(fs, path, f"times_{species}.npy")       
    
    P_clean, activated_cells = clean_transition_matrix(P, threshold=0.5)
    
    species_data[species] = {
        "P": P_clean, 
        "S_matrix": S_matrix, # le vecteur d'état n'est pas nettoyé à cette étape 
        "times": times,
        "activated": activated_cells 
    }
    print(f"   ✅ '{species}' — P nettoyée:{P_clean.shape}, Cellules actives: {activated_cells.sum()}")

# ── 3. Propagation Markov (Correction : S_new = P @ S) ───────────────────────
def propagate_markov(S0, P, times, start_idx, activated):
    row_start  = np.searchsorted(times, start_idx) # Retourne l'indice de start_idx dans le vecteur times
    times_full = times[row_start:]   # Récupère tous les intants à compter de l'instant initial.                     
    markov_indices = np.arange(0, len(times_full), tau)  # Récupère tous les indices par pas de temps de markov tau
    times_markov   = times_full[markov_indices]   # Récupère tous les instants pour les indices de markov correspondants

    S = S0.copy().astype(float)
    # le nettoyage permet de supprimer dans le vecteur d'état toutes cellules qui contiendraient moins d'une particule
    S[~activated] = 0.0 # Nettoyage de l'état initial
    
    trajectory = [S.copy()] 
    for _ in range(1, len(markov_indices)):
        S = P @ S  # P est left-stochastic (colonnes somment à 1)
        trajectory.append(S.copy())
    return np.array(trajectory), times_markov

markov_results = {} 
for species, data in species_data.items():
    row_start = np.searchsorted(data["times"], start)
    S0        = data["S_matrix"][row_start].astype(float)
    traj, times_markov = propagate_markov(S0, data["P"], data["times"], start, data["activated"])
    markov_results[species] = {"trajectory": traj, "times_markov": times_markov}
    print(f"    Markov '{species}' : {traj.shape[0]} pas | t={times_markov[0]}→{times_markov[-1]}")

# ── 4. Extraction DEM ────────────────────────────────────────────────────────
dem_results = {} 
for species, data in species_data.items():
    row_start   = np.searchsorted(data["times"], start)
    dem_results[species] = {
        "S_matrix": data["S_matrix"][row_start:],
        "times":    data["times"][row_start:],
    }

# ── 5. Fonctions de calcul (RSD et Entropie sur cellules actives) ────────────
def compute_rsd_from_S(S_traj, activated):
    S_active = S_traj[:, activated]
    mean = S_active.mean(axis=1)                          
    std  = S_active.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)

def compute_rsd_concentration(S_small, S_large, act_s, act_l):
    activated = act_s & act_l
    total = S_small[:, activated] + S_large[:, activated]
    C     = np.where(total > 0, S_small[:, activated] / total, 0.0)   
    mean  = C.mean(axis=1)
    std   = C.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)

def compute_entropy(S_traj, activated):
    """Entropie de Shannon de la distribution spatiale."""
    S_active = S_traj[:, activated]
    N_total = S_active.sum(axis=1, keepdims=True)
    N_total = np.where(N_total > 0, N_total, 1.0)
    p = S_active / N_total
    entropy = np.zeros(S_active.shape[0])
    for t in range(S_active.shape[0]):
        p_t = p[t]
        mask = p_t > 0
        if mask.any():
            entropy[t] = -np.sum(p_t[mask] * np.log(p_t[mask]))
    return entropy

def compute_entropy_concentration(S_small, S_large, act_s, act_l):
    """Entropie de la distribution de concentration (mélange binaire)."""
    activated = act_s & act_l
    total = S_small[:, activated] + S_large[:, activated]
    total = np.where(total > 0, total, 1.0)
    C = S_small[:, activated] / total
    entropy = np.zeros(C.shape[0])
    for t in range(C.shape[0]):
        C_t = C[t]
        valid = (C_t > 0) & (C_t < 1)
        if valid.any():
            C_valid = C_t[valid]
            entropy[t] = -np.sum(C_valid * np.log(C_valid) + (1 - C_valid) * np.log(1 - C_valid))
    return entropy

# ── 6. Figure 1 : États par partition ────────────────────────────────────────
for species in species_list:
    traj_markov   = markov_results[species]["trajectory"]    
    times_markov  = markov_results[species]["times_markov"]
    S_dem         = dem_results[species]["S_matrix"]         
    times_dem     = dem_results[species]["times"]
    n_states      = traj_markov.shape[1]

    fig, ax = plt.subplots(figsize=(12, 5))
    for k in range(min(3, n_states)):
        ax.plot(times_dem, S_dem[:, k], "--", linewidth=1, label=f"État {k} DEM")
        ax.plot(times_markov, traj_markov[:, k], "o-", markersize=3, linewidth=1, label=f"État {k} Markov")
    ax.set_ylabel("Nombre de particules")
    ax.set_xlabel("Temps (centièmes de secondes)")
    fig.suptitle(f"Comparaison Markov vs DEM — espèce '{species}'", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    plt.savefig(f"{LOCAL_OUTPUT}/images/etats/etats_{species}_{short_path}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── 7. Figure 2 : RSD et Entropie Globale ────────────────────────────────────
n_panels = len(species_list) + 2 
fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

for ax, species in zip(axes[:len(species_list)], species_list):
    traj_markov  = markov_results[species]["trajectory"]
    times_markov = markov_results[species]["times_markov"]
    S_dem        = dem_results[species]["S_matrix"]
    times_dem    = dem_results[species]["times"]
    activated    = species_data[species]["activated"]

    ax.plot(times_dem, compute_rsd_from_S(S_dem, activated), "--", label="DEM")
    ax.plot(times_markov, compute_rsd_from_S(traj_markov, activated), "o-", markersize=3, label="Markov")
    ax.set_title(f"RSD — espèce '{species}'", fontsize=9)
    ax.set_xlabel("Temps"); ax.set_ylabel("RSD"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Entropie globale
sp_a, sp_b = species_list[0], species_list[1]
n_min = min(len(markov_results[sp_a]["times_markov"]), len(markov_results[sp_b]["times_markov"]))
n_dem_min = min(len(dem_results[sp_a]["times"]), len(dem_results[sp_b]["times"]))

ent_m = compute_entropy(markov_results[sp_a]["trajectory"][:n_min], species_data[sp_a]["activated"]) + \
        compute_entropy(markov_results[sp_b]["trajectory"][:n_min], species_data[sp_b]["activated"])
ent_d = compute_entropy(dem_results[sp_a]["S_matrix"][:n_dem_min], species_data[sp_a]["activated"]) + \
        compute_entropy(dem_results[sp_b]["S_matrix"][:n_dem_min], species_data[sp_b]["activated"])

axes[len(species_list)].plot(dem_results[sp_a]["times"][:n_dem_min], ent_d, "--", label="DEM")
axes[len(species_list)].plot(markov_results[sp_a]["times_markov"][:n_min], ent_m, "o-", markersize=3, label="Markov")
axes[len(species_list)].set_title("Entropie de Shannon (Totale)", fontsize=9)
axes[len(species_list)].set_xlabel("Temps"); axes[len(species_list)].set_ylabel("Entropie"); axes[len(species_list)].legend(fontsize=8); axes[len(species_list)].grid(True, alpha=0.3)

# RSD Concentration
rsd_conc_m = compute_rsd_concentration(markov_results[sp_a]["trajectory"][:n_min], markov_results[sp_b]["trajectory"][:n_min], species_data[sp_a]["activated"], species_data[sp_b]["activated"])
rsd_conc_d = compute_rsd_concentration(dem_results[sp_a]["S_matrix"][:n_dem_min], dem_results[sp_b]["S_matrix"][:n_dem_min], species_data[sp_a]["activated"], species_data[sp_b]["activated"])

axes[-1].plot(dem_results[sp_a]["times"][:n_dem_min], rsd_conc_d, "--", label="DEM")
axes[-1].plot(markov_results[sp_a]["times_markov"][:n_min], rsd_conc_m, "o-", markersize=3, label="Markov")
axes[-1].set_title(f"RSD C({sp_a}) — bidisperse", fontsize=9)
axes[-1].set_xlabel("Temps"); axes[-1].set_ylabel("RSD Conc."); axes[-1].legend(fontsize=8); axes[-1].grid(True, alpha=0.3)

fig.suptitle(f"RSD & Entropie — {short_path}", fontsize=8)
fig.tight_layout()
plt.savefig(f"{LOCAL_OUTPUT}/images/rsd/rsd_entropy_{short_path}.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 8. Figure 3 : Entropie de Concentration ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ent_conc_m = compute_entropy_concentration(markov_results[sp_a]["trajectory"][:n_min], markov_results[sp_b]["trajectory"][:n_min], species_data[sp_a]["activated"], species_data[sp_b]["activated"])
ent_conc_d = compute_entropy_concentration(dem_results[sp_a]["S_matrix"][:n_dem_min], dem_results[sp_b]["S_matrix"][:n_dem_min], species_data[sp_a]["activated"], species_data[sp_b]["activated"])

ax.plot(dem_results[sp_a]["times"][:n_dem_min], ent_conc_d, "--", lw=2, label="DEM", alpha=0.7)
ax.plot(markov_results[sp_a]["times_markov"][:n_min], ent_conc_m, "o-", markersize=3, lw=2, label="Markov")
ax.set_title(f"Entropie de Concentration C({sp_a})\n{short_path}"); ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlabel("Temps"); ax.set_ylabel("Entropie de Concentration")
plt.tight_layout()
plt.savefig(f"{LOCAL_OUTPUT}/images/rsd/entropy_concentration_{short_path}.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 9. Diagnostic Spectral de P ──────────────────────────────────────────────
P_analysis = species_data["small"]["P"]
eigenvalues, eigenvectors = np.linalg.eig(P_analysis.T)
idx = np.argsort(np.abs(eigenvalues))[::-1]
eigvals = eigenvalues[idx]
eigvecs = eigenvectors[:, idx]

pi = np.abs(eigvecs[:, 0]); pi = pi / pi.sum()
activated_small = species_data["small"]["activated"]
pi_active = pi[activated_small]

print(f"\n🔬 Diagnostic Spectral :")
print(f"  Trou spectral = 1 - |λ1| = {1.0 - np.abs(eigvals[1]):.6f}")
print(f"  RSD stationnaire (active) = {pi_active.std()/pi_active.mean():.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].scatter(eigvals.real, eigvals.imag, s=20, alpha=0.7)
theta = np.linspace(0, 2*np.pi, 300)
axes[0].plot(np.cos(theta), np.sin(theta), 'r--', lw=0.8)
axes[0].set_title("Spectre de P"); axes[0].set_aspect('equal'); axes[0].grid(True, alpha=0.3)

axes[1].bar(range(len(pi)), pi)
axes[1].set_title(f"π\nRSD={pi_active.std()/pi_active.mean():.3f}"); axes[1].grid(True, alpha=0.3)

im = axes[2].imshow(P_analysis, aspect='auto', cmap='viridis', vmin=0, vmax=P_analysis.max())
plt.colorbar(im, ax=axes[2]); axes[2].set_title("Matrice P (nettoyée)")
plt.tight_layout()
plt.savefig(f"{LOCAL_OUTPUT}/images/diagnostic_P.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── 10. Sauvegarde et visualisation des matrices de transition ───────────────
print("\n💾 Sauvegarde des matrices de transition...")
for species in species_list:
    P_clean = species_data[species]["P"]
    
    # Sauvegarde du fichier .npy local
    npy_path = f"{LOCAL_OUTPUT}/fichiers/transitionmatrix_{species}_{short_path}.npy"
    np.save(npy_path, P_clean)
    
    # Visualisation et sauvegarde en image
    fig, ax = plt.subplots(figsize=(10, 10))
    max_val = np.max(P_clean[P_clean > 0]) if np.any(P_clean > 0) else 1.0
    im = ax.imshow(P_clean, cmap='viridis', vmin=0, vmax=max_val)
    
    n_cells = P_clean.shape[0]
    if n_cells <= 40:  # Ajouter le texte seulement si la matrice n'est pas trop grande
        for i in range(n_cells):
            for j in range(n_cells):
                val = P_clean[i, j]
                color = 'white' if val > (max_val / 2) else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=6, color=color)
                
    ax.set_title(f'Matrice de transition P nettoyée — {species}\n{short_path}')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    img_path = f"{LOCAL_OUTPUT}/images/transitions/transition_{species}_{short_path}.png"
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Matrice de transition '{species}' enregistrée (Numpy et PNG).")

print("\n🎉 Analyse complète terminée avec post-traitement, calcul d'entropie, maillage et matrices P !")