import asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
from directory import (BUCKET_BASE, BUCKET_ID, BUCKET_PREFIX)
from huggingface_hub import HfFileSystem
fs = HfFileSystem()
from DEM_MCM1.src.partitioners import create_partitioner
import numpy as np
import matplotlib.pyplot as plt
import io
import json
import pyvista as pv

pv.start_xvfb()
pv.OFF_SCREEN = True
LOCAL_OUTPUT = "/teamspace/studios/this_studio/MyStudio/outputs"

# ── Trouver le dossier ───────────────────────────────────────────────────────
dossiers = fs.ls(f"hf://buckets/{BUCKET_ID}/_Good/Experiment")
path = ""
for dossier in dossiers:
    if all(j in dossier.get('name', '') for j in
           [
            #    "gmm_full_10cells_vw0.5_NLT30_step10_dt2_tau157_start250",
               "spectral_100cells_vw0.5_k15_NLT30_step10_dt2_tau157_start250",
               ]):
        path = dossier['name']

print(f"📂 Dossier trouvé : {path}")

# ── Chargement des fichiers par espèce ───────────────────────────────────────
def load_npy(fs, path, filename):
    with fs.open(f"{path}/{filename}", "rb") as f:
        return np.load(io.BytesIO(f.read()))

def load_json(fs, path, filename):
    with fs.open(f"{path}/{filename}", "r") as f:
        return json.load(f)

config  = load_json(fs, path, "stats.json")   # stats contient species_list
species_list = config.get("species_list", ["small", "large"]) # tailles des particules(par défaut deux tailles)
exp_config   = load_json(fs, path, "config.json") # configurations du modèle (spatial et temporel)

start    = exp_config.get("start_index", 250) #start (debut de l'apprentissage) et instant du partitionnement
tau      = exp_config.get("tau", 50) # pas de temps de markov
tau_dem  = tau  # S_matrix est déjà échantillonnée à chaque timestep DEM # pas de temps dem

print(f"   start={start}, tau={tau}, espèces={species_list}")

# Charger P et S_matrix pour chaque espèce
species_data = {} # dictionnaire de matrice de transition  vecteur d'état et temps pour chaque taille
for species in species_list:
    P        = load_npy(fs, path, f"transitionmatrix_{species}.npy")
    S_matrix = load_npy(fs, path, f"S_matrix_{species}.npy")   # (n_timesteps, n_states)
    times    = load_npy(fs, path, f"times_{species}.npy")       # indices DEM triés tous les pas de temps de la DEM au cas où on travaillerait avec des données differentes 
    species_data[species] = {"P": P, "S_matrix": S_matrix, "times": times}
    print(f"   ✅ '{species}' — P:{P.shape}, S_matrix:{S_matrix.shape}, "
          f"P col sums: {np.round(P.sum(axis=0)[:5], 3)}...")

# ── Propagation Markov par espèce ────────────────────────────────────────────
def propagate_markov(S0, P, times, start_idx):
    """
    Propage S0 avec P sur les pas de temps tau à partir de start_idx.
    Retourne (trajectory, times_markov).
    """
    # Trouver la ligne correspondant à start_idx dans times
    row_start  = np.searchsorted(times, start_idx) # indice de markov de début dans les indices de la dem
    times_full = times[row_start:]                        # tous les timesteps DEM depuis start 

    # On propage à chaque pas tau (pas tous les timesteps DEM)
    markov_indices = np.arange(0, len(times_full), tau)  # indices pour la prédiction de markov ie séparé de tau
    times_markov   = times_full[markov_indices] # les temps correspondant à ces indices

    S = S0.copy().astype(float)
    trajectory = [S.copy()] # est une matrice de taille (times_markov,n_states) ou de taille (n_steps,n_states)
    for _ in range(1, len(markov_indices)):
        S =S@P
        trajectory.append(S.copy())

    return np.array(trajectory), times_markov             # (n_steps, n_states)


markov_results = {} # dictionnaire contenant le vecteur d'état et les times_markov pour chacune des espèces
for species, data in species_data.items():
    row_start = np.searchsorted(data["times"], start)
    S0        = data["S_matrix"][row_start].astype(float)

    traj, times_markov = propagate_markov(S0, data["P"], data["times"], start)
    markov_results[species] = {
        "trajectory":    traj,          # (n_steps_markov, n_states)
        "times_markov":  times_markov,
    }
    print(f"   🔁 Markov '{species}' : {traj.shape[0]} pas | "
          f"t={times_markov[0]}→{times_markov[-1]}")

# ── DEM : extraire S_matrix depuis start avec pas tau_dem ────────────────────
dem_results = {} # dictionnaire contenant les vecteurs d'état et les times pour chacune des espèces
for species, data in species_data.items():
    row_start   = np.searchsorted(data["times"], start)
    S_dem_full  = data["S_matrix"][row_start:]            # (n_timesteps_after_start, n_states)
    times_dem   = data["times"][row_start:]

    dem_results[species] = {
        "S_matrix": S_dem_full,
        "times":    times_dem,
    }

# ── RSD par espèce ───────────────────────────────────────────────────────────
def compute_rsd_from_S(S_traj):
    """
    RSD sur le nombre de particules par partition.
    S_traj : (n_steps, n_states)
    """
    mean = S_traj.mean(axis=1)                          # (n_steps,)
    std  = S_traj.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)


def compute_rsd_concentration(S_small, S_large):
    """
    RSD sur la concentration locale C_small = S_small / (S_small + S_large).
    Plus physique pour un mélange bidisperse.
    S_small, S_large : (n_steps, n_states)
    """
    total = S_small + S_large
    C     = np.where(total > 0, S_small / total, 0.0)   # (n_steps, n_states)
    mean  = C.mean(axis=1)
    std   = C.std(axis=1)
    return np.where(mean > 0, std / mean, 0.0)


short_path = path.replace(f"buckets/{BUCKET_ID}/_Good/Experiment/", "")

# ── Figure 1 : états par partition au cours du temps (une figure par espèce) ─
for species in species_list:
    traj_markov   = markov_results[species]["trajectory"]    # (n_steps_markov, n_states)
    times_markov  = markov_results[species]["times_markov"]
    S_dem         = dem_results[species]["S_matrix"]         # (n_timesteps_dem, n_states)
    times_dem     = dem_results[species]["times"]
    n_states      = traj_markov.shape[1]

    fig, ax = plt.subplots(figsize=(12, 5))
    for k in range(3):
        ax.plot(times_markov, traj_markov[:, k],
                "o-", markersize=3, linewidth=1,
                label=f"État {k} Markov" if n_states <= 10 else None)
        ax.plot(times_dem, S_dem[:, k],
                "--", linewidth=1,
                label=f"État {k} DEM" if n_states <= 10 else None)

    ax.set_ylabel("Nombre de particules")
    ax.set_xlabel("Temps (centièmes de secondes)")
    fig.suptitle(f"Comparaison Markov vs DEM — espèce '{species}'", fontsize=9)
    ax.set_title(short_path, fontsize=7)
    fig.tight_layout()
    if n_states <= 10:
        ax.legend(fontsize=6, ncol=2)
    plt.savefig(
        f"{LOCAL_OUTPUT}/images/etats/etats_{species}_{short_path}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(f"   💾 Figure états '{species}' sauvegardée")

# ── Figure 2 : RSD par espèce + RSD concentration ────────────────────────────
fig, axes = plt.subplots(1, len(species_list) + 1,
                         figsize=(5 * (len(species_list) + 1), 4))

for ax, species in zip(axes[:len(species_list)], species_list):
    traj_markov  = markov_results[species]["trajectory"]
    times_markov = markov_results[species]["times_markov"]
    S_dem        = dem_results[species]["S_matrix"]
    times_dem    = dem_results[species]["times"]

    rsd_markov = compute_rsd_from_S(traj_markov)
    rsd_dem    = compute_rsd_from_S(S_dem)

    ax.plot(times_markov, rsd_markov, "o-", markersize=3, label="Markov")
    ax.plot(times_dem,    rsd_dem,    "--",               label="DEM")
    ax.set_xlabel("Temps (centièmes de secondes)")
    ax.set_ylabel("RSD")
    ax.set_title(f"RSD — espèce '{species}'", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Dernier panneau : RSD de concentration si deux espèces
if len(species_list) == 2:
    sp_a, sp_b = species_list[0], species_list[1]
    ax_conc = axes[-1]

    # Markov : aligner sur le même axe temporel (le plus court)
    n_min = min(len(markov_results[sp_a]["times_markov"]),
                len(markov_results[sp_b]["times_markov"]))
    rsd_conc_markov = compute_rsd_concentration(
        markov_results[sp_a]["trajectory"][:n_min],
        markov_results[sp_b]["trajectory"][:n_min],
    )
    times_markov_common = markov_results[sp_a]["times_markov"][:n_min]

    # DEM
    n_dem_min = min(len(dem_results[sp_a]["times"]),
                    len(dem_results[sp_b]["times"]))
    rsd_conc_dem = compute_rsd_concentration(
        dem_results[sp_a]["S_matrix"][:n_dem_min],
        dem_results[sp_b]["S_matrix"][:n_dem_min],
    )
    times_dem_common = dem_results[sp_a]["times"][:n_dem_min]

    ax_conc.plot(times_markov_common, rsd_conc_markov, "o-", markersize=3, label="Markov")
    ax_conc.plot(times_dem_common,    rsd_conc_dem,    "--",               label="DEM")
    ax_conc.set_xlabel("Temps (centièmes de secondes)")
    ax_conc.set_ylabel("RSD concentration")
    ax_conc.set_title(f"RSD C({sp_a}) — bidisperse", fontsize=9)
    ax_conc.legend(fontsize=8)
    ax_conc.grid(True, alpha=0.3)

fig.suptitle(f"RSD — {short_path}", fontsize=8)
fig.tight_layout()
plt.savefig(
    f"{LOCAL_OUTPUT}/images/rsd/rsd_{short_path}.png",
    dpi=150, bbox_inches="tight"
)
plt.close(fig)
print(f"   💾 Figure RSD sauvegardée")



import numpy as np
import matplotlib.pyplot as plt

# ── Analyse spectrale de P ───────────────────────────────────────────────────
eigenvalues, eigenvectors = np.linalg.eig(P.T)

# Trier par valeur propre décroissante (en module)
idx     = np.argsort(np.abs(eigenvalues))[::-1]
eigvals = eigenvalues[idx]
eigvecs = eigenvectors[:, idx]

print("Top 10 valeurs propres (module) :")
for i, ev in enumerate(eigvals[:10]):
    print(f"  λ{i} = {ev.real:.6f} + {ev.imag:.4f}j  |λ|={np.abs(ev):.6f}")

# λ0 doit être ≈ 1 (distribution stationnaire)
# λ1 = "trou spectral" — contrôle la vitesse de mélange
spectral_gap = 1.0 - np.abs(eigvals[1])
print(f"\nTrou spectral = 1 - |λ1| = {spectral_gap:.6f}")
print(f"Temps de mélange estimé (pas Markov) ≈ {1/spectral_gap:.1f}")
print(f"Temps de mélange estimé (secondes)   ≈ {tau * 0.01 / spectral_gap:.2f}s")

# ── Distribution stationnaire ────────────────────────────────────────────────
pi = np.abs(eigvecs[:, 0])
pi = pi / pi.sum()
print(f"\nDistribution stationnaire π :")
print(f"  min={pi.min():.4f}  max={pi.max():.4f}  std={pi.std():.4f}")
print(f"  RSD stationnaire = {pi.std()/pi.mean():.4f}")
# Si RSD_stationnaire ≈ 0 → P converge vers uniforme → problème structurel

# ── Vérification colonnes vides ──────────────────────────────────────────────
col_sums    = P.sum(axis=0)
n_empty     = (col_sums == 0).sum()
n_not_one   = (np.abs(col_sums - 1) > 0.01).sum()
print(f"\nColonnes vides        : {n_empty}/{P.shape[0]}")
print(f"Colonnes non normalisées (|sum-1|>0.01) : {n_not_one}/{P.shape[0]}")

# ── Visualisation ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Spectre
axes[0].scatter(eigvals.real, eigvals.imag, s=20, alpha=0.7)
theta = np.linspace(0, 2*np.pi, 300)
axes[0].plot(np.cos(theta), np.sin(theta), 'r--', lw=0.8, label='|λ|=1')
axes[0].axvline(eigvals[1].real, color='orange', lw=1, label=f'λ1={eigvals[1].real:.4f}')
axes[0].set_title("Spectre de P")
axes[0].set_xlabel("Re(λ)")
axes[0].set_ylabel("Im(λ)")
axes[0].legend()
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Distribution stationnaire
axes[1].bar(range(len(pi)), pi)
axes[1].set_title(f"Distribution stationnaire π\nRSD={pi.std()/pi.mean():.3f}")
axes[1].set_xlabel("État")
axes[1].set_ylabel("π(k)")
axes[1].grid(True, alpha=0.3)

# Matrice P
im = axes[2].imshow(P, aspect='auto', cmap='viridis',
                    vmin=0, vmax=P.max())
plt.colorbar(im, ax=axes[2])
axes[2].set_title("Matrice P")
axes[2].set_xlabel("État j (colonne)")
axes[2].set_ylabel("État i (ligne)")

plt.tight_layout()
plt.savefig(f"{LOCAL_OUTPUT}/images/diagnostic_P.png", dpi=150, bbox_inches="tight")
# #plt.show()

# ── Simulation de convergence ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
S = species_data["small"]["S_matrix"][
    np.searchsorted(species_data["small"]["times"], start)
].astype(float)

rsd_trajectory = []
for t in range(200):
    mean_S = S.mean()
    rsd_trajectory.append(S.std() / mean_S if mean_S > 0 else 0)
    S = P.T @ S

ax.semilogy(rsd_trajectory)
ax.axhline(pi.std()/pi.mean(), color='r', ls='--',
           label=f'RSD stationnaire = {pi.std()/pi.mean():.4f}')
ax.set_xlabel("Pas Markov")
ax.set_ylabel("RSD (log)")
ax.set_title("Convergence RSD Markov (log)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(f"{LOCAL_OUTPUT}/images/convergence_rsd.png", dpi=150, bbox_inches="tight")
# #plt.show()


# ── MAUVAIS (actuel) ────────────────────────────────────────────────────────
# RSD sur le nombre de particules par cellule → converge vers uniforme
rsd = S.std() / S.mean()   # → 0 inévitablement

# ── CORRECT ─────────────────────────────────────────────────────────────────
# RSD sur la concentration locale C(k) = S_small(k) / (S_small(k) + S_large(k))
# C(k) mesure la ségrégation locale → ne converge PAS vers 0 si mélange imparfait
def rsd_concentration(S_small, S_large):
    total = S_small + S_large
    C     = np.where(total > 0, S_small / total, np.nan)
    C_active = C[~np.isnan(C)]
    mean = C_active.mean()
    return C_active.std() / mean if mean > 0 else 0.0
# ── Propagation correcte ────────────────────────────────────────────────────
row_start    = np.searchsorted(species_data["small"]["times"], start)
S_small      = species_data["small"]["S_matrix"][row_start].astype(float)
S_large      = species_data["large"]["S_matrix"][row_start].astype(float)
P_small      = species_data["small"]["P"]
P_large      = species_data["large"]["P"]

n_steps      = 200
rsd_markov   = np.zeros(n_steps)
rsd_dem      = np.zeros(n_steps)
times_markov = np.zeros(n_steps, dtype=int)
times_dem    = species_data["small"]["times"][row_start:]

S_s = S_small.copy()
S_l = S_large.copy()

for t in range(n_steps):
    times_markov[t] = start + t * tau

    # RSD Markov sur concentration
    rsd_markov[t] = rsd_concentration(S_s, S_l)

    # RSD DEM sur concentration au même instant
    # Trouver le timestep DEM le plus proche
    row_t = np.searchsorted(times_dem, times_markov[t])
    if row_t < len(times_dem):
        S_dem_s = species_data["small"]["S_matrix"][row_start + row_t]
        S_dem_l = species_data["large"]["S_matrix"][row_start + row_t]
        rsd_dem[t] = rsd_concentration(S_dem_s, S_dem_l)

    # Propagation
    S_s = P_small.T @ S_s
    S_l = P_large.T @ S_l

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times_markov, rsd_markov, "o-", markersize=3, lw=2, label="Markov")
ax.plot(times_markov, rsd_dem,    "o-", markersize=3, lw=2, label="DEM",
        alpha=0.7)
ax.set_xlabel("Temps (centièmes de secondes)")
ax.set_ylabel("RSD concentration C(small)")
ax.set_title(f"RSD mélange bidisperse\n{short_path}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{LOCAL_OUTPUT}/images/rsd/rsd_concentration_{short_path}.png",
            dpi=150, bbox_inches="tight")
# #plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── 1. Préparation des données DEM et Markov sur toute la durée ─────────────
row_start = np.searchsorted(species_data["small"]["times"], start)
S_small_dem = species_data["small"]["S_matrix"][row_start:]
S_large_dem = species_data["large"]["S_matrix"][row_start:]
times_dem   = species_data["small"]["times"][row_start:]

# RSD DEM (sur toute la durée disponible)
rsd_dem_full = compute_rsd_concentration(S_small_dem, S_large_dem)

# ── 2. Propagation Markov sur la même durée physique ────────────────────────
# Nombre de pas Markov pour couvrir la durée DEM
n_steps = len(times_dem) // tau 

S_s = species_data["small"]["S_matrix"][row_start].astype(float)
S_l = species_data["large"]["S_matrix"][row_start].astype(float)
P_small = species_data["small"]["P"]
P_large = species_data["large"]["P"]
n_states = S_s.shape[0]

# On stocke les trajectoires pour le calcul vectorisé du RSD
S_s_traj = np.zeros((n_steps, n_states))
S_l_traj = np.zeros((n_steps, n_states))
S_s_traj[0] = S_s
S_l_traj[0] = S_l

for k in range(1, n_steps):
    S_s_traj[k] = P_small.T @ S_s_traj[k-1]
    S_l_traj[k] = P_large.T @ S_l_traj[k-1]

rsd_markov_full = compute_rsd_concentration(S_s_traj, S_l_traj)

# ⚠️ IMPORTANT : L'équation de Zhou suppose que t=0 est le début du mélange.
# On doit donc soustraire 'start' pour avoir le temps écoulé.
t_dem_fit = times_dem - start
t_markov_fit = np.arange(n_steps) * tau

# ── 3. Modèle de régression de Zhou (Second-order regression) ───────────────
def rsd_model(t, RSD_inf, RSD_0, tau_c):
    # t est le temps écoulé depuis le début de l'observation
    t_tau = t / tau_c
    t_tau = np.clip(t_tau, -500, 500) # Éviter overflow dans l'exponentielle
    return RSD_inf + (RSD_0 - RSD_inf) * (1 + t_tau) * np.exp(-t_tau)

# Estimation initiale des paramètres (p0) pour aider l'optimiseur
p0_dem = [rsd_dem_full[-1], rsd_dem_full[0], len(t_dem_fit)*0.1]
p0_mc  = [rsd_markov_full[-1], rsd_markov_full[0], len(t_markov_fit)*0.1]

# Ajustement pour la DEM
try:
    popt_dem, _ = curve_fit(rsd_model, t_dem_fit, rsd_dem_full, p0=p0_dem, maxfev=10000)
    RSD_inf_dem, RSD_0_dem, tau_DEM = popt_dem
    print(f"✅ Ajustement DEM réussi : τ_DEM = {tau_DEM:.2f} pas DEM")
except Exception as e:
    print("⚠️ Ajustement DEM échoué, utilisation des valeurs par défaut.")
    tau_DEM = p0_dem[2]

# Ajustement pour le modèle Markov (non calibré, avec t_m = t_l)
try:
    popt_mc, _ = curve_fit(rsd_model, t_markov_fit, rsd_markov_full, p0=p0_mc, maxfev=10000)
    RSD_inf_mc, RSD_0_mc, tau_MC = popt_mc
    print(f"✅ Ajustement Markov (brut) réussi : τ_MC = {tau_MC:.2f} pas DEM")
except Exception as e:
    print("⚠️ Ajustement Markov échoué, utilisation des valeurs par défaut.")
    tau_MC = p0_mc[2]

# ── 4. Recalibrage du pas de temps de Markov (Formule de Zhou) ──────────────
# Formule : t_m = t_l * (tau_DEM / tau_MC)
# Ici t_l est votre 'tau' actuel
tau_calibrated = tau * (tau_DEM / tau_MC)
print(f"🔄 Recalibrage : tau original (t_l) = {tau}, tau calibré (t_m) = {tau_calibrated:.2f}")

# Le nouvel axe temporel écoulé pour Markov (on ne refait pas tourner la chaîne, 
# on change juste l'échelle de temps physique associée à chaque pas)
t_markov_calibrated_fit = np.arange(n_steps) * tau_calibrated

# ── 5. Visualisation des courbes et des ajustements ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))

# Pour le tracé, on remet le temps absolu (start + t) pour cohérence avec vos autres graphiques
times_markov_calibrated_plot = start + t_markov_calibrated_fit
times_markov_raw_plot = start + t_markov_fit

# 1. Courbe DEM et son ajustement
ax.plot(times_dem, rsd_dem_full, "k-", lw=2, label="DEM (Référence)")
t_fit_dem_plot = np.linspace(t_dem_fit[0], t_dem_fit[-1], 200) + start
ax.plot(t_fit_dem_plot, rsd_model(t_fit_dem_plot - start, *popt_dem), "k--", lw=1.5, 
        label=f"Ajustement DEM ($\\tau_{{DEM}}$={tau_DEM:.1f})")

# 2. Courbe Markov NON CALIBRÉE (t_m = t_l)
ax.plot(times_markov_raw_plot, rsd_markov_full, "b-", lw=1.5, alpha=0.6, 
        label=f"Markov non calibré ($t_m=t_l$={tau})")

# 3. Courbe Markov CALIBRÉE (t_m recalibré)
ax.plot(times_markov_calibrated_plot, rsd_markov_full, "r-", lw=2, 
        label=f"Markov recalibré ($t_m$={tau_calibrated:.1f})")

# 4. Ajustement sur le Markov recalibré (pour prouver que les tau correspondent)
try:
    popt_mc_cal, _ = curve_fit(rsd_model, t_markov_calibrated_fit, rsd_markov_full, p0=p0_mc, maxfev=10000)
    t_fit_mc_plot = np.linspace(t_markov_calibrated_fit[0], t_markov_calibrated_fit[-1], 200) + start
    ax.plot(t_fit_mc_plot, rsd_model(t_fit_mc_plot - start, *popt_mc_cal), "r--", lw=1.5,
            label=f"Ajustement Markov recalibré ($\\tau_{{MC'}}$={popt_mc_cal[2]:.1f})")
except:
    pass

ax.set_xlabel("Temps (indices DEM)", fontsize=12)
ax.set_ylabel("RSD de concentration", fontsize=12)
ax.set_title(f"Recalibrage du pas de temps Markov (Méthode de Zhou)\n{short_path}", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

save_path = f"{LOCAL_OUTPUT}/images/rsd/recalibration_zhou_{short_path}.png"
plt.savefig(save_path, dpi=150, bbox_inches="tight")
#plt.show()
print(f"💾 Figure de recalibrage sauvegardée dans {save_path}")