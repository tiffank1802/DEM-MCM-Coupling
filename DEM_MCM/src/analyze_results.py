"""
===================================================================================
ANALYSE MARKOVIENNE — Chargement et visualisation depuis le bucket HuggingFace
===================================================================================

Charge automatiquement toutes les expériences (voronoi, cartesian, cylindrical,
quantile, octree, physics) et propose des visualisations comparatives.

Usage:
    python analyze_results.py
    
Depuis un notebook:
    from analyze_results import MarkovAnalyzer
    analyzer = MarkovAnalyzer()
    analyzer.load_all()
    analyzer.compare_methods()
===================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import json
import io
from collections import defaultdict
from huggingface_hub import HfFileSystem

# Imports relatifs (notebooks) vs absolus (script direct)
try:
    from .import bucket_io as b_io
    from .utils import apply_species_mask
except ImportError:
    import bucket_io as b_io
    from utils import apply_species_mask
# =============================================================================
# CONFIGURATION
# =============================================================================

# BUCKET_ID = "ktongue/DEM_MCM"
# BUCKET_PREFIX = "markov_results"
# BUCKET_PREFIX = "ResultsDtMCM"
# BUCKET_PREFIX = "ResultsDtMCM"
BUCKET_ID = b_io.BUCKET_ID
BUCKET_PREFIX = b_io.BUCKET_PREFIX
BUCKET_BASE = b_io.BUCKET_BASE

# ✅ Option B: Tous les buckets possibles (cherchés dans l'ordre)
def _get_bucket_prefix_from_particle_diameter(particle_diameter):
    """
    Détermine le bucket selon particle_diameter (même logique que bucket_io.py).
    Utilisé pour charger les expériences du bon bucket (Option A).
    """
    if particle_diameter == 0.008:
        return "BIG"
    elif particle_diameter == 0.004:
        return "SMALL"
    else:
        return "Experiments"

# ✅ Liste de tous les buckets à chercher (par ordre de priorité)
# - D'abord les buckets actuels (Experiments, SMALL, BIG)
# - Ensuite les anciens buckets (OLD_BUCKET_PREFIX)
ALL_BUCKET_PREFIXES = ["Experiments", "SMALL", "BIG"]

# Anciennes données cartésiennes (dossier séparé) - cherché en dernier
# OLD_BUCKET_PREFIX = "markov_sweep_results"
# OLD_BUCKET_PREFIX = "NewResultsMCM"
OLD_BUCKET_PREFIX = "Experiments"  # Par défaut = même que bucket actuel
OLD_BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{OLD_BUCKET_PREFIX}"

# ✅ ALL_BUCKET_BASES pour Option B (cherche dans tous les buckets)
ALL_BUCKET_BASES = [f"hf://buckets/{BUCKET_ID}/{prefix}" for prefix in ALL_BUCKET_PREFIXES]
if OLD_BUCKET_BASE not in ALL_BUCKET_BASES:
    ALL_BUCKET_BASES.append(OLD_BUCKET_BASE)

BUCKET_ID=b_io.BUCKET_ID
BUCKET_PREFIX=b_io.BUCKET_PREFIX
BUCKET_BASE=b_io.BUCKET_BASE
# Méthodes connues et leurs préfixes
METHOD_PREFIXES = {
    "cartesian": ["cartesian_", "NLT_"],   # NLT_ = ancien format cartésien
    "cylindrical": ["cylindrical_"],
    "voronoi": ["voronoi_"],
    "quantile": ["quantile_"],
    "octree": ["octree_"],
    "physics": ["physics_"],
    "adaptive":["adaptive_"],
    "multizone":["multizone_"],
    "single":["single_"],
}

# Couleurs par méthode
METHOD_COLORS = {
    "cartesian": "#1f77b4",
    "cylindrical": "#ff7f0e",
    "voronoi": "#2ca02c",
    "quantile": "#d62728",
    "octree": "#9467bd",
    "physics": "#8c564b",
    "adaptive":"#af6c3c",
    "multizone":"#4b5d4c",
    "single":"#2b3e4ba8",
    "unknown": "#7f7f7f",
}


# =============================================================================
# CLASSE PRINCIPALE
# =============================================================================

class MarkovAnalyzer:
    """
    Chargeur et analyseur universel de résultats Markoviens.
    
    Gère tous les types de partitionnement et les deux formats
    (ancien cartésien + nouveau multi-méthode).
    """
    
    def __init__(self):
        self.fs = HfFileSystem()
        self.results = {}           
        self.by_method = defaultdict(dict)  
        
        # ═══ NOUVEAUX ATTRIBUTS PARTAGÉS ═══
        # Données DEM
        self.dem_snapshots = []
        self.dem_file_indices = []
        self.n_particles = 0
        self.dem_diameters = None
        self.species_labels = None
        
        # Partitionneurs
        self.current_partitioner = None
        self.partitioners = {}  # {name: partitioner}
        
        # Résultats RSD (stockage centralisé)
        self.dem_rsd_results = {}      # {partitioner_name: rsd_data}
        self.markov_rsd_results = {}   # {experiment_name: rsd_data}
        
        # Conditions initiales partagées
        self.initial_time = 250        # Temps de départ par défaut
        self.C0 = None                 # Concentration initiale de référence
        self.phi_A_0 = None            # Distribution espèce A initiale
        self.phi_total_0 = None        # Distribution totale initiale
    
    # ─────────────────────────────────────────────────────────────────────
    # DÉTECTION DE MÉTHODE
    # ─────────────────────────────────────────────────────────────────────
    
    def _detect_method(self, folder_name, params=None):
        """
        Détecte la méthode de partitionnement depuis le nom du dossier ou les params.
        
        Args:
            folder_name: nom du dossier
            params: dict de paramètres (optionnel)
        Returns:
            str: nom de la méthode
        """
        # Depuis les params/config
        if params:
            if "method" in params:
                return params["method"]
            # Ancien format cartésien (a nx/ny/nz mais pas de "method")
            if "nx" in params and "method" not in params:
                return "cartesian"
        
        # Depuis le nom du dossier
        for method, prefixes in METHOD_PREFIXES.items():
            for prefix in prefixes:
                if folder_name.startswith(prefix):
                    return method
        
        return "unknown"
    
    def _parse_experiment_info(self, folder_name, params, stats):
        """
        Extrait les infos clés d'une expérience de manière uniforme.
        
        Returns:
            dict avec n_states, nlt, step_size, start_index, description
        """
        info = {
            "folder": folder_name,
            "n_states": None,
            "nlt": None,
            "step_size": None,
            "start_index": None,
            "description": "",
        }
        
        # Depuis stats
        if stats:
            info["n_states"] = stats.get("n_states")
            info["nlt"] = stats.get("n_timesteps_used")
        
        # Depuis params/config
        if params:
            # Nouveau format (config.json)
            if "method_kwargs" in params:
                kwargs = params["method_kwargs"]
                info["description"] = str(kwargs)
            
            info["nlt"] = info["nlt"] or params.get("nlt") or params.get("NLT")
            info["step_size"] = params.get("step_size")
            info["start_index"] = params.get("start_index")
            
            # Ancien format cartésien
            if "nx" in params and "method" not in params:
                nx = params.get("nx", "?")
                ny = params.get("ny", "?")
                nz = params.get("nz", "?")
                info["description"] = f"nx={nx}, ny={ny}, nz={nz}"
                if info["n_states"] is None:
                    try:
                        info["n_states"] = int(nx) * int(ny) * int(nz)
                    except:
                        pass
        
        # Fallback depuis la matrice
        return info
    
    # ─────────────────────────────────────────────────────────────────────
    # CHARGEMENT
    # ─────────────────────────────────────────────────────────────────────
    
    def _load_npy(self, full_path):
        """Charge un .npy depuis le bucket."""
        with self.fs.open(full_path, "rb") as f:
            return np.load(io.BytesIO(f.read()))
    
    def _load_json(self, full_path):
        """Charge un .json depuis le bucket."""
        with self.fs.open(full_path, "r") as f:
            return json.load(f)
    
    def _load_partitioner_data(self, partitioner_path):
        """
        Charge les données du partitionnement depuis le bucket.
        
        Args:
            partitioner_path: chemin du dossier /partitioner
        
        Returns:
            dict avec métadonnées et données spécifiques (e.g., r_edges pour cylindrique)
        """
        # Charger les métadonnées
        meta_file = f"{partitioner_path}/partitioner_meta.json"
        meta = self._load_json(meta_file)
        
        partitioner_data = {
            "type": meta.get("type"),
            "label": meta.get("label"),
            "n_cells": meta.get("n_cells"),
        }
        
        # Charger les données spécifiques selon le type
        partitioner_type = meta.get("type")
        
        if partitioner_type == "CylindricalPartitioner":
            try:
                # Charger les paramètres cylindriques
                cyl_params = self._load_json(f"{partitioner_path}/cylindrical_params.json")
                partitioner_data.update(cyl_params)
                
                # Charger les edges des rayons
                r_edges = self._load_npy(f"{partitioner_path}/r_edges.npy")
                partitioner_data["r_edges"] = r_edges
            except Exception as e:
                print(f"⚠️  Impossible de charger les données cylindriques: {e}")
        
        elif partitioner_type == "CartesianPartitioner":
            try:
                cart_params = self._load_json(f"{partitioner_path}/cartesian_params.json")
                partitioner_data.update(cart_params)
            except Exception as e:
                print(f"⚠️  Impossible de charger les données cartésiennes: {e}")
        
        elif partitioner_type == "VoronoiPartitioner":
            try:
                vor_params = self._load_json(f"{partitioner_path}/voronoi_params.json")
                partitioner_data.update(vor_params)
                
                centroids = self._load_npy(f"{partitioner_path}/centroids.npy")
                partitioner_data["centroids"] = centroids
            except Exception as e:
                print(f"⚠️  Impossible de charger les données Voronoï: {e}")
        
        return partitioner_data
    
    def _list_folders(self, base_path=BUCKET_BASE):
        """Liste les sous-dossiers d'un chemin."""
        try:
            items = self.fs.ls(base_path)
            return sorted([
                item["name"].split("/")[-1]
                for item in items
                if item["type"] == "directory"
            ])
        except FileNotFoundError:
            return []
    
    def _load_experiment(self, base_path=BUCKET_BASE, folder_name=None):
        """
        Charge une expérience depuis un dossier du bucket.
        
        Option A: Lire particle_diameter depuis stats.json et construire le bon bucket path
        Option B: Essayer la base_path fournie, puis tous les autres buckets
        
        Gère les deux formats:
        - Ancien: params.json + stats.json + transition_matrix.npy
        - Nouveau: config.json + stats.json + transition_matrix.npy
        
        Essaie aussi de charger les données du partitionnement si disponibles.
        """
        # ✅ Option A: Essayer le bucket principal fourni
        prefix = f"{base_path}/{folder_name}"
        
        # Essayer de charger stats.json DABORD pour déterminer le bon bucket (Option A)
        stats = {}
        try:
            stats = self._load_json(f"{prefix}/stats.json")
            # ✅ Extraire particle_diameter et reconstruire le bucket si nécessaire
            particle_diameter = stats.get("particle_diameter")
            if particle_diameter is not None:
                correct_bucket_prefix = _get_bucket_prefix_from_particle_diameter(particle_diameter)
                correct_base_path = f"hf://buckets/{BUCKET_ID}/{correct_bucket_prefix}"
                if correct_base_path != base_path:
                    print(f"     ℹ️  bucket fourni {base_path} → rechargement depuis {correct_bucket_prefix} (particle_diameter={particle_diameter})")
                    base_path = correct_base_path
                    prefix = f"{base_path}/{folder_name}"
        except:
            pass  # Stats pas encore chargé, on continue
        
        # ✅ Option B: Lister les buckets à essayer (par ordre de priorité)
        buckets_to_try = [base_path] + [b for b in ALL_BUCKET_BASES if b != base_path]
        
        # ✅ NOUVEAU: Ajouter aussi le chemin direct (sans markov_results/)
        # ex: hf://buckets/ktongue/DEM_MCM/SMALL/dossier
        direct_base = base_path.replace("/markov_results", "")
        if direct_base not in buckets_to_try:
            buckets_to_try.append(direct_base)
        
        loaded = False
        last_error = None
        
        for attempt_path in buckets_to_try:
            prefix = f"{attempt_path}/{folder_name}"
            
            try:
                # Matrice (obligatoire)
                matrix = self._load_npy(f"{prefix}/transitionmatrix.npy")
                
                # Params (essayer config.json puis params.json)
                params = {}
                for fname in ["config.json", "params.json"]:
                    try:
                        params = self._load_json(f"{prefix}/{fname}")
                        break
                    except:
                        continue
                
                # Stats (recharger si première tentative échouée)
                if not stats:
                    try:
                        stats = self._load_json(f"{prefix}/stats.json")
                    except:
                        pass
                
                # Centroïdes (voronoi)
                centroids = None
                try:
                    centroids = self._load_npy(f"{prefix}/centroids.npy")
                except:
                    pass
                
                # Données de partitionnement
                partitioner_data = None
                try:
                    partitioner_data = self._load_partitioner_data(f"{prefix}/partitioner")
                except:
                    pass
                
                # Méthode
                method = self._detect_method(folder_name, params)
                
                # Infos
                info = self._parse_experiment_info(folder_name, params, stats)
                if info["n_states"] is None:
                    info["n_states"] = matrix.shape[0]
                
                loaded = True
                break  # ✅ Succès! Arrêter la boucle
                
            except Exception as e:
                last_error = e
                continue  # ✅ Essayer le bucket suivant
        
        # ✅ Si chargement réussi, retourner les données
        if loaded:
            return {
                "matrix": matrix,
                "params": params,
                "stats": stats,
                "method": method,
                "info": info,
                "centroids": centroids,
                "partitioner_data": partitioner_data,
            }
        else:
            # Lancer une exception avec tous les buckets essayés
            buckets_str = ", ".join([b.replace("hf://buckets/ktongue/DEM_MCM/", "") for b in buckets_to_try])
            raise Exception(f"Impossible de charger {folder_name} depuis les buckets: {buckets_str}. Erreur: {last_error}")
            
    def load_single(self, folder_name):
        """
        Charge UNIQUEMENT un dossier spécifique (pas de scan automatique).
        ✅ Option B: Cherche dans tous les buckets (ALL_BUCKET_BASES)
        """
        print(f"🔍 Chargement ciblé : {folder_name}")
        
        # ✅ Option B: Essayer TOUS les buckets
        for base_path in ALL_BUCKET_BASES:
            bucket_name = base_path.replace("hf://buckets/ktongue/DEM_MCM/", "")
            print(f"  → Test bucket {bucket_name}...", end=" ")
            
            try:
                # Utiliser _load_experiment qui gère Option A + fallback
                data = self._load_experiment(base_path=base_path, folder_name=folder_name)
                
                # Stocker
                self.results = {folder_name: data}
                print(f"✅ Trouvé")
                return True
                
            except Exception as e:
                print(f"❌")
                continue
        
        print(f"\n❌ {folder_name} introuvable dans tous les buckets")
        return False
    
    def load_single_folder(self, folder_name):
        """Charge un seul dossier même s'il ne match pas la méthode."""
        self.results = {}
        data = self._load_experiment(BUCKET_BASE, folder_name)
        self.results[folder_name] = data
        print(f"✅ {folder_name} chargé")
    def load_all(self, include_old=True):
        """
        Charge toutes les expériences depuis le bucket.
        ✅ Option B: Cherche dans TOUS les buckets (BIG, SMALL, Experiments, OLD)
        
        Args:
            include_old: inclure les anciennes données (non utilisé car cherche ALL_BUCKET_BASES)
        """
        self.results = {}
        self.by_method = defaultdict(dict)
        loaded_folders = set()  # ✅ Track dossiers déjà chargés pour éviter les doublons
        
        # ✅ Option B: Chercher dans TOUS les buckets
        for base_path in ALL_BUCKET_BASES:
            bucket_name = base_path.replace("hf://buckets/ktongue/DEM_MCM/", "")
            print(f"📂 Chargement depuis bucket '{bucket_name}'...")
            
            try:
                folders = self._list_folders(base_path)
                print(f"   {len(folders)} dossiers trouvés")
                
                for folder in folders:
                    if folder in loaded_folders:
                        continue  # ✅ Déjà chargé depuis un autre bucket
                    
                    try:
                        data = self._load_experiment(base_path, folder)
                        self.results[folder] = data
                        self.by_method[data["method"]][folder] = data
                        loaded_folders.add(folder)
                        print(f"   ✅ [{data['method']:12s}] {folder}: "
                              f"shape={data['matrix'].shape}")
                    except Exception as e:
                        print(f"   ⚠️  {folder}: {e}")
            except Exception as e:
                print(f"   ⚠️  Impossible de lister le bucket: {e}")
        
        # Résumé
        print(f"\n{'='*60}")
        print(f"RÉSUMÉ: {len(self.results)} expériences chargées")
        print(f"{'='*60}")
        for method, exps in sorted(self.by_method.items()):
            print(f"   {method:15s}: {len(exps):3d} expériences")
        print()
    
    def load_method(self, method):
        """
        Charge uniquement les expériences d'une méthode.
        ✅ Option B: Cherche dans TOUS les buckets (BIG, SMALL, Experiments, OLD)
        """
        self.results = {}
        self.by_method = defaultdict(dict)
        loaded_folders = set()  # ✅ Track dossiers déjà chargés
        
        # ✅ Option B: Chercher dans TOUS les buckets
        for base_path in ALL_BUCKET_BASES:
            bucket_name = base_path.replace("hf://buckets/ktongue/DEM_MCM/", "")
            
            try:
                folders = self._list_folders(base_path)
                for folder in folders:
                    if folder in loaded_folders:
                        continue  # ✅ Déjà chargé depuis un autre bucket
                    
                    detected = self._detect_method(folder)
                    if detected == method:
                        try:
                            data = self._load_experiment(base_path, folder)
                            self.results[folder] = data
                            self.by_method[method][folder] = data
                            loaded_folders.add(folder)
                            print(f"   ✅ {folder}: shape={data['matrix'].shape}")
                        except Exception as e:
                            print(f"   ⚠️  {folder}: {e}")
            except Exception as e:
                print(f"   ⚠️  Impossible de lister {bucket_name}: {e}")
        
        print(f"\n{len(self.results)} expériences {method} chargées")
    
    # ─────────────────────────────────────────────────────────────────────
    # ACCÈS AUX DONNÉES
    # ─────────────────────────────────────────────────────────────────────
    
    def get_methods(self):
        """Retourne la liste des méthodes disponibles."""
        return list(self.by_method.keys())
    
    def get_experiments(self, method=None):
        """
        Retourne les expériences, optionnellement filtrées par méthode.
        
        Args:
            method: str ou None (toutes)
        Returns:
            dict {folder_name: data}
        """
        if method is None:
            return self.results
        return dict(self.by_method.get(method, {}))
    
    def get_matrix(self, folder_name):
        """Accès rapide à une matrice."""
        return self.results[folder_name]["matrix"]
    
    def get_matrices_by_method(self, method):
        """Retourne {folder_name: matrix} pour une méthode."""
        return {
            name: data["matrix"]
            for name, data in self.by_method.get(method, {}).items()
        }
    
    def summary_table(self):
        """Retourne un tableau récapitulatif de toutes les expériences."""
        rows = []
        for name, data in self.results.items():
            M = data["matrix"]
            diag = np.diag(M)
            row_sums = M.sum(axis=0)
            visited = row_sums > 0
            
            rows.append({
                "name": name,
                "method": data["method"],
                "n_states": M.shape[0],
                "n_visited": int(visited.sum()),
                "nlt": data["info"]["nlt"],
                "step": data["info"]["step_size"],
                "start": data["info"]["start_index"],
                "diag_mean": float(diag.mean()),
                "diag_std": float(diag.std()),
                "row_sum_min": float(row_sums[visited].min()) if visited.any() else 0,
                "row_sum_max": float(row_sums[visited].max()) if visited.any() else 0,
            })
        
        rows.sort(key=lambda r: (r["method"], r["n_states"]))
        return rows
    
    def print_summary(self):
        """Affiche le résumé formaté."""
        rows = self.summary_table()
        
        print(f"\n{'Method':>12s} | {'Name':40s} | {'States':>6s} | {'Visit':>5s} | "
              f"{'NLT':>4s} | {'P(stay)':>8s} | {'ΣRow':>12s}")
        print("-" * 110)
        
        current_method = None
        for r in rows:
            if r["method"] != current_method:
                current_method = r["method"]
                print(f"{'─'*12}─┼{'─'*52}┼{'─'*8}┼{'─'*7}┼{'─'*6}┼{'─'*10}┼{'─'*14}")
            
            nlt_str = str(r["nlt"]) if r["nlt"] else "?"
            print(f"{r['method']:>12s} | {r['name'][:50]:50s} | {r['n_states']:6d} | "
                  f"{r['n_visited']:5d} | {nlt_str:>4s} | {r['diag_mean']:8.4f} | "
                  f"[{r['row_sum_min']:.3f}, {r['row_sum_max']:.3f}]")
    
    # ─────────────────────────────────────────────────────────────────────
    # SIMULATION
    # ─────────────────────────────────────────────────────────────────────
    
    def simulate_mixing(self, folder_name, n_steps=100, initial_split=0.5):
        """
        Simule le mélange à partir d'une matrice de transition.
        
        Args:
            folder_name: nom de l'expérience
            n_steps: nombre de pas de temps
            initial_split: fraction de la frontière initiale
        
        Returns:
            S_history: array (n_steps, n_states)
        """
        M = self.get_matrix(folder_name)
        n_states = M.shape[0]
        
        # État initial: séparation binaire
        S = np.zeros(n_states)
        mid = int(n_states * initial_split)
        S[:mid] = 1.0
        S[mid:] = 0.0
        S = S / S.sum() if S.sum() > 0 else S
        
        S_history = np.zeros((n_steps, n_states))
        for i in range(n_steps):
            S = S @ M
            S_history[i] = S
        
        return S_history
    
    # ─────────────────────────────────────────────────────────────────────
    # VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────
    
    def plot_matrix(self, folder_name, log_scale=False, figsize=(8, 7)):
        """Affiche la matrice de transition en heatmap."""
        data = self.results[folder_name]
        M = data["matrix"]
        method = data["method"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        kwargs = {"cmap": "viridis", "aspect": "auto"}
        if log_scale:
            kwargs["norm"] = LogNorm(vmin=max(M[M > 0].min(), 1e-6), vmax=M.max())
        
        im = ax.imshow(M, **kwargs)
        ax.set_xlabel("État destination")
        ax.set_ylabel("État source")
        ax.set_title(f"Matrice P — {method}\n{folder_name}")
        plt.colorbar(im, ax=ax, label="Probabilité de transition")
        plt.tight_layout()
        plt.show()
    
    # def plot_experiment(self, folder_name, n_steps=100, figsize=(16, 10)):
    #     """Visualisation complète d'une expérience."""
    #     data = self.results[folder_name]
    #     M = data["matrix"]
    #     method = data["method"]
    #     n_states = M.shape[0]
        
    #     # Simulation
    #     S_history = self.simulate_mixing(folder_name, n_steps)
        
    #     fig = plt.figure(figsize=figsize)
    #     fig.suptitle(f"{method.upper()} — {folder_name}", fontsize=14, fontweight="bold")
    #     gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        
    #     # 1. Matrice de transition
    #     ax1 = fig.add_subplot(gs[0, 0])
    #     im = ax1.imshow(M, cmap="viridis", aspect="auto")
    #     ax1.set_xlabel("Dest")
    #     ax1.set_ylabel("Source")
    #     ax1.set_title("Matrice P")
    #     plt.colorbar(im, ax=ax1, fraction=0.046)
        
    #     # 2. Diagonale
    #     ax2 = fig.add_subplot(gs[0, 1])
    #     diag = np.diag(M)
    #     ax2.bar(range(n_states), diag, color=METHOD_COLORS.get(method, "#333"), alpha=0.8)
    #     ax2.axhline(diag.mean(), color="red", ls="--", label=f"μ={diag.mean():.3f}")
    #     ax2.set_xlabel("État")
    #     ax2.set_ylabel("P(rester)")
    #     ax2.set_title("Diagonale de P")
    #     ax2.legend()
        
    #     # 3. Somme des lignes
    #     ax3 = fig.add_subplot(gs[0, 2])
    #     row_sums = M.sum(axis=1)
    #     ax3.bar(range(n_states), row_sums, color="steelblue", alpha=0.8)
    #     ax3.axhline(1.0, color="red", ls="--", alpha=0.5)
    #     ax3.set_xlabel("État")
    #     ax3.set_ylabel("ΣP")
    #     ax3.set_title(f"Somme des lignes\n[{row_sums[row_sums>0].min():.3f}, {row_sums.max():.3f}]")
        
    #     # 4. Évolution temporelle
    #     ax4 = fig.add_subplot(gs[1, 0:2])
    #     step = max(1, n_states // 10)
    #     for j in range(0, n_states, step):
    #         ax4.plot(range(n_steps), S_history[:, j], label=f"État {j}")
    #     ax4.set_xlabel("Pas de temps")
    #     ax4.set_ylabel("Probabilité")
    #     ax4.set_title("Évolution temporelle")
    #     ax4.legend(fontsize=7, ncol=2)
    #     ax4.grid(True, alpha=0.3)
        
    #     # 5. Distribution finale vs initiale
    #     ax5 = fig.add_subplot(gs[1, 2])
    #     mid = n_states // 2
    #     S0 = np.zeros(n_states)
    #     S0[:mid] = 1.0
    #     S0 = S0 / S0.sum()
        
    #     ax5.bar(range(n_states), S0, alpha=0.4, label="Initial", color="blue")
    #     ax5.bar(range(n_states), S_history[-1], alpha=0.4, label=f"t={n_steps}", color="red")
    #     ax5.set_xlabel("État")
    #     ax5.set_ylabel("Probabilité")
    #     ax5.set_title("Initiale vs Finale")
    #     ax5.legend()
        
    #     plt.savefig(f"analysis_{folder_name[:50]}.png", dpi=150, bbox_inches="tight")
    #     plt.show()
    
    def compare_methods(self, metric="diag_mean", figsize=(14, 6)):
        """
        Compare toutes les méthodes sur une métrique.
        
        Args:
            metric: "diag_mean", "n_states", "row_sum_range"
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x_offset = 0
        tick_positions = []
        tick_labels = []
        method_spans = []
        
        for method in sorted(self.by_method.keys()):
            exps = self.by_method[method]
            if not exps:
                continue
            
            start_x = x_offset
            color = METHOD_COLORS.get(method, "#333")
            
            # Trier par nombre d'états
            sorted_exps = sorted(exps.items(), key=lambda x: x[1]["matrix"].shape[0])
            
            for name, data in sorted_exps:
                M = data["matrix"]
                diag = np.diag(M)
                row_sums = M.sum(axis=1)
                visited = row_sums > 0
                
                if metric == "diag_mean":
                    value = diag.mean()
                elif metric == "n_states":
                    value = M.shape[0]
                elif metric == "row_sum_range":
                    value = row_sums[visited].max() - row_sums[visited].min() if visited.any() else 0
                elif metric == "n_visited":
                    value = visited.sum()
                else:
                    value = 0
                
                ax.bar(x_offset, value, color=color, alpha=0.8, width=0.8)
                
                # Label court
                short = name.replace(f"{method}_", "").replace("_NLT", "\nNLT")[:25]
                tick_positions.append(x_offset)
                tick_labels.append(short)
                x_offset += 1
            
            method_spans.append((start_x, x_offset - 1, method))
            x_offset += 1  # espace entre méthodes
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(f"Comparaison inter-méthodes: {metric}")
        
        # Légende des méthodes
        for start, end, method in method_spans:
            mid = (start + end) / 2
            ax.annotate(method.upper(), xy=(mid, ax.get_ylim()[1]),
                       ha="center", va="bottom", fontsize=10, fontweight="bold",
                       color=METHOD_COLORS.get(method, "#333"))
        
        ax.grid(True, alpha=0.2, axis="y")
        plt.tight_layout()
        plt.savefig(f"compare_methods_{metric}.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    def compare_within_method(self, method, sweep_param="n_states", figsize=(12, 8)):
        """
        Compare les expériences au sein d'une même méthode.
        
        Args:
            method: "voronoi", "cartesian", etc.
            sweep_param: "n_states", "nlt", "step_size"
        """
        exps = self.get_experiments(method)
        if not exps:
            print(f"Aucune expérience pour {method}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"{method.upper()} — Sweep sur {sweep_param}", fontsize=14)
        color = METHOD_COLORS.get(method, "#333")
        
        # Collecter les données
        data_points = []
        for name, data in exps.items():
            M = data["matrix"]
            info = data["info"]
            diag = np.diag(M)
            row_sums = M.sum(axis=1)
            visited = row_sums > 0
            
            if sweep_param == "n_states":
                x_val = M.shape[0]
            elif sweep_param == "nlt":
                x_val = info.get("nlt") or 0
            elif sweep_param == "step_size":
                x_val = info.get("step_size") or 0
            elif sweep_param == "start_index":
                x_val = info.get("start_index") or 0
            else:
                x_val = M.shape[0]
            
            data_points.append({
                "x": x_val,
                "name": name,
                "diag_mean": diag.mean(),
                "diag_std": diag.std(),
                "n_visited": int(visited.sum()),
                "n_states": M.shape[0],
                "row_sum_min": float(row_sums[visited].min()) if visited.any() else 0,
                "row_sum_max": float(row_sums[visited].max()) if visited.any() else 0,
            })
        
        data_points.sort(key=lambda d: d["x"])
        xs = [d["x"] for d in data_points]
        
        # 1. Diagonale moyenne
        ax = axes[0, 0]
        ax.plot(xs, [d["diag_mean"] for d in data_points], "o-", color=color)
        ax.fill_between(
            xs,
            [d["diag_mean"] - d["diag_std"] for d in data_points],
            [d["diag_mean"] + d["diag_std"] for d in data_points],
            alpha=0.2, color=color,
        )
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("P(rester)")
        ax.set_title("Diagonale moyenne ± σ")
        ax.grid(True, alpha=0.3)
        
        # 2. Fraction visitée
        ax = axes[0, 1]
        fracs = [d["n_visited"] / d["n_states"] for d in data_points]
        ax.plot(xs, fracs, "o-", color=color)
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("Fraction visitée")
        ax.set_title("États visités / total")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # 3. Somme des lignes (min/max)
        ax = axes[1, 0]
        ax.fill_between(
            xs,
            [d["row_sum_min"] for d in data_points],
            [d["row_sum_max"] for d in data_points],
            alpha=0.3, color=color, label="[min, max]",
        )
        ax.axhline(1.0, color="red", ls="--", alpha=0.5, label="Idéal = 1")
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("Σ lignes")
        ax.set_title("Somme des lignes [min, max]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Nombre d'états visités
        ax = axes[1, 1]
        ax.plot(xs, [d["n_visited"] for d in data_points], "s-", color=color, label="Visités")
        ax.plot(xs, [d["n_states"] for d in data_points], "x--", color="gray", label="Total")
        ax.set_xlabel(sweep_param)
        ax.set_ylabel("Nombre d'états")
        ax.set_title("États visités vs total")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"sweep_{method}_{sweep_param}.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    def compute_rsd(
    self,
    folder_name,
    n_steps=200,
    initial_time=None,
    partitioner=None,
    species_labels=None,
    use_dem_initial_conditions=True,  # ← NOUVEAU paramètre
):
        """
        Calcule le RSD prédit par Markov.
        
        **NOUVEAU** : Peut maintenant utiliser les mêmes conditions initiales que la DEM
        pour une comparaison cohérente.
        
        Args:
            folder_name: nom de l'expérience
            n_steps: nombre de pas de simulation
            initial_time: instant DEM pour C0 (None = self.initial_time)
            partitioner: partitionneur fitté
            species_labels: labels des espèces (None = self.species_labels)
            use_dem_initial_conditions: si True, calcule C0 depuis les données DEM
            
        Returns:
            dict avec rsd, concentration_history, entropy, etc.
        """
        M = self.get_matrix(folder_name)
        n_states = M.shape[0]
        
        # Temps de départ
        if initial_time is None:
            initial_time = self.initial_time
        
        # ══════════════════════════════════════════════════════════════
        # CONDITION INITIALE
        # ══════════════════════════════════════════════════════════════
        
        if use_dem_initial_conditions:
            # ── Utiliser les données DEM (synchronisé avec compute_dem_rsd) ──
            if partitioner is None:
                raise ValueError(
                    "partitioner requis pour calculer les conditions initiales DEM"
                )
            
            if species_labels is None:
                if self.species_labels is None:
                    raise ValueError("species_labels requis (appelez label_species())")
                species_labels = self.species_labels
            
            # Charger le snapshot à initial_time
            if not self.dem_snapshots or self.dem_snapshots[0]["t"] != initial_time:
                print(f"🔄 Chargement du snapshot DEM à t={initial_time}...")
                self.load_dem_snapshots(file_indices=[initial_time])
            
            snap0 = self.dem_snapshots[0]
            coords0 = snap0["coords"]
            actual_time = int(snap0["t"])
            
            # États des particules
            states0 = partitioner.compute_states(
                coords0[:, 0], coords0[:, 1], coords0[:, 2]
            )
            
            # Comptage par cellule
            ntotal = np.bincount(states0, minlength=n_states).astype(float)
            nA = np.bincount(states0[species_labels], minlength=n_states).astype(float)
            
            # Concentration initiale : C0 = nA / ntotal
            C = np.zeros(n_states)
            mask = ntotal > 0
            C[mask] = nA[mask] / ntotal[mask]
            
            # ── Stocker pour réutilisation ──
            self.C0 = C.copy()
            self.phi_A_0 = nA.copy()
            self.phi_total_0 = ntotal.copy()
            self.initial_time = actual_time
            
            print(f"✅ Conditions initiales DEM à t={actual_time}")
            print(f"   Concentration totale: {C.sum():.4f}")
            print(f"   Cellules actives: {mask.sum()}/{n_states}")
            
        else:
            # ── Condition initiale artificielle (ancienne méthode) ──
            C = np.zeros(n_states)
            mid = n_states // 2
            C[:mid] = 1.0
            C[mid:] = 0.0
            C = C / C.sum() if C.sum() > 0 else C
            
            print("⚠️  Utilisation d'une condition initiale artificielle (moitié/moitié)")
        
        # ══════════════════════════════════════════════════════════════
        # SIMULATION MARKOV
        # ══════════════════════════════════════════════════════════════
        
        concentration_history = np.zeros((n_steps, n_states))
        rsd = np.zeros(n_steps)
        entropy = np.zeros(n_steps)
        
        for t in range(n_steps):
            
            visited = C > 1e-12
            if visited.sum() > 1:
                mean_c = C[visited].mean()
                std_c = C[visited].std()
                rsd[t] = std_c / mean_c if mean_c > 0 else 0
            else:
                rsd[t] = 0
            concentration_history[t] = C
            C = C @ M
            
            # Entropie
            C_active = C[visited]
            if len(C_active) > 0:
                C_clip = np.clip(C_active, 1e-10, 1 - 1e-10)
                H = -np.mean(
                    C_clip * np.log(C_clip) + (1 - C_clip) * np.log(1 - C_clip)
                )
                entropy[t] = H / np.log(2)
            else:
                entropy[t] = 0
        
        # ══════════════════════════════════════════════════════════════
        # MÉTRIQUES
        # ══════════════════════════════════════════════════════════════
        
        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
        
        mixing_time_50 = None
        mixing_time_90 = None
        for t in range(n_steps):
            if mixing_time_50 is None and rsd[t] < 0.5 * rsd_0:
                mixing_time_50 = t
            if mixing_time_90 is None and rsd[t] < 0.1 * rsd_0:
                mixing_time_90 = t
        
        result = {
            "rsd": rsd,
            "rsd_percent": rsd * 100,
            "concentration_history": concentration_history,
            "entropy": entropy,
            "rsd_initial": float(rsd[0]),
            "rsd_final": float(rsd[-1]),
            "mixing_time_50": mixing_time_50,
            "mixing_time_90": mixing_time_90,
            "n_states": n_states,
            "C0": self.C0 if use_dem_initial_conditions else concentration_history[0],
            "initial_time": self.initial_time if use_dem_initial_conditions else 0,
            "source": "Markov (DEM IC)" if use_dem_initial_conditions else "Markov (artificial IC)",
        }
        
        # ── Stocker dans l'attribut de classe ──
        self.markov_rsd_results[folder_name] = result
        self.concentration_history = concentration_history
        self.rsd = rsd
        
        return result
    """
Ajoutez ces méthodes à la classe MarkovAnalyzer dans analyze_results.py
"""

# ═══════════════════════════════════════════════════════════════════
# CHARGEMENT DES DONNÉES DEM
# ═══════════════════════════════════════════════════════════════════
    def load_dem_snapshots(self, file_indices=None, sample_every=1):
        """
        Charge les positions des particules DEM à plusieurs instants.

        Les particules conservent leur index (ligne) entre les fichiers:
        la particule i dans le fichier t est la même particule physique
        que la particule i dans le fichier t+1.

        Args:
            file_indices: liste d'indices de fichiers (None = auto)
            sample_every: sous-échantillonnage spatial (1 = toutes)

        Returns:
            list de dict {t, coords} stocké dans self.dem_snapshots
        """
    
        import polars as pl

        if not hasattr(self, '_dem_fs'):
            self._dem_fs = HfFileSystem()
            self._dem_files = sorted(
                self._dem_fs.glob("hf://buckets/ktongue/DEM_MCM/Output Paraview/*.csv")
            )
            print(f"{len(self._dem_files)} fichiers DEM disponibles")

        if file_indices is None:
            file_indices = list(range(0, min(len(self._dem_files), 500), 10))

        self.dem_snapshots = []
        self.dem_file_indices = file_indices

        print(f"📂 Chargement de {len(file_indices)} snapshots DEM...")
        for i, idx in enumerate(file_indices):
            with self._dem_fs.open(self._dem_files[idx], "rb") as f:
                df = pl.read_csv(f)
                coords = np.column_stack([
                    df["coordinates:0"].to_numpy(),
                    df["coordinates:1"].to_numpy(),
                    df["coordinates:2"].to_numpy(),
                ])[::sample_every]

                # === NOUVEAU : récupération des diamètres (uniquement au premier snapshot) ===
                if i == 0:
                    self.dem_diameters = df["Diameter"].to_numpy()[::sample_every]
                    print(f"   Diamètres chargés : {len(self.dem_diameters)} particules")
                # ===========================================================================

                self.dem_snapshots.append({"t": idx, "coords": coords})

            if (i + 1) % 10 == 0 or i == len(file_indices) - 1:
                print(f"   [{i+1}/{len(file_indices)}] t={idx}: {len(coords)} particules")

        self.n_particles = len(self.dem_snapshots[0]["coords"])
        print(f"✅ {len(self.dem_snapshots)} snapshots | {self.n_particles} particules/snapshot")
        return self.dem_snapshots


    def label_species(self, criterion="small", custom_labels=None):
        """
        Étiquette chaque particule comme espèce A (True) ou B (False) à t=0.
        L'étiquette est PERMANENTE.
        
        Nouveau critère basé sur le diamètre des particules :
            - "large" : particules de diamètre 0.008 m -> True, 0.004 m -> False
            - "small" : inverse (0.004 m -> True)
            - "auto"  : détection automatique (la valeur la plus grande = True)
        """
        if custom_labels is not None:
            self.species_labels = np.asarray(custom_labels, dtype=bool)
            n_a = self.species_labels.sum()
            print(f"✅ Labels custom: {n_a} A / {len(self.species_labels) - n_a} B")
            return self.species_labels

        # if not hasattr(self, 'dem_diameters'):
        #     raise AttributeError("Les diamètres DEM n'ont pas été chargés. Exécutez load_dem_snapshots() d'abord.")

        diameters = self.dem_diameters

        # Détermination automatique des deux tailles
        unique_vals = np.unique(diameters)
        if len(unique_vals) != 2:
            print(f"⚠️ Attention : {len(unique_vals)} diamètres différents trouvés (attendu 2).")
            # On prend les deux valeurs extrêmes
            small_val, large_val = unique_vals[0], unique_vals[-1]
        else:
            small_val, large_val = unique_vals[0], unique_vals[1]

        print(f"📏 Diamètres détectés : {small_val:.4f} m et {large_val:.4f} m")

        if criterion == "large":
            labels = diameters == large_val # labels est une liste de booléen de taille celle des particules
        elif criterion == "small":
            labels = diameters == small_val
        elif criterion == "auto":
            labels = diameters == large_val   # par défaut, grande taille = True
        else:
            raise ValueError(f"Critère '{criterion}' non reconnu. Utilisez 'large', 'small' ou 'auto'.")

        self.species_labels = labels 
        n_a = labels.sum()
        print(f"✅ Espèces ({criterion}): {n_a} particules A / {len(labels) - n_a} particules B")
        return self.species_labels


    def create_partitioner_for_comparison(self, method, method_kwargs):
        """
        Crée et fit un partitionneur sur les données DEM.

        Args:
            method: "cartesian", "voronoi", "cylindrical", "quantile"
            method_kwargs: dict de paramètres

        Returns:
            partitioner fitté
        """
        # from src.partitioners import create_partitioner
        from .partitioners import create_partitioner

        # Agréger les données pour le fit
        all_coords = np.vstack([s["coords"] for s in self.dem_snapshots])

        part = create_partitioner(method, **method_kwargs)
        part.fit(all_coords)

        diag = part.diagnostics(all_coords)
        print(f"🔧 {part.label}: {part.n_cells} cellules | "
            f"{diag['n_visited']} visitées | "
            f"pop μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}")

        return part


    # ═══════════════════════════════════════════════════════════════════
    # CALCUL DU RSD — DONNÉES DEM RÉELLES
    # ═══════════════════════════════════════════════════════════════════

    def compute_dem_rsd(self, partitioner, species_labels=None, partitioner_name=None):
        """
        Calcule le RSD (Relative Standard Deviation) à partir des données DEM réelles.

        À chaque instant t:
        1. Assigner chaque particule à sa cellule via le partitionneur
        2. Pour chaque cellule i:
            C_i(t) = n_A(i,t) / n_total(i,t) 
            (concentration de l'espèce A dans la cellule i)
        3. RSD(t) = std(C_i) / mean(C_i)  sur les cellules non-vides

        **NOUVEAU** :
        - Stocke les résultats dans self.dem_rsd_results pour accès global
        - Stocke les conditions initiales (C0, phi_A_0, phi_total_0) pour synchronisation Markov
        - Gère automatiquement le chargement des snapshots si absent

        Args:
            partitioner: partitionneur fitté (obligatoire)
            species_labels: array bool (None = self.species_labels)
            partitioner_name: nom pour stockage (None = partitioner.label)

        Returns:
            dict avec:
                - times: array des temps (indices des fichiers DEM)
                - rsd: array des RSD
                - rsd_percent: idem en %
                - concentrations: array (n_snaps, n_states) des concentrations C_i(t)
                - populations: list de arrays (nombre de particules par cellule)
                - entropy: entropie normalisée H(t) / log(2)
                - intensity_of_segregation: I(t) = σ²(C) / (C̄(1-C̄))
                - rsd_initial: float
                - rsd_final: float
                - mixing_time_50: int ou None (temps pour RSD = 0.5 * RSD_0)
                - mixing_time_90: int ou None (temps pour RSD = 0.1 * RSD_0)
                - n_states: int (nombre de cellules)
                - source: "DEM"
        """
        
        # ══════════════════════════════════════════════════════════════
        # 1. VALIDATION ET PRÉPARATION
        # ══════════════════════════════════════════════════════════════
        
        # Vérifier le partitionneur
        if partitioner is None:
            raise ValueError("❌ partitioner est obligatoire pour compute_dem_rsd()")
        
        n_states = partitioner.n_cells
        
        # Labels des espèces
        if species_labels is None:
            if self.species_labels is None:
                print("⚠️  species_labels non fourni, appel automatique de label_species()")
                self.label_species()
            species_labels = self.species_labels
        
        # Vérifier les snapshots DEM
        if not hasattr(self, 'dem_snapshots') or not self.dem_snapshots:
            print("⚠️  Aucun snapshot DEM chargé, chargement automatique...")
            # Charger par défaut de t=250 à t=6000 par pas de 50
            self.load_dem_snapshots(file_indices=list(range(250, 6000, 50)))
        
        n_snaps = len(self.dem_snapshots)
        
        if n_snaps == 0:
            raise ValueError("❌ Aucun snapshot DEM disponible après chargement")
        
        # Nom du partitionneur pour stockage
        if partitioner_name is None:
            partitioner_name = partitioner.label
        
        print(f"\n{'═'*70}")
        print(f"📊 CALCUL DU RSD DEM")
        print(f"{'═'*70}")
        print(f"Partitionneur   : {partitioner_name}")
        print(f"Nombre d'états  : {n_states}")
        print(f"Snapshots DEM   : {n_snaps} (t={self.dem_snapshots[0]['t']} → {self.dem_snapshots[-1]['t']})")
        print(f"Espèce A        : {species_labels.sum()} particules / {len(species_labels)} total")
        print(f"{'─'*70}")
        
        # ══════════════════════════════════════════════════════════════
        # 2. INITIALISATION DES TABLEAUX DE RÉSULTATS
        # ══════════════════════════════════════════════════════════════
        
        times = np.zeros(n_snaps)
        rsd = np.zeros(n_snaps)
        entropy = np.zeros(n_snaps)
        intensity_seg = np.zeros(n_snaps)
        concentrations = []  # Liste de arrays C_i(t)
        populations = []     # Liste de arrays n_total(i,t)
        
        # ══════════════════════════════════════════════════════════════
        # 3. BOUCLE SUR LES SNAPSHOTS DEM
        # ══════════════════════════════════════════════════════════════
        
        for k, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            times[k] = snap["t"]
            
            # ── 3.1 Assigner les particules aux cellules ──
            states = partitioner.compute_states(
                coords[:, 0], coords[:, 1], coords[:, 2]
            )
            
            # ── 3.2 Compter par cellule: total et espèce A ──
            n_total = np.bincount(states, minlength=n_states).astype(float)
            n_A = np.bincount(states[species_labels], minlength=n_states).astype(float)
            
            # ── 3.3 Concentration C_i = n_A / n_total ──
            C = np.zeros(n_states)
            mask = n_total > 0
            C[mask] = n_A[mask] / n_total[mask]
            
            # Stocker
            concentrations.append(C.copy())
            populations.append(n_total.copy())
            
            # ── 3.4 RSD sur cellules non-vides ──
            C_active = C[mask]
            if len(C_active) > 1 and C_active.mean() > 0:
                rsd[k] = C_active.std() / C_active.mean()
            else:
                rsd[k] = 0
            
            # ── 3.5 Entropie de mélange normalisée ──
            # H = -Σ [C_i ln(C_i) + (1-C_i) ln(1-C_i)] / N_cells_actives
            # Normalisé par log(2) (entropie max pour distribution binaire)
            if len(C_active) > 0:
                C_clip = np.clip(C_active, 1e-10, 1 - 1e-10)
                H = -np.mean(C_clip * np.log(C_clip) + (1 - C_clip) * np.log(1 - C_clip))
                H_max = np.log(2)
                entropy[k] = H / H_max if H_max > 0 else 0
            else:
                entropy[k] = 0
            
            # ── 3.6 Intensité de ségrégation: I = σ²(C) / (C̄(1-C̄)) ──
            C_bar = C_active.mean()
            if 0 < C_bar < 1 and len(C_active) > 1:
                intensity_seg[k] = C_active.var() / (C_bar * (1 - C_bar))
            else:
                intensity_seg[k] = 0
            
            # Affichage progression
            if (k + 1) % 10 == 0 or k == 0 or k == n_snaps - 1:
                print(f"   [{k+1:4d}/{n_snaps}] t={int(times[k]):5d} | "
                    f"RSD={rsd[k]*100:6.2f}% | "
                    f"Entropy={entropy[k]:.4f} | "
                    f"Cellules actives={mask.sum():3d}/{n_states}")
        
        # ══════════════════════════════════════════════════════════════
        # 4. CALCUL DES TEMPS DE MÉLANGE
        # ══════════════════════════════════════════════════════════════
        
        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
        
        mixing_time_50 = None
        mixing_time_90 = None
        
        for k in range(n_snaps):
            if mixing_time_50 is None and rsd[k] < 0.5 * rsd_0:
                mixing_time_50 = int(times[k])
            if mixing_time_90 is None and rsd[k] < 0.1 * rsd_0:
                mixing_time_90 = int(times[k])
        
        # ══════════════════════════════════════════════════════════════
        # 5. STOCKAGE DES CONDITIONS INITIALES (pour synchronisation Markov)
        # ══════════════════════════════════════════════════════════════
        
        coords0 = self.dem_snapshots[0]["coords"]
        states0 = partitioner.compute_states(
            coords0[:, 0], coords0[:, 1], coords0[:, 2]
        )
        
        self.phi_total_0 = np.bincount(states0, minlength=n_states).astype(float)
        self.phi_A_0 = np.bincount(states0[species_labels], minlength=n_states).astype(float)
        
        mask0 = self.phi_total_0 > 0
        self.C0 = np.zeros(n_states)
        self.C0[mask0] = self.phi_A_0[mask0] / self.phi_total_0[mask0]
        
        self.initial_time = int(times[0])
        
        # ══════════════════════════════════════════════════════════════
        # 6. CONSTRUCTION DU RÉSULTAT
        # ══════════════════════════════════════════════════════════════
        
        result = {
            "times": times,
            "rsd": rsd,
            "rsd_percent": rsd * 100,
            "concentrations": np.array(concentrations),  # shape: (n_snaps, n_states)
            "populations": populations,
            "entropy": entropy,
            "intensity_of_segregation": intensity_seg,
            "rsd_initial": float(rsd[0]),
            "rsd_final": float(rsd[-1]),
            "mixing_time_50": mixing_time_50,
            "mixing_time_90": mixing_time_90,
            "n_states": n_states,
            "source": "DEM",
            "partitioner_name": partitioner_name,
            "n_snapshots": n_snaps,
        }
    
        # ══════════════════════════════════════════════════════════════
        # 7. STOCKAGE DANS L'ATTRIBUT DE CLASSE
        # ══════════════════════════════════════════════════════════════
        
        self.dem_rsd_results[partitioner_name] = result
        
        # Stocker aussi le partitionneur pour accès ultérieur
        self.partitioners[partitioner_name] = partitioner
        self.current_partitioner = partitioner
        
        # ══════════════════════════════════════════════════════════════
        # 8. AFFICHAGE DU RÉSUMÉ
        # ══════════════════════════════════════════════════════════════
        
        print(f"{'─'*70}")
        print(f"✅ RÉSULTATS DU CALCUL RSD DEM")
        print(f"{'─'*70}")
        print(f"RSD initial     : {result['rsd_initial']*100:6.2f}%")
        print(f"RSD final       : {result['rsd_final']*100:6.2f}%")
        print(f"Réduction RSD   : {(1 - result['rsd_final']/max(result['rsd_initial'], 1e-10))*100:6.2f}%")
        print(f"Entropie finale : {entropy[-1]:.4f} / 1.000 (max)")
        print(f"t₅₀ (RSD ÷ 2)  : {mixing_time_50 or 'Non atteint'}")
        print(f"t₉₀ (RSD ÷ 10) : {mixing_time_90 or 'Non atteint'}")
        print(f"{'─'*70}")
        print(f"Stocké dans     : self.dem_rsd_results['{partitioner_name}']")
        print(f"Conditions init : self.C0 (shape={self.C0.shape}) à t={self.initial_time}")
        print(f"{'═'*70}\n")
        
        return result

    def compare_rsd_synchronized(
    self,
    folder_name,
    partitioner,
    n_steps=200,
    initial_time=250,
    species_criterion="large",
    figsize=(16, 8),
):
        """
        Compare le RSD DEM vs Markov avec conditions initiales SYNCHRONISÉES.
        
        Cette méthode garantit que :
        1. Les deux calculs partent du même instant t=initial_time
        2. Les mêmes conditions initiales C0 sont utilisées
        3. Les résultats sont stockés pour accès ultérieur
        
        Args:
            folder_name: nom de l'expérience Markov
            partitioner: partitionneur fitté
            n_steps: nombre de pas Markov
            initial_time: instant DEM de départ
            species_criterion: critère de labeling
            
        Returns:
            dict avec {"dem": rsd_data, "markov": rsd_data, "comparison": metrics}
        """
        # ══════════════════════════════════════════════════════════════
        # 1. PRÉPARATION DES DONNÉES
        # ══════════════════════════════════════════════════════════════
        
        # Charger snapshots si nécessaire
        if not self.dem_snapshots:
            print("📂 Chargement des snapshots DEM...")
            self.load_dem_snapshots(file_indices=list(range(initial_time, 6000, 50)))
        
        # Labeler espèces si nécessaire
        if self.species_labels is None:
            print(f"🏷️  Labeling des espèces ({species_criterion})...")
            self.label_species(species_criterion)
        
        # Stocker le temps initial
        self.initial_time = initial_time
        
        # ══════════════════════════════════════════════════════════════
        # 2. CALCUL RSD DEM
        # ══════════════════════════════════════════════════════════════
        
        print(f"\n📊 Calcul RSD DEM (partitionneur: {partitioner.label})...")
        dem_rsd = self.compute_dem_rsd(
            partitioner, 
            partitioner_name=f"{partitioner.label}_t{initial_time}"
        )
        print(f"   RSD DEM: {dem_rsd['rsd_initial']*100:.2f}% → {dem_rsd['rsd_final']*100:.2f}%")
        
        # ══════════════════════════════════════════════════════════════
        # 3. CALCUL RSD MARKOV (mêmes conditions initiales)
        # ══════════════════════════════════════════════════════════════
        
        print(f"\n📊 Calcul RSD Markov ({folder_name})...")
        markov_rsd = self.compute_rsd(
            folder_name,
            n_steps=n_steps,
            initial_time=initial_time,
            partitioner=partitioner,
            species_labels=self.species_labels,
            use_dem_initial_conditions=True,  # ← ESSENTIEL
        )
        print(f"   RSD Markov: {markov_rsd['rsd_initial']*100:.2f}% → {markov_rsd['rsd_final']*100:.2f}%")
        
        # ══════════════════════════════════════════════════════════════
        # 4. VÉRIFICATION DE LA SYNCHRONISATION
        # ══════════════════════════════════════════════════════════════
        
        c0_diff = np.abs(self.C0 - markov_rsd['C0']).max()
        print(f"\n✅ Vérification synchronisation:")
        print(f"   Temps initial: DEM={dem_rsd['times'][0]}, Markov={markov_rsd['initial_time']}")
        print(f"   Différence C0 (max): {c0_diff:.2e}")
        print(f"   RSD initial: DEM={dem_rsd['rsd_initial']:.6f}, Markov={markov_rsd['rsd_initial']:.6f}")
        
        # ══════════════════════════════════════════════════════════════
        # 5. MÉTRIQUES DE COMPARAISON
        # ══════════════════════════════════════════════════════════════
        
        n = min(len(dem_rsd["rsd"]), len(markov_rsd["rsd"]))
        
        # Corrélation
        corr = np.corrcoef(dem_rsd["rsd"][:n], markov_rsd["rsd"][:n])[0, 1]
        
        # Erreurs
        abs_error = np.abs(dem_rsd["rsd"][:n] - markov_rsd["rsd"][:n]) * 100
        rel_error = np.abs(dem_rsd["rsd"][:n] - markov_rsd["rsd"][:n]) / (dem_rsd["rsd"][:n] + 1e-10) * 100
        rmse = np.sqrt(np.mean((dem_rsd["rsd"][:n] - markov_rsd["rsd"][:n])**2)) * 100
        
        comparison = {
            "correlation": corr,
            "rmse_percent": rmse,
            "mean_abs_error_percent": abs_error.mean(),
            "max_abs_error_percent": abs_error.max(),
            "mean_rel_error_percent": rel_error.mean(),
            "c0_max_diff": c0_diff,
            "synchronized": c0_diff < 1e-6,
        }
        
        # ══════════════════════════════════════════════════════════════
        # 6. VISUALISATION
        # ══════════════════════════════════════════════════════════════
        
        self._plot_synchronized_comparison(
            dem_rsd, markov_rsd, comparison, partitioner, folder_name, figsize
        )
        
        return {
            "dem": dem_rsd,
            "markov": markov_rsd,
            "comparison": comparison,
        }


    def _plot_synchronized_comparison(
        self, dem_rsd, markov_rsd, comparison, partitioner, experiment_name, figsize
    ):
        """Visualisation de la comparaison synchronisée."""
        import matplotlib.gridspec as gridspec
        
        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"COMPARAISON SYNCHRONISÉE DEM vs MARKOV\n"
            f"{partitioner.label} | Expérience: {experiment_name}\n"
            f"Corrélation: {comparison['correlation']:.4f} | RMSE: {comparison['rmse_percent']:.2f}%",
            fontsize=14, fontweight="bold",
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        times_dem = dem_rsd["times"]
        times_mkv = np.arange(len(markov_rsd["rsd"])) + dem_rsd["times"][0]
        
        # ── 1. RSD comparaison ──
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(times_dem, dem_rsd["rsd_percent"], "o-", color="#1f77b4",
                lw=2.5, markersize=5, label="DEM (réel)", alpha=0.8)
        ax.plot(times_mkv, markov_rsd["rsd_percent"], "s--", color="#ff7f0e",
                lw=2.5, markersize=5, label="Markov (prédit)", alpha=0.8)
        
        # Vérification point de départ
        ax.plot(times_dem[0], dem_rsd["rsd_percent"][0], "o", 
                color="green", markersize=12, alpha=0.5, label="Départ synchronisé")
        
        ax.set_xlabel("Temps (index fichier DEM)")
        ax.set_ylabel("RSD (%)")
        ax.set_title("RSD: DEM vs Markov")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # ── 2. Échelle log ──
        ax = fig.add_subplot(gs[0, 1])
        rsd_dem_pos = np.clip(dem_rsd["rsd_percent"], 1e-3, None)
        rsd_mkv_pos = np.clip(markov_rsd["rsd_percent"], 1e-3, None)
        ax.semilogy(times_dem, rsd_dem_pos, "o-", color="#1f77b4", lw=2, label="DEM")
        ax.semilogy(times_mkv, rsd_mkv_pos, "s--", color="#ff7f0e", lw=2, label="Markov")
        ax.set_xlabel("Temps")
        ax.set_ylabel("RSD (%) — log")
        ax.set_title("RSD (échelle logarithmique)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ── 3. Erreur absolue ──
        ax = fig.add_subplot(gs[0, 2])
        n = min(len(dem_rsd["rsd"]), len(markov_rsd["rsd"]))
        abs_error = np.abs(dem_rsd["rsd"][:n] - markov_rsd["rsd"][:n]) * 100
        
        ax.bar(times_dem[:n], abs_error, 
            width=(times_dem[1]-times_dem[0]) if n > 1 else 1,
            color="#d62728", alpha=0.7)
        ax.axhline(abs_error.mean(), color="black", ls="--", 
                label=f"Moyenne={abs_error.mean():.2f}%")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Erreur absolue (%)")
        ax.set_title("Erreur |RSD_DEM - RSD_Markov|")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ── 4. Concentrations initiales ──
        ax = fig.add_subplot(gs[1, 0])
        states = np.arange(len(self.C0))
        ax.bar(states, self.C0, alpha=0.7, color="#2ca02c", label="C0 (DEM)")
        if markov_rsd["C0"] is not None:
            ax.scatter(states, markov_rsd["C0"], color="red", s=50, 
                    marker="x", label="C0 (Markov)", zorder=10)
        ax.set_xlabel("État")
        ax.set_ylabel("Concentration")
        ax.set_title(f"Conditions initiales (t={self.initial_time})")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        
        # ── 5. Entropie ──
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(times_dem, dem_rsd["entropy"], "o-", color="#1f77b4", 
                lw=2, label="DEM")
        ax.plot(times_mkv, markov_rsd["entropy"], "s--", color="#ff7f0e",
                lw=2, label="Markov")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Parfait")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Entropie normalisée")
        ax.set_title("Entropie de mélange")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ── 6. Tableau récapitulatif ──
        ax = fig.add_subplot(gs[1, 2])
        ax.axis("off")
        
        table_data = [
            ["Métrique", "DEM", "Markov"],
            ["RSD initial (%)", f"{dem_rsd['rsd_initial']*100:.2f}", f"{markov_rsd['rsd_initial']*100:.2f}"],
            ["RSD final (%)", f"{dem_rsd['rsd_final']*100:.2f}", f"{markov_rsd['rsd_final']*100:.2f}"],
            ["t₅₀", f"{dem_rsd['mixing_time_50'] or 'N/A'}", f"{markov_rsd['mixing_time_50'] or 'N/A'}"],
            ["t₉₀", f"{dem_rsd['mixing_time_90'] or 'N/A'}", f"{markov_rsd['mixing_time_90'] or 'N/A'}"],
            ["", "", ""],
            ["Corrélation", f"{comparison['correlation']:.4f}", ""],
            ["RMSE (%)", f"{comparison['rmse_percent']:.2f}", ""],
            ["Erreur moy (%)", f"{comparison['mean_abs_error_percent']:.2f}", ""],
            ["Erreur max (%)", f"{comparison['max_abs_error_percent']:.2f}", ""],
            ["C0 sync", "✅" if comparison["synchronized"] else "❌", ""],
        ]
        
        table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                        colWidths=[0.45, 0.275, 0.275])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Style
        for j in range(3):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        
        if comparison["synchronized"]:
            latest_row=len(table_data)-1
            table[latest_row, 0].set_facecolor("#C6E0B4")
            table[latest_row, 1].set_facecolor("#C6E0B4")
        
        ax.set_title("Résumé", fontsize=12, fontweight="bold", pad=20)
        
        plt.savefig(f"sync_comparison_{experiment_name}.png", dpi=200, bbox_inches="tight")
        plt.show()

# ═══════════════════════════════════════════════════════════════════
# CALCUL DU RSD — PRÉDICTION MARKOV
# ═══════════════════════════════════════════════════════════════════

    def compute_markov_rsd_from_dem(self, P, partitioner, species_labels=None):
        """
        Calcule le RSD prédit par la chaîne de Markov à partir
        de la condition initiale DEM réelle.

        Principe:
        1. À t=0: compter φ_A(i,0) et φ_total(i,0) depuis le DEM
        2. Prédire: φ_A(i,t+1) = φ_A(t) @ P
                    φ_total(i,t+1) = φ_total(t) @ P
        3. C_i(t) = φ_A(i,t) / φ_total(i,t)
        4. RSD(t) = std(C) / mean(C)

        Args:
            P: matrice de transition
            partitioner: partitionneur fitté
            species_labels: array bool

        Returns:
            dict similaire à compute_dem_rsd
        """
        if species_labels is None:
            species_labels = self.species_labels

        n_states = partitioner.n_cells
        n_snaps = len(self.dem_snapshots)

        # ── Condition initiale depuis les données DEM ──
        coords_t0 = self.dem_snapshots[250]["coords"]
        states_t0 = partitioner.compute_states(
            coords_t0[:, 0], coords_t0[:, 1], coords_t0[:, 2]
        )

        # Distribution initiale des particules A et totales par cellule
        phi_total = np.bincount(states_t0, minlength=n_states).astype(float)
        phi_A = np.bincount(states_t0[species_labels], minlength=n_states).astype(float)

        # ── Prédiction Markov ──
        times = np.zeros(n_snaps)
        rsd = np.zeros(n_snaps)
        entropy = np.zeros(n_snaps)
        intensity_seg = np.zeros(n_snaps)
        concentrations = []

        # État courant
        current_phi_A = phi_A.copy()
        current_phi_total = phi_total.copy()

        for k in range(n_snaps):
            times[k] = self.dem_snapshots[k]["t"]

            if k > 0:
                # Nombre de pas Markov entre deux snapshots
                dt = self.dem_snapshots[k]["t"] - self.dem_snapshots[k - 1]["t"]
                for _ in range(int(dt)):
                    current_phi_A = current_phi_A @ P
                    current_phi_total = current_phi_total @ P

            # Concentration prédite
            C = np.zeros(n_states)
            mask = current_phi_total > 1e-10
            C[mask] = current_phi_A[mask] / current_phi_total[mask]
            C = np.clip(C, 0, 1)

            concentrations.append(C.copy())

            # RSD
            C_active = C[mask]
            if len(C_active) > 1 and C_active.mean() > 0:
                rsd[k] = C_active.std() / C_active.mean()
            else:
                rsd[k] = 0

            # Entropie
            C_clip = np.clip(C_active, 1e-10, 1 - 1e-10)
            H = -np.mean(C_clip * np.log(C_clip) + (1 - C_clip) * np.log(1 - C_clip))
            H_max = np.log(2)
            entropy[k] = H / H_max if H_max > 0 else 0

            # Intensité de ségrégation
            C_bar = C_active.mean()
            if 0 < C_bar < 1:
                intensity_seg[k] = C_active.var() / (C_bar * (1 - C_bar))
            else:
                intensity_seg[k] = 0

        # Temps de mélange
        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
        mixing_time_50 = None
        mixing_time_90 = None
        for k in range(n_snaps):
            if mixing_time_50 is None and rsd[k] < 0.5 * rsd_0:
                mixing_time_50 = int(times[k])
            if mixing_time_90 is None and rsd[k] < 0.1 * rsd_0:
                mixing_time_90 = int(times[k])

        return {
            "times": times,
            "rsd": rsd,
            "rsd_percent": rsd * 100,
            "concentrations": concentrations,
            "entropy": entropy,
            "intensity_of_segregation": intensity_seg,
            "rsd_initial": float(rsd[0]),
            "rsd_final": float(rsd[-1]),
            "mixing_time_50": mixing_time_50,
            "mixing_time_90": mixing_time_90,
            "n_states": n_states,
            "source": "Markov",
        }


# ═══════════════════════════════════════════════════════════════════
# COMPARAISON DEM vs MARKOV
# ═══════════════════════════════════════════════════════════════════

    def compare_dem_vs_markov(self, method, method_kwargs,
                            folder_name=None,
                            species_criterion="large",
                            file_indices=None,
                            figsize=(20, 16)):
        """
        Comparaison complète DEM vs Markov pour un partitionnement donné.

        1. Charge les snapshots DEM (si pas déjà fait)
        2. Crée le partitionneur et calcule le RSD DEM
        3. Charge (ou calcule) la matrice P
        4. Calcule le RSD Markov depuis la même condition initiale
        5. Affiche la comparaison

        Args:
            method: "cartesian", "voronoi", "cylindrical", "quantile"
            method_kwargs: paramètres du partitionneur
            folder_name: nom de l'expérience dans le bucket (None = recalculer P)
            species_criterion: critère d'étiquetage des espèces
            file_indices: indices des fichiers DEM à charger
            figsize: taille de la figure

        Returns:
            dict avec dem_rsd, markov_rsd
        """
        # from src.partitioners import create_partitioner
        from .partitioners import create_partitioner

        # ── 1. Charger les snapshots DEM ──
        if not hasattr(self, 'dem_snapshots') or not self.dem_snapshots:
            if file_indices is None:
                file_indices = list(range(0, 500, 5))
            self.load_dem_snapshots(file_indices)

        # ── 2. Étiqueter les espèces ──
        self.label_species(species_criterion)

        # ── 3. Créer le partitionneur ──
        partitioner = self.create_partitioner_for_comparison(method, method_kwargs)

        # ── 4. RSD DEM ──
        print("\n📊 Calcul RSD DEM...")
        dem_rsd = self.compute_dem_rsd(partitioner)
        print(f"   RSD DEM: {dem_rsd['rsd_initial']*100:.1f}% → {dem_rsd['rsd_final']*100:.1f}%")

        # ── 5. Matrice P ──
        if folder_name and folder_name in self.results:
            P = self.results[folder_name]["matrix"]
            print(f"   Matrice P chargée: {folder_name}")
        else:
            print("   Calcul de la matrice P depuis les données DEM...")
            P = self._compute_P_from_dem(partitioner)

        # ── 6. RSD Markov ──
        print("📊 Calcul RSD Markov...")
        markov_rsd = self.compute_markov_rsd_from_dem(P, partitioner)
        print(f"   RSD Markov: {markov_rsd['rsd_initial']*100:.1f}% → {markov_rsd['rsd_final']*100:.1f}%")

        # ── 7. Visualisation ──
        self._plot_dem_vs_markov_comparison(
            dem_rsd, markov_rsd, partitioner, method, figsize
        )

        return {"dem": dem_rsd, "markov": markov_rsd, "partitioner": partitioner, "P": P}


    def _compute_P_from_dem(self, partitioner, species_labels=None):
        """
        Calcule la matrice P directement depuis les snapshots DEM chargés.
        
        Si species_labels est fourni (bool array), filtre les particules et
        construit une P-matrice réduite pour le sous-ensemble spécifié.
        Sinon, construit la P-matrice complète [1030 x 1030].
        """
        n_states = partitioner.n_cells
        T = np.zeros((n_states, n_states))

        for k in range(len(self.dem_snapshots) - 1):
            coords_prev = self.dem_snapshots[k]["coords"]
            coords_curr = self.dem_snapshots[k + 1]["coords"]

            states_prev = partitioner.compute_states(
                coords_prev[:, 0], coords_prev[:, 1], coords_prev[:, 2]
            )
            states_curr = partitioner.compute_states(
                coords_curr[:, 0], coords_curr[:, 1], coords_curr[:, 2]
            )
            
            # Appliquer le filtre espèce si fourni
            if species_labels is not None:
                # species_labels est un masque bool [1030]
                states_prev = states_prev[species_labels]
                states_curr = states_curr[species_labels]

            n = min(len(states_prev), len(states_curr))
            for i in range(n):
                T[states_prev[i], states_curr[i]] += 1

        # Normaliser
        row_sums = T.sum(axis=1, keepdims=True)
        P = np.divide(T, row_sums, where=row_sums > 0, out=np.zeros_like(T))

        print(f"   P calculée: {T.shape[0]}×{T.shape[1]}, diag_mean={np.diag(P).mean():.3f}")
        return P


    def _plot_dem_vs_markov_comparison(self, dem_rsd, markov_rsd,
                                        partitioner, method, figsize=(20, 16)):
        """Affiche la comparaison complète DEM vs Markov."""
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"COMPARAISON DEM vs MARKOV — {method.upper()}\n"
            f"{partitioner.label} | {partitioner.n_cells} cellules",
            fontsize=15, fontweight="bold",
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

        # Extraire les temps (compute_dem_rsd retourne 'times', compute_rsd non)
        times_dem = dem_rsd.get("times", np.arange(len(dem_rsd["rsd"])))
        
        # Pour Markov, calculer les temps depuis initial_time
        # Extraire dt du nom du folder (step=dt) ou utiliser default
        folder_name = [k for k in self.results.keys()][0] if self.results else ""
        import re
        dt_match = re.search(r'step(\d+)_dt(\d+)', folder_name)
        markov_dt = int(dt_match.group(2)) if dt_match else 10
        
        markov_initial = markov_rsd.get("initial_time", self.initial_time)
        times_mkv = markov_initial + np.arange(len(markov_rsd["rsd"])) * markov_dt

        # ── 1. RSD: DEM vs Markov ──
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(times_dem, dem_rsd["rsd_percent"], "o-", color="#1f77b4",
                lw=2, markersize=4, label="DEM (réel)")
        ax.plot(times_mkv, markov_rsd["rsd_percent"], "s--", color="#ff7f0e",
                lw=2, markersize=4, label="Markov (prédit)")
        ax.set_xlabel("Temps (index fichier)")
        ax.set_ylabel("RSD (%)")
        ax.set_title("RSD: DEM vs Markov")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # ── 2. RSD en échelle log ──
        ax = fig.add_subplot(gs[0, 1])
        rsd_dem_pos = np.clip(dem_rsd["rsd_percent"], 1e-3, None)
        rsd_mkv_pos = np.clip(markov_rsd["rsd_percent"], 1e-3, None)
        ax.semilogy(times_dem, rsd_dem_pos, "o-", color="#1f77b4",
                    lw=2, markersize=4, label="DEM")
        ax.semilogy(times_mkv, rsd_mkv_pos, "s--", color="#ff7f0e",
                    lw=2, markersize=4, label="Markov")
        ax.set_xlabel("Temps")
        ax.set_ylabel("RSD (%) — log")
        ax.set_title("RSD (échelle logarithmique)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 3. Intensité de ségrégation ──
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(times_dem, dem_rsd["intensity_of_segregation"], "o-",
                color="#1f77b4", lw=2, markersize=4, label="DEM")
        ax.plot(times_mkv, markov_rsd["intensity_of_segregation"], "s--",
                color="#ff7f0e", lw=2, markersize=4, label="Markov")
        ax.set_xlabel("Temps")
        ax.set_ylabel("I(t)")
        ax.set_title("Intensité de ségrégation I(t) = σ²(C) / C̄(1-C̄)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        # ── 4. Entropie ──
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(times_dem, dem_rsd["entropy"], "o-", color="#1f77b4",
                lw=2, markersize=4, label="DEM")
        ax.plot(times_mkv, markov_rsd["entropy"], "s--", color="#ff7f0e",
                lw=2, markersize=4, label="Markov")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Mélange parfait")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Entropie normalisée")
        ax.set_title("Entropie de mélange")
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 5. Erreur relative ──
        ax = fig.add_subplot(gs[2, 0])
        rsd_dem = dem_rsd["rsd"]
        rsd_mkv = markov_rsd["rsd"]
        n = min(len(rsd_dem), len(rsd_mkv))

        abs_error = np.abs(rsd_dem[:n] - rsd_mkv[:n]) * 100
        rel_error = np.zeros(n)
        for k in range(n):
            if rsd_dem[k] > 1e-6:
                rel_error[k] = abs(rsd_dem[k] - rsd_mkv[k]) / rsd_dem[k] * 100

        ax.bar(times_dem[:n], abs_error, width=times_dem[1] - times_dem[0] if n > 1 else 1,
            color="#d62728", alpha=0.7, label="Erreur absolue (%)")
        ax.set_xlabel("Temps")
        ax.set_ylabel("Erreur RSD (%)")
        ax.set_title(f"Erreur |RSD_DEM - RSD_Markov| — "
                    f"moyenne={abs_error.mean():.2f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ── 6. Tableau récapitulatif ──
        ax = fig.add_subplot(gs[2, 1])
        ax.axis("off")

        # Corrélation
        corr = np.corrcoef(rsd_dem[:n], rsd_mkv[:n])[0, 1] if n > 2 else 0
        rmse = np.sqrt(np.mean((rsd_dem[:n] - rsd_mkv[:n])**2)) * 100

        table_data = [
            ["", "DEM", "Markov"],
            ["RSD initial (%)", f"{dem_rsd['rsd_initial']*100:.1f}", f"{markov_rsd['rsd_initial']*100:.1f}"],
            ["RSD final (%)", f"{dem_rsd['rsd_final']*100:.1f}", f"{markov_rsd['rsd_final']*100:.1f}"],
            ["t₅₀", f"{dem_rsd['mixing_time_50'] or 'N/A'}", f"{markov_rsd['mixing_time_50'] or 'N/A'}"],
            ["t₉₀", f"{dem_rsd['mixing_time_90'] or 'N/A'}", f"{markov_rsd['mixing_time_90'] or 'N/A'}"],
            ["", "", ""],
            ["Corrélation", f"{corr:.4f}", ""],
            ["RMSE (%)", f"{rmse:.2f}", ""],
            ["Erreur moy (%)", f"{abs_error.mean():.2f}", ""],
            ["Erreur max (%)", f"{abs_error.max():.2f}", ""],
        ]

        table = ax.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            colWidths=[0.4, 0.3, 0.3],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)

        # Style
        for j in range(3):
            table[0, j].set_facecolor("#4472C4")
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(1, len(table_data)):
            for j in range(3):
                if i == 5:
                    table[i, j].set_height(0.02)
                if i >= 6:
                    table[i, j].set_facecolor("#E2EFDA")

        ax.set_title("Résumé", fontsize=12, fontweight="bold", pad=20)

        plt.savefig(f"dem_vs_markov_{method}.png", dpi=200, bbox_inches="tight")
        plt.show()



    def plot_dem_vs_markov_simple(self, dem_rsd, partitioner, method, 
                                   folder_name=None, max_time_seconds=60, figsize=(14, 7), save_name=None):
        """
        ✅ Affiche SEULEMENT les courbes RSD DEM vs Markov sur [start, 60s].
        
        - Chaque fichier DEM = 0.01 secondes
        - DEM: fichiers de start à last_available (par pas de 50)
        - Markov: step fichiers = pas de temps en centièmes de secondes
        - Les deux courbes sont alignées sur le même axe de temps
        """
        import re
        from .partitioners import create_partitioner
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if folder_name is None:
            folder_name = [k for k in self.results.keys()][0] if self.results else ""
        
        start_match = re.search(r'start(\d+)', folder_name)
        step_match = re.search(r'step(\d+)', folder_name)
        
        start_file = int(start_match.group(1)) if start_match else 250
        markov_step = int(step_match.group(1)) if step_match else 10
        
        total_files = 5999
        
        print(f"\n📊 Paramètres:")
        print(f"   Start = fichier {start_file} ({start_file*0.01:.2f}s)")
        print(f"   Dernier fichier = {total_files} ({total_files*0.01:.2f}s)")
        print(f"   Markov step = {markov_step} fichiers = {markov_step*0.01:.3f}s par pas")
        
        # Charger les snapshots DEM de start à 6000
        file_indices = list(range(start_file, total_files + 1, 50))
        print(f"   Chargement DEM: {len(file_indices)} fichiers")
        
        self.load_dem_snapshots(file_indices=file_indices)
        if self.species_labels is None:
            self.label_species()
        
        # Créer partitionneur Markov
        nr_match = re.search(r'nr(\d+)', folder_name)
        nth_match = re.search(r'nth(\d+)', folder_name)
        nz_match = re.search(r'nz(\d+)', folder_name)
        nx_match = re.search(r'nx(\d+)', folder_name)
        ny_match = re.search(r'ny(\d+)', folder_name)
        
        if 'cylindrical' in method.lower() and nr_match and nth_match:
            nr = int(nr_match.group(1))
            nth = int(nth_match.group(1))
            nz_val = int(nz_match.group(1)) if nz_match else 1
            markov_part = create_partitioner("cylindrical", nr=nr, ntheta=nth, nz=nz_val)
        elif 'cartesian' in method.lower() and nx_match and ny_match:
            nx = int(nx_match.group(1))
            ny = int(ny_match.group(1))
            nz_val = int(nz_match.group(1)) if nz_match else 1
            markov_part = create_partitioner("cartesian", nx=nx, ny=ny, nz=nz_val)
        else:
            markov_part = partitioner
        
        all_coords = np.vstack([s["coords"] for s in self.dem_snapshots])
        markov_part.fit(all_coords)
        
        n_states = markov_part.n_cells
        species_labels = self.species_labels
        
        # Calcul RSD DEM
        n_snaps = len(self.dem_snapshots)
        rsd_dem = np.zeros(n_snaps)
        times_dem_files = np.array([s["t"] for s in self.dem_snapshots])
        
        for i, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            states = markov_part.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])
            C_i = np.zeros(n_states)
            for sid in range(n_states):
                mask = states == sid
                if mask.sum() > 0:
                    C_i[sid] = species_labels[mask].sum() / mask.sum()
            mask_active = C_i > 0
            if mask_active.sum() > 1:
                rsd_dem[i] = C_i[mask_active].std() / C_i[mask_active].mean()
        
        t_dem_seconds = times_dem_files * 0.01
        
        # Calcul RSD Markov
        M = self.get_matrix(folder_name)
        snap0 = self.dem_snapshots[0]
        coords0 = snap0["coords"]
        states0 = markov_part.compute_states(coords0[:, 0], coords0[:, 1], coords0[:, 2])
        
        C0 = np.zeros(n_states)
        phi_total_0 = np.zeros(n_states)
        for sid in range(n_states):
            mask = states0 == sid
            phi_total_0[sid] = mask.sum()
            if mask.sum() > 0:
                C0[sid] = species_labels[mask].sum()
        
        mask_active = phi_total_0 > 0
        if phi_total_0[mask_active].sum() > 0:
            C0[mask_active] /= phi_total_0[mask_active].sum()
        
        n_steps_markov = (total_files - start_file) // markov_step
        C = C0.copy()
        rsd_markov = np.zeros(n_steps_markov)
        
        for t in range(n_steps_markov):
            if mask_active.sum() > 1:
                rsd_markov[t] = C[mask_active].std() / C[mask_active].mean() if C[mask_active].mean() > 0 else 0
            C = C @ M
        
        t_markov_seconds = (start_file + np.arange(n_steps_markov) * markov_step) * 0.01
        
        print(f"   DEM: {n_snaps} points de {t_dem_seconds[0]:.2f}s à {t_dem_seconds[-1]:.2f}s")
        print(f"   Markov: {n_steps_markov} points de {t_markov_seconds[0]:.2f}s à {t_markov_seconds[-1]:.2f}s")
        
        # Plot
        ax.plot(t_dem_seconds, rsd_dem * 100,
               color="#1f77b4", marker='o', linewidth=2.5, markersize=6,
               label="RSD DEM (réel)", zorder=3, alpha=0.85)
        
        ax.plot(t_markov_seconds, rsd_markov * 100,
               color="#ff7f0e", marker='s', linewidth=2.5, markersize=5, linestyle='--',
               label=f"RSD Markov (step={markov_step})", zorder=2, alpha=0.85)
        
        ax.set_xlabel("Temps (s)", fontsize=13, fontweight='bold')
        ax.set_ylabel("RSD (%)", fontsize=13, fontweight='bold')
        ax.set_title(
            f"Comparaison RSD — DEM vs Markov\n"
            f"{method.upper()} | {partitioner.label} | {partitioner.n_cells} cellules",
            fontsize=14, fontweight='bold', pad=15
        )
        
        ax.legend(fontsize=12, loc='best', framealpha=0.95, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_xlim(t_dem_seconds[0], max_time_seconds)
        ax.set_ylim(bottom=0)
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f"rsd_{method}_{partitioner.n_cells}cells.png"
        
        plt.savefig(save_name, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Figure sauvegardée: {save_name}")
        plt.show()
        
        return fig, ax

    def plot_rsd_vs_tau_comparison(self, partitioner, method, folder_name_template,
                                     tau_list=None, max_time_seconds=60, figsize=(14, 8), save_name=None):
        """
        ✅ Étude de l'influence du pas de temps Markov (tau) sur la cinétique de mélange.
        
        Pour chaque tau, charge la matrice Markov correspondante et compare au RSD DEM.
        Toutes les courbes sont tracées sur le même graphe.
        
        Args:
            partitioner: partitionneur (~20 cellules)
            method: nom de la méthode
            folder_name_template: template avec {tau} (ex: "cylindrical_nr4_nth5_nz1_equal_area_NLT10_step50_dt2_tau{tau}_start250_d0004")
            tau_list: liste des tau à tester (None = [50, 100, 200, 500, 1000])
            max_time_seconds: temps final (par défaut 60s)
            figsize: taille de la figure
            save_name: nom du fichier de sortie
        """
        import re
        # Ne pas utiliser d'import relatif ici
        from partitioners import create_partitioner
        
        if tau_list is None:
            tau_list = [50, 100, 200, 500, 1000]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # ════════════════════════════════════════════════════════════════
        # 1. CALCUL RSD DEM
        # ════════════════════════════════════════════════════════════════
        
        start_file = 250
        total_files = 5999
        file_indices = list(range(start_file, total_files + 1, 50))
        
        print(f"\n📊 Calcul RSD DEM...")
        self.load_dem_snapshots(file_indices=file_indices)
        if self.species_labels is None:
            self.label_species()
        
        all_coords = np.vstack([s["coords"] for s in self.dem_snapshots])
        partitioner.fit(all_coords)
        
        n_states = partitioner.n_cells
        species_labels = self.species_labels
        n_snaps = len(self.dem_snapshots)
        rsd_dem = np.zeros(n_snaps)
        times_dem_files = np.array([s["t"] for s in self.dem_snapshots])
        
        for i, snap in enumerate(self.dem_snapshots):
            coords = snap["coords"]
            states = partitioner.compute_states(coords[:, 0], coords[:, 1], coords[:, 2])
            C_i = np.zeros(n_states)
            for sid in range(n_states):
                mask = states == sid
                if mask.sum() > 0:
                    C_i[sid] = species_labels[mask].sum() / mask.sum()
            mask_active = C_i > 0
            if mask_active.sum() > 1:
                rsd_dem[i] = C_i[mask_active].std() / C_i[mask_active].mean()
        
        t_dem_seconds = times_dem_files * 0.01
        print(f"   DEM: {n_snaps} points de {t_dem_seconds[0]:.2f}s à {t_dem_seconds[-1]:.2f}s")
        
        # Plot DEM curve
        ax.plot(t_dem_seconds, rsd_dem * 100,
               color="black", marker='o', linewidth=3, markersize=8,
               label="RSD DEM (réel)", zorder=10, alpha=0.9)
        
        # ════════════════════════════════════════════════════════════════
        # 2. CALCUL RSD MARKOV POUR CHAQUE TAU
        # ════════════════════════════════════════════════════════════════
        
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(tau_list)))
        
        print(f"\n📊 Calcul RSD Markov pour {len(tau_list)} tau...")
        
        for tau_idx, tau in enumerate(tau_list):
            folder_name = folder_name_template.format(tau=tau)
            dt_markov = tau * 0.01
            
            print(f"\n   ── tau = {tau} ({dt_markov:.3f}s par pas) ──")
            
            # Vérifier si le folder existe
            try:
                M = self.get_matrix(folder_name)
            except Exception as e:
                print(f"   ⚠️  Folder {folder_name} non trouvé: {e}")
                continue
            
            # Conditions initiales depuis le premier snapshot DEM
            snap0 = self.dem_snapshots[0]
            coords0 = snap0["coords"]
            states0 = partitioner.compute_states(coords0[:, 0], coords0[:, 1], coords0[:, 2])
            
            # ✅ Comptes de particules (A et total) par cellule
            phi_A_0 = np.zeros(n_states, dtype=float)
            phi_total_0 = np.zeros(n_states, dtype=float)
            for sid in range(n_states):
                mask = states0 == sid
                phi_total_0[sid] = mask.sum()
                phi_A_0[sid] = species_labels[mask].sum()
            
            mask_active = phi_total_0 > 0
            
            # Simulation Markov: évoluer les comptes, puis recalculer concentrations
            n_steps_markov = (total_files - start_file) // tau
            phi_A = phi_A_0.copy()
            phi_total = phi_total_0.copy()
            rsd_markov = np.zeros(n_steps_markov + 1)
            
            # t=0: concentrations initiales (identiques au DEM)
            C_t0 = np.zeros(n_states)
            C_t0[mask_active] = phi_A[mask_active] / phi_total[mask_active]
            if mask_active.sum() > 1 and C_t0[mask_active].mean() > 0:
                rsd_markov[0] = C_t0[mask_active].std() / C_t0[mask_active].mean()
            
            for t in range(1, n_steps_markov + 1):
                phi_A = phi_A @ M
                phi_total = phi_total @ M
                C_t = np.zeros(n_states)
                C_t[mask_active] = phi_A[mask_active] / phi_total[mask_active]
                if mask_active.sum() > 1:
                    rsd_markov[t] = C_t[mask_active].std() / C_t[mask_active].mean() if C_t[mask_active].mean() > 0 else 0
            
            t_markov_seconds = (start_file + np.arange(n_steps_markov + 1) * tau) * 0.01
            
            # Plot Markov curve
            ax.plot(t_markov_seconds, rsd_markov * 100,
                   color=colors[tau_idx], linewidth=2.5, linestyle='-',
                   label=f"Markov tau={tau} ({n_steps_markov+1} pts)", zorder=5, alpha=0.8)
            
            print(f"   ✅ {n_steps_markov+1} points de {t_markov_seconds[0]:.2f}s à {t_markov_seconds[-1]:.2f}s (incl. t=0)")
        
        # ════════════════════════════════════════════════════════════════
        # 3. STYLING
        # ════════════════════════════════════════════════════════════════
        
        ax.set_xlabel("Temps (s)", fontsize=13, fontweight='bold')
        ax.set_ylabel("RSD (%)", fontsize=13, fontweight='bold')
        ax.set_title(
            f"Influence du pas de temps Markov (tau) sur la cinétique de mélange\n"
            f"{method.upper()} | {partitioner.label} | {n_states} cellules",
            fontsize=14, fontweight='bold', pad=15
        )
        
        ax.legend(fontsize=10, loc='best', framealpha=0.95, edgecolor='black', ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
        ax.set_xlim(t_dem_seconds[0], max_time_seconds)
        ax.set_ylim(bottom=0)
        ax.minorticks_on()
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_name is None:
            save_name = f"rsd_tau_comparison_{method}_{n_states}cells.png"
        
        plt.savefig(save_name, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Figure sauvegardée: {save_name}")
        plt.show()
        
        return fig, ax

    def compare_all_methods_dem_vs_markov(self, species_criterion="z_median",
                                        file_indices=None, figsize=(16, 10)):
        """
        Compare DEM vs Markov pour TOUTES les méthodes sur un seul graphique.

        Args:
            species_criterion: critère de labeling
            file_indices: indices des fichiers DEM
        """
        from src.partitioners import create_partitioner

        # Charger les données
        if not hasattr(self, 'dem_snapshots') or not self.dem_snapshots:
            if file_indices is None:
                file_indices = list(range(0, 500, 5))
            self.load_dem_snapshots(file_indices)

        self.label_species(species_criterion)

        # Configurations à tester
        configs = {
            "Cartésien (5³)": {"method": "cartesian", "kwargs": {"nx": 15, "ny": 15, "nz": 15}},
            "Cylindrique": {"method": "cylindrical", "kwargs": {"nr": 5, "ntheta": 8, "nz": 5, "radial_mode": "equal_area"}},
            "Voronoï (125)": {"method": "voronoi", "kwargs": {"n_cells": 125}},
            "Quantile (5³)": {"method": "quantile", "kwargs": {"nx": 5, "ny": 5, "nz": 5}},
        }

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"DEM vs MARKOV — Toutes les méthodes\n"
            f"(espèces: {species_criterion} | {len(self.dem_snapshots)} snapshots)",
            fontsize=14, fontweight="bold",
        )

        all_results = {}

        colors_dem = "#1f77b4"
        colors_mkv = "#ff7f0e"

        for idx, (name, config) in enumerate(configs.items()):
            row, col = divmod(idx, 2)
            ax = axes[row, col]

            print(f"\n{'─'*50}")
            print(f"📐 {name}")

            # Créer partitionneur
            part = self.create_partitioner_for_comparison(config["method"], config["kwargs"])

            # RSD DEM
            dem_rsd = self.compute_dem_rsd(part)

            # Matrice P
            P = self._compute_P_from_dem(part)

            # RSD Markov
            mkv_rsd = self.compute_markov_rsd_from_dem(P, part)

            all_results[name] = {"dem": dem_rsd, "markov": mkv_rsd}

            # Plot
            t = dem_rsd["times"]
            ax.plot(t, dem_rsd["rsd_percent"], "o-", color=colors_dem,
                    lw=2, markersize=3, label="DEM", alpha=0.8)
            ax.plot(t, mkv_rsd["rsd_percent"], "s--", color=colors_mkv,
                    lw=2, markersize=3, label="Markov", alpha=0.8)

            # Corrélation
            n = min(len(dem_rsd["rsd"]), len(mkv_rsd["rsd"]))
            corr = np.corrcoef(dem_rsd["rsd"][:n], mkv_rsd["rsd"][:n])[0, 1] if n > 2 else 0

            ax.set_title(f"{name}\nCorr={corr:.3f}", fontsize=11)
            ax.set_xlabel("Temps")
            ax.set_ylabel("RSD (%)")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig("dem_vs_markov_all_methods.png", dpi=200, bbox_inches="tight")
        plt.show()

        return all_results
        
    def plot_experiment(self, folder_name, n_steps=200, figsize=(20, 16),partitioner=None):
        """
        Visualisation complète d'une expérience incluant le RSD.

        6 subplots:
            1. Matrice de transition P (heatmap)
            2. Diagonale de P
            3. Somme des lignes
            4. Évolution temporelle des concentrations
            5. RSD + Entropie au cours du temps
            6. Distribution initiale vs finale + RSD annoté
        """
        from matplotlib.colors import LogNorm
        import matplotlib.gridspec as gridspec

        data = self.results[folder_name]
        M = data["matrix"]
        method = data["method"]
        n_states = M.shape[0]

        # ── Calcul du RSD ──
        rsd_data = self.compute_rsd(folder_name, n_steps,partitioner=partitioner)
        C_history = rsd_data["concentration_history"]
        rsd_vals = rsd_data["rsd_percent"]
        entropy_vals = rsd_data["entropy"]

        # ── Figure ──
        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"{method.upper()} — {folder_name}\n"
            f"{n_states} états | RSD initial={rsd_data['rsd_initial']*100:.1f}% → "
            f"final={rsd_data['rsd_final']*100:.1f}%",
            fontsize=14, fontweight="bold",
        )
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

        # ── 1. Matrice P ──
        ax1 = fig.add_subplot(gs[0, 0])
        im = ax1.imshow(M, cmap="viridis", aspect="auto")
        ax1.set_xlabel("Destination")
        ax1.set_ylabel("Source")
        ax1.set_title("Matrice de transition P")
        plt.colorbar(im, ax=ax1, fraction=0.046, label="Probabilité")

        # ── 2. Diagonale ──
        ax2 = fig.add_subplot(gs[0, 1])
        diag = np.diag(M)
        color = METHOD_COLORS.get(method, "#333")
        ax2.bar(range(n_states), diag, color=color, alpha=0.8, width=1.0)
        ax2.axhline(diag.mean(), color="red", ls="--", lw=2,
                    label=f"μ={diag.mean():.3f}")
        ax2.axhline(diag.mean() + diag.std(), color="red", ls=":", alpha=0.5)
        ax2.axhline(diag.mean() - diag.std(), color="red", ls=":", alpha=0.5)
        ax2.set_xlabel("État")
        ax2.set_ylabel("P(rester)")
        ax2.set_title("Diagonale de P")
        ax2.legend()

        # ── 3. Évolution des concentrations ──
        ax3 = fig.add_subplot(gs[1, 0])
        step = max(1, n_states // 10)
        for j in range(0, n_states, step):
            ax3.plot(range(n_steps), C_history[:, j], label=f"Cellule {j}", alpha=0.8)

        # Ligne de concentration uniforme
        ax3.axhline(1.0 / n_states, color="gray", ls=":", alpha=0.5,
                    label=f"Uniforme={1/n_states:.4f}")
        ax3.set_xlabel("Pas de temps")
        ax3.set_ylabel("Concentration C_i(t)")
        ax3.set_title("Évolution de la concentration par cellule")
        ax3.legend(fontsize=7, ncol=2, loc="upper right")
        ax3.grid(True, alpha=0.3)

        # ── 4. RSD + Entropie ──
        ax4 = fig.add_subplot(gs[1, 1])

        color_rsd = "#d62728"
        color_entropy = "#2ca02c"

        ax4_twin = ax4.twinx()

        # RSD
        ax4.plot(range(n_steps), rsd_vals, color=color_rsd, lw=2.5, label="RSD (%)")
        ax4.fill_between(range(n_steps), rsd_vals, alpha=0.1, color=color_rsd)
        ax4.set_xlabel("Pas de temps")
        ax4.set_ylabel("RSD (%)", color=color_rsd)
        ax4.tick_params(axis="y", labelcolor=color_rsd)

        # Entropie
        ax4_twin.plot(range(n_steps), entropy_vals, color=color_entropy, lw=2.5,
                    ls="--", label="Entropie norm.")
        ax4_twin.set_ylabel("Entropie normalisée", color=color_entropy)
        ax4_twin.tick_params(axis="y", labelcolor=color_entropy)
        ax4_twin.set_ylim(0, 1.05)

        # Temps de mélange
        if rsd_data["mixing_time_50"] is not None:
            t50 = rsd_data["mixing_time_50"]
            ax4.axvline(t50, color="orange", ls="--", alpha=0.7,
                        label=f"t₅₀={t50} (RSD÷2)")
        if rsd_data["mixing_time_90"] is not None:
            t90 = rsd_data["mixing_time_90"]
            ax4.axvline(t90, color="purple", ls="--", alpha=0.7,
                        label=f"t₉₀={t90} (RSD÷10)")

        ax4.set_title("RSD et Entropie au cours du mélange")
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, alpha=0.3)

        # ── 5. Distribution initiale vs finale ──
        ax5 = fig.add_subplot(gs[2, 0])

        C_initial = np.zeros(n_states)
        mid = n_states // 2
        C_initial[:mid] = 1.0

        ax5.bar(range(n_states), C_initial, alpha=0.4, label="Initial (ségrégé)",
                color="blue", width=1.0)
        ax5.bar(range(n_states), C_history[-1], alpha=0.4,
                label=f"Final (t={n_steps})", color="red", width=1.0)
        ax5.axhline(1.0 / n_states, color="gray", ls=":", alpha=0.7,
                    label=f"Uniforme={1/n_states:.4f}")
        ax5.set_xlabel("Cellule")
        ax5.set_ylabel("Concentration")
        ax5.set_title(f"Distribution: initiale → finale | "
                    f"RSD={rsd_data['rsd_final']*100:.1f}%")
        ax5.legend(fontsize=8)

        # ── 6. Somme des lignes + annotation RSD ──
        ax6 = fig.add_subplot(gs[2, 1])
        row_sums = M.sum(axis=1)
        visited = row_sums > 0
        ax6.bar(range(n_states), row_sums, color="steelblue", alpha=0.8, width=1.0)
        ax6.axhline(1.0, color="red", ls="--", alpha=0.5, label="Idéal = 1")
        ax6.set_xlabel("État")
        ax6.set_ylabel("Σ P(i→j)")
        ax6.set_title(f"Somme des lignes\n"
                    f"[{row_sums[visited].min():.3f}, {row_sums[visited].max():.3f}]")
        ax6.legend()

        # Annotation avec les métriques RSD
        textstr = (
            f"━━━ Métriques de mélange ━━━\n"
            f"RSD initial:  {rsd_data['rsd_initial']*100:.1f}%\n"
            f"RSD final:    {rsd_data['rsd_final']*100:.1f}%\n"
            f"Réduction:    {(1 - rsd_data['rsd_final']/max(rsd_data['rsd_initial'], 1e-10))*100:.1f}%\n"
            f"t₅₀ (RSD÷2): {rsd_data['mixing_time_50'] or 'N/A'}\n"
            f"t₉₀ (RSD÷10):{rsd_data['mixing_time_90'] or 'N/A'}\n"
            f"Entropie fin: {entropy_vals[-1]:.4f}"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                    edgecolor="gray", alpha=0.9)
        ax6.text(0.95, 0.95, textstr, transform=ax6.transAxes,
                fontsize=9, verticalalignment="top", horizontalalignment="right",
                bbox=props, family="monospace")

        plt.savefig(f"experiment_{folder_name[:50]}.png", dpi=200, bbox_inches="tight")
        plt.show()


    def plot_rsd_comparison(self, folder_names=None, n_steps=200, figsize=(14, 10)):
        """
        Compare le RSD entre plusieurs expériences.

        Args:
            folder_names: liste de noms (None = une par méthode)
            n_steps: nombre de pas de simulation
        """
        if folder_names is None:
            folder_names = []
            for method in sorted(self.by_method.keys()):
                exps = sorted(
                    self.by_method[method].items(),
                    key=lambda x: x[1]["matrix"].shape[0],
                )
                if exps:
                    folder_names.append(exps[len(exps) // 2][0])

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Comparaison du RSD entre méthodes", fontsize=14, fontweight="bold")

        all_rsd_data = {}

        for name in folder_names:
            if name not in self.results:
                print(f"⚠️ {name} non trouvé")
                continue

            method = self.results[name]["method"]
            color = METHOD_COLORS.get(method, "#333")
            n_states = self.results[name]["matrix"].shape[0]

            rsd_data = self.compute_rsd(name, n_steps)
            all_rsd_data[name] = rsd_data
            label = f"{method} ({n_states})"

            # 1. RSD vs temps
            axes[0, 0].plot(range(n_steps), rsd_data["rsd_percent"],
                            # color=color,
                            lw=2, label=label)

            # 2. Entropie vs temps
            axes[0, 1].plot(range(n_steps), rsd_data["entropy"],
                            # color=color,
                             lw=2, label=label)

            # 3. RSD en log
            rsd_pos = rsd_data["rsd_percent"].copy()
            rsd_pos[rsd_pos < 1e-6] = 1e-6
            axes[1, 0].semilogy(range(n_steps), rsd_pos,
                                # color=color,
                                lw=2, label=label)

        # 1. RSD
        ax = axes[0, 0]
        ax.set_xlabel("Pas de temps")
        ax.set_ylabel("RSD (%)")
        ax.set_title("Décroissance du RSD")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", ls=":", alpha=0.3)

        # 2. Entropie
        ax = axes[0, 1]
        ax.set_xlabel("Pas de temps")
        ax.set_ylabel("Entropie normalisée")
        ax.set_title("Convergence entropique")
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="Mélange parfait")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 3. RSD log
        ax = axes[1, 0]
        ax.set_xlabel("Pas de temps")
        ax.set_ylabel("RSD (%) — échelle log")
        ax.set_title("RSD (échelle logarithmique)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 4. Tableau récapitulatif
        ax = axes[1, 1]
        ax.axis("off")

        table_data = []
        headers = ["Méthode", "N états", "RSD₀ %", "RSD_f %", "t₅₀", "t₉₀", "Entropie_f"]

        for name in folder_names:
            if name not in all_rsd_data:
                continue
            rd = all_rsd_data[name]
            method = self.results[name]["method"]
            n_st = rd["n_states"]
            table_data.append([
                f"{method}",
                f"{n_st}",
                f"{rd['rsd_initial']*100:.1f}",
                f"{rd['rsd_final']*100:.1f}",
                f"{rd['mixing_time_50'] or 'N/A'}",
                f"{rd['mixing_time_90'] or 'N/A'}",
                f"{rd['entropy'][-1]:.3f}",
            ])

        if table_data:
            table = ax.table(
                cellText=table_data,
                colLabels=headers,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)

            # Coloriser les en-têtes
            for j, header in enumerate(headers):
                table[0, j].set_facecolor("#4472C4")
                table[0, j].set_text_props(color="white", fontweight="bold")

            # Coloriser la meilleure valeur de RSD final
            rsd_finals = [float(row[3]) for row in table_data]
            best_idx = np.argmin(rsd_finals)
            for j in range(len(headers)):
                table[best_idx + 1, j].set_facecolor("#E2EFDA")

        ax.set_title("Résumé", fontsize=12, fontweight="bold", pad=20)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        plt.savefig("rsd_comparison.png", dpi=200, bbox_inches="tight")
        plt.show()

        return all_rsd_data


    def plot_rsd_vs_resolution(self, method, n_steps=200, figsize=(12, 5)):
        """
        RSD final en fonction de la résolution (nombre d'états) pour une méthode.

        Args:
            method: "cartesian", "voronoi", etc.
            n_steps: nombre de pas de simulation
        """
        exps = self.get_experiments(method)
        if not exps:
            print(f"Aucune expérience pour {method}")
            return

        data_points = []
        for name, exp_data in exps.items():
            rsd_data = self.compute_rsd(name, n_steps)
            data_points.append({
                "n_states": rsd_data["n_states"],
                "rsd_final": rsd_data["rsd_final"] * 100,
                "rsd_initial": rsd_data["rsd_initial"] * 100,
                "mixing_time_50": rsd_data["mixing_time_50"],
                "mixing_time_90": rsd_data["mixing_time_90"],
                "entropy_final": rsd_data["entropy"][-1],
                "name": name,
            })

        data_points.sort(key=lambda d: d["n_states"])

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"{method.upper()} — RSD vs Résolution", fontsize=14)
        color = METHOD_COLORS.get(method, "#333")

        xs = [d["n_states"] for d in data_points]

        # 1. RSD final
        ax = axes[0]
        ax.plot(xs, [d["rsd_final"] for d in data_points], "o-", color=color, lw=2)
        ax.set_xlabel("Nombre d'états")
        ax.set_ylabel("RSD final (%)")
        ax.set_title("RSD final")
        ax.grid(True, alpha=0.3)

        # 2. Temps de mélange
        ax = axes[1]
        t50s = [d["mixing_time_50"] if d["mixing_time_50"] else n_steps for d in data_points]
        t90s = [d["mixing_time_90"] if d["mixing_time_90"] else n_steps for d in data_points]
        ax.plot(xs, t50s, "o-", color="orange", lw=2, label="t₅₀")
        ax.plot(xs, t90s, "s-", color="purple", lw=2, label="t₉₀")
        ax.set_xlabel("Nombre d'états")
        ax.set_ylabel("Temps de mélange")
        ax.set_title("Temps de mélange")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Entropie finale
        ax = axes[2]
        ax.plot(xs, [d["entropy_final"] for d in data_points], "o-", color=color, lw=2)
        ax.axhline(1.0, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Nombre d'états")
        ax.set_ylabel("Entropie normalisée")
        ax.set_title("Entropie finale")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"rsd_vs_resolution_{method}.png", dpi=200, bbox_inches="tight")
        plt.show()
    
    
    def plot_mixing_comparison(self, folder_names=None, n_steps=200, figsize=(14, 6)):
        """
        Compare la convergence du mélange entre plusieurs expériences.
        
        Args:
            folder_names: liste de noms (None = une par méthode)
            n_steps: nombre de pas de simulation
        """
        if folder_names is None:
            # Prendre une expérience par méthode (la plus petite)
            folder_names = []
            for method in sorted(self.by_method.keys()):
                exps = sorted(
                    self.by_method[method].items(),
                    key=lambda x: x[1]["matrix"].shape[0],
                )
                if exps:
                    folder_names.append(exps[len(exps) // 2][0])  # taille médiane
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        for name in folder_names:
            if name not in self.results:
                print(f"⚠️ {name} non trouvé")
                continue
            
            data = self.results[name]
            method = data["method"]
            color = METHOD_COLORS.get(method, "#333")
            
            S_history = self.simulate_mixing(name, n_steps)
            n_states = S_history.shape[1]
            
            # Entropie normalisée (mesure de mélange)
            entropy = np.zeros(n_steps)
            for t in range(n_steps):
                S = S_history[t]
                S_pos = S[S > 0]
                if len(S_pos) > 0:
                    entropy[t] = -np.sum(S_pos * np.log(S_pos)) / np.log(n_states)
            
            # Variance (mesure de ségrégation)
            variance = S_history.var(axis=1)
            
            label = f"{method} ({n_states} états)"
            axes[0].plot(range(n_steps), entropy, label=label, color=color, linewidth=2)
            axes[1].plot(range(n_steps), variance, label=label, color=color, linewidth=2)
        
        axes[0].set_xlabel("Pas de temps")
        axes[0].set_ylabel("Entropie normalisée")
        axes[0].set_title("Convergence du mélange (entropie)")
        axes[0].axhline(1.0, color="gray", ls=":", alpha=0.5, label="Mélange parfait")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel("Pas de temps")
        axes[1].set_ylabel("Variance")
        axes[1].set_title("Décroissance de la ségrégation")
        axes[1].set_yscale("log")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("mixing_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()
    
    def plot_eigenvalues(self, folder_names=None, n_eigenvalues=20, figsize=(12, 5)):
        """
        Compare les valeurs propres des matrices de transition.
        
        Le 2ème plus grand eigenvalue contrôle la vitesse de mélange.
        """
        if folder_names is None:
            folder_names = []
            for method in sorted(self.by_method.keys()):
                exps = sorted(
                    self.by_method[method].items(),
                    key=lambda x: x[1]["matrix"].shape[0],
                )
                if exps:
                    folder_names.append(exps[len(exps) // 2][0])
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        lambda2_data = []
        
        for name in folder_names:
            if name not in self.results:
                continue
            
            data = self.results[name]
            method = data["method"]
            color = METHOD_COLORS.get(method, "#333")
            M = data["matrix"]
            
            # Valeurs propres (les n plus grandes)
            n_eig = min(n_eigenvalues, M.shape[0])
            eigenvalues = np.sort(np.abs(np.linalg.eigvals(M)))[::-1][:n_eig]
            
            label = f"{method} ({M.shape[0]})"
            axes[0].plot(range(len(eigenvalues)), eigenvalues, "o-",
                        label=label, color=color, markersize=4)
            
            if len(eigenvalues) > 1:
                lambda2_data.append({
                    "name": name, "method": method,
                    "lambda2": eigenvalues[1],
                    "n_states": M.shape[0],
                })
        
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("|λ|")
        axes[0].set_title("Spectre des valeurs propres")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        
        # 2ème eigenvalue
        if lambda2_data:
            methods = [d["method"] for d in lambda2_data]
            l2s = [d["lambda2"] for d in lambda2_data]
            colors = [METHOD_COLORS.get(m, "#333") for m in methods]
            labels = [f"{d['method']}\n({d['n_states']})" for d in lambda2_data]
            
            axes[1].bar(range(len(l2s)), l2s, color=colors, alpha=0.8)
            axes[1].set_xticks(range(len(l2s)))
            axes[1].set_xticklabels(labels, fontsize=8)
            axes[1].set_ylabel("|λ₂|")
            axes[1].set_title("2ème valeur propre\n(plus petit = mélange plus rapide)")
            axes[1].grid(True, alpha=0.3, axis="y")
        
        plt.tight_layout()
        plt.savefig("eigenvalues_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()
    def plot_rsd_vs_timestep(self, folder_name=None, timestep_range=None, dem_rsd=None, n_steps=200, figsize=(14, 8), use_dem_initial_conditions=False, partitioner=None, species_labels=None):
        """
        ✅ **NOUVELLE MÉTHODE** - Trace le RSD Markov pour différents timesteps d'initialisation,
        comparé à une courbe DEM constante.
        
        Args:
            folder_name: nom de l'expérience (None = première expérience trouvée)
            timestep_range: list de timesteps à tester (None = [5, 10, 20, 50])
            dem_rsd: RSD DEM constant (ligne horizontale). Si None, pas affiché
            n_steps: nombre de pas de simulation Markov
            figsize: taille de la figure
            use_dem_initial_conditions: si True, utilise les conditions initiales DEM (nécessite partitioner)
            partitioner: partitionneur fitté (requis si use_dem_initial_conditions=True)
            species_labels: labels des espèces (None = self.species_labels)
        
        Returns:
            dict: {timestep: rsd_data_dict, ...}
        
        Example:
            >>> analyzer = MarkovAnalyzer()
            >>> analyzer.load_all()
            >>> analyzer.plot_rsd_vs_timestep("cartesian_nx5_ny5_nz5", 
            ...                              timestep_range=[5, 10, 20, 50],
            ...                              dem_rsd=0.25)
        """
        # ════════════════════════════════════════════════════════════════
        # SÉLECTION DE L'EXPÉRIENCE
        # ════════════════════════════════════════════════════════════════
        
        if folder_name is None:
            # Prendre la première expérience trouvée
            if not self.results:
                print("❌ Aucune expérience chargée")
                return None
            folder_name = list(self.results.keys())[0]
            print(f"📌 Utilisant l'expérience: {folder_name}")
        
        if folder_name not in self.results:
            print(f"❌ Expérience {folder_name} non trouvée")
            return None
        
        # ════════════════════════════════════════════════════════════════
        # PARAMÈTRES PAR DÉFAUT
        # ════════════════════════════════════════════════════════════════
        
        if timestep_range is None:
            timestep_range = [5, 10, 20, 50]
        
        exp_data = self.results[folder_name]
        method = exp_data.get("method", "unknown")
        n_states = exp_data["matrix"].shape[0]
        
        # ════════════════════════════════════════════════════════════════
        # CALCUL DU RSD MARKOV POUR CHAQUE TIMESTEP
        # ════════════════════════════════════════════════════════════════
        
        rsd_by_timestep = {}
        print(f"\n📈 Calcul RSD Markov pour {len(timestep_range)} timesteps...")
        
        for t_init in timestep_range:
            try:
                rsd_data = self.compute_rsd(
                    folder_name, 
                    n_steps=n_steps, 
                    initial_time=t_init,
                    use_dem_initial_conditions=use_dem_initial_conditions,
                    partitioner=partitioner,
                    species_labels=species_labels
                )
                rsd_by_timestep[t_init] = rsd_data
                print(f"   ✅ t={t_init:3d}: RSD initial={rsd_data['rsd_initial']*100:6.2f}%, final={rsd_data['rsd_final']*100:6.2f}%")
            except Exception as e:
                print(f"   ⚠️  t={t_init}: {e}")
        
        if not rsd_by_timestep:
            print("❌ Aucun RSD calculé")
            return None
        
        # ════════════════════════════════════════════════════════════════
        # VISUALISATION
        # ════════════════════════════════════════════════════════════════
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"{method.upper()} — RSD vs Timestep initial ({n_states} états)", 
                    fontsize=14, fontweight='bold')
        
        # Couleurs pour les courbes
        colors = plt.cm.tab10(np.linspace(0, 1, len(rsd_by_timestep)))
        
        # ────── 1. RSD vs temps (linéaire) ──────
        ax = axes[0, 0]
        for (t_init, rsd_data), color in zip(sorted(rsd_by_timestep.items()), colors):
            ax.plot(range(n_steps), rsd_data['rsd_percent'], 
                   color=color, lw=2.5, label=f't_init={t_init}', alpha=0.8)
        
        if dem_rsd is not None:
            ax.axhline(dem_rsd * 100, color='red', linestyle='--', linewidth=2.5, 
                      label=f'DEM RSD = {dem_rsd*100:.2f}%', zorder=10)
        
        ax.set_xlabel('Pas de temps Markov', fontsize=11, fontweight='bold')
        ax.set_ylabel('RSD (%)', fontsize=11, fontweight='bold')
        ax.set_title('1. RSD vs Temps (linéaire)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # ────── 2. RSD vs temps (log) ──────
        ax = axes[0, 1]
        for (t_init, rsd_data), color in zip(sorted(rsd_by_timestep.items()), colors):
            rsd_pos = rsd_data['rsd_percent'].copy()
            rsd_pos[rsd_pos < 1e-6] = 1e-6
            ax.semilogy(range(n_steps), rsd_pos, 
                       color=color, lw=2.5, label=f't_init={t_init}', alpha=0.8)
        
        if dem_rsd is not None:
            ax.axhline(dem_rsd * 100, color='red', linestyle='--', linewidth=2.5, 
                      label=f'DEM RSD = {dem_rsd*100:.2f}%', zorder=10)
        
        ax.set_xlabel('Pas de temps Markov', fontsize=11, fontweight='bold')
        ax.set_ylabel('RSD (%) — log', fontsize=11, fontweight='bold')
        ax.set_title('2. RSD vs Temps (log)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # ────── 3. RSD initial vs timestep ──────
        ax = axes[1, 0]
        t_inits = sorted(rsd_by_timestep.keys())
        rsd_initials = [rsd_by_timestep[t]['rsd_initial'] * 100 for t in t_inits]
        rsd_finals = [rsd_by_timestep[t]['rsd_final'] * 100 for t in t_inits]
        
        ax.plot(t_inits, rsd_initials, 'o-', color='blue', linewidth=2.5, 
               markersize=8, label='RSD initial', alpha=0.7)
        ax.plot(t_inits, rsd_finals, 's-', color='green', linewidth=2.5, 
               markersize=8, label='RSD final', alpha=0.7)
        
        if dem_rsd is not None:
            ax.axhline(dem_rsd * 100, color='red', linestyle='--', linewidth=2.5, 
                      label=f'DEM RSD = {dem_rsd*100:.2f}%', zorder=10)
        
        ax.set_xlabel('Timestep initial (t)', fontsize=11, fontweight='bold')
        ax.set_ylabel('RSD (%)', fontsize=11, fontweight='bold')
        ax.set_title('3. RSD initial/final vs t_init', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
        
        # ────── 4. Entropie vs temps ──────
        ax = axes[1, 1]
        for (t_init, rsd_data), color in zip(sorted(rsd_by_timestep.items()), colors):
            ax.plot(range(n_steps), rsd_data['entropy'], 
                   color=color, lw=2.5, label=f't_init={t_init}', alpha=0.8)
        
        ax.set_xlabel('Pas de temps Markov', fontsize=11, fontweight='bold')
        ax.set_ylabel('Entropie normalisée', fontsize=11, fontweight='bold')
        ax.set_title('4. Entropie vs Temps', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        filename = f"rsd_vs_timestep_{method}_{n_states}states.png"
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"\n✅ Figure sauvegardée: {filename}")
        plt.show()
        
        return rsd_by_timestep


 
# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    analyzer = MarkovAnalyzer()
    
    # Charger tout
    analyzer.load_all()
    
    # Résumé
    analyzer.print_summary()
    
    # Comparaison inter-méthodes
    if len(analyzer.get_methods()) > 1:
        analyzer.compare_methods(metric="diag_mean")
    
    # Analyse par méthode
    for method in analyzer.get_methods():
        n_exps = len(analyzer.get_experiments(method))
        if n_exps > 2:
            print(f"\n📊 Sweep {method.upper()} ({n_exps} expériences):")
            analyzer.compare_within_method(method, sweep_param="n_states")
    
    # Comparaison du mélange
    analyzer.plot_mixing_comparison(n_steps=200)
    
    # Spectre des eigenvalues
    analyzer.plot_eigenvalues()
    
    # Visualisation détaillée d'une expérience
    if analyzer.results:
        first = list(analyzer.results.keys())[0]
        analyzer.plot_experiment(first, n_steps=100)
    
    print("\n✨ Analyse terminée!")
