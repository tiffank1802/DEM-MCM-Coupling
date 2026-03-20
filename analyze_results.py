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

# =============================================================================
# CONFIGURATION
# =============================================================================

BUCKET_ID = "ktongue/DEM_MCM"
BUCKET_PREFIX = "markov_results"
BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"

# Anciennes données cartésiennes (dossier séparé)
OLD_BUCKET_PREFIX = "markov_sweep_results"
OLD_BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{OLD_BUCKET_PREFIX}"

# Méthodes connues et leurs préfixes
METHOD_PREFIXES = {
    "cartesian": ["cartesian_", "NLT_"],   # NLT_ = ancien format cartésien
    "cylindrical": ["cylindrical_"],
    "voronoi": ["voronoi_"],
    "quantile": ["quantile_"],
    "octree": ["octree_"],
    "physics": ["physics_"],
}

# Couleurs par méthode
METHOD_COLORS = {
    "cartesian": "#1f77b4",
    "cylindrical": "#ff7f0e",
    "voronoi": "#2ca02c",
    "quantile": "#d62728",
    "octree": "#9467bd",
    "physics": "#8c564b",
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
        self.results = {}           # {folder_name: {matrix, params, stats, method}}
        self.by_method = defaultdict(dict)  # {method: {folder_name: data}}
    
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
    
    def _list_folders(self, base_path):
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
    
    def _load_experiment(self, base_path, folder_name):
        """
        Charge une expérience depuis un dossier du bucket.
        
        Gère les deux formats:
        - Ancien: params.json + stats.json + transition_matrix.npy
        - Nouveau: config.json + stats.json + transition_matrix.npy
        """
        prefix = f"{base_path}/{folder_name}"
        
        # Matrice (obligatoire)
        matrix = self._load_npy(f"{prefix}/transition_matrix.npy")
        
        # Params (essayer config.json puis params.json)
        params = {}
        for fname in ["config.json", "params.json"]:
            try:
                params = self._load_json(f"{prefix}/{fname}")
                break
            except:
                continue
        
        # Stats
        stats = {}
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
        
        # Méthode
        method = self._detect_method(folder_name, params)
        
        # Infos
        info = self._parse_experiment_info(folder_name, params, stats)
        if info["n_states"] is None:
            info["n_states"] = matrix.shape[0]
        
        return {
            "matrix": matrix,
            "params": params,
            "stats": stats,
            "method": method,
            "info": info,
            "centroids": centroids,
        }
    
    def load_all(self, include_old=True):
        """
        Charge toutes les expériences depuis le bucket.
        
        Args:
            include_old: inclure les anciennes données cartésiennes
        """
        self.results = {}
        self.by_method = defaultdict(dict)
        
        # ── Nouveau format ──
        print(f"📂 Chargement depuis {BUCKET_BASE}...")
        new_folders = self._list_folders(BUCKET_BASE)
        print(f"   {len(new_folders)} dossiers trouvés")
        
        for folder in new_folders:
            try:
                data = self._load_experiment(BUCKET_BASE, folder)
                self.results[folder] = data
                self.by_method[data["method"]][folder] = data
                print(f"   ✅ [{data['method']:12s}] {folder}: "
                      f"shape={data['matrix'].shape}")
            except Exception as e:
                print(f"   ⚠️  {folder}: {e}")
        
        # ── Ancien format cartésien ──
        if include_old:
            print(f"\n📂 Chargement depuis {OLD_BUCKET_BASE}...")
            old_folders = self._list_folders(OLD_BUCKET_BASE)
            print(f"   {len(old_folders)} dossiers trouvés")
            
            for folder in old_folders:
                if folder in self.results:
                    continue  # déjà chargé
                try:
                    data = self._load_experiment(OLD_BUCKET_BASE, folder)
                    self.results[folder] = data
                    self.by_method[data["method"]][folder] = data
                    print(f"   ✅ [{data['method']:12s}] {folder}: "
                          f"shape={data['matrix'].shape}")
                except Exception as e:
                    print(f"   ⚠️  {folder}: {e}")
        
        # Résumé
        print(f"\n{'='*60}")
        print(f"RÉSUMÉ: {len(self.results)} expériences chargées")
        print(f"{'='*60}")
        for method, exps in sorted(self.by_method.items()):
            print(f"   {method:15s}: {len(exps):3d} expériences")
        print()
    
    def load_method(self, method):
        """Charge uniquement les expériences d'une méthode."""
        self.results = {}
        self.by_method = defaultdict(dict)
        
        for base_path in [BUCKET_BASE, OLD_BUCKET_BASE]:
            folders = self._list_folders(base_path)
            for folder in folders:
                detected = self._detect_method(folder)
                if detected == method:
                    try:
                        data = self._load_experiment(base_path, folder)
                        self.results[folder] = data
                        self.by_method[method][folder] = data
                        print(f"   ✅ {folder}: shape={data['matrix'].shape}")
                    except Exception as e:
                        print(f"   ⚠️  {folder}: {e}")
        
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
            row_sums = M.sum(axis=1)
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
                print(f"{'─'*12}─┼{'─'*42}┼{'─'*8}┼{'─'*7}┼{'─'*6}┼{'─'*10}┼{'─'*14}")
            
            nlt_str = str(r["nlt"]) if r["nlt"] else "?"
            print(f"{r['method']:>12s} | {r['name'][:40]:40s} | {r['n_states']:6d} | "
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
    
    
    # Ajoutez ces méthodes à la classe MarkovAnalyzer dans analyze_results.py

    def compute_rsd(self, folder_name, n_steps=200, initial_split=0.5):
        """
        Calcule le RSD (Relative Standard Deviation) des particules
        dans chaque partition au cours du temps.

        Le RSD mesure l'homogénéité du mélange:
            RSD = 0%   → mélange parfait (distribution uniforme)
            RSD = 100% → ségrégation totale

        Formule: RSD(t) = σ(C_i(t)) / μ(C_i(t))
        où C_i(t) est la concentration (fraction de particules) dans la cellule i.

        Args:
            folder_name: nom de l'expérience
            n_steps: nombre de pas de simulation
            initial_split: fraction de la frontière initiale (0.5 = moitié/moitié)

        Returns:
            dict avec:
                - rsd: array (n_steps,) — RSD à chaque pas
                - rsd_percent: array (n_steps,) — RSD en pourcentage
                - concentration_history: array (n_steps, n_states) — C_i(t)
                - entropy: array (n_steps,) — entropie normalisée
                - rsd_initial: float — RSD initial
                - rsd_final: float — RSD final
                - mixing_time_50: int ou None — pas où RSD < 50% du RSD initial
                - mixing_time_90: int ou None — pas où RSD < 10% du RSD initial
        """
        M = self.get_matrix(folder_name)
        n_states = M.shape[0]

        # ── État initial ségrégé ──
        # Concentration initiale: espèce A dans la moitié gauche,
        # espèce B dans la moitié droite
        C = np.zeros(n_states)
        mid = int(n_states * initial_split)
        C[:mid] = 1.0    # 100% d'espèce A dans les cellules 0..mid
        C[mid:] = 0.0    # 0% d'espèce A dans les cellules mid..n

        # ── Simulation ──
        concentration_history = np.zeros((n_steps, n_states))
        rsd = np.zeros(n_steps)
        entropy = np.zeros(n_steps)

        for t in range(n_steps):
            C = C @ M

            # Stocker
            concentration_history[t] = C

            # RSD: σ/μ sur les cellules visitées (P > 0)
            visited = C > 1e-12
            if visited.sum() > 1:
                mean_c = C[visited].mean()
                std_c = C[visited].std()
                rsd[t] = std_c / mean_c if mean_c > 0 else 0
            else:
                rsd[t] = 0

            # Entropie normalisée
            C_pos = C[C > 1e-12]
            if len(C_pos) > 0 and n_states > 1:
                entropy[t] = -np.sum(C_pos * np.log(C_pos)) / np.log(n_states)
            else:
                entropy[t] = 0

        # ── Temps de mélange ──
        rsd_0 = rsd[0] if rsd[0] > 0 else 1.0

        mixing_time_50 = None
        mixing_time_90 = None
        for t in range(n_steps):
            if mixing_time_50 is None and rsd[t] < 0.5 * rsd_0:
                mixing_time_50 = t
            if mixing_time_90 is None and rsd[t] < 0.1 * rsd_0:
                mixing_time_90 = t

        return {
            "rsd": rsd,
            "rsd_percent": rsd * 100,
            "concentration_history": concentration_history,
            "entropy": entropy,
            "rsd_initial": float(rsd[0]),
            "rsd_final": float(rsd[-1]),
            "mixing_time_50": mixing_time_50,
            "mixing_time_90": mixing_time_90,
            "n_states": n_states,
        }


    def plot_experiment(self, folder_name, n_steps=200, figsize=(20, 16)):
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
        rsd_data = self.compute_rsd(folder_name, n_steps)
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
                            color=color, lw=2, label=label)

            # 2. Entropie vs temps
            axes[0, 1].plot(range(n_steps), rsd_data["entropy"],
                            color=color, lw=2, label=label)

            # 3. RSD en log
            rsd_pos = rsd_data["rsd_percent"].copy()
            rsd_pos[rsd_pos < 1e-6] = 1e-6
            axes[1, 0].semilogy(range(n_steps), rsd_pos,
                                color=color, lw=2, label=label)

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