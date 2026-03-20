"""
================================================================================
DOCUMENTATION INDEX — Accès rapide aux docstrings du projet MCM
================================================================================

Ce module ne contient PAS de code exécutable, mais référence 
tous les éléments documentés du projet avec des accès rapides.

Utilisation:
    python -c "import DOCUMENTATION_INDEX; 
               help(DOCUMENTATION_INDEX.markov_workflow)"

    ou

    python -m pydoc partitioners.VoronoiPartitioner
    python -m pydoc run_sweep.run_markov_sweep
    python -m pydoc analyze_results.MarkovAnalyzer.compute_rsd
"""

# ============================================================================
# GUIDE COMPLÈTE DU WORKFLOW
# ============================================================================

def markov_workflow():
    """
    Workflow complet du calcul de matrice de Markov.
    
    Étapes:
    
    1. PRÉPARATION DES DONNÉES
       └─ Données DEM brutes: positions (x, y, z) à différents timesteps
    
    2. PARTITIONNEMENT SPATIAL
       └─ Choisir une méthode (Voronoï recommandé)
       └─ Fit le partitionneur sur les coordonnées
       └─ Chaque particule → indice d'état (0 à n_states-1)
    
    3. SUIVI DES TRANSITIONS
       └─ Charger snapshot_t et snapshot_{t+1}
       └─ Assigner états à chaque particule fois
       └─ Compter transitions: state_t → state_{t+1}
    
    4. CONSTRUCTION DE MATRICE P
       └─ T[i,j] = nombre de particules i→j
       └─ P[i,j] = T[i,j] / sum(T[i,:])
       └─ Résultat: matrice stochastique
    
    5. ANALYSE
       └─ Calculer métriques: RSD, entropie, lambdas propres
       └─ Évaluer qualité du mélange
       └─ Comparer différentes méthodes de partitionnement
    
    Code Minimum:
    
        from partitioners import create_partitioner
        from run_sweep import run_experiment
        import numpy as np
        
        # 1. Charger données
        coordinates = np.load("coords.npy")  # (N, 3)
        
        # 2. Créer partitionneur
        part = create_partitioner("voronoi", n_cells=125)
        part.fit(coordinates)
        
        # 3. Calculer P (voir run_sweep.py pour détails)
        P, stats = run_experiment(config, part, files, fs, device)
        
        # 4. Analyser
        from analyze_results import MarkovAnalyzer
        analyzer = MarkovAnalyzer()
        analyzer.load_all()
        analyzer.compare_methods()
    """
    pass


# ============================================================================
# QUICK REFERENCE — CLASSES & FONCTIONS
# ============================================================================

class QuickReference:
    """
    Référence rapide de TOUS les éléments publics du projet.
    
    PARTITIONNEURS
    ──────────────
    
    create_partitioner(method, **kwargs)
        Factory pour créer un partitionneur
        Args:
            method: "cartesian", "cylindrical", "voronoi", "quantile", "octree", "physics"
            **kwargs: paramètres spécifiques
        Returns: instance de BasePartitioner
        Usage: part = create_partitioner("voronoi", n_cells=125)
    
    VoronoiPartitioner(n_cells=125, random_state=42)
        K-means clustering (recommandé)
        Pros: adaptatif, homogène, littérature MCM
        Cons: coûteux, non-déterministe
        Usage: part = VoronoiPartitioner(n_cells=125)
    
    CartesianPartitioner(nx=5, ny=5, nz=5)
        Grille régulière linéaire
        Pros: simple, rapide, reproductible
        Cons: population inégale
        Usage: part = CartesianPartitioner(nx=10, ny=10, nz=10)
    
    CylindricalPartitioner(nr=5, ntheta=8, nz=5, radial_mode="equal_area")
        Grille cylindrique
        Pros: adapté aux géométries cylindriques
        Cons: paramètrage plus complexe
        Usage: part = CylindricalPartitioner(nr=10, ntheta=12, nz=8)
    
    QuantileGridPartitioner(nx=5, ny=5, nz=5)
        Grille équi-population
        Pros: population homogène, simple
        Cons: forme cellules irrégulière
        Usage: part = QuantileGridPartitioner(nx=5, ny=5, nz=5)
    
    OctreePartitioner(max_particles=100, max_depth=5)
        Subdivision adaptative
        Pros: raffinage automatique en zones denses
        Cons: nombre d'états imprévisible
        Usage: part = OctreePartitioner(max_particles=100, max_depth=5)
    
    PhysicsAwarePartitioner(n_cells=125, velocity_weight=0.3)
        K-means sur [position, vitesse]
        Pros: capture l'information dynamique
        Cons: coûteux, nécessite vitesses
        Usage: part = PhysicsAwarePartitioner(n_cells=125)
    
    ─────────────────────────────────────────────────────────────
    
    BUCKET I/O
    ──────────
    
    save_experiment_to_bucket(folder_name, matrix, stats, config, partitioner_data=None)
        Sauvegarde résultats dans HF bucket
        Args:
            folder_name: "voronoi_125cells_NLT200"
            matrix: ndarray (n_states, n_states)
            stats: dict des statistiques
            config: dict de configuration
        Usage: save_experiment_to_bucket("voronoi_125cells", P, stats, cfg)
    
    load_experiment_from_bucket(folder_name)
        Charge une expérience complète
        Returns: dict avec "matrix", "stats", "config"
        Usage: data = load_experiment_from_bucket("voronoi_125cells")
    
    list_experiments()
        Liste toutes les expériences disponibles
        Returns: list[str]
        Usage: exps = list_experiments()
    
    ─────────────────────────────────────────────────────────────
    
    SWEEP & CALCUL
    ──────────────
    
    run_markov_sweep(method, configs=None, base_dir=BASE_OUTPUT_DIR)
        Lance les calculs pour une méthode donnée
        Args:
            method: "voronoi", "cartesian", ..., "all"
            configs: ExperimentConfig list (None = defaults)
        Usage: run_markov_sweep("voronoi")
    
    ─────────────────────────────────────────────────────────────
    
    ANALYSE & VISUALISATION
    ───────────────────────
    
    MarkovAnalyzer()
        Classe macro pour charger et analyser les résultats
        
        load_all()
            Charge tous les résultats du bucket
        
        load_method(method)
            Charge que les résultats d'une méthode
        
        print_summary()
            Affiche un tableau récapitulatif
        
        compute_rsd(folder_name, n_steps=200)
            Calcule la courbe RSD vs temps
            Returns: dict avec "rsd", "rsd_percent", "entropy", "mixing_time_50", ...
        
        compare_methods(metric="diag_mean")
            Graphique comparant toutes les méthodes
        
        plot_experiment(folder_name, n_steps=200, figsize=(20, 16))
            Visualisation complète d'une expérience (6 subplots)
        
        compare_dem_vs_markov(method, method_kwargs, ...)
            Comparaison prédiction Markov vs données DEM réelles
        
        plot_rsd_comparison(folder_names=None, n_steps=200)
            Compare RSD entre plusieurs expériences
        
        plot_eigenvalues(folder_names=None, n_eigenvalues=20)
            Affiche le spectre propre et λ₂
        
        Usage:
            analyzer = MarkovAnalyzer()
            analyzer.load_all()
            analyzer.plot_experiment("voronoi_125cells_NLT200")
    
    ─────────────────────────────────────────────────────────────
    """
    pass


# ============================================================================
# MÉTRIQUES & INTERPRÉTATION
# ============================================================================

class Metrics:
    """
    Définition des métriques clés du mélange.
    
    == RSD (Relative Standard Deviation) ==
    
    Formule: RSD(t) = σ(C_i(t)) / μ(C_i(t))
    où C_i(t) = concentration de l'espèce A dans cellule i au temps t
    
    Interprétation:
        RSD = 0%   → mélange parfait (distribution uniforme)
        RSD = 100% → ségrégation totale
        RSD = 50%  → demi-chemin
    
    Temps de mélange:
        t₅₀ = 1er temps où RSD < 50% × RSD_initial
        t₉₀ = 1er temps où RSD < 10% × RSD_initial
    Meilleur partitionnement → plus petits t₅₀, t₉₀
    
    == ENTROPIE ==
    
    Formule: H(t) = (1/n_cells) × Σ [C_i ln(C_i) + (1-C_i) ln(1-C_i)] / H_max
    où H_max = ln(2) pour système binaire
    
    Interprétation:
        H = 0   → ségrégation totale
        H = 1   → mélange parfait
    
    == DIAGONALE DE P ==
    
    P_diag = moyenne des éléments diagonaux
    
    Interprétation:
        P_diag proche 0.9   → particles stay long in cells (slow mixing)
        P_diag proche 0.5   → particles change frequently (fast mixing)
        P_diag proche 0.1   → almost no persistence (unrealistic)
    
    == VALEURS PROPRES ==
    
    λ_max = 1 (toujours, pour matrice stochastique)
    λ₂ = 2ème plus grande valeur propre
    
    Vitesse de convergence ∝ λ₂
    λ₂ proche 0 → mélange très rapide (bon)
    λ₂ proche 1 → mélange lent (mauvais)
    """
    pass


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

class CommonIssues:
    """
    Problèmes courants & solutions.
    
    ❌ CONNECTION TIMEOUT
        Cause: réseau HF instable
        Solution: 
            import socket
            socket.setdefaulttimeout(30)
    
    ❌ TOO MANY EMPTY CELLS
        Cause: partitionnement inadapté
        Solution:
            diag = part.diagnostics(coords)
            if diag['fraction_visited'] < 0.7:
                # Utiliser Voronoï ou augmenter résolution
    
    ❌ NAN IN RSD
        Cause: concentration moyenne = 0 dans cellules visitées
        Solution:
            # Vérifier condition initiale
            C_init = np.zeros(n_states)
            C_init[:n_states//2] = 1.0
            assert C_init.sum() > 0
    
    ❌ MATRIX P NOT STOCHASTIC
        Cause: normalization error
        Solution:
            row_sums = P.sum(axis=1)
            print(f"Min: {row_sums.min()}, Max: {row_sums.max()}")
            # Doit être ≈ 1 partout
    
    ❌ OUT OF MEMORY
        Cause: trop de particules ou trop d'états
        Solution:
            # Réduire n_cells ou utiliser step_size plus grand
            part = create_partitioner("voronoï", n_cells=1000)  # → 10000
            config.step_size = 2  # Sauter 1 timestep sur 2
    
    ❌ GPU OUT OF MEMORY
        Cause: batch size trop gros
        Solution:
            # Réduire dans run_sweep.py:
            T_acc = torch.zeros(..., device="cpu")  # fallback to CPU
    """
    pass


# ============================================================================
# DONNÉES & FORMATS
# ============================================================================

def data_formats():
    """
    Formats des données d'entrée/sortie.
    
    == DONNÉES DEM BRUTES ==
    
    Format: CSV (un par timestep)
    Fichier: file_0.csv, file_1.csv, ..., file_500.csv
    Colonnes:
        coordinates:0, coordinates:1, coordinates:2  (x, y, z)
        ...autres colonnes (vitesse, propriétés, etc.)
    
    Chargement:
        import polars as pl
        df = pl.read_csv("file_0.csv")
        coords = np.column_stack([
            df["coordinates:0"],
            df["coordinates:1"],
            df["coordinates:2"]
        ])
    
    == MATRICE P ==
    
    Format: .npy (NumPy binary)
    Shape: (n_states, n_states)
    dtype: float64
    Propriétés:
        - Éléments ∈ [0, 1]
        - Chaque ligne somme ≈ 1
        - Symétrique: non
        - Réversible: non (généralement)
    
    == STATS.JSON ==
    
    Exemple:
    {
      "n_timesteps_used": 200,
      "n_states": 125,
      "n_states_visited": 124,
      "n_states_empty": 1,
      "fraction_visited": 0.992,
      "row_sum_min": 0.8001,
      "row_sum_max": 1.0000,
      "row_sum_mean": 0.9950,
      "diagonal_mean": 0.328,
      "diagonal_std": 0.142,
      "method": "voronoi"
    }
    
    == CONFIG.JSON ==
    
    Exemple:
    {
      "method": "voronoi",
      "method_kwargs": {"n_cells": 125},
      "nlt": 200,
      "step_size": 1,
      "start_index": 0
    }
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("Pour explorer la documentation:")
    print("="*80)
    print("\npydoc partitioners")
    print("pydoc run_sweep")
    print("pydoc bucket_io")
    print("pydoc analyze_results")
    print("\nOu spécifiquement:")
    print("pydoc partitioners.VoronoiPartitioner")
    print("pydoc run_sweep.run_markov_sweep")
    print("pydoc analyze_results.MarkovAnalyzer.compute_rsd")
