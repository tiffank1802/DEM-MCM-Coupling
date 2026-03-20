"""
===================================================================================
BUCKET_IO — Lecture/écriture vers HuggingFace Hub
===================================================================================

Ce module gère l'interaction avec le bucket HuggingFace pour:
- Sauvegarder les résultats de calcul (matrice P + stats + config)
- Charger les résultats pour analyse

Structure du bucket:
    hf://buckets/ktongue/DEM_MCM/
    ├── markov_results/                    ← Résultats finaux
    │   ├── voronoi_125cells_NLT200_step1_start0/
    │   │   ├── transition_matrix.npy      ← Matrice (n_states, n_states)
    │   │   ├── stats.json                 ← Statistiques d'exécution
    │   │   ├── config.json                ← Paramètres (méthode, nlt, ...)
    │   │   └── centroids.npy              ← Données du partitionneur
    │   └── ...
    │
    └── Output Paraview/                   ← Données DEM brutes
        ├── file_0.csv
        ├── file_1.csv
        └── ...

Usage minimal:

    # Écrire
    from bucket_io import save_experiment_to_bucket
    save_experiment_to_bucket(
        folder_name="voronoi_125cells_NLT200",
        matrix=P,
        stats={"n_states": 125, ...},
        config={"method": "voronoi", "n_cells": 125, ...},
    )

    # Lire
    from bucket_io import load_experiment_from_bucket
    data = load_experiment_from_bucket("voronoi_125cells_NLT200")
    P = data["matrix"]
"""

import numpy as np
import json
import io
import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem

# Configuration globale
BUCKET_ID = "ktongue/DEM_MCM"
BUCKET_PREFIX = "markov_results"
BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"

_fs = None
_api = None


def get_fs() -> HfFileSystem:
    """
    Accès singleton au filesyst HuggingFace.
    
    Returns:
        HfFileSystem: instance singleton
    
    Details:
        - Lazy initialization (premier appel crée l'instance)
        - Réutilisation de la même connexion pour tous les appels suivants
        - Cache les tokens d'authentification
    
    Example:
        >>> fs = get_fs()
        >>> items = fs.ls("hf://buckets/ktongue/DEM_MCM/")
    """
    global _fs
    if _fs is None:
        _fs = HfFileSystem()
    return _fs


def get_api() -> HfApi:
    """
    Accès singleton à l'API HuggingFace.
    
    Returns:
        HfApi: instance singleton
    
    Details:
        Utilisée pour batch_bucket_files() : upload optimisé de plusieurs
        fichiers en parallèle.
    
    Example:
        >>> api = get_api()
        >>> api.batch_bucket_files(
        ...     bucket_id="ktongue/DEM_MCM",
        ...     add=[("local_file.npy", "bucket_path/file.npy"), ...]
        ... )
    """
    global _api
    if _api is None:
        _api = HfApi()
    return _api


# =============================================================================
# ÉCRITURE (EXPORT)
# =============================================================================

def save_experiment_to_bucket(
    folder_name: str,
    matrix: np.ndarray,
    stats: dict,
    config: dict,
    partitioner_data: dict = None,
) -> None:
    """
    Sauvegarde une expérience complète dans le bucket HuggingFace.
    
    Processus:
    1. Crée un répertoire temporaire local
    2. Sauvegarde tous les fichiers localement:
       - transition_matrix.npy
       - stats.json
       - config.json
       - <partitioner_data> (centroids, edges, etc.)
    3. Upload batch vers le bucket
    4. Nettoie le répertoire temporaire
    
    Args:
        folder_name (str): nom du dossier dans le bucket (sans chemin)
            Format recommandé: "{method}_{params}_NLT{nlt}_step{step}_start{start}"
            Exemple: "voronoi_125cells_NLT200_step1_start0"
        
        matrix (np.ndarray): matrice de transition P
            Shape: (n_states, n_states)
            dtype: float64
            Propriété: chaque ligne somme ≈ 1
        
        stats (dict): statistiques d'exécution
            Clés attendues (exemples):
                'n_timesteps_used' (int): nombre de transitions analysées
                'n_states' (int): nombre d'états
                'n_states_visited' (int): états ayant ≥ 1 particule
                'diagonal_mean' (float): P(rester)
                'row_sum_min', 'row_sum_max' (float): validation de P
        
        config (dict): paramètres de l'expérience
            Clés attendues:
                'method' (str): "voronoi", "cartesian", etc.
                'method_kwargs' (dict): paramètres du partitionneur
                'nlt' (int): nombre de timesteps d'observations
                'step_size' (int): sous-échantillonnage temporel
                'start_index' (int): indice de départ
        
        partitioner_data (dict, optional): données du partitionneur
            Clés possibles (varient selon la méthode):
                'centroids' (ndarray): centroides K-means (Voronoï)
                'centroids_variance' (ndarray): variances (Voronoï)
                'r_edges' (ndarray): bords radiaux (Cylindrique)
                'x_edges', 'y_edges', 'z_edges' (ndarray): grille quantile
                'leaves' (ndarray): feuilles octree
                'partitioner_meta' (dict): métadonnées
            Default: None (pas de données supplémentaires)
    
    Returns:
        None
    
    Raises:
        IOError: erreurs d'accès disque ou réseau
        ValueError: format de data invalide
    
    Side effects:
        - Upload les fichiers vers le bucket HuggingFace
        - Log des informations sur l'upload
    
    Example:
        >>> P = np.random.rand(125, 125)
        >>> P = P / P.sum(axis=1, keepdims=True)  # normaliser
        >>> stats = {"n_states": 125, "n_timesteps_used": 200}
        >>> config = {"method": "voronoi", "method_kwargs": {"n_cells": 125}}
        >>> save_experiment_to_bucket(
        ...     folder_name="voronoi_125cells_NLT200",
        ...     matrix=P,
        ...     stats=stats,
        ...     config=config,
        ... )
        # ✓ Fichiers uploadés vers le bucket
    
    Performance:
        - Réseau: limité par la vitesse de connexion (~1-2 MB/s)
        - Temps typique: 5-30 secondes par expérience
    """
    api = get_api()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        local_folder = Path(tmpdir)
        
        # Préparer tous les fichiers localement
        files_to_upload = []
        
        # 1. Matrice de transition
        matrix_path = local_folder / "transition_matrix.npy"
        np.save(matrix_path, matrix)
        files_to_upload.append(
            (str(matrix_path), f"{BUCKET_PREFIX}/{folder_name}/transition_matrix.npy")
        )
        
        # 2. Statistiques
        stats_path = local_folder / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        files_to_upload.append(
            (str(stats_path), f"{BUCKET_PREFIX}/{folder_name}/stats.json")
        )
        
        # 3. Configuration
        config_path = local_folder / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        files_to_upload.append(
            (str(config_path), f"{BUCKET_PREFIX}/{folder_name}/config.json")
        )
        
        # 4. Données du partitionneur (optionnel)
        if partitioner_data:
            for key, value in partitioner_data.items():
                if isinstance(value, np.ndarray):
                    file_path = local_folder / f"{key}.npy"
                    np.save(file_path, value)
                    files_to_upload.append(
                        (str(file_path), f"{BUCKET_PREFIX}/{folder_name}/{key}.npy")
                    )
                else:
                    # JSON pour les dicts et autres
                    file_path = local_folder / f"{key}.json"
                    with open(file_path, "w") as f:
                        json.dump(value, f, indent=2)
                    files_to_upload.append(
                        (str(file_path), f"{BUCKET_PREFIX}/{folder_name}/{key}.json")
                    )
        
        # 5. Upload batch
        api.batch_bucket_files(
            bucket_id=BUCKET_ID,
            add=[
                (local_path, path_in_bucket)
                for local_path, path_in_bucket in files_to_upload
            ],
        )


# =============================================================================
# LECTURE (IMPORT)
# =============================================================================

def load_matrix_from_bucket(path: str) -> np.ndarray:
    """
    Charge une matrice (.npy) depuis le bucket.
    
    Args:
        path (str): chemin relatif dans le bucket (sans "hf://buckets/..." prefix)
            Exemple: "voronoi_125cells/transition_matrix.npy"
    
    Returns:
        np.ndarray: matrice chargée
            Shape: généralement (n_states, n_states)
            dtype: match le fichier (habituellement float64)
    
    Raises:
        FileNotFoundError: fichier n'existe pas sur le bucket
        IOError: erreur réseau ou lecture
    
    Example:
        >>> P = load_matrix_from_bucket("voronoi_125cells/transition_matrix.npy")
        >>> print(P.shape)
        (125, 125)
    """
    fs = get_fs()
    full_path = f"{BUCKET_BASE}/{path}"
    with fs.open(full_path, "rb") as f:
        buffer = io.BytesIO(f.read())
    return np.load(buffer)


def load_json_from_bucket(path: str) -> dict:
    """
    Charge un JSON depuis le bucket.
    
    Args:
        path (str): chemin relatif
            Exemple: "voronoi_125cells/config.json"
    
    Returns:
        dict: structure chargée du JSON
    
    Raises:
        FileNotFoundError: fichier n'existe pas
        json.JSONDecodeError: JSON malformé
    
    Example:
        >>> config = load_json_from_bucket("voronoi_125cells/config.json")
        >>> print(config["method"])
        'voronoi'
    """
    fs = get_fs()
    full_path = f"{BUCKET_BASE}/{path}"
    with fs.open(full_path, "r") as f:
        return json.load(f)


def load_experiment_from_bucket(folder_name: str) -> dict:
    """
    Charge tous les fichiers d'une expérience.
    
    Fonction de haut niveau : charge la matrice, stats et config en une seule
    appel. Les données du partitionneur sont **ignorées** à moins d'appel
    spécialisé.
    
    Args:
        folder_name (str): nom du dossier dans markov_results/
            Exemple: "voronoi_125cells_NLT200"
    
    Returns:
        dict avec clés:
            'matrix' (ndarray): matrice P
            'stats' (dict): stats.json
            'config' (dict): config.json
    
    Raises:
        FileNotFoundError: dossier ou fichiers manquants
    
    Example:
        >>> data = load_experiment_from_bucket("voronoi_125cells_NLT200")
        >>> P = data["matrix"]
        >>> n_states = data["stats"]["n_states"]
        >>> method = data["config"]["method"]
    """
    return {
        "matrix": load_matrix_from_bucket(f"{folder_name}/transition_matrix.npy"),
        "stats": load_json_from_bucket(f"{folder_name}/stats.json"),
        "config": load_json_from_bucket(f"{folder_name}/config.json"),
    }


def list_experiments() -> list:
    """
    Liste tous les dossiers d'expériences disponibles.
    
    Returns:
        list[str]: noms des dossiers triés alphabétiquement
    
    Example:
        >>> exps = list_experiments()
        >>> print(exps)
        ['cartesian_nx5_ny5_nz5', 'voronoi_125cells', ...]
    
    Performance:
        ~1-2 secondes pour 100+ expériences (appel réseau)
    """
    fs = get_fs()
    try:
        items = fs.ls(BUCKET_BASE)
        return sorted([
            item["name"].split("/")[-1] 
            for item in items 
            if item["type"] == "directory"
        ])
    except FileNotFoundError:
        return []


def load_all_experiments() -> dict:
    """
    Charge TOUTES les expériences du bucket.
    
    Fonction de commodité : itère sur tous les dossiers et les charge.
    Mauvaises expériences sont loggées mais retournées quand même (avec None).
    
    Returns:
        dict[str, dict]: {folder_name: {"matrix": P, "stats": {...}, ...}}
    
    Example:
        >>> all_data = load_all_experiments()
        >>> for name, data in all_data.items():
        ...     print(f"{name}: {data['stats']['n_states']} états")
    
    Warning:
        Cette fonction est très lente (~1 minute pour 100+ expériences).
        Préférer load_method() ou list_experiments() + load sélectif.
    """
    results = {}
    for folder in list_experiments():
        try:
            results[folder] = load_experiment_from_bucket(folder)
        except Exception as e:
            print(f"⚠️ {folder}: {e}")
    return results
