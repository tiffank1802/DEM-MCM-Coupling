"""
bucket_io.py — Lecture/écriture directe vers HuggingFace bucket
    Lors de la lecture des fichiers aucun fichier n'est téléchargé en local, tous sont lus depuis huggingface et seul les variables contenant
les informations necessaires sont retournées
    Lors de l'écriture, chaque fichier est tranferé sur huggingface depuis un repertoire temporaire, et détruit une fois le transfert éffectué
"""

import numpy as np
import json
import io
import os
import tempfile # pour la sauvegarde temporaire des fichiers en local avant son tranfert vers le bucket
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem

# Configuration
BUCKET_ID = "ktongue/DEM_MCM"

# ============================================================================
# DÉTECTION DYNAMIQUE DU BUCKET SELON LE MASQUE APPLIQUÉ (particle_diameter)
# ============================================================================
def _get_bucket_prefix_from_particle_diameter(particle_diameter):
    """
    Détermine le BUCKET_PREFIX selon le type de masque (particle_diameter).
    
    Logique:
    - particle_diameter == 0.008 → "BIG" (grosses particules)
    - particle_diameter == 0.004 → "SMALL" (petites particules)
    - particle_diameter == None → "Experiments" (toutes les particules)
    
    Args:
        particle_diameter: Optional[float] - Le diamètre filtré (ou None)
    
    Returns:
        str: Le prefix du bucket ("BIG", "SMALL", ou "Experiments")
    """
    if particle_diameter == 0.008:
        return "_Good/BIG"
    elif particle_diameter == 0.004:
        return "_Good/SMALL"
    else:
        return "_Good/Experiment"

def _get_current_branch():
    """
    Détecte la branche git actuelle (pour information seulement, ne détermine plus le bucket).
    Utilisé à titre informatif uniquement.
    """
    try:
        current_dir = Path(__file__).resolve().parent
        for _ in range(5):
            if (current_dir / ".git").exists():
                git_root = current_dir
                break
            current_dir = current_dir.parent
        else:
            return None
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(git_root),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return branch
    except (subprocess.CalledProcessError, FileNotFoundError, Exception):
        return None

# ✅ NEW: Initialiser avec le bucket par défaut (sera override selon particle_diameter)
BUCKET_PREFIX = _get_bucket_prefix_from_particle_diameter(None)  # Par défaut: "Experiments"
BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"

# Afficher la branche git détectée à titre informatif
_branch = _get_current_branch()
if _branch:
    print(f"🔀 Branche git détectée: '{_branch}' (bucket déterminé par masque appliqué)")
else:
    print(f"🔀 Branche git: non détectée (bucket déterminé par masque appliqué)")

_fs = None
_api = None


def get_fs():
    global _fs
    if _fs is None:
        _fs = HfFileSystem()
    return _fs


def get_api():
    global _api
    if _api is None:
        _api = HfApi()
    return _api


# =============================================================================
# ÉCRITURE
# =============================================================================
def save_experiment_to_bucket(folder_name, matrix, stats, config,
                              partitioner_data=None, image_data=None, particle_diameter=None,states=None):
    """
    Sauvegarde une expérience dans le bucket HuggingFace.
    
    Args:
        folder_name: Nom du dossier de destination
        matrix: Matrice de transition numpy (n_states × n_states)
        stats: Dictionnaire des statistiques
        config: Configuration de l'expérience
        partitioner_data: Données du partitionneur (optionnel)
        image_data: Images (optionnel)
        particle_diameter: Diamètre filtré (optionnel) - détermine le bucket
    """
    # ✅ NEW: Déterminer le bucket selon particle_diameter
    bucket_prefix = _get_bucket_prefix_from_particle_diameter(particle_diameter)
    bucket_base = f"hf://buckets/{BUCKET_ID}/{bucket_prefix}"
    
    api = get_api()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_folder = Path(tmpdir)
        files_to_upload = []

        # ✅ Utiliser bucket_prefix dynamique
        matrix_path = local_folder / "transitionmatrix.npy"
        np.save(matrix_path, matrix)
        files_to_upload.append((str(matrix_path), f"{bucket_prefix}/{folder_name}/transitionmatrix.npy"))

        states_path = local_folder / "states.npy"
        np.save(states_path, states)
        files_to_upload.append((str(states_path), f"{bucket_prefix}/{folder_name}/states.npy"))

        stats_path = local_folder / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        files_to_upload.append((str(stats_path), f"{bucket_prefix}/{folder_name}/stats.json"))

        config_path = local_folder / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        files_to_upload.append((str(config_path), f"{bucket_prefix}/{folder_name}/config.json"))

        # données partitionneur
        if partitioner_data:
            for key, value in partitioner_data.items():
                if isinstance(value, np.ndarray):
                    p = local_folder / f"{key}.npy"
                    np.save(p, value)
                    files_to_upload.append((str(p), f"{bucket_prefix}/{folder_name}/{key}.npy"))
                else:
                    p = local_folder / f"{key}.json"
                    with open(p, "w") as f:
                        json.dump(value, f, indent=2)
                    files_to_upload.append((str(p), f"{bucket_prefix}/{folder_name}/{key}.json"))

        # ✅ Images en mémoire → fichiers temporaires → upload
        if image_data:
            for img_name, img_bytes in image_data.items():
                img_path = local_folder / img_name
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                files_to_upload.append(
                    (str(img_path), f"{bucket_prefix}/{folder_name}/images/{img_name}")
                )
                
        # Upload batch
        api.batch_bucket_files(
            bucket_id=BUCKET_ID,
            add=[(local_path, path_in_bucket) for local_path, path_in_bucket in files_to_upload],
        )
        
        # ✅ Afficher le bucket utilisé
        bucket_info = f"{bucket_prefix}"
        if particle_diameter is not None:
            bucket_info += f" (diamètre={particle_diameter})"
        print(f"   ✅ {len(files_to_upload)} fichiers uploadés vers {bucket_info}/{folder_name}/")

# =============================================================================
# LECTURE
# =============================================================================

def load_matrix_from_bucket(path, base_path=None):
    fs = get_fs()
    if base_path is None:
        base_path = BUCKET_BASE
    full_path = f"{base_path}/{path}"
    with fs.open(full_path, "rb") as f:
        buffer = io.BytesIO(fs.read())
    return np.load(buffer)

def load_json_from_bucket(path, base_path=None):
    fs = get_fs()
    if base_path is None:
        base_path = BUCKET_BASE
    full_path = f"{base_path}/{path}"
    with fs.open(full_path, "r") as f:
        return json.load(f)

def load_experiment_from_bucket(folder_name):
    """Charge  depuis le bucket huggingface:
            - la matrice de transition
            - les statistiques de l'experience
            - la configuration de l'experience
    
    ✅ NOUVEAU: Cherche dans le bucket correspondant au diamètre,
    puis dans /Experiments/ si pas trouvé.
    
    Args:
        folder_name (str): est le nom du dossier de l'experience à charger depuis le bucket
    
    Returns:
        dict: un dictionnaire comportant:
                -Matrice de transition correspondant à l'experience chargée
                - les statistiques
                - les configurations
    """
    # 1. Déterminer le bucket prefix selon le diamètre dans le nom du dossier
    bucket_prefix = None
    if "_d" in folder_name:
        # Extraire le diamètre du nom (ex: _d0004 → 0.004)
        import re
        match = re.search(r'_d(\d+)', folder_name)
        if match:
            diameter_str = match.group(1)
            if diameter_str == "0004":
                bucket_prefix = "SMALL"
            elif diameter_str == "0008":
                bucket_prefix = "BIG"
    
    if bucket_prefix is None:
        bucket_prefix = "Experiments"  # Fallback ancien format
    
    # 2. Essayer différents chemins
    fs = get_fs()
    bucket_base = f"hf://buckets/{BUCKET_ID}/{bucket_prefix}"
    
    # Chemins à essayer (par ordre de priorité)
    paths_to_try = [
        f"{bucket_base}/markov_results/{folder_name}",  # Nouveau format avec markov_results/
        f"{bucket_base}/Experiments/{folder_name}",    # Ancien format avec Experiments/
        f"{bucket_base}/{folder_name}",             # Directement dans le bucket
    ]
    
    for path in paths_to_try:
        if fs.exists(path):
            return {
                "matrix": load_matrix_from_bucket(f"{folder_name}/transitionmatrix.npy", base_path=bucket_base),
                "stats": load_json_from_bucket(f"{folder_name}/stats.json", base_path=bucket_base),
                "config": load_json_from_bucket(f"{folder_name}/config.json", base_path=bucket_base),
            }
    
    # Si toujours pas trouvé, essayer avec le prefixe opposé
    alt_prefix = "BIG" if bucket_prefix == "SMALL" else "SMALL" if bucket_prefix == "BIG" else None
    if alt_prefix:
        alt_base = f"hf://buckets/{BUCKET_ID}/{alt_prefix}"
        alt_paths = [
            f"{alt_base}/markov_results/{folder_name}",
            f"{alt_base}/Experiments/{folder_name}",
            f"{alt_base}/{folder_name}",
        ]
        for path in alt_paths:
            if fs.exists(path):
                return {
                    "matrix": load_matrix_from_bucket(f"{folder_name}/transitionmatrix.npy", base_path=alt_base),
                    "stats": load_json_from_bucket(f"{folder_name}/stats.json", base_path=alt_base),
                    "config": load_json_from_bucket(f"{folder_name}/config.json", base_path=alt_base),
                }
    
    raise FileNotFoundError(f"❌ Dossier introuvable: {folder_name} dans {bucket_prefix} ou {alt_prefix}")


def list_experiments():
    """Liste les expériences dans un dossier BUCKET_PREFIX (celui correspondant à la route BUCKET_BASE) qui est le dossier par où toutes les entrées sorties des expériences se font
    

    Returns:
        list[str]: Une liste des expériences se trouvent dans le dossier BUCKET_PREFIX correspondant à la route BUCKET_BASE
    """
    fs = get_fs()
    try:
        items = fs.ls(BUCKET_BASE)
        return sorted([
            item["name"].split("/")[-1]  # type: ignore
            for item in items 
            if item["type"] == "directory" # type: ignore
        ])
    except FileNotFoundError:
        return []


def load_all_experiments():
    """Charge les fichiers se sortie de chacune des expérience se trouvant dans le dossier BUCKET_PREFIX dont la route est BUCKET_BAE

    Returns:
        list[dict]: Retourne une liste de dictionnaires dont chaque dictionnaire de la liste correspond aux résultats de chaque expérience dans le dossier BUCKET_PREFIX
    """
    results = {}
    for folder in list_experiments():
        try:
            results[folder] = load_experiment_from_bucket(folder)
        except Exception as e:
            print(f"⚠️ {folder}: {e}")
    return results
