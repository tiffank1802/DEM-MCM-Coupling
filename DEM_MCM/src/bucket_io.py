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
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem

# Configuration
BUCKET_ID = "ktongue/DEM_MCM"
# BUCKET_PREFIX = "ResultsDtMCM"
# BUCKET_PREFIX = "NewResultsMCM"
# BUCKET_PREFIX = "RaffinageTemporel"
BUCKET_PREFIX = "Experiments"
# BUCKET_PREFIX = "BIG"
# BUCKET_PREFIX = "SMALL"
BUCKET_BASE = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"

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
                              partitioner_data=None, image_data=None):  # ← Changé image_paths en image_data
    api = get_api()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_folder = Path(tmpdir)
        files_to_upload = []

        # ✅ Corriger les chemins avec / entre BUCKET_PREFIX et folder_name
        matrix_path = local_folder / "transitionmatrix.npy"
        np.save(matrix_path, matrix)
        files_to_upload.append((str(matrix_path), f"{BUCKET_PREFIX}/{folder_name}/transitionmatrix.npy"))

        stats_path = local_folder / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        files_to_upload.append((str(stats_path), f"{BUCKET_PREFIX}/{folder_name}/stats.json"))

        config_path = local_folder / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        files_to_upload.append((str(config_path), f"{BUCKET_PREFIX}/{folder_name}/config.json"))

        # données partitionneur
        if partitioner_data:
            for key, value in partitioner_data.items():
                if isinstance(value, np.ndarray):
                    p = local_folder / f"{key}.npy"
                    np.save(p, value)
                    files_to_upload.append((str(p), f"{BUCKET_PREFIX}/{folder_name}/{key}.npy"))
                else:
                    p = local_folder / f"{key}.json"
                    with open(p, "w") as f:
                        json.dump(value, f, indent=2)
                    files_to_upload.append((str(p), f"{BUCKET_PREFIX}/{folder_name}/{key}.json"))

        # ✅ Images en mémoire → fichiers temporaires → upload
        if image_data:
            for img_name, img_bytes in image_data.items():
                img_path = local_folder / img_name
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                files_to_upload.append(
                    (str(img_path), f"{BUCKET_PREFIX}/{folder_name}/images/{img_name}")
                )
                
        # Upload batch
        api.batch_bucket_files(
            bucket_id=BUCKET_ID,
            add=[(local_path, path_in_bucket) for local_path, path_in_bucket in files_to_upload],
        )
        
        print(f"   ✅ {len(files_to_upload)} fichiers uploadés vers {BUCKET_PREFIX}/{folder_name}/")
# =============================================================================
# LECTURE
# =============================================================================

def load_matrix_from_bucket(path):
    fs = get_fs()
    full_path = f"{BUCKET_BASE}/{path}"
    with fs.open(full_path, "rb") as f:
        buffer = io.BytesIO(f.read())
    return np.load(buffer)


def load_json_from_bucket(path):
    fs = get_fs()
    full_path = f"{BUCKET_BASE}/{path}"
    with fs.open(full_path, "r") as f:
        return json.load(f)


def load_experiment_from_bucket(folder_name):
    """Charge  depuis le bucket huggingface:
            - la matrice de transition
            - les statistiques de l'experience
            - la configuration de l'experience

    Args:
        folder_name (str): est le nom du dossier de l'experience à charger depuis le bucket

    Returns:
        dict: un dictionnare comportant:
                -Matrice de transition correspondant à l'expérience chargée
                - les statistiques
                - les configurations
    """
    return {
        "matrix": load_matrix_from_bucket(f"{folder_name}/transitionmatrix.npy"),
        "stats": load_json_from_bucket(f"{folder_name}/stats.json"),
        "config": load_json_from_bucket(f"{folder_name}/config.json"),
    }


def list_experiments():
    """Liste les expériences dans un dossier BUCKET_PREFIX (celui correspondant à la route BUCKET_BASE) qui est le dossier par où toutes les entrées sorties des expériences se font
    

    Returns:
        list[str]: Une liste des expériences se trouvent dans le dossier BUCKET_PREFIX correspondant à la route BUCKET_BASE
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
