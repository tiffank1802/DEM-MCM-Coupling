"""
bucket_io.py — Lecture/écriture directe vers HuggingFace bucket

  Lors de la lecture : aucun fichier n'est téléchargé en local,
    tout est lu depuis HuggingFace et seules les variables utiles sont retournées.
  Lors de l'écriture : chaque fichier transite par un répertoire temporaire,
    puis est transféré vers le bucket et détruit localement.

  Organisation du bucket :
    _Good/Experiment/
      voronoi_simulations/     ← voronoi_*
      cartesian_simulations/   ← cartesian_*
      cylindrical_simulations/ ← cylindrical_*
      gmm_simulations/         ← gmm_*
      spectral_simulations/    ← spectral_*
      adaptive_simulations/    ← adaptive_*
      physics_simulations/     ← physics_*, physics_full_vel_*
      quantile_simulations/    ← quantile_*
      octree_simulations/      ← octree_*
      multizone_simulations/   ← multizone_*
      single_simulations/      ← single_*
      summaries/               ← _summary*
      other_simulations/       ← tout le reste
      postraitement/           ← sorties de post-traitement (non touché)
"""

import numpy as np
import json
import io
import os
import shutil
import tempfile
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, HfFileSystem


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

BUCKET_ID = "ktongue/DEM_MCM"

# Correspondance préfixe de nom → sous-dossier de catégorie
# L'ordre est important : les préfixes les plus longs/spécifiques en premier.
CATEGORY_MAP = {
    "physics_full_vel_": "physics_simulations",
    "voronoi_":          "voronoi_simulations",
    "cartesian_":        "cartesian_simulations",
    "cylindrical_":      "cylindrical_simulations",
    "gmm_":              "gmm_simulations",
    "spectral_":         "spectral_simulations",
    "adaptive_":         "adaptive_simulations",
    "physics_":          "physics_simulations",
    "quantile_":         "quantile_simulations",
    "octree_":           "octree_simulations",
    "multizone_":        "multizone_simulations",
    "single_":           "single_simulations",
    "_summary":          "summaries",
}

# Dossiers qui ne sont jamais déplacés / catégorisés
_SKIP_FOLDERS = {"postraitement"}

# Toutes les catégories connues (utile pour list_experiments)
ALL_CATEGORIES = list(dict.fromkeys(CATEGORY_MAP.values())) + ["other_simulations"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNES
# ─────────────────────────────────────────────────────────────────────────────

def get_simulation_category(folder_name: str) -> str:
    """Détermine la catégorie de simulation à partir du nom du dossier."""
    for prefix, category in CATEGORY_MAP.items():
        if folder_name.startswith(prefix):
            return category
    return "other_simulations"


def _get_bucket_prefix_from_particle_diameter(particle_diameter) -> str:
    if particle_diameter == 0.008:
        return "_Good/BIG"
    elif particle_diameter == 0.004:
        return "_Good/SMALL"
    else:
        return "_Good/Experiment"


def _get_current_branch():
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
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return branch
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GLOBALS (singletons légers)
# ─────────────────────────────────────────────────────────────────────────────

BUCKET_PREFIX = _get_bucket_prefix_from_particle_diameter(None)
BUCKET_BASE   = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"

_branch = _get_current_branch()
if _branch:
    print(f"🔀 Branche git détectée : '{_branch}'")

_fs  = None
_api = None


def get_fs() -> HfFileSystem:
    global _fs
    if _fs is None:
        _fs = HfFileSystem()
    return _fs


def get_api() -> HfApi:
    global _api
    if _api is None:
        _api = HfApi()
    return _api


# ─────────────────────────────────────────────────────────────────────────────
# MIGRATION (utilitaire à appeler une seule fois)
# ─────────────────────────────────────────────────────────────────────────────

def migrate_bucket(bucket_prefix: str = "_Good/Experiment", dry_run: bool = False):
    """
    Parcourt la racine d'un bucket_prefix et déplace les dossiers de simulation
    vers leurs sous-dossiers de catégorie.

    Args:
        bucket_prefix : chemin relatif au bucket (ex. "_Good/Experiment").
        dry_run       : si True, affiche seulement les déplacements sans les faire.
    """
    fs   = get_fs()
    base = f"buckets/{BUCKET_ID}/{bucket_prefix}"

    items = [i for i in fs.ls(base) if i["type"] == "directory"]
    print(f"📦 {len(items)} dossiers détectés dans {base}\n")

    moved = 0
    for item in items:
        name = item["name"].split("/")[-1]

        if name in _SKIP_FOLDERS or name in ALL_CATEGORIES:
            print(f"⏭️  Skip '{name}'")
            continue

        cat = get_simulation_category(name)
        src = f"{base}/{name}"
        dst = f"{base}/{cat}/{name}"
        print(f"{'[DRY] ' if dry_run else ''}➡️  {name}  →  {cat}/")

        if not dry_run:
            try:
                fs.mv(src, dst, recursive=True)
                moved += 1
            except Exception as e:
                print(f"  ❌ Erreur : {e}")

    print(f"\n✅ Migration {'simulée' if dry_run else 'terminée'} — {moved} dossiers déplacés.")


# ─────────────────────────────────────────────────────────────────────────────
# ÉCRITURE
# ─────────────────────────────────────────────────────────────────────────────

def save_experiment_to_bucket(
    folder_name,
    stats,
    config,
    species_data=None,
    partitioner_data=None,
    image_data=None,
    particle_diameter=None,
):
    """
    Sauvegarde une expérience dans le bucket, dans le bon sous-dossier de catégorie.

    Chemin final : {bucket_prefix}/{category}/{folder_name}/
    """
    bucket_prefix = _get_bucket_prefix_from_particle_diameter(particle_diameter)
    category      = get_simulation_category(folder_name)
    bucket_base_path = f"{bucket_prefix}/{category}/{folder_name}"
    api = get_api()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_folder   = Path(tmpdir)
        files_to_upload = []

        # Arrays par espèce
        if species_data:
            for array_name, array in species_data.items():
                p = local_folder / f"{array_name}.npy"
                np.save(p, array)
                files_to_upload.append((str(p), f"{bucket_base_path}/{array_name}.npy"))

        # stats.json
        stats_path = local_folder / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        files_to_upload.append((str(stats_path), f"{bucket_base_path}/stats.json"))

        # config.json
        config_path = local_folder / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        files_to_upload.append((str(config_path), f"{bucket_base_path}/config.json"))

        # Données du partitionneur
        if partitioner_data:
            part_dir = local_folder / "partitioner"
            part_dir.mkdir()
            for key, value in partitioner_data.items():
                if isinstance(value, np.ndarray):
                    p = part_dir / f"{key}.npy"
                    np.save(p, value)
                    files_to_upload.append(
                        (str(p), f"{bucket_base_path}/partitioner/{key}.npy")
                    )
                else:
                    p = part_dir / f"{key}.json"
                    with open(p, "w") as f:
                        json.dump(value, f, indent=2)
                    files_to_upload.append(
                        (str(p), f"{bucket_base_path}/partitioner/{key}.json")
                    )

        # Images
        if image_data:
            for img_name, img_bytes in image_data.items():
                img_path = local_folder / img_name
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                files_to_upload.append(
                    (str(img_path), f"{bucket_base_path}/images/{img_name}")
                )

        api.batch_bucket_files(
            bucket_id=BUCKET_ID,
            add=[(lp, bp) for lp, bp in files_to_upload],
        )
        print(f"   ✅ {len(files_to_upload)} fichiers uploadés → {bucket_base_path}/")


def upload_postprocessing_to_bucket(
    local_dir="outputs",
    bucket_subfolder="postraitement",
    particle_diameter=None,
    cleanup=False,
):
    """
    Envoie tous les fichiers du dossier local vers le bucket (sous postraitement/).
    Conserve l'arborescence exacte.
    """
    bucket_prefix = _get_bucket_prefix_from_particle_diameter(particle_diameter)
    api = get_api()

    local_path = Path(local_dir).resolve()
    if not local_path.exists():
        print(f"❌ Dossier local introuvable : {local_path}")
        return

    files_to_upload = []
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            rel_path    = file_path.relative_to(local_path)
            bucket_path = f"{bucket_prefix}/{bucket_subfolder}/{rel_path.as_posix()}"
            files_to_upload.append((str(file_path), bucket_path))

    if not files_to_upload:
        print(f"⚠️  Aucun fichier trouvé dans {local_path}")
        return

    api.batch_bucket_files(
        bucket_id=BUCKET_ID,
        add=[(lp, bp) for lp, bp in files_to_upload],
    )
    print(
        f"✅ {len(files_to_upload)} fichiers de post-traitement uploadés → "
        f"{bucket_prefix}/{bucket_subfolder}/"
    )

    if cleanup:
        shutil.rmtree(local_path)
        print(f"🧹 Dossier local supprimé : {local_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LECTURE
# ─────────────────────────────────────────────────────────────────────────────

def load_experiment_from_bucket(folder_name: str, bucket_prefix: str = None) -> dict:
    """
    Charge une expérience depuis le bucket.

    Stratégie de recherche (par ordre) :
      1. {bucket_prefix}/{category}/{folder_name}/   ← nouveau format
      2. {bucket_prefix}/{folder_name}/              ← ancien format (avant migration)
      3. Idem dans le bucket opposé (BIG ↔ SMALL)

    Returns:
        {
            "species": {
                "small": {"P": ndarray, "S_matrix": ndarray, "times": ndarray},
                "large": {"P": ndarray, "S_matrix": ndarray, "times": ndarray},
            },
            "stats":  dict,
            "config": dict,
        }
    """
    fs       = get_fs()
    category = get_simulation_category(folder_name)

    if bucket_prefix is None:
        if "_d0004" in folder_name:
            bucket_prefix = "_Good/SMALL"
        elif "_d0008" in folder_name:
            bucket_prefix = "_Good/BIG"
        else:
            bucket_prefix = "_Good/Experiment"

    def _candidate_prefixes(bp):
        return [
            f"hf://buckets/{BUCKET_ID}/{bp}/{category}/{folder_name}",  # nouveau
            f"hf://buckets/{BUCKET_ID}/{bp}/{folder_name}",             # ancien
        ]

    alt_prefix = (
        "_Good/BIG"   if bucket_prefix == "_Good/SMALL" else
        "_Good/SMALL" if bucket_prefix == "_Good/BIG"   else
        None
    )

    candidates = _candidate_prefixes(bucket_prefix)
    if alt_prefix:
        candidates += _candidate_prefixes(alt_prefix)

    for prefix in candidates:
        if not fs.exists(f"{prefix}/stats.json"):
            continue

        def _load_npy(name):
            with fs.open(f"{prefix}/{name}", "rb") as f:
                return np.load(io.BytesIO(f.read()))

        def _load_json(name):
            with fs.open(f"{prefix}/{name}", "r") as f:
                return json.load(f)

        stats  = _load_json("stats.json")
        config = _load_json("config.json")

        species_list = stats.get("species_list", ["small", "large"])
        species_out  = {}
        for species in species_list:
            species_out[species] = {
                "P":        _load_npy(f"transitionmatrix_{species}.npy"),
                "S_matrix": _load_npy(f"S_matrix_{species}.npy"),
                "times":    _load_npy(f"times_{species}.npy"),
            }

        return {"species": species_out, "stats": stats, "config": config}

    raise FileNotFoundError(
        f"❌ Introuvable : '{folder_name}' (catégorie : {category}) "
        f"dans {bucket_prefix}" + (f" ni {alt_prefix}" if alt_prefix else "")
    )


def list_experiments(bucket_prefix: str = None) -> list[str]:
    """
    Liste toutes les expériences en parcourant les sous-dossiers de catégorie,
    plus un fallback sur les dossiers encore à la racine (avant migration).

    Args:
        bucket_prefix : force un bucket précis ; sinon utilise BUCKET_BASE.

    Returns:
        Liste triée des noms de dossiers d'expérience.
    """
    fs   = get_fs()
    base = f"hf://buckets/{BUCKET_ID}/{bucket_prefix}" if bucket_prefix else BUCKET_BASE

    experiments = set()

    # 1. Sous-dossiers de catégorie (nouveau format)
    for cat in ALL_CATEGORIES:
        try:
            for item in fs.ls(f"{base}/{cat}"):
                if item["type"] == "directory":
                    experiments.add(item["name"].split("/")[-1])
        except FileNotFoundError:
            pass

    # 2. Fallback : dossiers encore à la racine (avant migration)
    try:
        for item in fs.ls(base):
            if item["type"] == "directory":
                name = item["name"].split("/")[-1]
                if name not in set(ALL_CATEGORIES) | _SKIP_FOLDERS:
                    experiments.add(name)
    except FileNotFoundError:
        pass

    return sorted(experiments)


def list_experiments_by_category(bucket_prefix: str = None) -> dict[str, list[str]]:
    """
    Même chose que list_experiments() mais retourne un dict catégorie → [noms].
    Utile pour afficher des groupes dans l'interface Streamlit.
    """
    fs   = get_fs()
    base = f"hf://buckets/{BUCKET_ID}/{bucket_prefix}" if bucket_prefix else BUCKET_BASE

    result = {cat: [] for cat in ALL_CATEGORIES}
    result["(racine)"] = []  # fallback

    for cat in ALL_CATEGORIES:
        try:
            for item in fs.ls(f"{base}/{cat}"):
                if item["type"] == "directory":
                    result[cat].append(item["name"].split("/")[-1])
        except FileNotFoundError:
            pass

    # Fallback racine
    try:
        for item in fs.ls(base):
            if item["type"] == "directory":
                name = item["name"].split("/")[-1]
                if name not in set(ALL_CATEGORIES) | _SKIP_FOLDERS:
                    result["(racine)"].append(name)
    except FileNotFoundError:
        pass

    # Trier chaque liste et supprimer les catégories vides
    return {
        cat: sorted(names)
        for cat, names in result.items()
        if names
    }


def load_all_experiments(bucket_prefix: str = None) -> dict:
    """Charge toutes les expériences listées par list_experiments()."""
    results = {}
    for folder in list_experiments(bucket_prefix):
        try:
            results[folder] = load_experiment_from_bucket(folder)
        except Exception as e:
            print(f"⚠️  {folder} : {e}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT MANAGER — post-traitement
# ─────────────────────────────────────────────────────────────────────────────

class PostprocessingBucketUploader:
    """
    Gestionnaire de contexte : génère les fichiers dans un dossier temporaire,
    les envoie vers le bucket, puis supprime le dossier temporaire.

    Usage :
        with PostprocessingBucketUploader(bucket_subfolder="postraitement") as tmp:
            (tmp / "images").mkdir(parents=True, exist_ok=True)
            fig.savefig(tmp / "images" / "plot.png")
        # ← upload automatique + nettoyage
    """

    def __init__(self, bucket_subfolder: str = "postraitement", particle_diameter=None):
        self.bucket_subfolder   = bucket_subfolder
        self.particle_diameter  = particle_diameter
        self.local_path: Path | None = None

    def __enter__(self) -> Path:
        self.local_path = Path(tempfile.mkdtemp(prefix="dem_mcm_postproc_"))
        print(f"📂 Répertoire temporaire créé : {self.local_path}")
        return self.local_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.local_path and self.local_path.exists():
            print("\n🚀 Envoi des fichiers vers le bucket...")
            upload_postprocessing_to_bucket(
                local_dir=str(self.local_path),
                bucket_subfolder=self.bucket_subfolder,
                particle_diameter=self.particle_diameter,
                cleanup=True,
            )