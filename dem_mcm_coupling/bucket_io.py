"""Reading/writing helpers for the Hugging Face bucket.

When reading, no file is downloaded locally: everything is streamed from the
Hub and only the requested variables are returned. When writing, files first
transit through a temporary directory, are uploaded to the bucket, then
destroyed locally.

Bucket layout::

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
      Inhomogènes/             ← inhomogeneous_*
      summaries/               ← _summary*
      other_simulations/       ← everything else
      postraitement/           ← post-processing outputs (untouched)
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, HfFileSystem

from dem_mcm_coupling._config import BUCKET_ID, get_bucket_prefix

# =============================================================================
# CONSTANTS
# =============================================================================

#: Name-prefix → category sub-folder correspondence.
#: The order matters: the longest/most specific prefixes must come first.
CATEGORY_MAP: dict[str, str] = {
    "inhomogeneous_": "Inhomogènes",
    "physics_full_vel_": "physics_simulations",
    "voronoi_": "voronoi_simulations",
    "cartesian_": "cartesian_simulations",
    "cylindrical_": "cylindrical_simulations",
    "gmm_": "gmm_simulations",
    "spectral_": "spectral_simulations",
    "adaptive_": "adaptive_simulations",
    "physics_": "physics_simulations",
    "quantile_": "quantile_simulations",
    "octree_": "octree_simulations",
    "multizone_": "multizone_simulations",
    "single_": "single_simulations",
    "_summary": "summaries",
}

#: Folders that are never categorised/moved.
_SKIP_FOLDERS: frozenset[str] = frozenset({"postraitement"})

#: All known categories (used when listing experiments).
ALL_CATEGORIES: list[str] = [
    *list(dict.fromkeys(CATEGORY_MAP.values())),
    "other_simulations",
]

#: Default bucket prefix (experiments with both particle diameters).
BUCKET_PREFIX: str = get_bucket_prefix(None)

#: Base ``hf://`` path of the default bucket prefix.
BUCKET_BASE: str = f"hf://buckets/{BUCKET_ID}/{BUCKET_PREFIX}"


def _list_directory_names(fs: HfFileSystem, path: str) -> list[str]:
    """Return the names of the directories under a bucket path.

    :meth:`HfFileSystem.ls` returns ``list[str | dict]`` depending on the
    Hub version; both shapes are accepted here.
    """
    names: list[str] = []
    for item in fs.ls(path):
        if isinstance(item, dict):
            if item.get("type") == "directory":
                names.append(str(item["name"]).split("/")[-1])
        else:
            # Plain-path form: directory entries end with "/".
            stripped = item.rstrip("/")
            if stripped:
                names.append(stripped.split("/")[-1])
    return names


def get_simulation_category(folder_name: str) -> str:
    """Return the category of a simulation from its folder name.

    Args:
        folder_name: Name of the simulation folder.

    Returns:
        The category name (one of :data:`ALL_CATEGORIES`);
        ``"other_simulations"`` when no prefix matches.
    """
    for prefix, category in CATEGORY_MAP.items():
        if folder_name.startswith(prefix):
            return category
    return "other_simulations"


# =============================================================================
# SINGLETONS (lightweight)
# =============================================================================

_fs: HfFileSystem | None = None
_api: HfApi | None = None


def get_fs() -> HfFileSystem:
    """Return the shared :class:`~huggingface_hub.HfFileSystem` instance."""
    global _fs
    if _fs is None:
        _fs = HfFileSystem()
    return _fs


def get_api() -> HfApi:
    """Return the shared :class:`~huggingface_hub.HfApi` instance."""
    global _api
    if _api is None:
        _api = HfApi()
    return _api


# =============================================================================
# WRITING
# =============================================================================


def save_experiment_to_bucket(
    folder_name: str,
    stats: dict[str, Any],
    config: dict[str, Any],
    species_data: dict[str, np.ndarray] | None = None,
    partitioner_data: dict[str, Any] | None = None,
    image_data: dict[str, bytes] | None = None,
    particle_diameter: float | None = None,
    inhomogeneous_metadata: dict[str, Any] | None = None,
) -> None:
    """Save an experiment into the right category sub-folder of the bucket.

    Final path: ``{bucket_prefix}/{category}/{folder_name}/``.

    Args:
        folder_name: Name of the experiment folder.
        stats: Statistics dictionary, saved as ``stats.json``.
        config: Configuration dictionary, saved as ``config.json``.
        species_data: Mapping ``array_name -> ndarray`` saved as ``.npy``.
        partitioner_data: Partitioner data; numpy arrays go to
            ``partitioner/*.npy``, other values to ``partitioner/*.json``.
        image_data: Mapping ``image_name -> bytes`` saved under ``images/``.
        particle_diameter: Optional diameter filter selecting the bucket
            prefix (``0.004`` → ``_Good/SMALL``, ``0.008`` → ``_Good/BIG``).
        inhomogeneous_metadata: Optional metadata of an inhomogeneous
            experiment, saved as ``inhomogeneous_metadata.json``.

    Raises:
        TypeError: If ``inhomogeneous_metadata`` is not a dictionary.
    """
    bucket_prefix = get_bucket_prefix(particle_diameter)
    category = get_simulation_category(folder_name)
    bucket_base_path = f"{bucket_prefix}/{category}/{folder_name}"
    api = get_api()

    with tempfile.TemporaryDirectory() as tmpdir:
        local_folder = Path(tmpdir)
        files_to_upload: list[tuple[str, str]] = []

        # Per-species numpy arrays.
        if species_data:
            for array_name, array in species_data.items():
                path = local_folder / f"{array_name}.npy"
                np.save(path, array)
                files_to_upload.append(
                    (str(path), f"{bucket_base_path}/{array_name}.npy")
                )

        # stats.json
        stats_path = local_folder / "stats.json"
        with stats_path.open("w") as fh:
            json.dump(stats, fh, indent=2)
        files_to_upload.append((str(stats_path), f"{bucket_base_path}/stats.json"))

        # config.json
        config_path = local_folder / "config.json"
        with config_path.open("w") as fh:
            json.dump(config, fh, indent=2)
        files_to_upload.append((str(config_path), f"{bucket_base_path}/config.json"))

        # Partitioner data.
        if partitioner_data:
            part_dir = local_folder / "partitioner"
            part_dir.mkdir()
            for key, value in partitioner_data.items():
                if isinstance(value, np.ndarray):
                    path = part_dir / f"{key}.npy"
                    np.save(path, value)
                    files_to_upload.append(
                        (str(path), f"{bucket_base_path}/partitioner/{key}.npy")
                    )
                else:
                    path = part_dir / f"{key}.json"
                    with path.open("w") as fh:
                        json.dump(value, fh, indent=2)
                    files_to_upload.append(
                        (str(path), f"{bucket_base_path}/partitioner/{key}.json")
                    )

        # Images.
        if image_data:
            for img_name, img_bytes in image_data.items():
                img_path = local_folder / img_name
                img_path.write_bytes(img_bytes)
                files_to_upload.append(
                    (str(img_path), f"{bucket_base_path}/images/{img_name}")
                )

        # Inhomogeneous metadata.
        if inhomogeneous_metadata is not None:
            if not isinstance(inhomogeneous_metadata, dict):
                raise TypeError(
                    "inhomogeneous_metadata must be a dict, "
                    f"not {type(inhomogeneous_metadata).__name__}"
                )
            meta_path = local_folder / "inhomogeneous_metadata.json"
            with meta_path.open("w") as fh:
                json.dump(inhomogeneous_metadata, fh, indent=2)
            files_to_upload.append(
                (str(meta_path), f"{bucket_base_path}/inhomogeneous_metadata.json")
            )

        api.batch_bucket_files(
            bucket_id=BUCKET_ID,
            add=[(local, remote) for local, remote in files_to_upload],
        )
        print(f"   ✅ {len(files_to_upload)} files uploaded → {bucket_base_path}/")


def upload_postprocessing_to_bucket(
    local_dir: str = "outputs",
    bucket_subfolder: str = "postraitement",
    particle_diameter: float | None = None,
    cleanup: bool = False,
) -> None:
    """Upload every file of a local directory under ``postraitement/``.

    The directory tree is preserved exactly. VTK files (``.vtp``, ``.vtu``,
    ``.vtk``) are additionally pushed immediately through ``fs.put``.

    Args:
        local_dir: Local directory to upload.
        bucket_subfolder: Destination sub-folder in the bucket.
        particle_diameter: Optional diameter filter selecting the bucket
            prefix.
        cleanup: When ``True``, delete ``local_dir`` after a successful
            upload.
    """
    bucket_prefix = get_bucket_prefix(particle_diameter)
    api = get_api()
    fs = get_fs()

    local_path = Path(local_dir).resolve()
    if not local_path.exists():
        print(f"❌ Local directory not found: {local_path}")
        return

    files_to_upload: list[tuple[str, str]] = []
    vtk_files_count = 0

    for file_path in local_path.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(local_path)
        bucket_path = f"{bucket_prefix}/{bucket_subfolder}/{rel_path.as_posix()}"
        files_to_upload.append((str(file_path), bucket_path))

        # VTK files are also uploaded immediately via fs.put.
        if file_path.suffix in (".vtp", ".vtu", ".vtk"):
            try:
                fs.put(str(file_path), f"hf://buckets/{BUCKET_ID}/{bucket_path}")
                print(f"   📤 {rel_path} (immediate VTK upload)")
                vtk_files_count += 1
            except Exception as exc:
                print(f"   ⚠️  VTK upload failed for {rel_path}: {exc}")

    if not files_to_upload:
        print(f"⚠️  No file found in {local_path}")
        return

    api.batch_bucket_files(
        bucket_id=BUCKET_ID,
        add=[(local, remote) for local, remote in files_to_upload],
    )

    print(
        f"✅ {len(files_to_upload)} post-processing files uploaded → "
        f"{bucket_prefix}/{bucket_subfolder}/ "
        f"({vtk_files_count} VTK files)"
    )

    if cleanup:
        shutil.rmtree(local_path)
        print(f"🧹 Local directory removed: {local_path}")


# =============================================================================
# READING
# =============================================================================


def load_experiment_from_bucket(
    folder_name: str, bucket_prefix: str | None = None
) -> dict[str, Any]:
    """Load an experiment from the bucket.

    Search strategy (in order):
      1. ``{bucket_prefix}/{category}/{folder_name}/`` — current layout;
      2. ``{bucket_prefix}/{folder_name}/`` — legacy layout (before migration);
      3. The same paths in the opposite bucket (BIG ↔ SMALL) when the folder
         name encodes a diameter (``_d0004``/``_d0008``).

    Args:
        folder_name: Name of the experiment folder.
        bucket_prefix: Optional bucket prefix; when ``None`` it is inferred
            from the folder name (``_d0004`` → ``_Good/SMALL``,
            ``_d0008`` → ``_Good/BIG``).

    Returns:
        Dictionary with ``"species"`` (per-species ``P``/``P_blocks``,
        ``S_matrix``, ``times`` arrays), ``"stats"``, ``"config"`` and the
        inhomogeneous flags.

    Raises:
        FileNotFoundError: If the experiment cannot be found anywhere.
    """
    fs = get_fs()
    category = get_simulation_category(folder_name)

    if bucket_prefix is None:
        if "_d0004" in folder_name:
            bucket_prefix = get_bucket_prefix(0.004)
        elif "_d0008" in folder_name:
            bucket_prefix = get_bucket_prefix(0.008)
        else:
            bucket_prefix = get_bucket_prefix(None)

    def _candidate_prefixes(bp: str) -> list[str]:
        return [
            f"hf://buckets/{BUCKET_ID}/{bp}/{category}/{folder_name}",  # current
            f"hf://buckets/{BUCKET_ID}/{bp}/{folder_name}",  # legacy
        ]

    alt_prefix = (
        get_bucket_prefix(0.008)
        if bucket_prefix == get_bucket_prefix(0.004)
        else get_bucket_prefix(0.004)
        if bucket_prefix == get_bucket_prefix(0.008)
        else None
    )

    candidates = _candidate_prefixes(bucket_prefix)
    if alt_prefix:
        candidates += _candidate_prefixes(alt_prefix)

    for prefix in candidates:
        if not fs.exists(f"{prefix}/stats.json"):
            continue

        def _load_npy(name: str) -> np.ndarray:
            with fs.open(f"{prefix}/{name}", "rb") as fh:
                return np.load(io.BytesIO(fh.read()))

        def _load_json(name: str) -> dict[str, Any]:
            with fs.open(f"{prefix}/{name}", "r") as fh:
                return json.load(fh)

        stats = _load_json("stats.json")
        config = _load_json("config.json")

        # Automatic detection of the inhomogeneous format.
        inhomogeneous = fs.exists(f"{prefix}/inhomogeneous_metadata.json")
        inhomogeneous_metadata = (
            _load_json("inhomogeneous_metadata.json") if inhomogeneous else None
        )

        species_list = stats.get("species_list", ["small", "large"])
        species_out: dict[str, dict[str, np.ndarray]] = {}
        for species in species_list:
            if inhomogeneous:
                species_out[species] = {
                    "P_blocks": _load_npy(f"P_blocks_{species}.npy"),
                    "S_matrix": _load_npy(f"S_matrix_{species}.npy"),
                    "times": _load_npy(f"times_{species}.npy"),
                }
            else:
                species_out[species] = {
                    "P": _load_npy(f"transitionmatrix_{species}.npy"),
                    "S_matrix": _load_npy(f"S_matrix_{species}.npy"),
                    "times": _load_npy(f"times_{species}.npy"),
                }

        return {
            "species": species_out,
            "stats": stats,
            "config": config,
            "inhomogeneous": inhomogeneous,
            "inhomogeneous_metadata": inhomogeneous_metadata,
        }

    raise FileNotFoundError(
        f"❌ Not found: '{folder_name}' (category: {category}) "
        f"in {bucket_prefix}" + (f" nor {alt_prefix}" if alt_prefix else "")
    )


def list_experiments(bucket_prefix: str | None = None) -> list[str]:
    """List every experiment by walking the category sub-folders.

    Also falls back on folders still at the root of the prefix (pre-migration
    layout).

    Args:
        bucket_prefix: Optional bucket prefix; defaults to
            :data:`BUCKET_BASE`.

    Returns:
        Sorted list of experiment folder names.
    """
    fs = get_fs()
    base = f"hf://buckets/{BUCKET_ID}/{bucket_prefix}" if bucket_prefix else BUCKET_BASE

    experiments: set[str] = set()

    # 1. Category sub-folders (current layout).
    for category in ALL_CATEGORIES:
        with contextlib.suppress(FileNotFoundError):
            experiments.update(_list_directory_names(fs, f"{base}/{category}"))

    # 2. Folders still at the root (pre-migration layout).
    with contextlib.suppress(FileNotFoundError):
        for name in _list_directory_names(fs, base):
            if name not in set(ALL_CATEGORIES) | _SKIP_FOLDERS:
                experiments.add(name)

    return sorted(experiments)


# =============================================================================
# CONTEXT MANAGER — post-processing
# =============================================================================


class PostprocessingBucketUploader:
    """Context manager for bucket uploads of post-processing outputs.

    Generates files in a temporary directory, uploads them to the bucket,
    then deletes the temporary directory.

    Usage::

        with PostprocessingBucketUploader(bucket_subfolder="postraitement") as tmp:
            (tmp / "images").mkdir(parents=True, exist_ok=True)
            fig.savefig(tmp / "images" / "plot.png")
        # ← automatic upload + cleanup
    """

    def __init__(
        self,
        bucket_subfolder: str = "postraitement",
        particle_diameter: float | None = None,
    ) -> None:
        self.bucket_subfolder = bucket_subfolder
        self.particle_diameter = particle_diameter
        self.local_path: Path | None = None

    def __enter__(self) -> Path:
        self.local_path = Path(tempfile.mkdtemp(prefix="dem_mcm_postproc_"))
        print(f"📂 Temporary directory created: {self.local_path}")
        return self.local_path

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self.local_path and self.local_path.exists():
            print("\n🚀 Uploading files to the bucket...")
            upload_postprocessing_to_bucket(
                local_dir=str(self.local_path),
                bucket_subfolder=self.bucket_subfolder,
                particle_diameter=self.particle_diameter,
                cleanup=True,
            )
