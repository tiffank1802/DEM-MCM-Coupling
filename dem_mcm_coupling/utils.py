"""utils.py — Utilitaires généraux pour le module dem_mcm_coupling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


def load_parquet_as_timestep_dict(
    parquet_path: str,
    fs: HfFileSystem,
) -> dict[int, pd.DataFrame]:
    """
    Charge le fichier Parquet entier et retourne un dict {idx: DataFrame}.

    L'index est extrait de la colonne Fichier_Source ("data_{idx}.csv").

    Args:
        parquet_path : chemin HuggingFace (hf://buckets/...)
        fs           : HfFileSystem déjà instancié

    Returns:
        dict[int, pd.DataFrame] — une entrée par timestep disponible
    """
    with fs.open(parquet_path, "rb") as fh:
        pf = pq.ParquetFile(fh)
        list_dfs = []
        with tqdm(
            total=pf.num_row_groups,
            desc="   Chargement parquet",
            unit="bloc",
        ) as bar:
            for i in range(pf.num_row_groups):
                list_dfs.append(pf.read_row_group(i).to_pandas())
                bar.update(1)

    df_full = pd.concat(list_dfs, ignore_index=True)

    timestep_dict: dict[int, pd.DataFrame] = {}
    for source, group_df in df_full.groupby("Fichier_Source"):
        # "data_42.csv" → 42
        idx = int(str(source).replace("data_", "").replace(".csv", ""))
        timestep_dict[idx] = group_df.reset_index(drop=True)

    print(
        f"   📦 {len(timestep_dict)} timesteps indexés "
        f"(index {min(timestep_dict)} → {max(timestep_dict)})"
    )
    return timestep_dict


def apply_species_mask(
    states: np.ndarray,
    species_labels: np.ndarray | None,
) -> np.ndarray:
    """
    Filtre un vecteur d'états pour garder seulement les particules
    correspondant au masque species_labels.

    Args:
        states         : (n_particles,) — états assignés
        species_labels : (n_unique_particles,) bool ou None

    Returns:
        np.ndarray — états filtrés, ou states inchangé si species_labels is None
    """
    if species_labels is None:
        return states

    n_repeats = len(states) // len(species_labels)
    mask = np.tile(species_labels, n_repeats)

    # Gérer le cas où len(states) n'est pas un multiple exact
    if len(mask) < len(states):
        remaining = len(states) - len(mask)
        mask = np.concatenate([mask, species_labels[:remaining]])

    return states[mask]


def filter_by_diameter(
    df: pl.DataFrame,
    diameter: float,
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Filtre un DataFrame Polars par diamètre de particule.

    Args:
        df       : DataFrame Polars chargé depuis un fichier CSV
        diameter : diamètre cible en mètres (0.004 ou 0.008)

    Returns:
        (filtered_df, particle_ids_kept)
    """
    valid_diameters = [0.004, 0.008]
    if diameter not in valid_diameters:
        raise ValueError(f"diameter doit être dans {valid_diameters}, reçu {diameter}")

    mask = df["Diameter"] == diameter
    filtered_df = df.filter(mask)
    particle_ids_kept = filtered_df["Particle_ID"].to_numpy()

    return filtered_df, particle_ids_kept
