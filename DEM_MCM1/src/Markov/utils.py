"""
Utilitaires généraux pour le module DEM_MCM.
"""

import numpy as np
import polars as pl
from typing import Optional, Tuple


def apply_species_mask(states, species_labels):
    """
    Filtre un vecteur d'états pour garder seulement les particules matchant le masque species_labels.
    
    Cette fonction est évolutive et fonctionne avec n'importe quel nombre de particules
    et de snapshots en répétant le masque au besoin.
    
    Args:
        states (np.ndarray): Vecteur d'états assignées aux particules. Shape: (n_particles,)
                            Valeurs entières représentant les cellules/partitions.
        species_labels (np.ndarray or None): Masque booléen pour filtrer les espèces.
                                            Shape: (n_unique_particles,)
                                            Si None, aucun filtrage n'est appliqué.
    
    Returns:
        np.ndarray: Vecteur d'états filtrées. 
                   - Si species_labels is None: returns states (pas de filtrage)
                   - Sinon: returns states[mask] avec le masque répété au besoin
    
    Examples:
        >>> states = np.array([0, 1, 2, 0, 1])  # 5 particules
        >>> labels = np.array([True, False, True])  # 3 particules uniques
        >>> # Répète le masque: [True, False, True, True, False]
        >>> result = apply_species_mask(states, labels)
        >>> # Garde seulement les indices 0, 2, 3 → states[[0,2,3]]
    """
    if species_labels is None:
        return states  # Pas de filtrage si pas de masque
    
    # Calculer le nombre de répétitions nécessaires
    n_repeats = len(states) // len(species_labels)
    mask = np.tile(species_labels, n_repeats)
    
    # Gérer le cas où len(states) n'est pas un multiple exact
    if len(mask) < len(states):
        remaining = len(states) - len(mask)
        mask = np.concatenate([mask, species_labels[:remaining]])
    
    return states[mask]


def filter_by_diameter(
    df: pl.DataFrame, 
    diameter: float
) -> Tuple[pl.DataFrame, np.ndarray]:
    """
    Filtre les particules par diamètre et retourne le dataframe filtré + IDs conservés.
    
    Args:
        df: DataFrame Polars chargé depuis un fichier CSV (1030 lignes × 24 colonnes)
        diameter: Diamètre cible en mètres (0.004 ou 0.008)
    
    Returns:
        tuple:
            - filtered_df: DataFrame avec seulement les particules matchant le diamètre
            - particle_ids_kept: np.ndarray des valeurs Particle_ID conservées
    
    Raises:
        ValueError: Si le diamètre n'est pas dans [0.004, 0.008]
    
    Example:
        >>> df_filtered, ids = filter_by_diameter(df, diameter=0.004)
        >>> len(df_filtered)  # ~515 particules (environ la moitié)
        515
        >>> len(ids)
        515
        >>> assert all(d == 0.004 for d in df_filtered["Diameter"])
    """
    # Valider le diamètre
    valid_diameters = [0.004, 0.008]
    if diameter not in valid_diameters:
        raise ValueError(f"diameter doit être dans {valid_diameters}, reçu {diameter}")
    
    # Filtrer par diamètre
    mask = df["Diameter"] == diameter
    filtered_df = df.filter(mask)
    
    # Extraire les Particle_IDs (comme tableau numpy pour les métadonnées)
    particle_ids_kept = filtered_df["Particle_ID"].to_numpy()
    
    return filtered_df, particle_ids_kept
