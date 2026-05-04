"""
Utilitaires généraux pour le module DEM_MCM.
"""

import numpy as np


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
