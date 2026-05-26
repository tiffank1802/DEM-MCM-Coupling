"""
markov_math.py
==============
Utilitaires mathématiques pour opérations sur les matrices et vecteurs Markoviens.

Responsabilités:
- Opérations sur matrices transition (normalization, spectral analysis)
- Calculs RSD et métriques de mélange
- Validation de conditions Markoviennes
- Comparaison entre trajectoires

Ce module est INDEPENDANT de Streamlit (pur math).

Examples:
    >>> from markov_math import analyze_transition_matrix, compute_rsd
    >>> M = np.random.rand(10, 10)
    >>> M /= M.sum(axis=1, keepdims=True)
    >>> 
    >>> props = analyze_transition_matrix(M)
    >>> print(f"Eigenvalues: {props['largest_eigenvalue']}")
    >>> 
    >>> phi = np.array([100, 50, 30, ...])
    >>> rsd = compute_rsd(phi)
    >>> print(f"RSD: {rsd:.4f}")
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


# ============================================================================
# MATRIX ANALYSIS
# ============================================================================

def analyze_transition_matrix(
    transition_matrix: np.ndarray,
    threshold: float = 1e-10,
) -> Dict[str, Any]:
    """
    Analyser les propriétés spectrales d'une matrice transition.
    
    Args:
        transition_matrix: Array (n_states, n_states)
        threshold: Seuil pour "petites" eigenvalues
        
    Returns:
        Dict avec propriétés:
        - eigenvalues: Tous les eigenvalues
        - largest_eigenvalue: Max |λ|
        - second_eigenvalue: Deuxième max |λ|
        - condition_number: κ(M)
        - trace: tr(M)
        - is_row_stochastic: Check ∑ row = 1
        - steady_state: Vecteur stationnaire (optionnel)
        
    Examples:
        >>> M = np.eye(5) * 0.9 + np.ones((5,5)) * 0.02
        >>> M /= M.sum(axis=1, keepdims=True)
        >>> props = analyze_transition_matrix(M)
        >>> print(f"λ₁ = {props['largest_eigenvalue']:.4f}")
        >>> print(f"κ = {props['condition_number']:.2f}")
    """
    # Eigenvalues
    eigenvalues = np.linalg.eigvals(transition_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Décroissant
    
    # Largest eigenvalue (doit être ~1 pour stochastique)
    abs_eigs = np.abs(eigenvalues)
    largest_idx = np.argmax(abs_eigs)
    largest_eig = eigenvalues[largest_idx]
    
    # Deuxième plus grand
    second_eig = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
    
    # Condition number
    try:
        cond_number = np.linalg.cond(transition_matrix)
    except:
        cond_number = np.inf
    
    # Vérifier row-stochastique
    row_sums = transition_matrix.sum(axis=1)
    is_row_stochastic = np.allclose(row_sums, 1.0, atol=1e-6)
    
    # Vecteur stationnaire (eigenvector de λ=1)
    steady_state = None
    if is_row_stochastic:
        try:
            _, eigenvecs = np.linalg.eig(transition_matrix.T)
            idx_dominant = np.argmax(np.abs(np.linalg.eigvals(transition_matrix.T)))
            steady_state = np.real(eigenvecs[:, idx_dominant])
            steady_state /= steady_state.sum()
        except:
            steady_state = None
    
    return {
        'eigenvalues': eigenvalues,
        'largest_eigenvalue': float(largest_eig),
        'second_eigenvalue': float(second_eig),
        'spectral_gap': float(1.0 - abs(second_eig)),  # Pour convergence
        'condition_number': float(cond_number),
        'trace': float(np.trace(transition_matrix)),
        'is_row_stochastic': bool(is_row_stochastic),
        'row_sum_deviations': row_sums - 1.0,  # Écarts normalisation
        'steady_state': steady_state,
    }


def normalize_transition_matrix(
    matrix: np.ndarray,
    method: str = "row",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normaliser une matrice pour qu'elle soit stochastique.
    
    Args:
        matrix: Array (n_states, n_states) non normalisée
        method: "row" pour row-stochastic (par défaut), 
                "column" pour column-stochastic
        
    Returns:
        (matrix_normalized, stats)
        
    Examples:
        >>> M_raw = np.random.rand(5, 5)
        >>> M_norm, stats = normalize_transition_matrix(M_raw)
        >>> print(f"Row sums: {M_norm.sum(axis=1)}")  # [1, 1, ...]
    """
    if method == "row":
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Éviter division par zéro
        M_norm = matrix / row_sums
    elif method == "column":
        col_sums = matrix.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        M_norm = matrix / col_sums
    else:
        raise ValueError(f"method doit être 'row' ou 'column', reçu {method}")
    
    stats = {
        'method': method,
        'max_row_deviation': np.max(np.abs(M_norm.sum(axis=1) - 1.0)),
        'is_valid': np.allclose(M_norm.sum(axis=1), 1.0) if method == "row" else True,
    }
    
    return M_norm, stats


# ============================================================================
# STATE VECTOR METRICS
# ============================================================================

def compute_rsd(
    state_vector: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Calculer le Relative Standard Deviation (RSD) d'un vecteur état.
    
    RSD = std(φ) / mean(φ)
    
    Mesure l'hétérogénéité: RSD=0 → homogène, RSD>1 → hétérogène
    
    Args:
        state_vector: Array (n_states,)
        mask: Masque optionnel des états "actifs"
        
    Returns:
        float: RSD (peut être NaN si mean=0)
        
    Examples:
        >>> phi = np.array([100, 100, 100])  # Homogène
        >>> compute_rsd(phi)  # 0.0
        >>>
        >>> phi2 = np.array([1, 10, 100])  # Hétérogène
        >>> compute_rsd(phi2)  # ~0.95
    """
    if mask is not None:
        phi_active = state_vector[mask]
    else:
        phi_active = state_vector[state_vector > 0]
    
    if len(phi_active) <= 1:
        return 0.0
    
    mean_val = phi_active.mean()
    if mean_val == 0:
        return np.nan
    
    rsd = phi_active.std() / mean_val
    return float(rsd)


def compute_entropy(
    state_vector: np.ndarray,
    normalized: bool = False,
) -> float:
    """
    Calculer l'entropie de Shannon du vecteur état.
    
    H = -∑ p_i ln(p_i), où p_i = φ_i / ∑φ
    
    Args:
        state_vector: Array (n_states,)
        normalized: Si True, diviser par H_max = ln(n_states)
        
    Returns:
        float: Entropie (normalisée si demandé)
        
    Examples:
        >>> phi = np.array([100, 100, 100])  # Uniforme
        >>> compute_entropy(phi, normalized=True)  # ~1.0
        >>>
        >>> phi2 = np.array([300, 0, 0])  # Concentrée
        >>> compute_entropy(phi2, normalized=True)  # ~0.0
    """
    total = state_vector.sum()
    if total == 0:
        return 0.0
    
    p = state_vector / total
    p = p[p > 0]  # Ignorer zéros
    
    H = -np.sum(p * np.log(p))
    
    if normalized:
        H_max = np.log(len(state_vector))
        H = H / H_max if H_max > 0 else 0.0
    
    return float(H)


def compute_segregation_intensity(
    state_vector: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Calculer l'intensité de ségrégation pour un état binaire.
    
    Metric pour mesurer mélange de deux espèces:
    I_S = Var(C) / (C̄ * (1 - C̄))
    
    où C est la concentration dans chaque cellule.
    
    Args:
        state_vector: Array (n_states,) - nombre particules type-A
        mask: Masque optionnel des états "actifs"
        
    Returns:
        float: Intensité ségrégation (I_S = 0 parfaitement mélangé)
    """
    if mask is not None:
        phi_active = state_vector[mask]
    else:
        phi_active = state_vector[state_vector > 0]
    
    if len(phi_active) <= 1:
        return 0.0
    
    C_mean = phi_active.mean()
    if C_mean == 0 or C_mean == 1:
        return 0.0
    
    I_S = phi_active.var() / (C_mean * (1 - C_mean))
    return float(I_S)


def validate_normalization(
    state_trajectory: np.ndarray,
    total_particles: float,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Valider que la conservation de particules est respectée.
    
    Args:
        state_trajectory: Array (n_timesteps, n_states)
        total_particles: N attendu (∑φ(t))
        tolerance: Tolérance relative
        
    Returns:
        Dict avec:
        - is_valid: Tous les timesteps OK
        - total_particles: Array (n_timesteps,)
        - deviations: Écarts relatifs
        - max_deviation: Écart maximal
        - all_timesteps_valid: Liste bool
        
    Examples:
        >>> traj = np.random.rand(100, 10)
        >>> traj *= 100 / traj.sum(axis=1, keepdims=True)
        >>> validation = validate_normalization(traj, 100)
        >>> print(f"Valid: {validation['is_valid']}")
    """
    totals = state_trajectory.sum(axis=1)
    deviations = np.abs(totals - total_particles) / total_particles
    all_valid = np.all(deviations <= tolerance)
    
    return {
        'is_valid': bool(all_valid),
        'total_particles': totals,
        'expected': float(total_particles),
        'deviations': deviations,
        'deviations_percent': deviations * 100,
        'max_deviation': float(np.max(deviations)),
        'mean_deviation': float(np.mean(deviations)),
        'all_timesteps_valid': (deviations <= tolerance).tolist(),
        'tolerance': tolerance,
    }


# ============================================================================
# COMPARISON METRICS
# ============================================================================

def compare_trajectories(
    traj1: np.ndarray,
    traj2: np.ndarray,
    method: str = "l2",
) -> Dict[str, Any]:
    """
    Comparer deux trajectoires d'état.
    
    Args:
        traj1: Array (n_timesteps, n_states)
        traj2: Array (n_timesteps, n_states)
        method: "l2" (Euclidean), "l1" (Manhattan), "cosine"
        
    Returns:
        Dict avec distances/similarités
        
    Examples:
        >>> traj_dem = np.random.rand(50, 10)
        >>> traj_markov = np.random.rand(50, 10)
        >>> comp = compare_trajectories(traj_dem, traj_markov)
        >>> print(f"Distance L2: {comp['mean_distance']:.4f}")
    """
    if traj1.shape != traj2.shape:
        raise ValueError(
            f"Trajectoires incompatibles: {traj1.shape} vs {traj2.shape}"
        )
    
    if method == "l2":
        distances = np.linalg.norm(traj1 - traj2, axis=1)
    elif method == "l1":
        distances = np.sum(np.abs(traj1 - traj2), axis=1)
    elif method == "cosine":
        # Cosine distance par timestep
        distances = []
        for t in range(len(traj1)):
            v1, v2 = traj1[t], traj2[t]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                distances.append(1 - cos_sim)  # Distance = 1 - similarité
            else:
                distances.append(0.0)
        distances = np.array(distances)
    else:
        raise ValueError(f"method unknown: {method}")
    
    return {
        'method': method,
        'distances': distances,
        'mean_distance': float(np.mean(distances)),
        'std_distance': float(np.std(distances)),
        'max_distance': float(np.max(distances)),
        'min_distance': float(np.min(distances)),
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """Exemples d'utilisation."""
    
    logger.info("=" * 70)
    logger.info("MARKOV MATH - EXAMPLES")
    logger.info("=" * 70)
    
    # Exemple 1: Analyse matrice
    logger.info("\n[Example 1] Analyse transition matrix")
    M = np.eye(5) * 0.8 + np.ones((5, 5)) * 0.04
    M /= M.sum(axis=1, keepdims=True)
    
    props = analyze_transition_matrix(M)
    print(f"Largest eigenvalue: {props['largest_eigenvalue']:.4f}")
    print(f"Condition number: {props['condition_number']:.2f}")
    print(f"Is row-stochastic: {props['is_row_stochastic']}")
    
    # Exemple 2: RSD
    logger.info("\n[Example 2] Calculer RSD")
    phi_homogeneous = np.array([100.0] * 10)
    phi_heterogeneous = np.array([10.0, 20.0, 50.0, 30.0, 90.0] + [25.0] * 5)
    
    rsd_homo = compute_rsd(phi_homogeneous)
    rsd_hetero = compute_rsd(phi_heterogeneous)
    
    print(f"RSD homogène: {rsd_homo:.4f}")
    print(f"RSD hétérogène: {rsd_hetero:.4f}")
    
    # Exemple 3: Validation normalisation
    logger.info("\n[Example 3] Validation normalisation")
    traj = np.random.rand(20, 10)
    traj *= 100 / traj.sum(axis=1, keepdims=True)
    
    validation = validate_normalization(traj, 100.0, tolerance=1e-6)
    print(f"Valid: {validation['is_valid']}")
    print(f"Max deviation: {validation['max_deviation_percent']:.6f}%")
