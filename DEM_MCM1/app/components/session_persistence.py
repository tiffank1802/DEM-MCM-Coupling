"""
session_persistence.py
======================
Persistance du contexte Streamlit (save/load session).

Permet de:
1. Sauvegarder l'état de la session (modèles chargés, filtres, etc)
2. Restaurer la session entre fermeture/réouverture app
3. Exporter la session en JSON pour analyse ultérieure

Stockage: .streamlit_cache/session_state.json (local)

Usage:
    from components.session_persistence import save_session, load_session, export_session
    
    # Sauvegarder automatiquement avant fermeture
    save_session()
    
    # Restaurer au démarrage
    load_session()
    
    # Exporter pour partage/archive
    export_session("my_analysis_session.json")

Examples:
    >>> from components.session_persistence import *
    >>> save_session()  # Sauvegarde dans .streamlit_cache/
    >>> load_session()  # Restaure au redémarrage
    >>> export_session("export.json")  # Export complet
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

from src.Markov._config import (
    AppContext,
    SESSION_CACHE_DIR,
    SESSION_CONFIG_FILE,
)
from .session_manager import (
    get_app_context,
    initialize_session_state,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CACHE DIRECTORY SETUP
# ============================================================================

def _get_cache_dir() -> Path:
    """
    Obtenir le répertoire cache (crée s'il n'existe pas).
    
    Returns:
        Path: Répertoire .streamlit_cache/
    """
    cache_dir = Path(SESSION_CACHE_DIR)
    cache_dir.mkdir(exist_ok=True, parents=True)
    return cache_dir


def _get_session_file() -> Path:
    """
    Obtenir le chemin du fichier session.
    
    Returns:
        Path: .streamlit_cache/session_state.json
    """
    return _get_cache_dir() / SESSION_CONFIG_FILE


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

class NumpyEncoder(json.JSONEncoder):
    """
    Encoder JSON custom pour gérer numpy arrays.
    
    Convertit:
    - np.ndarray → list
    - np.floating → float
    - np.integer → int
    """
    
    def default(self, obj: Any) -> Any:
        """Sérialiser objets non-JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        return super().default(obj)


# ============================================================================
# SAVE / LOAD SESSION
# ============================================================================

def save_session() -> bool:
    """
    Sauvegarder l'état de la session actuellement en mémoire.
    
    Sauvegarde:
    - Modèles chargés (métadata + params)
    - Filtres actuels
    - Mode comparaison
    - Timestamps
    
    Returns:
        bool: True si succès, False sinon
        
    Examples:
        >>> if save_session():
        ...     logger.info("Session sauvegardée")
    """
    try:
        initialize_session_state()
        ctx = get_app_context()
        
        # Sérialiser contexte
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'context': ctx.to_dict(),
        }
        
        # Écrire fichier
        session_file = _get_session_file()
        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(
            f"✅ Session sauvegardée: {session_file} "
            f"({len(ctx.selected_models)} modèles)"
        )
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde session: {e}")
        return False


def load_session() -> bool:
    """
    Restaurer l'état de la session depuis fichier sauvegardé.
    
    Restaure:
    - Modèles chargés (métadata seulement)
    - Filtres
    - Mode comparaison
    - Index actif
    
    ⚠️  Les matrices transition NE sont PAS restaurées (lazy-loaded)
    
    Returns:
        bool: True si session restaurée, False si aucune sauvegarde
        
    Examples:
        >>> if load_session():
        ...     st.success("Session restaurée")
        >>> else:
        ...     st.info("Aucune session sauvegardée")
    """
    try:
        session_file = _get_session_file()
        
        if not session_file.exists():
            logger.info("Aucune session sauvegardée trouvée")
            return False
        
        # Charger fichier
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        # Restaurer contexte
        ctx_dict = session_data.get('context', {})
        ctx = AppContext.from_dict(ctx_dict)
        
        # Mettre à jour session state
        import streamlit as st
        st.session_state.app_context = ctx
        
        timestamp = session_data.get('timestamp', 'unknown')
        logger.info(
            f"✅ Session restaurée: {len(ctx.selected_models)} modèles "
            f"(saved {timestamp})"
        )
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur restauration session: {e}")
        return False


# ============================================================================
# EXPORT / IMPORT
# ============================================================================

def export_session(filepath: str) -> bool:
    """
    Exporter la session en JSON (pour archive/partage).
    
    Exporte:
    - Tout l'état du contexte
    - Métadata des modèles chargés
    - Timestamp export
    
    ⚠️  Ne sauvegarde PAS les matrices (fichiers trop gros)
    
    Args:
        filepath: Chemin destination (ex: "my_session.json")
        
    Returns:
        bool: True si succès
        
    Examples:
        >>> export_session("analysis_session_2024.json")
        >>> # Peut être partagé ou archivé
    """
    try:
        initialize_session_state()
        ctx = get_app_context()
        
        export_data = {
            'app_name': 'DEM_MCM1 Markovian Analysis',
            'export_timestamp': datetime.now().isoformat(),
            'context': ctx.to_dict(),
            'num_models': len(ctx.selected_models),
            'metadata': {
                'compare_mode': ctx.compare_mode,
                'active_model': ctx.active_model_index,
                'models': [
                    {
                        'folder': m.folder_name,
                        'method': m.method,
                        'n_states': m.n_states,
                        'diameter': m.particle_diameter,
                    }
                    for m in ctx.selected_models
                ]
            }
        }
        
        output_path = Path(filepath)
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"✅ Session exportée: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur export session: {e}")
        return False


def import_session(filepath: str) -> bool:
    """
    Importer une session depuis fichier exporté.
    
    Args:
        filepath: Chemin du fichier JSON
        
    Returns:
        bool: True si succès
    """
    try:
        import streamlit as st
        
        input_path = Path(filepath)
        if not input_path.exists():
            raise FileNotFoundError(f"Fichier non trouvé: {filepath}")
        
        with open(input_path, 'r') as f:
            export_data = json.load(f)
        
        ctx_dict = export_data.get('context', {})
        ctx = AppContext.from_dict(ctx_dict)
        
        st.session_state.app_context = ctx
        
        logger.info(
            f"✅ Session importée: {len(ctx.selected_models)} modèles"
        )
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur import session: {e}")
        return False


# ============================================================================
# AUTO-SAVE HOOK
# ============================================================================

def setup_auto_save() -> None:
    """
    Configurer la sauvegarde automatique à chaque changement.
    
    À appeler une seule fois dans app.py.
    
    Examples:
        >>> import streamlit as st
        >>> from components.session_persistence import setup_auto_save
        >>> setup_auto_save()
    """
    import streamlit as st
    
    # Sauvegarder à chaque rerund si contexte a changé
    if 'last_save_version' not in st.session_state:
        st.session_state.last_save_version = -1
    
    from .session_manager import get_context_version
    current_version = get_context_version()
    
    if current_version > st.session_state.last_save_version:
        save_session()
        st.session_state.last_save_version = current_version


# ============================================================================
# UI HELPERS
# ============================================================================

def show_session_export_button() -> Optional[str]:
    """
    Afficher bouton "Exporter session" dans Streamlit.
    
    Returns:
        str: Contenu JSON si user a cliqué "Download"
        
    Examples:
        >>> json_str = show_session_export_button()
        >>> if json_str:
        ...     st.success("Session prête à exporter")
    """
    import streamlit as st
    
    if st.button("📥 Exporter session (JSON)"):
        initialize_session_state()
        ctx = get_app_context()
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'context': ctx.to_dict(),
        }
        
        json_str = json.dumps(export_data, indent=2, cls=NumpyEncoder)
        
        st.download_button(
            label="📥 Télécharger session",
            data=json_str,
            file_name=f"dem_mcm1_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        return json_str
    
    return None


def show_session_load_button() -> None:
    """
    Afficher bouton pour charger une session depuis fichier.
    
    Examples:
        >>> show_session_load_button()
    """
    import streamlit as st
    
    uploaded_file = st.file_uploader(
        "📤 Charger une session sauvegardée",
        type="json",
        key="session_uploader"
    )
    
    if uploaded_file is not None:
        try:
            session_data = json.load(uploaded_file)
            ctx_dict = session_data.get('context', {})
            ctx = AppContext.from_dict(ctx_dict)
            
            st.session_state.app_context = ctx
            st.success(f"✅ Session chargée: {len(ctx.selected_models)} modèles")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Erreur chargement session: {e}")


# ============================================================================
# CLEANUP
# ============================================================================

def cleanup_old_sessions(days: int = 7) -> int:
    """
    Nettoyer les anciennes sessions (optionnel).
    
    Args:
        days: Supprimer sessions plus vieilles que N jours
        
    Returns:
        int: Nombre fichiers supprimés
    """
    from datetime import timedelta
    import time
    
    cache_dir = _get_cache_dir()
    cutoff_time = time.time() - (days * 86400)
    
    deleted_count = 0
    for file in cache_dir.glob("session_*.json"):
        if file.stat().st_mtime < cutoff_time:
            file.unlink()
            deleted_count += 1
    
    if deleted_count > 0:
        logger.info(f"Nettoyage: {deleted_count} anciennes session(s) supprimée(s)")
    
    return deleted_count
