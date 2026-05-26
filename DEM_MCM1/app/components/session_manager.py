"""
session_manager.py
==================
Gestionnaire du contexte global (session state) de l'app Streamlit.

Responsabilités:
- Initialiser AppContext dans st.session_state
- Détecter les changements (filters)
- Notifier les pages des changements
- Accès thread-safe au contexte

Le contexte (AppContext) est l'unique "source de vérité" pour l'app.

Usage dans les pages:
    from components.session_manager import (
        initialize_session_state,
        get_app_context,
        get_context_version,
        show_refresh_notification,
    )
    
    # Une seule fois au démarrage app
    initialize_session_state()
    
    # Dans n'importe quelle page
    ctx = get_app_context()
    if ctx.selected_models:
        model = ctx.get_active_model()
        # ...

Examples:
    >>> from components.session_manager import *
    >>> initialize_session_state()
    >>> ctx = get_app_context()
    >>> print(ctx.summary())
"""

from __future__ import annotations
import streamlit as st
import time
import logging
from typing import Optional, Dict, Any

from src.Markov._config import AppContext, LoadedModel

logger = logging.getLogger(__name__)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

@st.cache_resource
def _get_session_initialized() -> bool:
    """
    Marker pour éviter ré-initialisation du contexte.
    
    Utilisé internement par initialize_session_state() pour s'assurer
    qu'on initialise qu'une SEULE FOIS par session Streamlit.
    """
    return True


def initialize_session_state() -> None:
    """
    Initialiser le session_state avec AppContext et autres variables globales.
    
    Appelé UNE SEULE FOIS au démarrage de l'app (dans app.py).
    
    Initialise:
    - app_context: AppContext (singleton)
    - context_version: int (incrémenté à chaque changement)
    - last_config_change_page: str (source du dernier changement)
    - config_change_detected: bool (pour UI notifications)
    
    Examples:
        >>> initialize_session_state()
        >>> assert 'app_context' in st.session_state
    """
    if _get_session_initialized():
        if 'app_context' not in st.session_state:
            st.session_state.app_context = AppContext()
            st.session_state.context_version = 0
            st.session_state.last_config_change_page = None
            st.session_state.config_change_detected = False
            logger.info("[SessionManager] ✅ Context initialisé")


def get_app_context() -> AppContext:
    """
    Accéder au contexte global (singleton).
    
    Utilisable depuis n'importe quelle page. Toutes les pages
    reçoivent la MÊME instance (via st.session_state).
    
    Returns:
        AppContext: Le contexte global unique
        
    Raises:
        RuntimeError: Si initialize_session_state() n'a pas été appelé
        
    Examples:
        >>> ctx = get_app_context()
        >>> print(f"Modèles chargés: {len(ctx.selected_models)}")
        >>> active = ctx.get_active_model()
    """
    if 'app_context' not in st.session_state:
        raise RuntimeError(
            "Session state non initialisé. "
            "Appelez initialize_session_state() d'abord (dans app.py)"
        )
    
    return st.session_state.app_context


def get_context_version() -> int:
    """
    Obtenir la version du contexte.
    
    La version est incrémentée à chaque changement du contexte.
    Utile pour détecter si contexte a changé depuis dernier appel.
    
    Returns:
        int: Numéro de version (commence à 0)
        
    Examples:
        >>> v1 = get_context_version()
        >>> # ... user changes filters ...
        >>> v2 = get_context_version()
        >>> if v2 > v1:
        ...     print("Context changed, refresh needed")
    """
    if 'context_version' not in st.session_state:
        st.session_state.context_version = 0
    
    return st.session_state.context_version


def increment_context_version() -> int:
    """
    Incrémenter la version du contexte (marqué comme modifié).
    
    Appelé en interne chaque fois que le contexte change.
    
    Returns:
        int: Nouvelle version
    """
    st.session_state.context_version = get_context_version() + 1
    return st.session_state.context_version


# ============================================================================
# CHANGE DETECTION
# ============================================================================

def detect_config_changes(
    current_filters: Dict[str, Any],
) -> bool:
    """
    Détecter si les filtres ont changé (appelé page 1).
    
    Mécanisme pour détecter quand l'utilisateur modifie les filtres
    (diamètre, méthode, etc.) et marquer le contexte comme changed.
    
    Args:
        current_filters: Filtres actuels (ex: {'diameters': [0.004], 'methods': ['voronoi']})
        
    Returns:
        bool: True si changement détecté, False sinon
        
    Examples:
        >>> filters = {'diameters': [0.004], 'methods': ['voronoi']}
        >>> changed = detect_config_changes(filters)
        >>> if changed:
        ...     st.info("Filtres mis à jour")
    """
    ctx = get_app_context()
    
    if ctx.current_filters != current_filters:
        ctx.current_filters = current_filters
        increment_context_version()
        st.session_state.config_change_detected = True
        return True
    
    return False


def notify_config_changed(source_page: str) -> None:
    """
    Notifier que config a changé (appelé par pages qui modifient contexte).
    
    Args:
        source_page: Nom page qui triggered changement (ex: "Page 1 Load Models")
        
    Examples:
        >>> notify_config_changed("Page 1️⃣ Load Models")
    """
    st.session_state.last_config_change_page = source_page
    st.session_state.config_change_detected = True
    increment_context_version()


# ============================================================================
# UI HELPERS - DISPLAY
# ============================================================================

def show_config_status_bar() -> None:
    """
    Afficher barre d'info sur l'état du contexte (pour toutes pages).
    
    À utiliser dans app.py (au top de toutes pages).
    
    Affiche:
    - Nombre de modèles chargés
    - Mode (simple vs comparaison)
    - Modèle actuellement actif
    
    Examples:
        >>> show_config_status_bar()
    """
    ctx = get_app_context()
    
    if not ctx.selected_models:
        st.info("ℹ️  Aucun modèle chargé. Allez **page 1️⃣ Load Models** pour charger.")
    else:
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            st.metric(
                label="Modèles chargés",
                value=len(ctx.selected_models)
            )
        
        with col2:
            mode = "Comparaison 🔀" if ctx.compare_mode else "Simple 📌"
            st.metric(label="Mode", value=mode)
        
        with col3:
            active = ctx.get_active_model()
            if active:
                st.write(f"**📍 Actif:** {active.method}")


def show_refresh_notification() -> bool:
    """
    Afficher notification "Config a changé" + bouton Refresh.
    
    À utiliser dans pages 2, 3, 4 pour notifier des changements
    provenant d'autres pages.
    
    Returns:
        bool: True si user a cliqué "Refresh"
        
    Examples:
        >>> if show_refresh_notification():
        ...     st.rerun()  # Reload la page
    """
    if not st.session_state.get('config_change_detected', False):
        return False
    
    source = st.session_state.get('last_config_change_page', 'Unknown page')
    
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.warning(
            f"⚠️  Configuration a changé (depuis {source})",
            icon="🔄"
        )
    with col2:
        if st.button("🔄 Rafraîchir", key=f"refresh_btn_{get_context_version()}"):
            st.session_state.config_change_detected = False
            return True
    
    return False


def show_models_summary() -> None:
    """
    Afficher tableau résumé des modèles actuellement chargés.
    
    À utiliser dans pages 2, 3, 4 pour afficher quels modèles
    sont en mémoire et peuvent être analysés.
    
    Examples:
        >>> show_models_summary()
    """
    ctx = get_app_context()
    
    if not ctx.selected_models:
        st.info("Aucun modèle chargé")
        return
    
    st.subheader(f"📦 {len(ctx.selected_models)} modèle(s) chargé(s)")
    
    import pandas as pd
    
    data = []
    for i, model in enumerate(ctx.selected_models):
        data.append({
            'Index': i,
            'Method': model.method,
            'Diameter': f"{model.particle_diameter} m" if model.particle_diameter else "All",
            'States': model.n_states,
            'τ': model.tau or "?",
            'Data Loaded': "✅" if model.is_data_loaded() else "❌"
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)


def show_context_debug_info() -> None:
    """
    Afficher infos debug (version contexte, etc).
    
    À utiliser pendant développement.
    
    Examples:
        >>> if st.checkbox("Debug info"):
        ...     show_context_debug_info()
    """
    ctx = get_app_context()
    
    st.write("**Debug Info:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Version:** {get_context_version()}")
    
    with col2:
        last_change = ctx.last_modified
        import datetime
        dt = datetime.datetime.fromtimestamp(last_change)
        st.write(f"**Last modified:** {dt.strftime('%H:%M:%S')}")
    
    with col3:
        st.write(f"**Compare mode:** {ctx.compare_mode}")


# ============================================================================
# MODEL MANAGEMENT HELPERS
# ============================================================================

def add_model_to_context(model: LoadedModel) -> None:
    """
    Ajouter un modèle au contexte (update + notify).
    
    Args:
        model: LoadedModel à ajouter
        
    Examples:
        >>> model = LoadedModel(folder_name="exp1", ...)
        >>> add_model_to_context(model)
    """
    ctx = get_app_context()
    ctx.add_model(model)
    notify_config_changed(source_page="Model added programmatically")


def remove_model_from_context(folder_name: str) -> None:
    """
    Retirer un modèle du contexte.
    
    Args:
        folder_name: Clé du modèle à retirer
    """
    ctx = get_app_context()
    ctx.remove_model(folder_name)
    notify_config_changed(source_page="Model removed")


def clear_all_models() -> None:
    """Vider tous les modèles du contexte."""
    ctx = get_app_context()
    ctx.clear_models()
    notify_config_changed(source_page="All models cleared")


def get_models_by_method(method: str) -> list[LoadedModel]:
    """
    Filtrer les modèles par méthode.
    
    Args:
        method: Méthode à filtrer
        
    Returns:
        list: Modèles correspondants
        
    Examples:
        >>> models_voronoi = get_models_by_method("voronoi")
    """
    ctx = get_app_context()
    return [m for m in ctx.selected_models if m.method == method]


def get_models_by_diameter(diameter: float) -> list[LoadedModel]:
    """
    Filtrer les modèles par diamètre.
    
    Args:
        diameter: Diamètre (0.004, 0.008, ou None)
        
    Returns:
        list: Modèles correspondants
    """
    ctx = get_app_context()
    return [m for m in ctx.selected_models if m.particle_diameter == diameter]
