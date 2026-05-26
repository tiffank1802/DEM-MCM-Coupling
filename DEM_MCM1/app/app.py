"""
app.py
======
Orchestration principale de l'app Streamlit - DEM_MCM1 Comparative Analysis.

Responsabilités:
- Initialiser la session au démarrage
- Afficher header et status bar global
- Configurer auto-save
- Charger les pages automatiquement (Streamlit multipage)

Architecture:
- Page 0: Overview (accueil)
- Page 1: Load Models (source de vérité contexte)
- Page 2: Visualize 3D (consommatrice contexte)
- Page 3: Analyze Matrices (consommatrice contexte)
- Page 4: State Evolution (consommatrice contexte)

Usage:
    streamlit run app/app.py

Examples:
    Les pages sont chargées automatiquement depuis app/pages/*.py
    grâce au système multipage de Streamlit.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path to enable 'src' imports
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import logging

# Internal imports
from components.session_manager import (
    initialize_session_state,
    show_config_status_bar,
)
from components.session_persistence import (
    setup_auto_save,
    load_session,
)

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="DEM_MCM1 - Comparative Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/anomalyco/opencode",
        "Report a bug": "https://github.com/anomalyco/opencode/issues",
        "About": "DEM_MCM1 Markovian Chain Model Comparative Analysis Platform"
    }
)


# ============================================================================
# SESSION INITIALIZATION (UNE SEULE FOIS)
# ============================================================================

initialize_session_state()
load_session()  # Restaurer session sauvegardée si exists
setup_auto_save()  # Auto-save à chaque changement


# ============================================================================
# GLOBAL HEADER & SIDEBAR
# ============================================================================

st.title("🔬 DEM_MCM1: Markovian Chain Model Analysis")

with st.sidebar:
    st.markdown("""
    ---
    **Plateforme d'analyse comparative** pour Markovian Chains 
    en systèmes particulaires (DEM simulations).
    
    ### 📖 Guide rapide:
    1. **Page 1️⃣ Load Models** - Sélectionner configurations
    2. **Page 2️⃣ Visualize 3D** - Voir découpage du mélangeur
    3. **Page 3️⃣ Analyze Matrices** - Étudier matrices transition
    4. **Page 4️⃣ State Evolution** - Vérifier évolution états
    
    ### 🎯 Caractéristiques:
    - ✅ Multi-configuration comparison
    - ✅ Synchronisation pages (session state)
    - ✅ Persistance session (save/restore)
    - ✅ Export comparaison (CSV/JSON/PNG)
    
    ---
    """)
    
    st.write("### ⚙️ Session Control")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Session"):
            from components.session_persistence import save_session
            if save_session():
                st.success("✅ Sauvegardée")
            else:
                st.error("❌ Erreur")
    
    with col2:
        if st.button("📥 Export"):
            from components.session_persistence import show_session_export_button
            show_session_export_button()


# ============================================================================
# MAIN CONTENT - STATUS BAR
# ============================================================================

st.divider()

st.subheader("📊 Session Status")
show_config_status_bar()

st.divider()


# ============================================================================
# INTRO TEXT (visible sur page 0)
# ============================================================================

with st.expander("📖 About this platform", expanded=False):
    st.markdown("""
    ## DEM_MCM1: Comparative Markovian Analysis Platform
    
    ### Objectif
    Analyser et comparer **plusieurs configurations** d'un modèle Markovien
    pour étudier l'influence de paramètres (découpage, tau, type de particules)
    sur la dynamique de mélange.
    
    ### Workflow typique
    
    **1. Load Models** (page 1️⃣)
    - Filtrer par diamètre particule (0.004, 0.008 m, toutes)
    - Filtrer par méthode découpage (Voronoi, Cartesian, etc.)
    - Sélectionner modèles pour comparaison multi-config
    
    **2. Visualize 3D** (page 2️⃣)
    - Voir le découpage appliqué au mélangeur (3D)
    - Comparer visuellement différentes méthodes
    - Affichage: grille côte-à-côte, toggle, ou transparent overlay
    
    **3. Analyze Matrices** (page 3️⃣)
    - Étudier propriétés spectrales (eigenvalues, condition number)
    - Comparer heatmaps des matrices transition
    - Analyser cinétique RSD vs tau
    
    **4. State Evolution** (page 4️⃣)
    - Suivre évolution vecteur d'état φ(t) au cours du temps
    - **Validation critique:** ∑φ(t) = N (conservation particules)
    - Comparer trajectoires entre configurations
    
    ### 📐 Architecture Technique
    
    - **Markov**: Classe pour gestion UN découpage
    - **MarkovAnalyzer**: Orchestration PLUSIEURS expériences HF
    - **AppContext**: Contexte global synchro (session state)
    - **Session Persistence**: Save/restore session entre sessions
    
    ### ⚡ Synchronisation Pages
    
    Toutes les pages sont **synchronisées via `st.session_state`**:
    - Page 1 (Load Models) = source de vérité
    - Pages 2,3,4 (consumers) = lisent depuis session_state
    - Changements page 1 → notifs auto pages 2,3,4
    
    ### 💾 Persistance Session
    
    Votre session est **sauvegardée automatiquement**:
    - Modèles chargés, filtres, mode comparaison
    - Restaurée si fermeture/réouverture app
    - Exportable en JSON pour archive
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🔗 Links:**")
    st.markdown("[OpenCode Docs](https://opencode.ai/docs) | [GitHub](https://github.com/anomalyco/opencode)")

with col2:
    st.markdown("**📧 Support:**")
    st.markdown("[Report issue](https://github.com/anomalyco/opencode/issues)")

with col3:
    st.markdown("**📝 Version:**")
    st.markdown("DEM_MCM1 v2.0 - Professional Refactor")


logger.info("=" * 70)
logger.info("APP STARTED - DEM_MCM1 Comparative Analysis")
logger.info("=" * 70)
