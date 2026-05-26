"""
pages/4_📈_State_Evolution.py
=============================
Page 4️⃣: State Vector Evolution & Normalization Check

RÔLE: Consommatrice du contexte global.

Analyser l'évolution du vecteur d'état φ(t) au cours du temps:
- Multi-courbes φᵢ(t) vs t
- Heatmaps (partition vs temps)
- **VALIDATION CRITIQUE**: ∑φ(t) = N (conservation particules)
- Comparaison entre configurations
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging

from components.session_manager import (
    get_app_context,
    show_refresh_notification,
    show_models_summary,
)
from src.Markov.markov_math import validate_normalization, compute_rsd

logger = logging.getLogger(__name__)


st.title("4️⃣ State Vector Evolution & Normalization")

st.markdown("""
**Analyser l'évolution du vecteur d'état** φ(t) et 
**valider la conservation des particules** ∑φ(t) = N.

Cette page est **CRITIQUE** pour vérifier la qualité des simulations Markov.
""")

st.divider()


# ============================================================================
# REFRESH & CONTEXT
# ============================================================================

if show_refresh_notification():
    st.rerun()

ctx = get_app_context()

if not ctx.selected_models:
    st.warning(
        "⚠️  Aucun modèle chargé. Allez **page 1️⃣ Load Models** d'abord.",
        icon="🔧"
    )
else:
    show_models_summary()
    st.divider()
    
    # ============================================================================
    # SOURCE VECTOR INITIAL
    # ============================================================================
    
    st.subheader("📍 Vecteur d'État Initial φ(0)")
    
    state_source = st.radio(
        "Source de φ(0):",
        options=["From DEM data (t=250)", "From saved config"],
        horizontal=True,
    )
    
    if state_source == "From DEM data (t=250)":
        timestep_0 = st.slider(
            "Timestep initial:",
            min_value=250,
            max_value=500,
            value=250,
            step=50
        )
        st.info(
            f"φ(0) sera calculé depuis snapshot DEM à t={timestep_0}",
            icon="ℹ️"
        )
    else:
        st.info(
            "φ(0) depuis fichier config.json de chaque expérience",
            icon="ℹ️"
        )
    
    st.divider()
    
    # ============================================================================
    # EVOLUTION TRAJECTORIES
    # ============================================================================
    
    st.subheader("📈 Évolution φ(t) au cours du temps")
    
    tab1, tab2 = st.tabs(["Multi-curves", "Heatmaps"])
    
    with tab1:
        st.write("Trajectoires superposées pour tous les modèles")
        
        st.info("""
        🚧 **À implémenter:**
        - Charger trajectoires φ(t) depuis HF
        - Ploter multi-courbes (une par modèle)
        - Option filtrage par partition
        """)
    
    with tab2:
        st.write("Heatmaps (partition vs temps) pour chaque modèle")
        
        for model in ctx.selected_models:
            st.write(f"**{model}**")
            st.info("🚧 Heatmap (en développement)")
    
    st.divider()
    
    # ============================================================================
    # NORMALIZATION CHECK (CRITICAL)
    # ============================================================================
    
    st.subheader("✅ Validation Normalisation: ∑φ(t) = N")
    st.write("""
    **Condition CRITIQUE**: La somme du vecteur d'état doit rester égale
    au nombre total de particules à chaque pas de temps.
    """)
    
    col1, col2 = st.columns(2)
    
    # ---- GRAPHIQUE NORMALISATION ----
    with col1:
        st.write("**Graphique: Total Particles vs Time**")
        
        st.info("""
        🚧 **À implémenter:**
        - Ploter ∑φ(t) pour chaque modèle
        - Ligne rouge = N (attendu)
        - Identifier écarts normalisation
        """)
    
    # ---- STATISTIQUES ----
    with col2:
        st.write("**Statistiques Normalisation**")
        
        st.info("""
        🚧 **À implémenter:**
        - Max écart
        - Écart moyen
        - % timesteps valides
        - Alertes si écarts > seuil
        """)
    
    st.divider()
    
    # ============================================================================
    # COMPARATIVE METRICS
    # ============================================================================
    
    st.subheader("📊 Métriques Comparatives")
    
    if st.button("🔄 Calculer RSD"):
        st.write("RSD evolution pour chaque modèle:")
        
        with st.spinner("Calcul en cours..."):
            # TODO: Charger trajectoires et calculer RSD
            st.info("🚧 RSD calculation (en développement)")
    
    st.divider()
    
    # ============================================================================
    # DETAILED ANALYSIS
    # ============================================================================
    
    with st.expander("📊 Analyse détaillée par modèle"):
        
        for i, model in enumerate(ctx.selected_models):
            st.write(f"### [{i}] {model.method} (d={model.particle_diameter})")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("N (expected)", model.n_particles)
            
            with col2:
                st.metric("States", model.n_states)
            
            with col3:
                if model.transition_matrix is not None:
                    st.metric("Data", "✅ Loaded")
                else:
                    st.metric("Data", "❌ Not loaded")
    
    st.divider()
    
    # ============================================================================
    # EXPORT
    # ============================================================================
    
    st.subheader("💾 Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export states (CSV)"):
            st.info("🚧 CSV export (en développement)")
    
    with col2:
        if st.button("📸 Export graphs (PNG)"):
            st.info("🚧 PNG export (en développement)")


st.divider()

st.markdown("""
### 🎯 Objectifs de cette page

1. **Visualiser** comment l'état évolue avec le temps
2. **Valider** que ∑φ(t) = N (conservation)
3. **Comparer** trajectoires entre configurations
4. **Détecter** anomalies ou dérives numériques

### ⚠️ Signaux d'alerte

- ∑φ(t) décroît → problème de normalisation matrice M
- RSD augmente → mixage non-homogène
- Certains états vides → partitioning mal adapté

### 🚧 À venir

- Chargement dynamique trajectoires
- Heatmaps interactives
- Export batch comparaisons
""")
