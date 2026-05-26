"""
pages/3_📊_Analyze_Matrices.py
==============================
Page 3️⃣: Transition Matrices Analysis

RÔLE: Consommatrice du contexte global.

Analyser les propriétés spectrales des matrices transition:
- Heatmaps côte-à-côte
- Eigenvalue spectrum
- RSD evolution vs tau
- Steady-state analysis
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
from components.plotter import MatrixPlotter, EvolutionPlotter
from components.model_loader import get_model_loader
from src.Markov.markov_math import analyze_transition_matrix

logger = logging.getLogger(__name__)


st.title("3️⃣ Transition Matrices Analysis")

st.markdown("""
**Analyser les propriétés spectrales** des matrices transition.

Comparer comment différentes configurations construisent leurs
matrices et à quel point elles garantissent la convergence.
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
    # LOAD TRANSITION MATRICES
    # ============================================================================
    
    loader = get_model_loader()
    matrices = {}
    
    with st.spinner("📥 Chargement matrices..."):
        for model in ctx.selected_models:
            try:
                matrix = loader.load_matrix(model.folder_name)
                matrices[model.folder_name] = matrix
                logger.info(f"✅ Loaded matrix for {model.folder_name}: {matrix.shape}")
            except Exception as e:
                st.warning(f"⚠️  Could not load matrix for {model.folder_name}: {e}")
                logger.warning(f"Matrix loading failed: {e}")
    
    if not matrices:
        st.error("❌ No matrices could be loaded")
        st.stop()
    
    st.success(f"✅ Chargé {len(matrices)} matrices transition")
    
    st.divider()
    
    # ============================================================================
    # TABS
    # ============================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Heatmaps",
        "🎯 Eigenvalues",
        "📈 Spectral Properties",
        "📊 Detailed Stats"
    ])
    
    # ============================================================================
    # TAB 1: HEATMAPS
    # ============================================================================
    
    with tab1:
        st.subheader("🔥 Transition Matrix Heatmaps")
        st.write("""
        Visualiser les probabilités de transition entre états.
        
        **Interprétation:**
        - Diagonal bright = états stables (autocorrelation forte)
        - Off-diagonal bright = transitions fréquentes
        - Lignes uniformes = mélange rapide
        """)
        
        st.divider()
        
        # Single heatmap selector
        col1, col2 = st.columns([2, 1])
        
        with col1:
            model_names = [m.folder_name for m in ctx.selected_models]
            selected_model_name = st.selectbox(
                "Sélectionner modèle:",
                model_names,
                format_func=lambda x: [m.method for m in ctx.selected_models if m.folder_name == x][0]
            )
        
        with col2:
            if st.button("🎨 Générer heatmap"):
                try:
                    matrix = matrices[selected_model_name]
                    model = [m for m in ctx.selected_models if m.folder_name == selected_model_name][0]
                    
                    plotter = MatrixPlotter()
                    fig = plotter.plot_heatmap(model, matrix)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error generating heatmap: {e}")
                    logger.error(f"Heatmap error: {e}", exc_info=True)
        
        st.divider()
        
        # Comparison heatmaps
        st.subheader("📊 Comparaison côte-à-côte")
        
        if st.button("🔄 Générer comparaison"):
            try:
                plotter = MatrixPlotter(figsize_default=(15, 4))
                fig = plotter.plot_heatmap_comparison(ctx.selected_models, matrices)
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"❌ Erreur comparaison: {e}")
                logger.error(f"Comparison error: {e}", exc_info=True)
    
    # ============================================================================
    # TAB 2: EIGENVALUES
    # ============================================================================
    
    with tab2:
        st.subheader("🎯 Eigenvalue Spectrum")
        st.write("""
        Analyser les eigenvalues pour évaluer la convergence.
        
        **Critères:**
        - Largest eigenvalue λ₁ ≈ 1.0 (stochastique)
        - Spectral gap = 1 - λ₂ (vitesse mélange)
        - Tous |λᵢ| ≤ 1.0 (stabilité)
        """)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📈 Spectre d'un modèle"):
                try:
                    model_names = [m.folder_name for m in ctx.selected_models]
                    selected = st.selectbox(
                        "Choisir modèle:",
                        model_names,
                        key="eigenvalue_select",
                        format_func=lambda x: [m.method for m in ctx.selected_models if m.folder_name == x][0]
                    )
                    
                    matrix = matrices[selected]
                    model = [m for m in ctx.selected_models if m.folder_name == selected][0]
                    
                    plotter = MatrixPlotter()
                    fig = plotter.plot_eigenvalues(model, matrix)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("🔀 Comparer spectres"):
                try:
                    plotter = MatrixPlotter()
                    fig = plotter.plot_eigenvalue_comparison(ctx.selected_models, matrices)
                    st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"❌ Erreur comparaison: {e}")
    
    # ============================================================================
    # TAB 3: SPECTRAL PROPERTIES
    # ============================================================================
    
    with tab3:
        st.subheader("📈 Spectral Properties")
        
        # Calculate properties for all models
        properties_list = []
        
        for model in ctx.selected_models:
            matrix = matrices.get(model.folder_name)
            if matrix is None:
                continue
            
            try:
                props = analyze_transition_matrix(matrix)
                
                properties_list.append({
                    "Method": model.method,
                    "Diameter": model.particle_diameter,
                    "N States": model.n_states,
                    "λ₁": f"{props.get('largest_eigenvalue', 0):.6f}",
                    "λ₂": f"{props.get('second_largest_eigenvalue', 0):.6f}",
                    "Spectral Gap": f"{props.get('spectral_gap', 0):.6f}",
                    "Condition Number": f"{props.get('condition_number', np.inf):.2f}",
                    "Mixing Time": f"{props.get('mixing_time', np.nan):.1f}",
                })
            except Exception as e:
                logger.warning(f"Could not analyze {model.folder_name}: {e}")
        
        if properties_list:
            df_props = pd.DataFrame(properties_list)
            st.dataframe(df_props, use_container_width=True)
            
            # Download as CSV
            csv = df_props.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger (CSV)",
                data=csv,
                file_name="spectral_properties.csv",
                mime="text/csv"
            )
        else:
            st.warning("No properties to display")
    
    # ============================================================================
    # TAB 4: DETAILED ANALYSIS
    # ============================================================================
    
    with tab4:
        st.subheader("📊 Analyse détaillée")
        st.write("Propriétés spectrales détaillées pour chaque modèle")
        
        for model in ctx.selected_models:
            with st.expander(
                f"📋 {model.method} (d={model.particle_diameter}, {model.n_states} states)",
                expanded=False
            ):
                matrix = matrices.get(model.folder_name)
                
                if matrix is None:
                    st.warning("❌ Matrice non chargée")
                    continue
                
                try:
                    props = analyze_transition_matrix(matrix)
                    
                    # Layout in columns
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("λ₁", f"{props.get('largest_eigenvalue', 0):.6f}")
                        st.metric("λ₂", f"{props.get('second_largest_eigenvalue', 0):.6f}")
                    
                    with col2:
                        st.metric("Spectral Gap", f"{props.get('spectral_gap', 0):.6f}")
                        st.metric("Condition Number", f"{props.get('condition_number', np.inf):.2f}")
                    
                    with col3:
                        st.metric("Mixing Time", f"{props.get('mixing_time', np.nan):.1f}")
                        st.metric("Frobenius Norm", f"{props.get('frobenius_norm', 0):.4f}")
                    
                    st.divider()
                    
                    # Eigenvalues list
                    eigenvalues = np.linalg.eigvals(matrix)
                    eigenvalues_sorted = eigenvalues[np.argsort(-np.abs(eigenvalues))]
                    
                    st.write(f"**Top 10 Eigenvalues (by magnitude):**")
                    for i, ev in enumerate(eigenvalues_sorted[:10]):
                        real = np.real(ev)
                        imag = np.imag(ev)
                        mag = np.abs(ev)
                        
                        if abs(imag) < 1e-10:
                            st.write(f"  λ₍{i}₎ = {real:.6f} (|λ| = {mag:.6f})")
                        else:
                            st.write(f"  λ₍{i}₎ = {real:.4f} + {imag:.4f}i (|λ| = {mag:.6f})")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing matrix: {e}")
                    logger.error(f"Analysis error: {e}", exc_info=True)


st.divider()

st.markdown("""
### 📚 Concepts Clés

- **λ₁** (largest eigenvalue): Doit être ≈ 1.0 pour matrice stochastique
- **λ₂** (second eigenvalue): Plus petit en magnitude → meilleure convergence
- **Spectral gap (1 - λ₂)**: Plus grand = mélange plus rapide
- **Condition number κ(M)**: Mesure sensibilité aux erreurs numériques
- **Mixing time**: Estimation du temps pour convergence complète

### 🎯 Interprétation

- **Good matrix**: λ₁=1, λ₂≈0.8-0.95, gap>0.05, κ<100
- **Bad matrix**: λ₂→1, spectral gap→0, κ>>100 = convergence très lente
- **Unstable**: Quelconque |λᵢ| > 1 = divergence!

### 🚧 À venir

- Export analyse complète
- Comparaison cross-diameter
- Prédiction temps mélange
""")

