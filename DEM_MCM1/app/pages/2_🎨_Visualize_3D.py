"""
pages/2_🎨_Visualize_3D.py
==========================
Page 2️⃣: 3D Visualization - Partitioner Comparison

RÔLE: Consommatrice du contexte global (lecture seule).

Visualiser et comparer les partitionings en 3D:
- Affichage du mélangeur avec découpage appliqué
- Particules coloriées par partition
- Multiple view modes (grid, toggle, transparent overlay)
- Plans de coupe optionnels (xy, yz, xz)
"""

import streamlit as st
import numpy as np
import logging

from components.session_manager import (
    get_app_context,
    get_context_version,
    show_refresh_notification,
    show_models_summary,
)
from components.visualizer import PartitioningVisualizer
from src.Markov.markov_core import Markov
from src.Markov.bucket_io import get_fs
from app.components.model_loader import get_model_loader

logger = logging.getLogger(__name__)


st.title("2️⃣ 3D Visualization")

st.markdown("""
**Visualiser et comparer les partitionings** du mélangeur en 3D.

Découvrez comment différentes méthodes de découpage divisent l'espace
et affectent l'assignation des particules.
""")

st.divider()


# ============================================================================
# REFRESH DETECTION
# ============================================================================

if show_refresh_notification():
    st.rerun()


# ============================================================================
# CONTEXT READING
# ============================================================================

ctx = get_app_context()

if not ctx.selected_models:
    st.warning(
        "⚠️  Aucun modèle chargé. Allez **page 1️⃣ Load Models** d'abord.",
        icon="🔧"
    )
else:
    # ---- Afficher modèles en mémoire ----
    show_models_summary()
    
    st.divider()
    
    # ============================================================================
    # LOAD DEM DATA
    # ============================================================================
    
    st.subheader("📦 Chargement données")
    
    # Get first model to determine particle diameter
    first_model = ctx.selected_models[0]
    
    with st.spinner("📥 Chargement DEM..."):
        try:
            # Build Markov for first model to load DEM data
            loader = get_model_loader()
            mk = loader.build_markov(first_model.folder_name)
            
            # Load DEM data
            dem_data = mk.load_dem_data(
                particle_diameter=first_model.particle_diameter
            )
            
            # Get timestep 250 (initial)
            if 250 in dem_data:
                coords = dem_data[250][['coordinates:0', 'coordinates:1', 'coordinates:2']].values
                st.success(f"✅ Chargé {coords.shape[0]} particules à t=250")
            else:
                # Get first available timestep
                first_ts = min(dem_data.keys())
                coords = dem_data[first_ts][['coordinates:0', 'coordinates:1', 'coordinates:2']].values
                st.success(f"✅ Chargé {coords.shape[0]} particules à t={first_ts}")
        except Exception as e:
            st.error(f"❌ Erreur chargement DEM: {e}")
            logger.error(f"DEM loading error: {e}")
            coords = None
    
    if coords is None:
        st.stop()
    
    st.divider()
    
    # ============================================================================
    # VISUALIZATION MODE
    # ============================================================================
    
    st.subheader("🎨 Mode Affichage")
    
    view_mode = st.radio(
        "Choisir mode visualisation:",
        options=["Grid (2x2)", "Toggle (Sliders)", "Transparent Overlay"],
        horizontal=True,
    )
    
    # ---- RENDER OPTIONS ----
    with st.expander("⚙️ Options de rendu", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            render_height = st.slider("Hauteur (px):", 300, 800, 500)
            render_width = st.slider("Largeur (px):", 300, 1000, 700)
            point_size = st.slider("Taille particules:", 1.0, 10.0, 5.0)
        
        with col2:
            background_color = st.selectbox("Couleur fond:", ["white", "black", "gray"])
            show_grid_axes = st.checkbox("Montrer grille", value=True)
            camera_view = st.selectbox(
                "Vue caméra:",
                ["isometric", "xy", "xz", "yz"]
            )
    
    try:
        # Initialize visualizer
        visualizer = PartitioningVisualizer()
        
        # ---- GRID MODE ----
        if view_mode == "Grid (2x2)":
            st.write("**Affichage côte-à-côte** de tous les modèles chargés")
            
            n_models = len(ctx.selected_models)
            
            # Show rendering status
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Render all models
            images = visualizer.render_grid(
                models=ctx.selected_models,
                coords=coords,
                n_cols=2,
                height=render_height,
                width=render_width,
                background=background_color,
                point_size=point_size,
                show_grid=show_grid_axes,
                camera_position=camera_view,
            )
            
            # Display in columns
            n_cols_display = min(2, n_models)
            
            for i, model in enumerate(ctx.selected_models):
                if i % n_cols_display == 0:
                    cols = st.columns(n_cols_display)
                
                with cols[i % n_cols_display]:
                    st.subheader(f"{model.method}")
                    st.write(f"**Config:**")
                    st.write(f"- Diamètre: {model.particle_diameter}")
                    st.write(f"- États: {model.n_states}")
                    st.write(f"- Particules: {coords.shape[0]}")
                    
                    img = images.get(model.folder_name)
                    if img is not None:
                        st.image(img, use_column_width=True)
                    else:
                        st.warning("⚠️  Rendu échoué")
        
        # ---- TOGGLE MODE ----
        elif view_mode == "Toggle (Sliders)":
            st.write("**Toggle entre modèles** avec sliders")
            
            model_idx = st.slider(
                "Sélectionner modèle:",
                0, len(ctx.selected_models) - 1,
                ctx.active_model_index
            )
            
            ctx.active_model_index = model_idx
            model = ctx.selected_models[model_idx]
            
            st.subheader(f"Visualisation: {model.method}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Diamètre", model.particle_diameter)
            with col2:
                st.metric("États", model.n_states)
            with col3:
                st.metric("Particules", coords.shape[0])
            
            # Render single model
            with st.spinner("🎨 Rendu en cours..."):
                img = visualizer.render_single(
                    model=model,
                    coords=coords,
                    height=render_height,
                    width=render_width,
                    background=background_color,
                    point_size=point_size,
                    show_grid=show_grid_axes,
                    camera_position=camera_view,
                )
            
            st.image(img, use_column_width=True)
        
        # ---- TRANSPARENT OVERLAY ----
        else:
            st.write("**Superposition transparente** de tous les modèles")
            
            # Opacity sliders for each model
            opacities = {}
            col_count = min(3, len(ctx.selected_models))
            cols = st.columns(col_count)
            
            for i, model in enumerate(ctx.selected_models):
                with cols[i % col_count]:
                    opacity = st.slider(
                        f"{model.method}",
                        0.0, 1.0,
                        0.7,
                        key=f"opacity_{model.folder_name}"
                    )
                    opacities[model.folder_name] = opacity
            
            # Render overlay
            with st.spinner("🎨 Rendu overlay en cours..."):
                img = visualizer.render_overlay(
                    models=ctx.selected_models,
                    coords=coords,
                    opacities=opacities,
                    height=render_height,
                    width=render_width,
                    background=background_color,
                    show_grid=show_grid_axes,
                )
            
            st.image(img, use_column_width=True)
        
    except Exception as e:
        st.error(f"❌ Erreur visualisation: {e}")
        logger.error(f"Visualization error: {e}", exc_info=True)
    
    st.divider()
    
    # ---- CLIPPING PLANE ----
    with st.expander("✂️ Plan de coupe (clipping)", expanded=False):
        st.write("Couper l'affichage avec un plan pour voir l'intérieur")
        
        enable_clipping = st.checkbox("Activer plan de coupe")
        
        if enable_clipping:
            col1, col2 = st.columns(2)
            
            with col1:
                clipping_axis = st.selectbox(
                    "Direction du plan:",
                    options=['x', 'y', 'z']
                )
            
            with col2:
                clip_value = st.slider(
                    f"Position ({clipping_axis}-axis):",
                    0.0, 1.0, 0.5
                )
            
            # Render with clipping
            if st.button("🎨 Afficher avec découpe"):
                try:
                    model = ctx.selected_models[ctx.active_model_index]
                    
                    with st.spinner("🎨 Rendu avec découpe..."):
                        img = visualizer.render_with_clipping(
                            model=model,
                            coords=coords,
                            clip_plane=clipping_axis,
                            clip_value=clip_value,
                            height=render_height,
                            width=render_width,
                            background=background_color,
                            show_grid=show_grid_axes,
                        )
                    
                    st.image(img, use_column_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Erreur rendu découpe: {e}")
                    logger.error(f"Clipping render error: {e}", exc_info=True)


st.divider()

st.markdown("""
### 💡 Guide

- **Grid mode**: Idéal pour comparer 2-4 configurations côte-à-côte
- **Toggle mode**: Parfait pour examiner une configuration en détail
- **Overlay mode**: Voir comment les partitionings se superposent
- **Clipping**: Couper avec un plan pour voir la structure interne

### 🎯 Interprétation

- **Couleurs uniformes**: Bonne distribution partitions
- **Zones vides**: Partitions inutilisées
- **Clusters**: Regroupement naturel des particules
- **Comparaison overlay**: Quelle méthode capture mieux le mélange?

### ⚙️ Performance

- Chaque rendu 3D prend ~5-10s selon résolution
- Utilisez une résolution basse (400-500px) pour itération rapide
- Augmentez pour export final

### 🚧 À venir

- Animation rotation automatique
- Export en PNG haute résolution
- Export vidéo MP4 (rotation 360°)
""")

