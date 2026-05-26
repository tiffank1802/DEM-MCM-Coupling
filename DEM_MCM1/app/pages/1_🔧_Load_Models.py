"""
pages/1_🔧_Load_Models.py
==========================
Page 1️⃣: Load & Select Models

RÔLE CRITIQUE: C'est la SEULE page qui MODIFIE le contexte global.

Responsabilités:
1. Lister modèles disponibles sur HuggingFace
2. Filtrer par diamètre et méthode
3. Multi-sélection modèles
4. Charger métadata + matrices
5. Notifier autres pages des changements

Architecture:
- Source de vérité pour AppContext
- Pages 2,3,4 lisent depuis session_state
- Changements ici → détectés pages 2,3,4 → auto-refresh
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path to enable 'src' imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import streamlit as st
import pandas as pd
import logging

from src.Markov.analyze_results import MarkovAnalyzer
from src.Markov._config import LoadedModel
from components.session_manager import (
    get_app_context,
    detect_config_changes,
    notify_config_changed,
    show_models_summary,
)

logger = logging.getLogger(__name__)


st.title("1️⃣ Load & Select Models")

st.markdown("""
**Sélectionner les configurations** à analyser et comparer.

Cette page est la **source de vérité** pour le contexte global:
- Changements ici sont vus automatiquement par les autres pages
- Les autres pages reçoivent une notification de refresh
""")

st.divider()


# ============================================================================
# FILTRES (déclencheurs de changements)
# ============================================================================

st.subheader("🔍 Filtres")

col1, col2 = st.columns(2)

with col1:
    selected_diameters = st.multiselect(
        "**Particle Diameters:**",
        options=[None, 0.004, 0.008],
        default=[None],
        format_func=lambda x: "All particles" if x is None else f"{x} m"
    )

with col2:
    selected_methods = st.multiselect(
        "**Partitioning Methods:**",
        options=[
            "cartesian",
            "cylindrical",
            "voronoi",
            "quantile",
            "octree",
            "physics",
            "adaptive",
            "multizone",
            "single"
        ],
        default=["cartesian", "voronoi"]
    )

# ---- Détecter changements filtres ----
filters = {
    'diameters': selected_diameters,
    'methods': selected_methods,
}

changed = detect_config_changes(filters)

if changed:
    st.info("✅ Filtres mis à jour", icon="🔍")
    notify_config_changed(source_page="Page 1️⃣ Load Models")


# ============================================================================
# CHARGE MODÈLES DISPONIBLES
# ============================================================================

st.subheader("📦 Modèles Disponibles")

with st.spinner("🔄 Recherche modèles sur HuggingFace..."):
    try:
        analyzer = MarkovAnalyzer()
        models_list = []
        for diameter in selected_diameters:
            for method in selected_methods:
                models = analyzer.list_available_models(
                    method=method,
                    particle_diameter=diameter,
                )
                models_list.extend(models)
    except Exception as e:
        st.error(f"❌ Erreur: {e}")
        models_list = []


if not models_list:
    st.warning("⚠️  Aucun modèle trouvé avec ces critères", icon="🔍")
else:
    # ---- Afficher comme tableau ----
    st.write(f"**Trouvé: {len(models_list)} modèles**")
    
    df_display = pd.DataFrame([{
        'Method': m['method'].upper(),
        'Diameter': f"{m.get('diameter')} m" if m.get('diameter') else "All",
        'States': m.get('n_states', '?'),
        'τ (timesteps)': m.get('tau', '?'),
        'NLT': m.get('nlt', '?'),
        'Fraction Visited': f"{m.get('fraction_visited', 1.0):.2f}",
    } for m in models_list])
    
    st.dataframe(df_display, use_container_width=True)
    
    # ---- Multi-sélection ----
    st.subheader("✅ Sélectionner pour comparaison")
    
    selected_indices = st.multiselect(
        "Choisir modèles à analyser:",
        options=range(len(models_list)),
        default=[],
        format_func=lambda i: (
            f"{models_list[i]['method'].upper()} | "
            f"d={models_list[i].get('diameter')} | "
            f"n={models_list[i].get('n_states')} | "
            f"τ={models_list[i].get('tau', '?')}"
        ),
    )
    
    if selected_indices:
        st.divider()
        
        # ---- UPDATE CONTEXT (CRITICAL) ----
        ctx = get_app_context()
        ctx.clear_models()
        
        with st.spinner("📦 Chargement configurations..."):
            for idx in selected_indices:
                m = models_list[idx]
                model = LoadedModel(
                    folder_name=m['folder'],
                    method=m['method'],
                    particle_diameter=m.get('diameter'),
                    n_states=m.get('n_states', 0),
                    n_particles=m.get('n_particles', 0),
                    nlt=m.get('nlt'),
                    tau=m.get('tau'),
                    fraction_visited=m.get('fraction_visited', 1.0),
                    description=m.get('description'),
                )
                ctx.add_model(model)
        
        ctx.compare_mode = len(selected_indices) > 1
        notify_config_changed(source_page="Page 1️⃣ Load Models")
        
        # ---- Afficher résumé ----
        st.success(f"✅ {len(selected_indices)} modèle(s) chargé(s)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Modèles chargés", len(ctx.selected_models))
        
        with col2:
            mode = "🔀 Comparaison" if ctx.compare_mode else "📌 Simple"
            st.metric("Mode", mode)
        
        # ---- Tableau résumé ----
        st.subheader("📊 Résumé configurations chargées")
        show_models_summary()
        
        # ---- Infos supplémentaires ----
        with st.expander("ℹ️ Détails", expanded=False):
            st.write(f"**Context summary:** {ctx.summary()}")
            
            for i, model in enumerate(ctx.selected_models):
                st.write(f"**[{i}] {model.method} (d={model.particle_diameter})**")
                st.write(f"  - Folder: `{model.folder_name}`")
                st.write(f"  - States: {model.n_states}")
                st.write(f"  - τ: {model.tau}, NLT: {model.nlt}")
                st.write(f"  - Fraction visited: {model.fraction_visited}")
    else:
        st.info("👈 Sélectionner au moins un modèle pour continuer")


st.divider()

st.markdown("""
### 📝 Instructions

1. **Filtrer** par diamètre particule et type de découpage
2. **Sélectionner** les configurations à comparer
3. Aller aux pages suivantes pour analyser:
   - Page 2️⃣: Visualisation 3D du découpage
   - Page 3️⃣: Analyse matrices transition
   - Page 4️⃣: Évolution vecteurs d'état

### 💡 Conseils

- **Comparaison optimale:** 2-3 modèles maximum (UI reste fluide)
- **Combinaisons utiles:** 
  - Même méthode, différents diamètres → impact taille particule
  - Même diamètre, différentes méthodes → impact type découpage
  - Même config, différents τ → analyse sensibilité
""")
