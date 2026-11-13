# Architecture Modulaire CAO — DEM-MCM Studio

> Document d'architecture logicielle : séparation stricte entre le noyau scientifique pure-Rust (`src/gui/`), la couche d'affichage modulaire inspirée des logiciels de CAO (**SolidWorks**, **Fusion 360**, **Onshape**) servie par `dem-mcm-gui` (`src/gui/web/`), et le client natif **`egui` (`gui-egui/`)**.

---

## 1. Principes structurants

1. **Ergonomie CAO à panneaux pliables / dépliables** :
   - **Barre de commande haute** (`DEM-MCM Studio`, sans mention « Liquid Glass ») : source DEM, bouton **Exécuter**, et boutons pour replier/déplier l'arbre latéral, le ruban contextuel et la frise temporelle.
   - **Ruban contextuel pliable** *(façon Fusion 360)* regroupant des menus déroulants pliables (`Caméra 3D`, `Calques & Tailles`, `Critère & Horizon`) et la télémétrie compacte.
   - **Arbre de modèle latéral gauche pliable** *(façon FeatureManager / Onshape)* composé de nœuds indépendants pliables/dépliables (`Source DEM`, `Maillage 3D`, `Construction Markov`, `Inspecteur & Sélection`).
   - **Frise paramétrique pliable** en bas avec fenêtre verte de construction du modèle et curseur sur la durée complète de la DEM.
   - **Barre d'onglets de vues en pied d'écran** *(façon Document Tabs Onshape)*.
2. **Vue Sommaire + Vues Spécifiques Plein Format (« en très grand »)** :
   - **Vue `Sommaire`** : affiche de manière compacte les 6 caractéristiques clés, chacune dotée d'un bouton `⤢ En très grand` ouvrant la vue spécifique dédiée.
   - **7 Vues Spécifiques Plein Format** : `Mélangeur 3D`, `Teneur (Espèce)`, `Particules (N)`, `Erreurs L¹ & Discrepancy`, `Évolution P_inhom(t)`, `Matrices P & ΔP`, `Comparateur & Benchmark`.
3. **Figure dédiée à l'évolution des composants des matrices inhomogènes (`p_{i→j}(t)`)** :
   - Calculée par `src/gui/matrix_evolution.rs` selon `postprocessing/postprocess_inhomogeneous.py::fig_matrix_components_evolution` (seuil `0.01`, lois polynomiales jusqu'au degré 2, échelle Y commune `global_vmax * 1.15`, normes inter-blocs $\Delta P_k$).
   - Rendue en très grand format dans deux modes interactifs : **Figure unique superposée** (filtrable par portée, cellule source, espèce, seuil et tri) et **Grille à échelle commune** (paginée 16 panneaux/page avec zoom plein écran au clic).

---

## 2. Organisation modulaire des fichiers

```text
src/gui/
├── mod.rs                 # Exports publics
├── state.rs               # WorkspaceTab, CollapsiblePanelsState, MatrixEvolutionFilter, SimulationRequest
├── camera3d.rs            # Caméra orbite / pan / zoom / presets
├── mesh3d.rs              # Extraction polyédrique 3D des 14 approches + plans de coupe
├── metrics_eval.rs        # Métriques postprocess.py (Teneur, L1, RMS discrepancy, heatmaps, benchmark)
├── matrix_evolution.rs    # Évolution p_{i->j}(t) sur P_blocks, fits polynomiaux, global_vmax, ΔP_k
├── simulation.rs          # Orchestrateur & Cache Arc<BTreeMap<usize, DemFrame>>
└── web/                   # Frontend Web modulaire assemblé par include_str! (zéro dépendance externe)
    ├── mod.rs             # Assemble index_html() avec OnceLock
    ├── styles.css         # Thème sombre doux translucide (panneaux/menus pliables, vues plein format)
    ├── shell.html         # Squelette CAO 5 zones
    ├── store.js           # État, persistance localStorage, API & fallback statique
    ├── viewport3d.js      # Moteur 3D canvas (4mm/8mm échelle 1:2, maillage, flux, picking)
    ├── charts.js          # Primitives Canvas haute DPI + tooltips interactifs
    ├── app.js             # Orchestration des événements
    └── views/             # Modules de vues indépendants (summary, mixer3d, teneur, particles,
                           # errors, matrix_evol, matrices, benchmark, plan)

gui-egui/src/
├── main.rs                # Point d'entrée minimal eframe
├── app.rs                 # Shell CAO (ruban, arbre, frise, onglets de vues)
├── feature_tree.rs        # Arbre de modèle pliable (egui::CollapsingHeader)
├── ribbon.rs              # Ruban pliable avec menus déroulants
├── viewport3d.rs          # Moteur 3D egui::Painter
├── plots.rs               # Primitives egui_plot partagées
└── views/                 # Modules de vues indépendants
```
