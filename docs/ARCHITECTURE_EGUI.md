# Architecture Modulaire CAO & Recherche — DEM-MCM Studio

> Document d'architecture logicielle : séparation stricte entre le noyau scientifique pure-Rust (`src/research/`, `src/gui/`), l'interface modulaire Web (`src/gui/web/`) pilotée par un automate d'états sans encombrement (« zero-clutter »), et le client natif **`egui` (`gui-egui/`)**.

---

## 1. Principes structurants

1. **Ergonomie CAO à panneaux pliables / dépliables & 3 modes exclusifs** :
   - **Commutateur de modes exclusifs** dans la barre haute (`Explorer` · `Études de Recherche` · `Comparer A/B`) piloté par une machine à états centralisée (`src/gui/web/ui_machine.js` et `WorkbenchMode` côté Rust/egui) pour ne jamais afficher simultanément des centaines de boutons sans rapport.
   - **Barre de commande haute** (`DEM-MCM Studio`, sans mention « Liquid Glass ») : source DEM, bouton d'action unique contextuel (`Exécuter` / `Lancer l'étude` / `Lancer Benchmark`) verrouillé pendant le calcul, et bascules pour replier/déplier l'arbre, le ruban et la frise lorsque le mode actif les utilise.
   - **Ruban contextuel pliable** *(façon Fusion 360)* : les menus `Caméra 3D` et `Calques & Tailles` ne s'affichent que sur les vues 3D.
   - **Arbre de modèle latéral gauche pliable** *(façon FeatureManager / Onshape)* composé de nœuds indépendants pliables/dépliables (`Source DEM`, `Maillage 3D`, `Construction Markov`, `Inspecteur & Sélection`).
   - **Frise paramétrique pliable** en bas avec fenêtre verte de construction du modèle et curseur sur la durée complète de la DEM.
   - **Barre d'onglets de vues en pied d'écran** *(façon Document Tabs Onshape)*, affichée uniquement en mode `Explorer`.
2. **Découplage Spatial / Temporel & Moteur d'Études (`src/research/`)** :
   - `ResearchSessionCache` & `PreparedMethodSession` calculent le découpage spatial (`fit`, `states_matrix`, `S_matrices`, repère principal du lit `BedFrameBasis` optionnel) **une seule fois** par configuration spatiale et le réutilisent pour tous les balayages temporels.
   - `StudyRunner` exécute des études 1D et des grilles croisées 2D (`start_index`, `nlt`, `step`, `dt`, `tau`, `method`, `spatial_resolution`, `velocity_weight`, `species_mode`, `pred_start`) et calcule les métriques de convergence matricielle (Frobenius relative vs référence, lignes NaN, `diag_mean`, `|λ|_max`, `t50`).
   - **Approximation multi-lois `p_{i→j}(t)`** (`src/research/laws.rs`) : compare 7 familles mathématiques (`Constante`, `Linéaire`, `Quadratique`, `Cubique`, `Exponentielle`, `Logarithmique`, `Puissance`) sur temps normalisé décalé avec `RMSE`, `R²`, `AICc` et validation hors-bloc temporelle (`out_of_block_rmse`).
3. **Vue Sommaire + Vues Spécifiques Plein Format (« en très grand »)** :
   - **Vue `Sommaire`** : affiche de manière compacte les 6 caractéristiques clés, chacune dotée d'un bouton `⤢ En très grand` ouvrant la vue spécifique dédiée.
   - **Vues Spécifiques Plein Format** : `Mélangeur 3D`, `Teneur (Espèce)`, `Particules (N)`, `Erreurs L¹ & Discrepancy`, `Évolution P_inhom(t)` (enrichie des multi-lois par transition), `Matrices P & ΔP`, `Atelier d'Études Paramétriques`, `Comparateur & Benchmark`.

---

## 2. Organisation modulaire des fichiers

```text
src/research/
├── mod.rs                 # Exports publics du pôle d'études scientifiques
├── session.rs             # PreparedMethodSession, ResearchSessionCache (LRU 48), BedFrameBasis PCA
├── laws.rs                # 7 familles de lois p_{i->j}(t), AICc, validation out-of-block
├── study.rs               # StudySpec, StudyAxisKind (10 axes), grilles 1D/2D & StudyResultBundle
└── presets.rs             # 8 presets reproductibles (start, NLT/Frobenius, dt/NaN, tau, méthodes, lois 2D)

src/gui/
├── mod.rs                 # Exports publics
├── state.rs               # WorkbenchMode, WorkspaceTab, CollapsiblePanelsState, SimulationRequest
├── camera3d.rs            # Caméra orbite / pan / zoom / presets
├── mesh3d.rs              # Extraction polyédrique 3D des 14 approches + plans de coupe
├── metrics_eval.rs        # Métriques postprocess.py (Teneur, L1, RMS discrepancy, heatmaps, benchmark)
├── matrix_evolution.rs    # Évolution p_{i->j}(t) sur P_blocks + candidats multi-lois par transition
├── simulation.rs          # Orchestrateur partagé avec ResearchSessionCache
└── web/                   # Frontend Web modulaire assemblé par include_str! (zéro dépendance externe)
    ├── mod.rs             # Assemble index_html() avec OnceLock
    ├── ui_machine.js      # Automate d'états UI exclusifs & visibilité conditionnelle
    ├── styles.css         # Thème sombre doux translucide (panneaux/menus pliables, vues plein format)
    ├── shell.html         # Squelette CAO + Atelier d'Études schema-driven
    ├── store.js           # État, persistance localStorage, API /api/study/* & fallback statique
    ├── viewport3d.js      # Moteur 3D canvas (4mm/8mm échelle 1:2, maillage, flux, picking)
    ├── charts.js          # Primitives Canvas haute DPI + carte 2D interactive des lois
    ├── app.js             # Orchestration des événements
    └── views/             # Modules de vues indépendants (summary, mixer3d, teneur, particles,
                           # errors, matrix_evol, matrices, benchmark, study, plan)

gui-egui/src/
├── main.rs                # Point d'entrée minimal eframe
├── app.rs                 # Shell CAO (modes exclusifs, ResearchSessionCache, ruban, arbre, frise)
├── feature_tree.rs        # Arbre de modèle pliable (egui::CollapsingHeader)
├── ribbon.rs              # Ruban pliable piloté par WorkbenchMode & vue 3D active
├── viewport3d.rs          # Moteur 3D egui::Painter
├── plots.rs               # Primitives egui_plot partagées
└── views/                 # Modules de vues indépendants dont views/study.rs
```
