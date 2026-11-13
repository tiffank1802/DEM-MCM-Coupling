# Plan d'Architecture Modulaire (Façon CAO) — DEM-MCM Studio

> Refonte structurelle de l'interface et du code GUI (`src/gui/` et `gui-egui/`), inspirée de **SolidWorks**, **Fusion 360** et **Onshape**, visant une modularité stricte, des panneaux et menus entièrement pliables/dépliables, une vue d'ensemble sommaire couplée à des vues spécifiques plein format, et une figure dédiée pour l'évolution des composants des matrices du modèle inhomogène.

---

## 1. Principes directeurs de la refonte

| Exigence | Décision architecturale |
| :--- | :--- |
| **Suppression du terme « Liquid Glass » dans les titres** | Conserver le style visuel sombre doux et translucide, mais supprimer toute mention *Liquid Glass* dans `<title>`, le nom d'en-tête, les badges et les libellés d'interface. Titre sobre : **`DEM-MCM Studio`**. |
| **Menus déroulants et panneaux pliables / dépliables** | Chaque menu, groupe de paramètres, tiroir d'options, barre de commandes (ruban) et panneau latéral peut être plié/déplié individuellement (ou mémorisé dans l'état UI), libérant jusqu'à **95 % de l'écran** pour la caractéristique observée. |
| **Voir tout sommairement + voir chaque caractéristique en très grand** | Séparation nette entre une **Vue Sommaire** (tableau de bord synthétique avec miniatures cliquables munies d'un bouton `⤢ Agrandir`) et **7 Vues Spécifiques Plein Format** où la caractéristique choisie occupe 100 % du viewport central. |
| **Évolution des composants de $P_{\text{inhom}}(t)$ sur une figure dédiée** | Création d'une vue spécifique et d'un module scientifique dédiés reproduisant et enrichissant `postprocessing/postprocess_inhomogeneous.py::fig_matrix_components_evolution` (toutes transitions significatives $p_{i \to j}(t)$, lois polynomiales ajustées, échelle Y commune, vue grille multi-panneaux et vue superposée filtrable). |
| **Structure modulaire type CAO (SolidWorks / Fusion 360 / Onshape)** | Découpage du workspace en 5 zones orthogonales (Barre d'ateliers, Ruban pliable, Arbre de modèle/contrôle pliable à gauche, Viewport principal 100 % dynamique, Frise temporelle paramétrique pliable en bas) + découpage strict des fichiers source par responsabilité. |

---

## 2. Anatomie structurelle de l'interface (modèle CAO)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ [DEM-MCM Studio]  Source DEM ▾   [ Exécuter ]          [Ruban ▾] [Arbre ◂]   │  ← Top Command Strip
├──────────────────────────────────────────────────────────────────────────────┤
│ (Ruban contextuel — PLIABLE / DÉPLIABLE en 1 clic, groupes accordéons)       │  ← Collapsible Ribbon
├───────────────┬──────────────────────────────────────────────────────────────┤
│ Arbre Modèle  │                                                              │
│ (PLIABLE ◂/▸) │                                                              │
│               │                                                              │
│ ▾ Source DEM  │                 VIEWPORT PRINCIPAL                           │
│   • 0..5999   │        (100 % de l'espace dédié à la vue active)             │
│   • 4mm / 8mm │                                                              │
│ ▾ Maillage 3D │   • En mode « Sommaire » : vue compacte multi-vignettes      │
│   • Méthode   │     avec bouton [⤢ Agrandir] sur chaque vignette.            │
│   • Params    │   • En mode « Vue Spécifique » : la caractéristique observée │
│   • Calques   │     occupe tout le viewport (zéro carte superflue).          │
│ ▾ Modèle MK   │                                                              │
│   • NLT/step  │                                                              │
│   • tau/start │                                                              │
│ ▾ Critère     │                                                              │
│   • Teneur    │                                                              │
│   • Durée DEM │                                                              │
│ ▾ Inspecteur  │                                                              │
│   • Cellule i │                                                              │
│   • Edge i→j  │                                                              │
├───────────────┴──────────────────────────────────────────────────────────────┤
│ [Lecture]  t = 120  ├══════[Fenêtre modèle]══════════════════════════┤  [▾]  │  ← Frise Paramétrique (pliable)
├──────────────────────────────────────────────────────────────────────────────┤
│ [Sommaire] [Mélangeur 3D] [Teneur] [Particules N] [Erreurs L¹/RMS]           │  ← Barre de Vues Spécifiques
│ [Évolution P_inhom(t)] [Matrices P / ΔP] [Comparateur & Benchmark] [Plan]    │    (façon onglets Onshape)
└──────────────────────────────────────────────────────────────────────────────┘
```

### Comportement des éléments pliables / dépliables
1. **Arbre de modèle latéral gauche (FeatureManager — SolidWorks / Fusion 360)** :
   - Peut être **complètement rétracté (`◂` / `▸`)** pour laisser toute la largeur au viewport.
   - Contient des **nœuds pliables/dépliables** (accordéons indépendants) :
     - *Source DEM & Espèces*
     - *Découpage spatial / Maillage (14 approches + paramètres dynamiques)*
     - *Construction Markov (`NLT`, `step`, `dt`, `tau`, `start_index`)*
     - *Horizon de prédiction & Critère (`Teneur small/large`, `Durée DEM complète`)*
     - *Affichage 3D (`4 mm`, `8 mm`, `Maillage`, `Faces`, `Vitesses`, `Flux P_ij`, `Tambour`)*
     - *Inspecteur de sélection (Cellule `#i` ou transition `p_{i→j}` cliquée)*
2. **Ruban contextuel haut** :
   - **Pliable / dépliable** via chevron (`▴ / ▾`, façon Fusion 360 « Minimize Ribbon ») pour ne garder qu'une barre de `36 px` lors de l'inspection plein écran.
   - Chaque groupe de contrôles ou menu d'options avancées (`Paramètres méthode`, `Options 3D`, `Filtres transitions`) utilise des **menus déroulants / popovers pliables**, jamais d'empilement permanent.
3. **Frise temporelle paramétrique en bas (Timeline Fusion 360)** :
   - Pliable sur les vues purement analytiques, toujours dépliable pour animer la DEM (`t_min .. t_max`) avec indication visuelle de la fenêtre de construction des modèles.
4. **Barre de Vues Spécifiques en pied d'interface (Document Tabs Onshape)** :
   - Permet de passer en un clic entre la vue **Sommaire** et chaque **Vue Spécifique Plein Format**, sans modifier les réglages de simulation ni perdre le contexte.

---

## 3. Catalogue des vues : Sommaire vs Vues Spécifiques (en très grand)

### A. Vue `Sommaire` (synthèse compacte)
- Affiche en un coup d'œil les **6 caractéristiques essentielles** dans des panneaux minimalistes sans titre lourd (juste un bandeau discret avec un bouton **`⤢ Agrandir`** qui ouvre immédiatement la vue spécifique correspondante) :
  1. **Aperçu 3D du mélangeur** (deux tailles `4 mm` / `8 mm` + maillage) → `[⤢ Mélangeur 3D]`
  2. **Teneur en espèce & RSD** (DEM vs Hom vs Inhom sur durée complète) → `[⤢ Teneur]`
  3. **Conservation du nombre de particules ($N$)** → `[⤢ Particules N]`
  4. **Erreurs $L^1$ et Discrepancy RMS** → `[⤢ Erreurs]`
  5. **Évolution $p_{ij}(t)$ des matrices inhomogènes** → `[⤢ Évolution P_inhom(t)]`
  6. **Indicateurs clés comparatifs** (tableau compact Homogène vs Inhomogène) → `[⤢ Benchmark]`

### B. Vues Spécifiques (100 % du viewport pour une caractéristique)
1. **`Mélangeur 3D`** :
   - Scène 3D plein écran, **deux tailles de particules (`d = 0.004 m` vs `d = 0.008 m`)** à l'échelle exacte `1:2`, `ViewCube` / boutons `Iso / XY / XZ / YZ`, coloration des cellules selon *Teneur DEM / Hom / Inhom*, *Erreur Teneur*, *Discrepancy RMSE*, *Population $N_i$*, ou *Zone*.
2. **`Teneur (Espèce)`** :
   - Graphique plein écran dédié à l'évolution temporelle de la **teneur en espèce cible** ($T_i^{\text{sp}}(t) = S_i^{\text{sp}} / (S_i^{\text{small}} + S_i^{\text{large}})$) sur les top cellules ou une cellule choisie, couplé au graphe plein format du **RSD de teneur** (avec $t_{50}, t_{90}$) et au tiroir pliable des erreurs (`RMSE`, `MAE`, `MaxErr`).
3. **`Particules (N)`** :
   - Vue dédiée plein format sur le **nombre de particules** : $N_{\text{total}}(t)$, $N_{\text{small}}(t)$ (`4 mm`), $N_{\text{large}}(t)$ (`8 mm`), dérive de masse `%`, et trajectoires d'occupation $N_i(t)$ des top cellules peuplées (DEM vs Homogène vs Inhomogène).
4. **`Erreurs (L¹ & Discrepancy)`** :
   - Vue plein format pour les métriques d'erreur de `postprocess.py` : courbe $E_{L1}(t) = \sum_i |p_i^{\text{DEM}} - p_i^{\text{MC}}|$, courbe d'écart RMS dans le temps, et **heatmap plein écran Temps × Cellules** (`|p_MC - p_DEM|`) sélectionnable Homogène / Inhomogène.
5. **`Évolution P_inhom(t)` (nouvelle figure dédiée en très grand)** :
   - Calquée directement sur `postprocessing/postprocess_inhomogeneous.py::fig_matrix_components_evolution`, mais interactive et en très grand format :
     - **Extraction complète côté Rust (`src/gui/matrix_evolution.rs`)** : retourne **toutes** les transitions significatives ($p_{i \to j} > \text{seuil}$, défaut `0.01`, pour `small` et `large`) sur les $B$ blocs $P^{(0)} \dots P^{(B-1)}$, avec :
       - valeurs mesurées par bloc $[p_{ij}(t_0), \dots, p_{ij}(t_{B-1})]$,
       - loi polynomiale ajustée (`fit_probability_evolution` jusqu'au degré configuré : coefficients, `degree`, `rmse`, `r2`, `law_label`, courbe fine échantillonnée pour le tracé lisse),
       - `global_vmax` pour tracer toutes les composantes à **échelle Y commune** (comparabilité visuelle directe entre transitions!).
     - **Deux modes d'affichage sur la figure plein écran** :
       - **Mode « Multi-courbes sur 1 grande figure »** : affiche sur un seul grand graphique les évolutions $p_{ij}(t)$ sélectionnées (avec ajustement polynomial en pointillé, infobulle au survol donnant $p_{ij}(t_b)$, loi et $R^2$). Filtres pliables en haut du graphe :
         - *Portée* : `Toutes significatives (top-K)` | `Par cellule source i` ($\sum_j p_{ij}(t)$) | `Diagonales p_ii(t)` (rétention) | `Hors-diagonale (échanges)` | `Sélection manuelle`
         - *Axe X* : `Temps DEM (t_b)` | `Indice de bloc (B_k)`
         - *Seuil* (`0.01`, `0.02`, `0.05`) et case `Afficher fits polynomiaux`.
       - **Mode « Grille à échelle commune »** : grille paginée (façon `P_components_evolution_p1.png`, jusqu'à `4×4` sous-figures par page avec **ordonnée commune** `[0, 1.15 * vmax]`), chaque sous-figure montrant `p_{i→j}(t)`, courbe mesurée + fit pointillé + équation + `RMSE`. Cliquer sur un sous-graphique l'ouvre immédiatement en **zoom plein écran** !
6. **`Matrices P & Blocs`** :
   - Inspection en très grand de $P_{\text{hom}}$, des matrices par bloc $P^{(0)} \dots P^{(B-1)}$ en grille ou individuellement, des différences inter-blocs $\Delta P_k = P_{k+1} - P_k$ (échelle symétrique divergente + norme relative $\|P_{k+1} - P_k\|_1 / \|P_k\|_1$), et du spectre $|\lambda|$.
7. **`Comparateur & Benchmark`** :
   - Comparateur 3D A/B synchronisé en grand format + benchmark multi-méthodes complet (erreurs sur teneur, $L^1$, discrepancy RMS, particules conservées, $t_{50}$).

---

## 4. Refonte modulaire du code (fin des fichiers monolithiques)

Aujourd'hui `src/gui/web_ui.rs` (76 Ko) et `src/gui/simulation.rs` (67 Ko) concentrent toute l'application. La refonte découpe proprement les responsabilités entre **Cœur Rust**, **Frontend Web modulaire** et **Frontend `egui` modulaire** :

### 4.1 Cœur Rust (`src/gui/`)
```
src/gui/
├── mod.rs                 # Exports publics propres
├── state.rs               # WorkspaceView (8 vues), CollapsiblePanelsState, MatrixEvolutionFilter,
│                          # CellColorMode, SimulationRequest
├── camera3d.rs            # Caméra orbite / pan / zoom / presets
├── mesh3d.rs              # Extraction polyédrique des 14 méthodes + plans de coupe
├── metrics_eval.rs        # [NOUVEAU] ChainPredictionMetrics, L1/RMS discrepancy, heatmaps,
│                          # MethodBenchmarkRow (extrait de simulation.rs)
├── matrix_evolution.rs    # [NOUVEAU] TransitionComponentEvolution, fit polynomial jusqu'au degré D,
│                          # filtrage par seuil > 0.01, échelle commune global_vmax, ΔP_k
├── simulation.rs          # Orchestrateur allégé (Cache Arc, synthetic/real, run_gui_comparison, benchmark)
└── web_ui.rs -> web/      # [REFACTORISÉ] Assemblage modulaire via include_str! (zéro dépendance externe)
    ├── mod.rs             # Compose INDEX_HTML depuis styles/shell/modules JS pour serveur & --export-static
    ├── styles.css         # Thème sombre doux translucide (zéro libellé "Liquid Glass"), arbres/rubans pliables
    ├── shell.html         # Structure CAO (Top Strip + Ruban pliable + Arbre pliable + Viewport + Timeline + Tabs)
    ├── store.js           # État global, sauvegarde localStorage de l'état plié/déplié, API + mode statique
    ├── viewport3d.js      # Rendu 3D canvas modulaire (particules 4mm/8mm 1:2, maillage, flux, picking)
    ├── charts.js          # Primitives de tracé haute DPI (courbes, grilles échelle commune, heatmaps, barres, tooltips)
    └── views/             # Un fichier indépendant par vue :
        ├── summary.js         # Vue Sommaire + boutons [⤢ Agrandir]
        ├── mixer3d.js         # Vue spécifique Mélangeur 3D plein écran
        ├── teneur.js          # Vue spécifique Teneur en espèce & RSD
        ├── particles.js       # Vue spécifique Nombre de particules & top cellules
        ├── errors.js          # Vue spécifique Erreur L1, Discrepancy RMS & heatmap
        ├── matrix_evol.js     # Vue spécifique Évolution P_inhom(t) (mode Superposé + mode Grille échelle commune)
        ├── matrices.js        # Vue spécifique Matrices P, blocs P^(b) et ΔP_k
        ├── benchmark.js       # Vue spécifique Comparateur 3D A/B & Benchmark multi-méthodes
        └── plan.js            # Vue Plan d'architecture
```

### 4.2 Frontend natif `egui` (`gui-egui/src/`)
Miroir strict de la même architecture modulaire :
```
gui-egui/src/
├── main.rs                # Point d'entrée minimal eframe
├── app.rs                 # Shell CAO (TopBottomPanel pliable, SidePanel FeatureTree pliable, barre de vues)
├── feature_tree.rs        # Arbre de modèle SolidWorks/Onshape via egui::CollapsingHeader
├── ribbon.rs              # Ruban pliable avec menus déroulants rétractables
├── viewport3d.rs          # Moteur 3D egui::Painter réutilisable (miniature Sommaire + Plein écran)
├── plots.rs               # Primitives egui_plot partagées + tooltips
└── views/
    ├── mod.rs
    ├── summary.rs
    ├── mixer3d.rs
    ├── teneur.rs
    ├── particles.rs
    ├── errors.rs
    ├── matrix_evolution.rs  # Figure dédiée P_inhom(t) (superposé & grille échelle commune)
    ├── matrices.rs
    └── benchmark.rs
```

---

## 5. Plan d'implémentation par phases (sûr et incrémental)

1. **Phase 1 — Extraction & enrichissement du moteur scientifique (`src/gui/`)** :
   - Extraire `src/gui/metrics_eval.rs` depuis `simulation.rs`.
   - Créer `src/gui/matrix_evolution.rs` pour calculer **toutes** les transitions $p_{ij}(t)$ significatives ($> 0.01$) sur les $B$ blocs pour `small` et `large` (plus courbes polynomiales lissées + `global_vmax`) et les exposer dans `ComparisonBundle::to_json()`.
   - Mettre à jour `src/gui/state.rs` (`WorkspaceView`, états pliables/dépliables).
   - Ajouter des tests unitaires/intégration dédiés dans `tests_rs/test_gui_mesh_and_chains.rs`.
2. **Phase 2 — Découpage modulaire du client Web (`src/gui/web/`) & ergonomie CAO** :
   - Remplacer `web_ui.rs` par la structure `src/gui/web/{mod.rs,styles.css,shell.html,store.js,viewport3d.js,charts.js,views/*.js}`.
   - Supprimer toute mention de « Liquid Glass » dans les titres et badges ; appliquer la barre d'atelier + ruban pliable + arbre latéral pliable + menus déroulants pliables + frise pliable + barre de vues en bas.
   - Implémenter la **Vue Sommaire** (avec boutons `⤢ Agrandir` sur chaque vignette) et les **Vues Spécifiques Plein Écran**, dont la figure dédiée **`Évolution P_inhom(t)`** (mode superposé filtrable + mode grille à ordonnée commune + zoom sur une transition).
   - Vérifier la syntaxe JS de chaque module (`node --check`) et le fonctionnement en mode serveur dynamique ET en mode `--export-static`.
3. **Phase 3 — Synchronisation modulaire de `gui-egui/`** :
   - Découper `gui-egui/src/` en `app.rs`, `feature_tree.rs`, `ribbon.rs`, `views/*.rs` avec `egui::CollapsingHeader` pour tous les menus et panneaux pliables.
4. **Phase 4 — Validation complète (Rust + HTTP + CI)** :
   - `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --release`.
   - Vérification HTTP de toutes les routes et redémarrage du serveur de prévisualisation.
