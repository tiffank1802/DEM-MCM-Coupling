# Architecture & Planification GUI `egui` + 3D — DEM-MCM Studio

> Document d'architecture logicielle : séparation stricte entre le noyau scientifique pure-Rust (`src/gui/`), la couche d'affichage épurée **Ribbon + Liquid Glass sombre léger**, et le client natif/WASM **`egui` (`gui-egui/`)** / serveur HTTP (`dem-mcm-gui`).

---

## 1. Principes directeurs de la refonte

| Exigence | Traduction architecturale |
| :--- | :--- |
| **Ergonomie à rubans (Ribbon UI)** | Suppression des cartes titrées redondantes et de la barre latérale fixe. Un ruban unique regroupe les fonctionnalités communes en groupes horizontaux compacts : *Découpage (14 méthodes)*, *Construction modèles & prédiction DEM complète*, *Mélangeur 3D & deux tailles*. |
| **Thème Liquid Glass sombre léger** | Fond ardoise nocturne doux (`#15181e`), panneaux verre dépoli translucides (`rgba(255,255,255,0.042–0.072)`, flou d'arrière-plan, filet lumineux `1px`), palette apaisée sans couleurs criardes (argent givré, menthe douce, perle glaciaire, ambre doux). |
| **Critère physique : Teneur en espèce** | Toute l'analyse compare la **teneur en l'espèce cible** $T_i^{\text{sp}}(t) = S_i^{\text{sp}}(t) / (S_i^{\text{small}}(t) + S_i^{\text{large}}(t)) \in [0, 1]$ (`small` 4 mm ou `large` 8 mm), au lieu d'une concentration abstraite. |
| **Construction locale vs Prédiction DEM totale** | La phase de **construction** des modèles $P$ et $\{P_0,\dots,P_{\text{NLT}-1}\}$ reste identique à `run_sweep` (fenêtre $[t_{\text{start}}, t_{\text{start}} + \text{NLT}(\text{step}+\tau))$), puis les chaînes homogène et inhomogène sont propagées sur **toute la durée de la DEM** (`[ts_min .. ts_max]`, ex. `0..359` ou `0..5999`). |
| **Mélangeur 3D bifractionnel** | Rendu fidèle des **deux tailles de particules** au rapport réel $1:2$ (`d = 0.004 m` turquoise givré vs `d = 0.008 m` ambre doux), avec bascules individuelles `4 mm` / `8 mm`, vitesses, maillage 3D, faces semi-transparentes et flux $P_{ij}$. |
| **Métriques `postprocess.py` & Benchmark** | Intégration complète des indicateurs de référence : **nombre de particules** ($N_{\text{small}}, N_{\text{large}}, N_{\text{total}}$, dérive %), **erreur $L^1$ normalisée**, **discrepancy RMS** (temporelle et heatmap Temps × Cellules), **RMSE / MAE / MaxErr de teneur**, **RSD & temps $t_{50}, t_{90}$**, **lois polynomiales $p_{ij}(t)$** et **benchmark multi-méthodes (`POST /api/benchmark`)**. |

---

## 2. Arborescence & séparation des responsabilités

```
src/
├── data/local.rs               # Lecture parallèle std::thread::scope + décompression rapide (RFC 1951 / GZIP)
├── gui/
│   ├── camera3d.rs             # Caméra orbite/pan/zoom + presets Iso / XY / XZ / YZ
│   ├── mesh3d.rs               # Extraction polyédrique réelle des 14 méthodes + plans de coupe
│   ├── state.rs                # Ribbon WorkspaceTab, CellColorMode (Teneur / Erreur / RMS / Pop), SimulationRequest
│   ├── simulation.rs           # Cache Arc<BTreeMap<usize, DemFrame>>, construction modèle vs prédiction DEM complète,
│   │                           # ChainPredictionMetrics, TransitionLawItem, MethodBenchmarkRow
│   └── web_ui.rs               # Client HTML5 Canvas + Liquid Glass Ribbon UI sans dépendance externe
└── bin/dem_mcm_gui.rs          # Serveur HTTP pure-Rust (/api/health, /api/methods, /api/simulate, /api/benchmark)

gui-egui/                       # Frontend natif & WASM (eframe + egui + egui_plot)
└── src/{main,viewport3d,plots}.rs
```

---

## 3. Pipeline scientifique exact (`simulation.rs`)

1. **Construction des modèles (inchangée)** :
   - Ajustement du partitionneur sur la phase permanente (`ts >= permanent_start`), avec sous-échantillonnage temporel borné pour maintenir la réactivité sur 6 000 pas de temps.
   - Estimation de la matrice homogène $P$ (`run_experiment`) et des $\text{NLT}$ blocs inhomogènes $\{P_0,\dots,P_{\text{NLT}-1}\}$ (`run_inhomogeneous_experiment`) sur `[start_index, start_index + NLT*(step + tau))`.
2. **Prédiction sur toute la durée de la DEM (`predict_full_duration = true`)** :
   - Initialisation du vecteur d'état $S(t_0)$ à $t_0 = t_{\min}^{\text{DEM}}$ et propagation par pas $\tau$ sur l'intégralité de `[ts_min .. ts_max]`.
   - Le dernier bloc $P_{\text{NLT}-1}$ prend le relais après la fin de la fenêtre de construction pour la chaîne inhomogène, conformément à `postprocessing/metrics.py`.
3. **Critères d'évaluation & Benchmark (`postprocess.py` + `postprocess_inhomogeneous.py`)** :
   - **Teneur en espèce cible** : `RMSE(Teneur)`, `MAE(Teneur)`, `MaxErr(Teneur)` sur toutes les cellules actives, plus trajectoires par cellule la plus peuplée (`fig_teneur`, `fig_compare_teneur`).
   - **Nombre de particules** : $N_{\text{small}}(t), N_{\text{large}}(t), N_{\text{total}}(t)$ DEM vs Homogène vs Inhomogène (`fig_compare_n_particles`, `fig_states_totale`).
   - **Erreur $L^1$ normalisée** : $E_{L1}(t) = \sum_{i \in \mathcal{A}} |p_i^{\text{DEM}}(t) - p_i^{\text{MC}}(t)| \in [0, 2]$ (`fig_compare_hom_vs_inhom`).
   - **Discrepancy par cellule & dans le temps** : $\text{RMS}(t) = \sqrt{\frac{1}{|\mathcal{A}|}\sum_i (p_i^{\text{MC}} - p_i^{\text{DEM}})^2}$ + matrice heatmap Temps × Cellules (`fig_discrepancy_analysis`).
   - **Diagnostics inhomogènes** : différences relatives inter-blocs $\|P_{k+1} - P_k\|_1 / \|P_k\|_1$ (`fig_matrix_differences`) et lois polynomiales $p_{ij}(t)$ (`fit_probability_evolution`, `fig_matrix_components_evolution`).
   - **Comparateur & Benchmark** : évaluation parallèle/séquentielle de plusieurs méthodes sur une même DEM (`POST /api/benchmark`) restituant erreurs sur la teneur, $L^1$, discrepancy RMS, effectifs de particules et $t_{50}$.
