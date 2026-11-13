# Plan d'Architecture Logicielle Orientée Recherche — DEM-MCM Studio

> Diagnostic de l'architecture existante et conception d'une architecture logicielle orientée **expérimentation scientifique dynamique** (études paramétriques homogènes/inhomogènes, comparaison de méthodes sur le RSD, construction/sélection multi-lois des coefficients $p_{i\to j}(t)$, et gestion rigoureuse des états UI sans inflation de boutons).

---

## 1. Diagnostic de l'architecture actuelle

### 1.1 Forces déjà présentes dans le dépôt
- **Cœur numérique pure-Rust complet** (`src/partitioners/`, `src/run_sweep.rs`, `src/metrics.rs`) avec parité exacte vis-à-vis de la librairie Python (`postprocessing/metrics.py`).
- **Preuve de concept recherche déjà validée côté Python** (`postprocessing/etudes_librairie.py::EtudeMethode`, `postprocessing/driver_etudes_nlt_dt_tau.py`) :
  - Calcul unique de `states_matrix` et `S_matrices` par `(méthode, kwargs_spatiaux)`, puis réutilisation instantanée pour balayer `start_index`, `nlt`, `step`, `dt`, `tau`, `no_species` ou des études de convergence de Frobenius.
- **Interface CAO modulaire existante** (`src/gui/web/`, `gui-egui/src/`) avec vue `Sommaire`, vues spécifiques plein format, arbre/ruban/frise pliables et module `src/gui/matrix_evolution.rs`.

### 1.2 Limites structurelles identifiées
1. **Couplage rigide « 1 requête = 1 exécution complète » dans `src/gui/simulation.rs`** :
   - `run_gui_comparison` ré-exécute à chaque appel l'échantillonnage, `partitioner.fit`, `compute_state_matrices` sur toute la trajectoire DEM et la génération des frames 3D, même lorsqu'on ne fait varier que `start_index`, `nlt`, `step`, `dt` ou `tau`.
   - Aucun cache de session spatiale (`states_matrix`, `S_matrices`) n'existe en Rust.
2. **Absence de moteur d'études paramétriques dynamiques (`StudyEngine`)** :
   - `run_method_benchmark` ne fait que comparer une liste fixe de méthodes sur un point temporel figé ; il ne permet pas de balayer dynamiquement `start_index`, `nlt`, `step`, `dt`, `tau`, la résolution spatiale (`n_cells`, `velocity_weight`), ni des grilles croisées 2D (ex. `start_index × nlt`).
   - Les métriques de convergence matricielle type Doucet et al. (erreur relative de Frobenius $\|P - P_{\text{réf}}\|_F / \|P_{\text{réf}}\|_F$, comptage de lignes `NaN` vs `dt`, écart moyen $|\mathrm{RSD}_{\text{MK}} - \mathrm{RSD}_{\text{DEM}}|$) ne sont pas encore exposées dans l'application interactive.
3. **Ajustement des coefficients $p_{i\to j}(t)$ limité à une seule famille polynomiale** :
   - `src/metrics.rs::fit_probability_evolution` ne teste qu'un polynôme de degré $\le d_{\max}$ sur un seul réglage `(start_index, nlt, step)` choisi à la main.
   - Impossible aujourd'hui de construire et comparer **plusieurs familles de lois** (Constante, Linéaire, Quadratique, Cubique, Exponentielle relaxante, Loi puissance/logarithmique), ni d'étudier automatiquement comment `start_index` et le nombre de blocs `nlt` modifient les coefficients et la loi optimale.
4. **Multiplication des contrôles dans l'interface (`src/gui/web/shell.html`)** :
   - La coexistence permanente de tous les contrôles 3D, paramètres de maillage, filtres de vues et futurs paramètres d'études paramétriques expose rapidement des dizaines de boutons/listes simultanément à l'écran.
   - Il manque une **machine à états UI centralisée** qui pilote l'affichage contextuel et l'exclusivité des commandes.

---

## 2. Architecture Logicielle Orientée Recherche Cible

Le principe fondamental est de séparer la pile en **4 couches indépendantes** et d'introduire un **noyau de recherche réutilisable (`src/research/`)** entre le moteur DEM/MCM existant et les frontends (`web`, `egui`, `CLI`).

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Couche 4 · Présentation & Automate à États UI                          │
│  src/gui/web/ (ui_machine.js + 3 modes exclusifs)  |  gui-egui/src/     │
│  -> N'affiche QUE les boutons/champs valides pour le mode et l'axe      │
├─────────────────────────────────────────────────────────────────────────┤
│  Couche 3 · Orchestrateur Applicatif & API HTTP / CLI                   │
│  src/gui/simulation.rs  |  src/bin/dem_mcm_gui.rs  |  dem_mcm_sweep.rs  │
│  -> POST /api/study/run, GET /api/study/presets, export statique        │
├─────────────────────────────────────────────────────────────────────────┤
│  Couche 2 · Noyau Orienté Recherche (`src/research/`)  [NOUVEAU]        │
│  • session.rs : PreparedMethodSession + SessionCache (fit & S 1 fois)   │
│  • study.rs   : StudySpec, StudyAxis (1D/2D), StudyRunner dynamique     │
│  • laws.rs    : Multi-familles p_{ij}(t) + AICc + R² + validation bloc  │
│  • presets.rs : Presets d'études reproductibles (homogène/inhomogène)   │
├─────────────────────────────────────────────────────────────────────────┤
│  Couche 1 · Moteur Scientifique Existant (inchangé sur le fond)         │
│  partitioners/  |  run_sweep.rs  |  metrics.rs  |  linalg.rs  |  data/  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Détail des modules de recherche (`src/research/`)

### 3.1 `session.rs` — Séparation Spatial / Temporel (`PreparedMethodSession`)
Miroir Rust optimisé de `EtudeMethode` (`postprocessing/etudes_librairie.py`) :
- **`SessionKey`** : `(dataset_id, method, canonical_kwargs_json, use_bed_frame: bool)`.
  - Note : `use_bed_frame` reprend le changement de repère principal du lit (`make_frame` / alignement surface libre) déjà validé en Python pour les méthodes géométriques, sélectionnable en option.
- **`PreparedMethodSession`** (stocké dans un cache `Arc` LRU/borné dans `ResearchSessionCache`) :
  - `mesh: Mesh3DData`, `diagnostics: PartitionerDiagnostics`
  - `sorted_timesteps: Vec<usize>`, `idx_to_row: HashMap<usize, usize>`
  - `states_matrix: Vec<Vec<i64>>` (calculé **une seule fois** par méthode spatiale !)
  - `species_masks: Vec<(String, Vec<bool>)>`
  - `s_matrices: HashMap<String, Matrix>` (`small`, `large`, `all`)
- **Gain architectural & temps de calcul** :
  - Changer `start_index`, `nlt`, `step`, `dt`, `tau`, `no_species`, `pred_start` ou lancer un balayage de 20 configurations temporelles réutilise la même `PreparedMethodSession` sans re-fitter ni relabelliser les particules.
  - Les snapshots 3D lourds (`frames`) ne sont générés **que** pour les vues d'exploration 3D, jamais pendant un balayage d'étude numérique.

### 3.2 `laws.rs` — Approximation Multi-Lois des Coefficients $p_{i\to j}(t)$
Extension modulaire de `fit_probability_evolution` (`src/metrics.rs`) :
- **`LawFamily`** (énumération extensible de familles de lois) :
  1. `Constant` : $p(t) = c_0$ (référence stationnaire / homogène)
  2. `Linear` : $p(t) = a_1 \tilde{t} + a_0$
  3. `Quadratic` : $p(t) = a_2 \tilde{t}^2 + a_1 \tilde{t} + a_0$
  4. `Cubic` : $p(t) = a_3 \tilde{t}^3 + a_2 \tilde{t}^2 + a_1 \tilde{t} + a_0$ (si $n_{\text{blocs}} \ge 4$)
  5. `ExponentialRelax` : $p(t) = p_\infty + (p_0 - p_\infty) e^{-\alpha \tilde{t}}$ ($\alpha > 0$ optimisé par balayage logarithmique 1D + régression linéaire exacte sur $(p_\infty, \Delta p)$ — zéro dépendance externe!)
  6. `PowerOrLog` : $p(t) = a + b \ln(1 + \tilde{t})$ (tendance sous-linéaire / logarithmique)
  *(où $\tilde{t} = (t - t_0) / \max(1, \Delta t)$ pour un conditionnement numérique parfait même lorsque $t \in [0..6000]$)*
- **`LawCandidateFit`** retourné pour chaque famille testée :
  - `family`, `n_params`, `coefficients`, `formula_label`
  - `rmse`, `mae`, `r2`, **`aicc`** (critère d'Akaike corrigé pour petits nombres de blocs $B$, évitant qu'un polynôme de degré élevé gagne artificiellement sur $B=3$ ou $4$ blocs!)
  - `out_of_block_rmse: Option<f64>` (validation croisée temporelle : ajustement sur les $B-1$ ou $B-2$ premiers blocs, prédiction sur les blocs suivants non vus !)
  - `dense_times`, `dense_predicted`
- **Sortie multi-lois** : pour chaque transition $(i \to j)$ significative, on conserve **toutes** les lois candidates + la sélection de la meilleure loi selon le critère choisi (`AICc`, `RMSE`, `Out-of-block RMSE`).

### 3.3 `study.rs` & `presets.rs` — Moteur d'Études Paramétriques Dynamiques (`StudyRunner`)
Un moteur déclaratif unique exécute toute étude scientifique définie par une requête `StudySpec` :

```rust
pub struct StudySpec {
    pub dataset: String,
    pub base_request: SimulationRequest,     // Configuration de base figée hors axes balayés
    pub chain_mode: StudyChainMode,          // Homogeneous | Inhomogeneous | Both
    pub primary_axis: StudyAxisSpec,         // Axe balayé (nom + valeurs dynamiques)
    pub secondary_axis: Option<StudyAxisSpec>, // Optionnel : grille 2D (ex: start_index × nlt)
    pub law_families: Vec<LawFamily>,        // Pour études d'approximation des lois
    pub focus_transition: Option<(usize, usize)>,
    pub reference_policy: ReferencePolicy,   // Ex: LastPoint (NLT=18) pour Frobenius Doucet
}
```

#### Axes dynamiques supportés (`StudyAxisKind`)
| Catégorie | Axe (`StudyAxisKind`) | Étude couverte | Métriques clés produites |
| :--- | :--- | :--- | :--- |
| **Apprentissage Homogène** | `StartIndex` | Départ apprentissage (transitoire vs régime permanent) | `RSD(t)` superposés, `err_rsd_mean`, `rmse_teneur`, `t50`, `nan_rows` |
| **Apprentissage Homogène** | `Nlt` | Nombre d'instants / blocs d'apprentissage | `err_rsd_mean`, `frobenius_rel_err` vs $P_{\text{réf}}$, `nan_rows` |
| **Apprentissage Homogène** | `Step` | Écart entre instants/blocs d'apprentissage | `err_rsd_mean`, `rmse_teneur`, `l1_mean`, `n_pairs_used` |
| **Apprentissage Homogène** | `Dt` | Pas d'échantillonnage intra-bloc (`dt`) | `nan_rows` (conformité stochastique), `diag_mean`, `err_rsd_mean` |
| **Modèle Temporel** | `Tau` | Horizon de transition $\tau$ (+ couplage optionnel `step=tau, dt=max(1,tau/10)`) | `RSD(t)` superposés pour chaque $\tau$, `t50_markov`, `err_rsd`, $|\lambda|_{\max}$ |
| **Modèle Spatial** | `Method` | Impact de la méthode de découpage sur le RSD | `RSD(t)` DEM vs Markov par méthode, `rmse_rsd`, `rmse_teneur`, `fraction_visited` |
| **Modèle Spatial** | `SpatialResolution` | Nombre de cellules (`n_cells`, `nx`, `nr`) d'une méthode | Trade-off résolution spatiale vs erreur RSD / `nan_rows` / `L1` |
| **Modèle Physique** | `VelocityWeight` | Poids cinématique $w_V$ (`physics`, `spectral`, `gmm`) | Impact cinématique sur `rmse_rsd`, `rmse_teneur`, `diag_mean` |
| **Modèle Espèces** | `SpeciesMode` | Matrice unique (`no_species`) vs matrices par espèce | `RSD(t)` comparés, écart moyen ségrégation |
| **Approximation Lois** | `StartIndex × Nlt` *(2D)* | Influence de `(start, NLT)` sur les lois $p_{i\to j}(t)$ | Carte 2D meilleure loi, évolution coefficients, courbes multi-lois, `AICc` & erreur out-of-block |

#### Sortie unifiée `StudyResultBundle`
1. **`points: Vec<StudyPointMetrics>`** : toutes les métriques scalaires pour chaque point `(p1[, p2])` (erreurs RSD, Teneur, $L^1$, RMS, $t_{50}$, `nan_rows`, `diag_mean`, `lambda_max`, `frobenius_rel_err`).
2. **`curves: Vec<StudyCurveSeries>`** : courbes temporelles superposées (`times`, `dem_rsd`, `markov_rsd` pour chaque point de l'axe, directement traçables sur une figure unique).
3. **`grid_2d: Option<StudyGrid2D>`** : matrice `(axis1 × axis2)` pour heatmaps de sensibilité (ex. `start_index × NLT`).
4. **`law_study: Option<LawComparisonBundle>`** : résultats comparatifs multi-familles (`Constant`, `Linear`, `Quadratic`, `Cubic`, `ExponentialRelax`, `PowerOrLog`) + évolution des coefficients et loi gagnante sur toute la grille `(start, NLT)`.

---

## 4. Gestion Rigoureuse des États de Boutons (« Zéro Fouillis »)

Pour garantir qu'on **n'ait jamais 1 000 boutons affichés en même temps**, l'interface (Web & `egui`) est gouvernée par un **Automate à États UI centralisé** (`src/gui/web/ui_machine.js` + `src/gui/state.rs`).

### 4.1 Trois modes exclusifs au sommet (`WorkbenchMode`)
Un **commutateur à 3 états exclusifs** dans la barre haute remplace l'accumulation d'outils :
1. **`Explorer`** *(Inspection d'une simulation)* : n'affiche que les vues d'exploration (`Sommaire`, `Mélangeur 3D`, `Teneur`, `Particules`, `Erreurs`, `Évolution P_inhom(t)`, `Matrices`). Les réglages 3D sont **masqués** dès qu'on quitte `Mélangeur 3D` ou `Sommaire`.
2. **`Études de Recherche`** *(Atelier paramétrique dynamique)* : masque l'arbre 3D, la frise d'animation et les barres d'outils 3D, pour ne laisser que **l'Atelier d'Études** (1 constructeur dynamique + 2 grandes zones de figures + tableau d'export).
3. **`Comparer A/B`** *(Dual 3D & Benchmark)* : n'affiche que la comparaison côte à côte et le benchmark.

### 4.2 Règles strictes de visibilité et d'état (`deriveUiVisibility(state)`)
Toute visibilité et tout état `disabled` / `active` passent par **une fonction pure unique** :

| Règle | Mécanisme logiciel | Effet concret pour l'utilisateur |
| :--- | :--- | :--- |
| **R1 · Formulaire piloté par schéma** | Un seul bloc « Axe d'étude » génère dynamiquement ses champs selon `primaryAxis`. | On n'affiche jamais simultanément les listes de valeurs de `tau`, `start`, `NLT`, `step`, `dt` et `n_cells` : **seuls les 2–3 champs de l'axe actif sont rendus**. |
| **R2 · Couplage contextuel des options** | Si `primaryAxis === 'tau'`, un seul interrupteur contextuel apparaît : *« Coupler step = τ et dt = τ/10 »*. Si `primaryAxis === 'method'`, seul le menu multi-méthodes apparaît. | Zéro option orpheline ou inutile à l'écran. |
| **R3 · Filtrage des paramètres spatiaux** | `methodCapabilities(method)` déclare les paramètres valides de la méthode choisie. | Sélectionner `cartesian` n'affiche que `nx, ny, nz` ; `physics` n'affiche que `n_cells, w_V`. |
| **R4 · Exclusivité Chaîne / Lois** | Si `studyKind !== 'law_fit'`, le sélecteur de familles de lois et la grille 2D sont démontés/masqués. | En étude homogène sur `tau` ou `start`, l'interface reste ultra-épurée. |
| **R5 · Bouton d'action unique & verrouillage** | Un seul bouton primaire contextuel (`Exécuter` vs `Lancer l'étude`) géré par `executionStatus ∈ {idle, computing, error}`. | Pendant le calcul : bouton désactivé + libellé `Calcul…` + blocage des doubles clics ; aucune prolifération de boutons « Run ». |
| **R6 · Détails à la demande** | Les presets rapides sont dans **un seul menu déroulant**, et les réglages avancés (ex. repère du lit, seuils, familles de lois) dans des popovers repliables (`<details>`). | Au maximum **5 contrôles de premier niveau** visibles à tout instant hors figures. |

---

## 5. Atelier UI « Études de Recherche » (Vue Dynamique Unique)

Au lieu d'écrire une page séparée par question scientifique, **une seule vue modulaire `views/study.js` (`gui-egui/src/views/study.rs`)** s'adapte automatiquement à l'étude choisie :

1. **Bandeau de sélection rapide (≤ 4 contrôles)** :
   - **Preset scientifique** (liste unique) :
     1. *Homogène : Impact du départ (`start_index`)*
     2. *Homogène : Nombre d'instants (`NLT`) & convergence Frobenius*
     3. *Homogène : Écart (`step`) & raffinage (`dt` / NaN)*
     4. *Modèle : Impact de $\tau$ (cinétique RSD + temps de mélange)*
     5. *Découpage : Comparaison dynamique des méthodes sur le RSD*
     6. *Découpage : Sensibilité à la résolution spatiale (`n_cells` / `w_V`)*
     7. *Inhomogène : Lois $p_{i\to j}(t)$ & influence de `(start × NLT)`*
     8. *Personnalisé (1D ou grille 2D libre)*
   - **Bouton unique** : `Lancer l'étude` (et bouton discret `Exporter CSV / JSON`).
2. **Constructeur dynamique dans le panneau latéral contextuel** *(repliable)* :
   - *Paramètres fixes (base)* : résumés en un bloc pliable.
   - *Axe primaire (X)* : choix de l'axe + valeurs (mode *Liste* ex. `10, 25, 50, 100, 157, 300` ou mode *Plage* `min..max / pas`) + mode `Homogène` / `Inhomogène` / `Comparé`.
   - *Axe secondaire (Y, optionnel)* : pour activer une **étude croisée 2D** (ex. `start_index × NLT`).
   - *Options spécifiques au contexte* : familles de lois actives & transition $(i \to j)$ si étude de lois.
3. **Zone de restitution plein format (adaptative selon `studyKind`)** :
   - **En étude 1D** :
     - *Grand graphe gauche* : **Superposition temporelle** (`RSD(t)` ou `Teneur(t)` ou `L¹(t)` — DEM en trait continu + chaque valeur du paramètre balayé en courbe colorée!).
     - *Grand graphe droit* : **Courbe de sensibilité paramétrique** (métrique au choix : `Erreur RSD`, `RMSE Teneur`, `Erreur Frobenius`, `Lignes NaN`, `t50`, `|λ|_max` en fonction du paramètre X).
   - **En étude 2D (`Axis1 × Axis2`)** :
     - *Grand graphe gauche* : **Heatmap 2D interactive** (ex. `start_index × NLT` colorée par `RMSE`, `AICc` ou `Famille de loi gagnante`) ; **cliquer une case de la heatmap** sélectionne ce couple `(start, NLT)` et met à jour le graphe de droite !
     - *Grand graphe droit* : Courbes détaillées pour la case sélectionnée.
   - **En étude Approximation de lois $p_{i\to j}(t)$** :
     - *Grand graphe gauche* : Points mesurés sur les blocs + **superposition des lois construites** (`Constante`, `Linéaire`, `Quadratique`, `Cubique`, `Exponentielle`, `Puissance`) pour la transition sélectionnée (ou top transitions).
     - *Grand graphe droit* : Évolution des coefficients / $R^2$ / `AICc` / erreur *out-of-block* selon `start_index` et `NLT` (ou heatmap 2D `(start × NLT)` de la loi optimale!).
     - *Tableau bas pliable* : Comparatif chiffré de toutes les lois testées (équations, coefficients, $R^2$, RMSE, AICc, out-of-block RMSE).

---

## 6. Plan d'implémentation incrémental (Phases & Critères de validation)

1. **Phase 1 — Noyau `src/research/` (Session cachée + Lois multi-familles + Moteur 1D/2D)** :
   - Créer `src/research/{mod,session,laws,study,presets}.rs` avec zéro dépendance externe.
   - Refactoriser `src/gui/simulation.rs` pour qu'il s'appuie sur `ResearchSessionCache` (suppression du re-fit redondant lors des changements temporels).
   - Tests unitaires & d'intégration vérifiant : (a) invariance exacte des résultats entre `Etude` cachée et `run_experiment`, (b) convergence monotone de Frobenius quand `NLT` augmente, (c) réduction des `NaN` quand `dt` diminue, (d) ajustement et classement `AICc` des 6 familles de lois.
2. **Phase 2 — API HTTP & Export Statique (`src/bin/dem_mcm_gui.rs`)** :
   - Ajouter `GET /api/study/presets`, `POST /api/study/run`, et pré-exporter les études presets clés dans `--export-static`.
3. **Phase 3 — Automate d'états UI & Atelier d'Étude Web (`src/gui/web/`)** :
   - Créer `src/gui/web/ui_machine.js` (automate strict `deriveUiVisibility`) et `src/gui/web/views/study.js`.
   - Mettre à jour `shell.html`, `store.js`, `charts.js` (support heatmap 2D catégorielle/continue & courbes multi-lois) et `app.js` avec les 3 modes exclusifs (`Explorer`, `Études`, `Comparer A/B`).
   - Vérifier la syntaxe (`node --check`) et valider qu'aucun écran ne dépasse la budgetisation stricte des contrôles visibles.
4. **Phase 4 — Synchronisation `gui-egui/` & Validation CI complète** :
   - Ajouter `gui-egui/src/views/study.rs` en miroir exact de la machine à états et du `StudyRunner`.
   - Valider `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --release`, `--export-static` et la CI GitHub Actions (`egui check` + `pytest`).
