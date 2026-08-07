# Projet MCM — Calcul de Chaînes de Markov pour Mélange

## Vue d'ensemble

Ce projet calcule des **matrices de transition Markoviennes** pour modéliser 
le mélange de particules dans un mélangeur DEM. L'objectif est d'évaluer 
différentes méthodes de partitionnement spatial (cartésien, cylindrique, 
Voronoï, etc.) et de déterminer laquelle capture le mieux la dynamique 
de mélange.

### Workflow général

```
Données DEM (position des particules)
        ↓
[1. Partitionnement spatial]  ← Choisir une méthode
        ↓
[2. Assignation d'états]      ← Chaque particule → un état (cellule)
        ↓
[3. Suivi des transitions]    ← Compter: état_t → état_{t+1}
        ↓
[4. Normalisation]            ← P(i→j) = n(i→j) / n(i)
        ↓
Matrice P (n_states × n_states)
        ↓
[5. Analyse du mélange]       ← RSD, Entropie, temps de mélange
```

---

## 📊 Architecture du Projet

### Fichiers principaux

| Fichier | Rôle |
|---------|------|
| `partitioners.py` | 6 méthodes de partitionnement spatial |
| `bucket_io.py` | Lecture/écriture vers HuggingFace Hub |
| `run_sweep.py` | Lance les calculs (sweep multi-paramètres) |
| `analyze_results.py` | Chargement, comparaison, visualisation |

### Structure des données

```
HuggingFace Bucket: ktongue/DEM_MCM
├── markov_results/                    ← Résultats finaux
│   ├── cartesian_nx5_ny5_nz5_NLT100_step1_start0/
│   │   ├── transition_matrix.npy      ← Matrice P (5³×5³)
│   │   ├── stats.json                 ← Statistiques
│   │   ├── config.json                ← Paramètres d'exécution
│   │   └── centroids.npy              ← Données du partitionneur
│   └── voronoi_125cells_NLT200.../
│
└── Output Paraview/                   ← Données DEM brutes
    ├── file_0.csv                     ← Snapshot t=0
    ├── file_1.csv                     ← Snapshot t=1
    └── ... (500+ fichiers)
```

---

## 🔍 Les 6 Méthodes de Partitionnement

### 1️⃣ Cartesian (Grille régulière)

**Concept:** Découpe le domaine en nx × ny × nz cellules de taille égale.

**Pros:**
- Simple, rapide
- Facile à interpréter

**Cons:**
- Inadapté aux géométries cylindriques (coins vides)
- Population cellulaire très inégale

**Exemple:**
```python
from partitioners import create_partitioner

part = create_partitioner("cartesian", nx=5, ny=5, nz=5)
# → 125 cellules en grille régulière
```

---

### 2️⃣ Cylindrical (Grille radiale)

**Concept:** Partitionnement en coordonnées cylindriques (r, θ, z).

**Deux modes radiaux:**
- `"equal_dr"`: Δr constant (débutant)
- `"equal_area"`: aire de section constante (recommandé)

**Pros:**
- Adapté aux mélangeurs cylindriques
- Conservation de la symétrie axiale

**Cons:**
- Configuration plus complexe

**Formule equal_area:**
```
r_i = R_max × √(i / n_r)
```
Chaque anneau a la même aire: π(r_{i+1}² - r_i²) = constant

**Exemple:**
```python
part = create_partitioner(
    "cylindrical",
    nr=5, ntheta=8, nz=5,
    radial_mode="equal_area"
)
# → 5 × 8 × 5 = 200 cellules
```

---

### 3️⃣ Voronoï (K-means)

**Concept:** K-means + clustering. Chaque cellule = bassin d'attraction 
du centroïde le plus proche.

**Pros:**
- **Référence en MCM** (Fan et al., Doucet et al.)
- S'adapte naturellement à la densité
- Population plus homogène

**Cons:**
- Coûteux en calcul (KDTree queries)
- Non-déterministe (même avec seed)

**Exemple:**
```python
part = create_partitioner("voronoi", n_cells=125)
# → 125 cellules par K-means
```

---

### 4️⃣ Quantile (Grille équi-population)

**Concept:** Grille dont les bords sont les quantiles des données.

**Pros:**
- Chaque cellule contient approximativement le même nombre de particules
- Meilleure homogénéité que cartésien régulier

**Cons:**
- Cellules de formes irrégulières

**Exemple:**
```python
part = create_partitioner("quantile", nx=5, ny=5, nz=5)
# Bords = quantiles [0%, 20%, 40%, 60%, 80%, 100%]
```

---

### 5️⃣ Octree (Adaptatif)

**Concept:** Subdivision récursive. Chaque cellule ayant > max_particles 
particules est divisée en 8 sous-cellules.

**Pros:**
- Raffinage local de la résolution
- Nombre de cellules adapté au domaine

**Cons:**
- Nombre de cellules imprévisible a priori
- Peut créer des imbalances

**Exemple:**
```python
part = create_partitioner(
    "octree",
    max_particles=100,   # subdiviser si > 100 particules
    max_depth=5          # max 5 niveaux de profondeur
)
```

---

### 6️⃣ Physics-aware (Position + vitesse)

**Concept:** K-means sur [position, vitesse]. Deux states proches en 
position mais avec vitesses différentes → cellules séparées.

**Pros:**
- Capture l'information dynamique
- Plus physique

**Cons:**
- Plus coûteux
- Nécessite les données de vitesse

**Exemple:**
```python
part = create_partitioner("physics", n_cells=125, velocity_weight=0.3)
part.fit_with_physics(positions, velocities)
states = part.compute_states_with_physics(x, y, z, vx, vy, vz)
```

---

## 🚀 Workflow Complet

### Étape 1 : Lancer un calcul

```bash
# Calcul Voronoï
python run_sweep.py --method voronoi

# Calcul tous les partitionnements
python run_sweep.py --method all

# Lister les configs sans lancer
python run_sweep.py --method cylindrical --list
```

### Étape 2 : Charger et analyser

```python
from analyze_results import MarkovAnalyzer

analyzer = MarkovAnalyzer()

# Charger tous les résultats
analyzer.load_all()

# Afficher un résumé
analyzer.print_summary()

# Visualiser une expérience
analyzer.plot_experiment("voronoi_125cells_NLT200", n_steps=200)

# Comparer les méthodes
analyzer.compare_methods(metric="diag_mean")

# Analyser le mélange (RSD)
rsd_data = analyzer.compute_rsd("voronoi_125cells_NLT200")
print(f"RSD final: {rsd_data['rsd_final']*100:.1f}%")
```

---

## 📈 Métriques de Mélange

### 1. RSD (Relative Standard Deviation)

**Formule:**
```
RSD(t) = σ(C_i(t)) / μ(C_i(t))
```

où C_i(t) = concentration de l'espèce A dans la cellule i au temps t.

**Interprétation:**
- RSD = 0%  → mélange parfait (distribution uniforme)
- RSD = 100% → ségrégation totale

**Temps de mélange:**
- **t₅₀**: temps où RSD < 50% du RSD initial
- **t₉₀**: temps où RSD < 10% du RSD initial

### 2. Entropie Normalisée

**Formule:**
```
H(t) = -Σ C_i(t) ln(C_i(t)) / H_max
```

où H_max = ln(2) pour un système binaire.

**Interprétation:**
- H = 0   → ségrégation totale
- H = 1   → mélange parfait (distribution uniforme)

### 3. Intensité de Ségrégation

**Formule:**
```
I(t) = σ²(C) / (C̄(1-C̄))
```

où C̄ = concentration moyenne (≈ 0.5 pour 50/50).

**Interprétation:**
- I = 0   → mélange parfait
- I → ∞   → très ségrégé

### 4. Diagonale de P

```
P_diag = moyenne des éléments diagonaux de P
```

**Interprétation:**
- P_diag proche de 1 → les particules restent longtemps dans leurs cellules (mauvais mélange)
- P_diag proche de 0 → les particules changent rapidement de cellule (bon mélange)

---

## 💾 Format des Données

### transition_matrix.npy

```
Matrice P (n_states × n_states)
P[i,j] = nombre de transitions i→j normalisé par le nombre total 
         de transitions depuis i

Propriétés:
- Éléments ∈ [0, 1]
- Somme de chaque ligne ≈ 1 (≠ 1 si états non visités)
- Diagonale = P(rester dans le même état)
```

### stats.json

```json
{
  "n_timesteps_used": 100,
  "n_states": 125,
  "n_states_visited": 124,
  "n_states_empty": 1,
  "fraction_visited": 0.992,
  "row_sum_min": 0.8,
  "row_sum_max": 1.0,
  "row_sum_mean": 0.99,
  "diagonal_mean": 0.35,
  "diagonal_std": 0.12,
  "method": "voronoi"
}
```

### config.json

```json
{
  "method": "voronoi",
  "method_kwargs": {"n_cells": 125},
  "nlt": 100,
  "step_size": 1,
  "start_index": 0
}
```

---

## 🔗 Comparaison DEM vs Markov

La classe `MarkovAnalyzer` permet de comparer la prédiction Markov 
avec les données DEM réelles :

```python
# Comparaison pour une méthode donnée
results = analyzer.compare_dem_vs_markov(
    method="cartesian",
    method_kwargs={"nx": 5, "ny": 5, "nz": 5},
    species_criterion="z_median",
    file_indices=list(range(0, 500, 5))
)

print(f"RSD DEM:   {results['dem']['rsd_final']*100:.1f}%")
print(f"RSD Markov:{results['markov']['rsd_final']*100:.1f}%")
print(f"Corrélation: {results['correlation']:.3f}")
```

**Processus:**

1. Charge les snapshots DEM
2. Crée le partitionneur
3. Assigne chaque particule à une cellule
4. Calcule le RSD depuis les données réelles
5. Prédit le RSD avec la matrice P
6. Compare

---

## 🎓 Conseils d'Utilisation

### Pour déboguer

```python
# Diagnostics du partitionneur
part = create_partitioner("cartesian", nx=5, ny=5, nz=5)
part.fit(coordinates)
diag = part.diagnostics(coordinates)

print(f"Population par cellule:")
print(f"  Min: {diag['pop_min']}")
print(f"  Max: {diag['pop_max']}")
print(f"  Mean: {diag['pop_mean']:.0f}")
print(f"  StdDev: {diag['pop_std']:.0f}")
print(f"  Fraction visitée: {diag['fraction_visited']*100:.1f}%")
```

### Pour valider la matrice P

```python
# Vérifier que P est stochastique
row_sums = P.sum(axis=1)
print(f"Somme des lignes: [{row_sums.min():.3f}, {row_sums.max():.3f}]")
# Doit être proche de 1 partout (sauf pour les états jamais visités)
```

### Pour optimiser les paramètres

```python
# Tester plusieurs résolutions
analyzer.compare_within_method(
    method="cartesian",
    sweep_param="n_states"  # ou "nlt", "step_size"
)
# Affiche RSD final vs résolution
```
