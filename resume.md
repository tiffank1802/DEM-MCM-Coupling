# 📚 INDEX COMPLET DE DOCUMENTATION

## Structure du Projet

```
.
├── partitioners.py              [↓ 400+ lignes]
│   ├── BasePartitioner          (classe abstraite)
│   ├── CartesianPartitioner     (5 méthodes publiques)
│   ├── CylindricalPartitioner   (5 + helper)
│   ├── VoronoiPartitioner       (5 + KDTree)
│   ├── QuantileGridPartitioner  (5)
│   ├── OctreePartitioner        (5 + récursion)
│   ├── PhysicsAwarePartitioner  (8 + normalization)
│   └── create_partitioner()     (factory)
│
├── bucket_io.py                 [↓ 100+ lignes]
│   ├── get_fs() / get_api()     (singletons)
│   ├── save_experiment_to_bucket()
│   ├── load_matrix_from_bucket()
│   ├── load_experiment_from_bucket()
│   ├── list_experiments()
│   └── load_all_experiments()
│
├── run_sweep.py                [↓ 600+ lignes]
│   ├── ExperimentConfig (dataclass)
│   ├── get_configs()            (par méthode)
│   ├── compute_P_matrix_torch() (GPU)
│   ├── run_experiment()         (exécution)
│   └── run_markov_sweep()       (main)
│
└── analyze_results.py           [↓ 1500+ lignes]
    ├── MarkovAnalyzer (classe macro)
    ├── Chargement: load_all(), load_method()
    ├── Analyse: compute_rsd(), compare_methods()
    ├── DEM vs Markov: compute_dem_rsd(), compare_dem_vs_markov()
    └── Visualisations: plot_experiment(), plot_rsd_comparison()
```

## Matrice de Rôles

| Fichier | Responsabilité | Entrée | Sortie |
|---------|---|---|---|
| partitioners.py | Découper l'espace | coords (N,3) | états int64 |
| run_sweep.py | Calculer P | fichiers DEM | matrice P |
| bucket_io.py | Persister | P + config | fichiers .npy/.json |
| analyze_results.py | Analyser | P du bucket | graphiques + stats |

## Flux de données

```
Données DEM (file_0.csv ... file_500.csv)
    ↓ [partitioners]
États (qui → quoi, timestep)
    ↓ [run_sweep]
Matrice P (n_states × n_states)
    ↓ [bucket_io]
Bucket HuggingFace
    ↓ [analyze_results]
Visualisations (plots) + Métriques (RSD, t_mix)
```

---

## 🔑 Concepts Clés

### 1. État (Cell / Compartment)
- Une région de l'espace
- Numéroté 0 à n_states-1
- Chaque particule appartient à exactement 1 état à chaque timestep

### 2. Transition
- Passaged'une particule d'un état à un autre
- Quantifiée dans la matrice T (non-normalisée)
- La probabilité P(i→j) = T(i,j) / sum(T(i,:))

### 3. Matrice P
- n_states × n_states
- Stochastique (lignes somment à 1)
- P(i,j) = probabilité de transition i → j en 1 pas de temps

### 4. RSD (Relative Standard Deviation)
- Mesure l'hétérogénéité de mélange
- RSD = 0 → mélange parfait
- RSD = 100% → ségrégation totale

### 5. Timestep (NLT)
- Nombre of Lagrangian Timesteps
- Combien de transitions observées
- Plus élevé = plus de données = P plus précis

---

## Checklists

### Pour lancer un calcul

- [ ] Données DEM disponibles sur le bucket HF
- [ ] Token HF authentifié
- [ ] GPU disponible (cuda)
- [ ] Choisir la méthode de partitionnement
- [ ] Passer les bons hyper-paramètres
- [ ] ```bash
        python run_sweep.py --method <method>
      ```

### Pour analyser les résultats

- [ ] Résultats téléchargés depuis le bucket
- [ ] Matrice P chargée
- [ ] Stats vérifiées (% d'états visitées > 70%)
- [ ] ```python
        analyzer.load_all()
        analyzer.print_summary()
        analyzer.plot_experiment(...)
```

### Avant de publier

- [ ] Valider que P est stochastique
- [ ] Comparer au moins 2 conditions
- [ ] Documenter le choix de partitionnement
- [ ] Reporter n_states, NLT, step_size
- [ ] Inclure RSD + entreopy + eignevalues

---

##🚀 Améliorations Futures

1. **Parallélisation multi-GPU** pour run_sweep
2. **Caching smart** dans analyze_results
3. **Tests unitaires** pour chaque partitionneur
4. **CLI enrichie** avec tqdm progressbar
5. **Export résultats** en DataFrame Polars
6. **Dashboard interactif** Flask/Streamlit
