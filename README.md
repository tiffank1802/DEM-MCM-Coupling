# 🔬 DEM-MCM-Coupling

**Modélisation de la ségrégation granulaire par chaînes de Markov (homogènes et inhomogènes) couplées à des simulations DEM (Discrete Element Method).**

`dem_mcm_coupling` est une **librairie Python** permettant de :

- **découper** le domaine du mélangeur granulaire en états discrets (voronoi, cartésien, cylindrique, octree, quantile, physique, spectral, GMM, DBSCAN, adaptatif, multi-zones…) ;
- **construire** les matrices de transition de Markov (homogènes ou inhomogènes, une matrice par bloc NLT) ;
- **propager** un état initial et **analyser** les cinétiques de mélange (RSD vs τ, entropie, temps de mélange) ;
- **se connecter à différentes sources de données** (bucket Hugging Face, dossier local, données en mémoire) via une interface unique.

La documentation complète (bilingue FR/EN) du stage est disponible dans le **wiki** :
👉 **https://github.com/tiffank1802/DEM-MCM-Coupling/wiki**

---

## 🚀 Installation

```bash
pip install dem-mcm-coupling                     # noyau (numpy, scipy, scikit-learn, …)
pip install dem-mcm-coupling[hf]                 # + accès au bucket Hugging Face
pip install dem-mcm-coupling[torch]              # + calcul des matrices via PyTorch
pip install dem-mcm-coupling[viz]                # + figures matplotlib
pip install dem-mcm-coupling[app]                # + visualisation Streamlit/PyVista
pip install dem-mcm-coupling[full]               # tout en une fois
```

En développement :

```bash
git clone https://github.com/tiffank1802/DEM-MCM-Coupling.git
cd DEM-MCM-Coupling
pip install -e ".[full,dev]"
python -m pytest tests/       # lance les tests
ruff check . && ruff format . # qualité de code
mypy dem_mcm_coupling/        # typage statique
```

---

## 🧩 Sources de données

Le point d'entrée de la librairie est l'interface [`DataSource`](dem_mcm_coupling/data/base.py) :
tout le pipeline Markov (modèle, sweeps, analyse) consomme **la même interface**,
quelle que soit l'origine des données.

| Source | Classe | Usage |
|---|---|---|
| Bucket Hugging Face | `HuggingFaceDataSource` | données DEM (`simulation_complete.parquet`) et expériences pré-calculées du dépôt `ktongue/DEM_MCM` |
| Dossier local | `LocalDataSource` | parquet local + dossiers d'expériences sur disque |
| En mémoire | `InMemoryDataSource` | tests, notebooks, prototypage |

```python
from dem_mcm_coupling.data import HuggingFaceDataSource, LocalDataSource, InMemoryDataSource
from dem_mcm_coupling import Markov

# 1. Données depuis le bucket Hugging Face (aucun téléchargement local)
source = HuggingFaceDataSource(particle_diameter=0.004)

# 2. Données depuis un dossier local
source = LocalDataSource("chemin/vers/mes_donnees")

# 3. Données en mémoire (dict {timestep: DataFrame})
source = InMemoryDataSource(timesteps={250: df_250, 300: df_300})

# Le modèle est identique quelle que soit la source :
model = Markov(method="voronoi", method_kwargs={"n_cells": 125}, data_source=source)
model.load_dem_data()
coords = model.get_coords([250, 300, 350])
model.fit_partitioner(coords)
state0 = model.build_initial_state_vector(250)
trajectory = model.propagate_markov(state0.phi, M, n_steps=100)
```

Un assistant `data_source_from_uri("hf://ktongue/DEM_MCM" | "memory://" | "<dossier>")`
construit la bonne source à partir d'une simple chaîne.

---

## 🎯 Convention de la matrice de transition

La librairie suit une **convention unique** :

- `P[i, j]` = probabilité de transition de l'état `i` vers l'état `j` ;
- les lignes sont stochastiques : `P.sum(axis=1) == 1` ;
- un vecteur d'état évolue par **multiplication à droite** : `phi_next = phi @ P`.

Cette convention est appliquée partout : `run_sweep.compute_P_matrix_torch`,
`markov_core.propagate_markov`, `analyze_results` et les scripts de
post-traitement.

---

## 🗂️ Structure du dépôt

```
dem_mcm_coupling/
├── __init__.py          # API publique + __version__
├── _config.py           # constantes, types partagés, dataclasses d'état
├── data/                # ← couche d'accès aux données (pluggable)
│   ├── base.py          #   DataSource (interface) + DemSnapshot
│   ├── huggingface.py   #   backend Hugging Face Hub
│   ├── local.py         #   backend dossier local
│   └── memory.py        #   backend en mémoire
├── partitioners.py      # découpages du mélangeur (REGISTRY + create_partitioner)
├── markov_core.py       # modèle Markov : état initial, propagation, visualisation
├── run_sweep.py         # sweeps homogènes/inhomogènes + CLI (dem-mcm-sweep)
├── analyze_results.py   # analyse RSD vs τ, entropie, temps de mélange
├── bucket_io.py         # lecture/écriture bas niveau du bucket Hugging Face
└── utils.py             # utilitaires généraux
postprocessing/          # outils de post-traitement (non packagés dans PyPI)
├── metrics.py           # ← physique du mélange : convention des matrices,
│                        #   propagation, RSD/entropie/ségrégation, validation,
│                        #   homogénéisation par N, interpolation de p_ij(t)
├── style.py             # ← code couleur global (méthode & espèce) + style figures
├── figures.py           # ← figures scientifiques annotées (t50/t90, unités SI)
├── validate_bucket.py   # ← validation physique des expériences du bucket
├── postprocess.py       # pipeline homogène + CLI
│                        #   · fig_mesh → série VTK temporelle : positions mobiles
│                        #     ET vecteur d'état évolutif + .pvd (temps en s)
│                        #   · erreurs DEM/Markov normalisées par le nombre de
│                        #     particules (écarts relatifs en fractions)
├── postprocess_inhomogeneous.py  # pipeline inhomogène (P_blocks) + CLI
│                        #   · p_ij(t) sur une échelle commune + loi
│                        #     d'interpolation ajustée (linéaire, quadratique…)
├── tools/               # scripts de maintenance ponctuels
└── run_parallel.sh
tests/                   # tests pytest (124 tests)
docs/                    # guides, méthodes et notebooks d'analyse
pyproject.toml           # packaging PyPI, dépendances, ruff, mypy
```

## ✅ Validation physique des résultats du bucket

Les expériences du bucket peuvent être vérifiées contre la physique du
mélange (probabilités positives, lignes stochastiques, conservation de la
masse, distribution stationnaire, RSD ∈ [0, 1] et décroissant) :

```bash
python -m postprocessing.validate_bucket --method voronoi --max 5
python -m postprocessing.validate_bucket --synthetic   # démo hors-ligne
```

**Compatibilité des anciennes données** : les matrices stockées avant
l'unification de la convention (stochastiques en colonnes) sont
automatiquement détectées et transposées au chargement
(`postprocessing.metrics.standardize_transition_matrix`) — les anciennes
expériences gardent donc leur sens physique avec le nouveau code.

## 🖥️ Ligne de commande

```bash
dem-mcm-sweep --method voronoi --list            # liste les configurations
dem-mcm-sweep --method voronoi                   # lance le sweep homogène
dem-mcm-sweep --method cartesian --inhomogeneous # sweep inhomogène (P par bloc NLT)
```

## 📌 Notes

- **Données** : les sources DEM et les résultats Markov ne sont pas dans ce
  dépôt — ils sont stockés sur le bucket Hugging Face (`ktongue/DEM_MCM`,
  dépôt privé/gated : nécessite `huggingface-cli login` avec un compte ayant
  accès) et chargés via la couche `data`.
- Les principaux points d'entrée du post-traitement sont
  `postprocessing/postprocess.py` et
  `postprocessing/postprocess_inhomogeneous.py` ; les nouvelles figures
  scientifiques annotées sont dans `postprocessing/figures.py`.

