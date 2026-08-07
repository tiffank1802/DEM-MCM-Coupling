# 🔬 DEM-MCM-Coupling

**Modélisation de la ségrégation granulaire par chaînes de Markov (homogènes et inhomogènes) couplées à des simulations DEM (Discrete Element Method).**

Ce dépôt contient les sources qui m'ont permis de construire le modèle de Markov à partir de simulations DEM d'un mélangeur granulaire. Le code est organisé comme une **librairie Python** (`dem_mcm_coupling`) accompagnée d'outils de **post-traitement** (`postprocessing`).

---

## 📖 Documentation

La documentation complète (bilingue FR/EN) du stage est disponible dans le **wiki** :

👉 **https://github.com/tiffank1802/DEM-MCM-Coupling/wiki**

---

## 🗂️ Structure du dépôt

```
DEM-MCM-Coupling/
├── dem_mcm_coupling/        # Librairie principale
│   ├── partitioners.py      #   Création des méthodes de découpage du mélangeur (voronoi, cartésien, cylindrique, octree, …)
│   ├── run_sweep.py         #   Configurations des modèles de Markov + construction des matrices de transition (homogène & inhomogène)
│   ├── bucket_io.py         #   Chargement des simulations DEM depuis un bucket HuggingFace + téléversement des résultats
│   ├── analyze_results.py   #   Chargement & comparaison des courbes RSD vs τ
│   ├── markov_core.py       #   Noyau de calcul markovien
│   ├── utils.py             #   Fonctions utilitaires
│   └── _config.py           #   Configuration et types
├── postprocessing/          # Scripts de post-traitement des expériences
│   ├── postprocess.py               #   Post-traitement automatisé (homogène)
│   ├── postprocess_inhomogeneous.py #   Post-traitement des chaînes inhomogènes (P_blocks)
│   ├── calibrage.py / create_hf_dir.py / directory.py
│   ├── run_parallel.sh              #   Post-traitement parallèle par catégorie
│   └── …                            #   Scripts de maintenance / notebooks
├── tests/                   # Tests pytest
├── docs/                    # Guides, méthodes et notebooks d'analyse
└── pyproject.toml           # Package pip-installable (dem-mcm-coupling)
```

## 🚀 Installation

```bash
cd DEM-MCM-Coupling
pip install -e .          # installe la librairie dem_mcm_coupling
python -m pytest tests/   # lance les tests
```

## 📌 Notes

- **Données** : vous ne trouverez ni les sources de données DEM, ni les résultats des modèles de Markov dans ce dépôt — ils sont stockés sur un **bucket HuggingFace** (`ktongue/DEM_MCM`) et chargés via `bucket_io.py`.
- Les principaux points d'entrée sont `postprocessing/postprocess.py` et `postprocessing/postprocess_inhomogeneous.py`.
