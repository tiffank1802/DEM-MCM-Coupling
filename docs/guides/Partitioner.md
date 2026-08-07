# Integration Guide - MCM Markov Analyzer

## Installation

```bash
pip install numpy polars scikit-learn scipy matplotlib huggingface-hub torch
```

## Premier lancement

### 1. Configuration HuggingFace

```python
from huggingface_hub import login
login()  # Entrer votre token HF
```

### 2. Lancer un calcul sweep

```bash
python run_sweep.py --method voronoi
```

### 3. Charger et analyser

```python
from analyze_results import MarkovAnalyzer

analyzer = MarkovAnalyzer()
analyzer.load_all()
analyzer.print_summary()
analyzer.plot_experiment("voronoi_125cells_NLT200")
analyzer.plot_rsd_comparison()
```

---

## Architecture des Classes

### Hiérarchie d'héritage

```
BasePartitioner (ABC)
├─ CartesianPartitioner
├─ CylindricalPartitioner
├─ VoronoiPartitioner
├─ QuantileGridPartitioner
├─ OctreePartitioner
└─ PhysicsAwarePartitioner
```

### Algorithme de génération de matrice P

```
Pour chaque timestep t ∈ [start_index, start_index + nlt*step_size]:
    Charger snapshot_t et snapshot_{t+1}
    
    Pour chaque particule i:
        state_prev[i] = partitioner.compute_states(x[i], y[i], z[i])
        state_curr[i] = partitioner.compute_states(x'[i], y'[i], z'[i])
    
    Compter transitions:
        T[state_prev[i], state_curr[i]] += 1
    
Normaliser:
    Pour chaque état i:
        P[i, :] = T[i, :] / sum(T[i, :])
```

### Calcul du RSD

```
À t=0: condition initiale ségrégée
    C[0:n//2] = 1  (espèce A dans la moitié "gauche")
    C[n//2:]  = 0  (espèce B dans la moitié "droite")

À t > 0:
    C(t+1) = C(t) @ P
    RSD(t) = std(C[visited]) / mean(C[visited])
    
Mélange "meilleur" quand RSD → 0
```

---

## Troubleshooting

### Problème: "Connection timeout"
**Cause:** Réseau HuggingFace instable
**Solution:** 
```python
import socket
socket.setdefaulttimeout(30)  # Augmenter le timeout
```

### Problème: "Cells never visited"
**Cause:** Partitionneur inadapté au domaine
**Solution:** 
```python
# Diagnostiquer
diag = partitioner.diagnostics(coordinates)
print(f"Fraction visitée: {diag['fraction_visited']*100:.1f}%")

# Si < 70%, augmenter n_cells ou utiliser Voronoï
```

### Problème: "NaN in RSD"
**Cause:** Concentration moyenne = 0 dans les cellules visitées
**Solution:**
```python
# Vérifier la condition initiale
C_initial = np.zeros(n_states)
mid = n_states // 2
C_initial[:mid] = 1.0
print(f"Somme initiale: {C_initial.sum()}")  # Doit être > 0
```
