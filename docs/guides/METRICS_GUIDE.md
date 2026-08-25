# 📊 Guide des Indicateurs — Analyse Markovienne

## Vue d'ensemble

Les indicateurs ci-dessous évaluent la **qualité des matrices de transition** et leur capacité à représenter le mélange des particules.

---

## 1️⃣ **P(rester)** — Diagonale Moyenne

**Formule:** `μ = mean(diag(P))` ± `std(diag(P))`

**Interprétation:**
- **Valeurs idéales:** 0.5 - 0.9
- **P(rester) trop bas** (<0.3): Trop de transitions → instabilité
- **P(rester) trop haut** (>0.95): Peu de transitions → mélange lent
- **σ faible**: Comportement homogène aux frontières
- **σ fort**: Hétérogénéité - certaines cellules "collantes"

**Exemple:**
```
μ = 0.750 ± 0.120  [0.421, 0.998]
→ Mélange raisonnablement équilibré
```

---

## 2️⃣ **Normalisation des Lignes** — Σ Lignes

**Formule:** `row_sums = P.sum(axis=1)` 

**Idéal:** Tous les `row_sums ≈ 1.0` (matrice stochastique)

**Interprétation:**
- **Σ lignes = 1.0**: Même probabilité totale de quitter chaque cellule
- **Σ < 1.0**: Perte de probabilité (certains états sont "puits")
- **Σ > 1.0**: Gain de probabilité (incorrectness)
- **Min/Max proches**: Distribution homogène
- **Min/Max éloignés**: Forte hétérogénéité = mauvaise ségrégation

**Exemple:**
```
[0.734, 1.001]
→ Certains états acceptent peu de transitions (min trop bas)
```

---

## 3️⃣ **États Visitées** — Visitabilité / Accessibility

**Formule:** `fraction_visited = n_visited / n_total`

**Interprétation:**
- **100%**: Tous les états sont accessibles (bon)
- **<95%**: Certains états jamais visités = partitionnement inadapté
- **Structure avec poches isolées**: Indique des problèmes de ségrégation

**Exemple:**
```
95.3% → 3 cellules jamais atteintes sur 64
```

---

## 4️⃣ **Valeur Propre Dominante** — λ₂

**Signification:** 2ème plus grande valeur propre (en valeur absolue)

**Interprétation:**
- **λ₂ proche de 0**: Convergence très rapide (excellent)
- **λ₂ ≈ 0.5**: Convergence modérée (acceptable)
- **λ₂ proche de 1.0**: Convergence très lente (mauvais)
- **λ₂ = λ₁ = 1.0**: Matrice singulière

**Physique:** λ₂ décrit la "persistance de la ségrégation":
- Basse valeur = mélange rapide et irréversible
- Haute valeur = mélange lent et peut-être réversible

---

## 5️⃣ **Spectral Gap** — Écart Spectral

**Formule:** `gap = 1 - λ₂`

**Interprétation:**
- **Gap proche de 1**: Excellente convergence exponentiellement rapide
- **Gap ≈ 0.3**: Convergence acceptable  
- **Gap proche de 0**: Convergence très lente

**Exemple de temps de mélange:**
```
Temps pour atteindre 90% du mélange: ~log(0.1) / log(λ₂)

λ₂ = 0.1   → gap = 0.9  → ~2 pas de temps
λ₂ = 0.5   → gap = 0.5  → ~7 pas de temps  
λ₂ = 0.95  → gap = 0.05 → ~60 pas de temps
```

---

## 6️⃣ **RSD** — Relative Standard Deviation (Mélange)

**Formule:**
```
C(t) = C(0) × P^t    (concentration à temps t)
RSD(t) = σ(C) / μ(C)
```

**Interprétation:**
- **RSD initial**: Homogénéité au départ (généralement 1.0 ou |équi-pop|)
- **RSD final**: Homogénéité après convergence (idéalement → 0)
- **T₅₀**: Temps pour atteindre 50% du mélange
- **T₉₀**: Temps pour atteindre 90% du mélange

**Bon profil RSD:**
```
Initial: 1.00
50%:     0.50 (rapide)
Final:   0.05 (bien mélangé)
```

**Mauvais profil RSD:**
```
Initial: 1.00
50%:     0.90 (lent)
Final:   0.30 (mal mélangé)
```

---

## 7️⃣ **Coefficient de Variation** — Population par Cellule (CV)

**Formule:** `CV = σ(pop) / μ(pop)`

**Interprétation (pour partitions):**
- **CV < 0.2**: Population très homogène (excellent équilibrage)
- **CV ≈ 0.5**: Hétérogénéité modérée (acceptable)
- **CV > 1.0**: Forte disparité de taille (mauvais partitionnement)

**Exemple:**
```
Cartesian 5×5×5: CV ≈ 0.15 (zones régulières, population égale)
Voronoï 125:     CV ≈ 0.35 (zones de taille variée mais plus adaptées)
```

---

## 🎯 **Recommandations d'Interprétation**

### ✅ **Matrice de Qualité Excellente:**
- P(rester) ≈ 0.6-0.8
- Σ lignes dans [0.95, 1.05]
- Visitabilité ≥ 99%
- λ₂ < 0.3 (spectral gap > 0.7)
- RSD final < 0.1
- CV (partition) < 0.25

### ⚠️ **Matrice de Qualité Acceptable:**
- P(rester) ≈ 0.5-0.9
- Σ lignes dans [0.8, 1.2]
- Visitabilité ≥ 95%
- λ₂ < 0.6
- RSD final < 0.2
- CV < 0.5

### ❌ **Matrice de Qualité Mauvaise:**
- P(rester) < 0.3 ou > 0.95
- Σ lignes hors [0.7, 1.3]
- Visitabilité < 90%
- λ₂ > 0.8
- RSD final > 0.3
- CV > 1.0

---

## 📈 **Comparaison Inter-Méthodes**

| Indicateur | Cartesian | Cylindrique | Voronoï | Quantile | Octree |
|-----------|-----------|------------|---------|----------|--------|
| **P(rester)** | ~0.75 | ~0.70 | ~0.65 | ~0.72 | ~0.68 |
| **λ₂** | 0.45 | 0.48 | 0.35 | 0.42 | 0.38 |
| **Visitabilité** | 100% | 100% | 98% | 99% | 95% |
| **CV** | 0.05 | 0.12 | 0.35 | 0.08 | 0.40 |
| **Avantage** | Simple | Adaptation cylindrique | Adaptative | Équitable | Densité |

---

## 🔬 **Sources et Références**

Ces indicateurs proviennent de:
- `analyze_results.py` — Chargeur universel de matrices
- `visualize_partitioning.py` — Diagnostiques de partitions
- Théorie des chaînes de Markov: temps de mélange, spectral gaps

Pour plus d'infos:
```python
from analyze_results import MarkovAnalyzer
analyzer = MarkovAnalyzer()
analyzer.print_summary()  # Affiche tous les indicateurs
```
