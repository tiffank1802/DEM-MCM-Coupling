# Méthodes de coupe pour l'Octree adaptatif

## Principe général

L'**OctreePartitioner** subdivise récursivement l'espace 3D en cellules de plus en plus petites
dans les zones denses en particules. À chaque nœud interne, un **plan de coupe** sépare les
particules en deux sous-ensembles. La méthode choisie détermine **comment** ce plan est calculé.

Paramètres de contrôle :
- `max_particles` : si une cellule contient moins de ce seuil, on arrête de la subdiviser
- `max_depth` : profondeur maximale de récursion (limite le nombre total de cellules)
- `oblique_method` : la méthode de calcul du plan de coupe (voir ci-dessous)

---

## 1. `axis` — Coupe axiale (octree classique)

**Principe :** on coupe selon les plans médians **alignés sur les axes** (x, y, z).

```
À chaque niveau:
  1. Calculer la médiane des coordonnées x, y, z des particules
  2. Couper en 8 octants selon les 3 médianes
  3. Répéter récursivement dans chaque octant
```

**Avantages :** simple, rapide, préserve la structure cartésienne, visualisation facile.
**Inconvénients :** peut créer des divisions inefficaces si la densité varie selon une direction oblique.
**Utilisation :** méthode par défaut (`oblique_method=None` ou `"axis"`).

---

## 2. `pca` — Plan orthogonal à la variance maximale

**Principe :** on cherche la direction dans laquelle les particules sont **le plus dispersées**
(analyse en composantes principales), et on coupe orthogonalement à cette direction.

```
  1. Calculer la matrice de covariance des positions (3×3)
  2. Décomposer en valeurs/vecteurs propres (eigh)
  3. La normale du plan = vecteur propre associé à la valeur propre maximale
  4. Le décalage (offset) = médiane de la projection des points sur cette normale
```

**Avantages :** s'adapte à l'orientation locale du nuage de points ; coupe là où ça sépare le mieux.
**Inconvénients :** coût du calcul de covariance + décomposition pour chaque nœud.
**Utilisation recommandée :** bon compromis général pour des nuages anisotropes.

---

## 3. `kmeans2` — Plan médiateur entre 2 centroïdes k-means

**Principe :** on regroupe les particules en **2 clusters** par k-means, puis on place le plan
de coupe à **équidistance** des deux centroïdes.

```
  1. Appliquer KMeans(n_clusters=2) sur les positions (échantillonnage à 10 000 max)
  2. Récupérer les centroïdes c1, c2
  3. Normale du plan = vecteur (c2 - c1) normalisé
  4. Offset = projection du point milieu (c1 + c2) / 2 sur la normale
```

**Avantages :** suit naturellement la structure des clusters locaux.
**Inconvénients :** dépendant de l'initialisation aléatoire ; coût du k-means.
**Remarque :** `n_init=3` pour stabiliser légèrement le résultat.

---

## 4. `2medians` — Plan médiateur entre 2 médianes de cluster

**Principe :** variante robuste du k-means qui utilise les **médianes** (L1) au lieu des
moyennes (L2), moins sensible aux outliers.

```
  1. Initialiser 2 centres avec 2 points aléatoires distincts
  2. Pour 5 itérations:
     a. Assigner chaque point au centre le plus proche (distance L2)
     b. Recalculer chaque centre comme la médiane des points assignés
  3. Normale du plan = vecteur (c2 - c1) normalisé
  4. Offset = projection du point milieu sur la normale
```

**Avantages :** robuste aux outliers (la médiane ne se laisse pas dévier par des points extrêmes).
**Inconvénients :** plus coûteux que la médiane simple ; 5 itérations suffisent généralement.
**Comparaison avec kmeans2 :** les médianes donnent des coupes plus stables en présence de données bruitées.

---

## 5. `random` — Plan aléatoire

**Principe :** direction aléatoire uniforme sur la sphère unité, coupure à la médiane.

```
  1. Tirer un vecteur direction ~ N(0, I), normaliser (distribution uniforme sur la sphère)
  2. Projeter les points sur cette direction
  3. Offset = médiane des projections
```

**Avantages :** très rapide ; utile comme baseline aléatoire ou pour des forêts d'arbres aléatoires.
**Inconvénients :** aucune adaptation à la géométrie locale ; peut couper n'importe où.
**Remarque :** seed fixe (`RandomState(42)`) pour reproductibilité.

---

## 6. `svm` — Plan de marge maximale (SVM linéaire)

**Principe :** on utilise PCA pour générer des étiquettes binaires, puis un SVM linéaire
trouve le séparateur à **marge maximale** entre les deux groupes.

```
  1. PCA : projeter sur la direction de variance maximale
  2. Étiquettes : points au-dessus de la médiane PCA → classe 1, en dessous → classe 0
  3. Entraîner LinearSVC(C=1.0) sur (positions, étiquettes)
  4. Normale = vecteur des coefficients SVM normalisé
  5. Offset = -intercepte / ||w||
```

**Avantages :** plan de séparation optimal au sens de la marge ; peut généraliser mieux.
**Inconvénients :** plus coûteux ; dépend des étiquettes PCA (biais d'orientation).
**Remarque :** `dual='auto'` laisse scikit-learn choisir la formulation duale ou primale.

---

## Tableau récapitulatif

| Méthode     | Direction du plan            | Coût par nœud | Robustesse | Déterministe |
|-------------|------------------------------|---------------|------------|--------------|
| `axis`      | Axes x/y/z (médianes)        | O(n)          | Haute      | Oui          |
| `pca`       | Variance maximale            | O(n·d²)       | Moyenne    | Oui          |
| `kmeans2`   | Médiatrice de 2 centroïdes   | O(n·d·iter)   | Faible     | Non*         |
| `2medians`  | Médiatrice de 2 médianes     | O(n·d·iter)   | Haute      | Non*         |
| `random`    | Direction aléatoire          | O(n)          | —          | Non*         |
| `svm`       | Marge maximale (SVM + PCA)   | O(n·d²)       | Moyenne    | Oui          |

*déterministe ici grâce à `RandomState(42)`, mais la méthode est stochastique par nature.

---

## Structure de l'arbre oblique

Dans l'implémentation, les coupes obliques construisent un **arbre binaire** (pas un octree 8-aire) :

```
Noeud interne:
  - normal    : vecteur normal (3,) au plan de coupe
  - offset    : décalage du plan (scalaire)
  - left      : sous-arbre gauche (proj <= offset)
  - right     : sous-arbre droit  (proj > offset)

Feuille:
  - type      : "leaf"
  - bounds    : bounding box (xmin, xmax, ymin, ymax, zmin, zmax)
  - centroid  : centre de masse des particules dans la feuille
  - halfspaces: liste des (normal, offset, sens) depuis la racine
```

Les feuilles stockent l'historique complet des demi-espaces depuis la racine,
ce qui permet d'assigner un état à une particule en évaluant simplement
tous les plans de coupe le long du chemin.
