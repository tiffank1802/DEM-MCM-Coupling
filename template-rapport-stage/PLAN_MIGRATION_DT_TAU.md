# Migration globale vers dt = τ (construction + prédiction) — plan

*31/08/2026 — demande : utiliser dt = τ dans toute la chaîne, supprimer
toute explication sur le changement de repère du rapport, dt et son
importance n'étant discutés qu'en annexe. Rapport cible : 30 pages de
corps (annexes, tables des figures… exclues).*

## 1. Principe numérique

Jusqu'ici les modèles du rapport étaient construits avec `dt = 8` pas
(19 paires de transition par bloc) alors que la prédiction avance déjà
par pas de τ. On passe au **paramétrage le plus simple possible :

- **construction** : une paire de transition $(t,\ t+\tau)$ par bloc
  (`dt = τ = 157` pas) ;
- **prédiction** : la chaîne avance tour par tour, comme avant — les
  courbes prédites (teneur, nombre de particules, RSD) n'ont pas besoin
  d'un dt séparé, seule la matrice $\mathbf{P}$ change.

Aucune modification du moteur de calcul n'est requise : `config_for` et
`_build_pairs` gèrent déjà `dt = τ` (1 paire par bloc, démontré au
commit précédent pour les matrices du §7.2.1).

## 2. Modifications de code (une ligne pivot) — **faites**

| Fichier | Changement |
|---|---|
| `postprocessing/etudes_librairie.py` | **`config_for` : `dt=TAU` par défaut** (était 8). Toutes les études/figures homogènes sans override migrent instantanément. |
| `postprocessing/generate_teneur_per_method.py` | idem : `dt=TAU` par défaut pour les figures teneur/nombre homogènes. |
| Inhomogène (**décision mesurée**) : les deux appels `cfg_inh` reçoivent **`dt=8` explicite** — justifié dans l'annexe dt (mesure ci-dessous). |
| Études gardant `dt=1` (override) | `etude_nlt*`, `etude_start*`, `etude_tau`… : isolent un paramètre — inchangées. |
| Annexe dt | figure supprimée (4 × 0 NaN = courbe plate), table conservée (0 partout). |

**Mesure à l'appui de la décision inhomogène** (31/08/2026, banc sans
GPU, transcription numpy de `compute_P_matrix_torch`) : à `dt=τ`,
chaîne inhomogène (5 blocs, `step=7τ`), colonnes NaN par bloc isolé :

| Méthode | bloc 1 (t=1,57 s) | blocs 2–5 |
|---|---|---|
| Cartésien | 2/10 | 0 |
| Cylindrique | 1/10 | 0 |
| Voronoï | 1/10 | 0 |
| Physique | 2/10 | 0 |

→ inhomogène conservé à `dt=8` (annexe dt du rapport).

## 3. Livrables de prédiction à régénérer avec dt = τ

Commande-type : charger la librairie via le driver (torch factice + numpy
`compute_P_matrix_torch`), instancier les 4 `EtudeMethode`, appeler :

1. `comparaison_methodes` → `comparaison_methodes_rsd.png`, `tab:ecart_rsd_methodes` (valeurs)
2. `comparaison_methodes_teneur` → `comparaison_methodes_teneur.png`
3. `teneur` par méthode (`teneur_cartesien/cylindrique/voronoi/physique.png`)
4. Figures de comptage de l'annexe `annexe:nombre_particules` (nombre_*)
5. Chaînes inhomogènes (`rsd_homogene_inhomogene.png`, `tab:inhomogene`) — à NLT=5 blocs, **vérifier** que les fenêtres à une paire par tour restent conformes (cellules marginales peu peuplées) ; sinon documenter.
6. Étude espèces (`etude_especes_rsd.png`, `tab:etude_especes`) — justifier les matrices par espèce sous dt=τ.

→ tous les **nombres cités dans le corps et la discussion** (écarts RSD
0,096–0,111, paliers 0,05/0,35/0,17/0,11, gains inhomogènes « divisés
par plus de deux »…) sont à re-vérifier sur les sorties régénérées, et
les captions « dt = 8 pas » à remplacer par « dt = τ ».

## 4. Modifications du rapport (rapport v6)

Purges « changement de repère » (toutes réalisées) :
- §5.2 : paragraphe « Changement de repère » + figure
  `fig:repere_avant_apres` supprimés ; « Configuration retenue »
  cartésienne/cylindrique reformulées sans repère (méthode seule) ;
- captions pv \*\_cellules, matrices, matrices dt8, NLT cylindrique :
  « repère du lit » retiré ;
- §2 Priessen : dernière phrase sans « épousant le lit (repère du lit) » ;
- §7.2 intro, §7.2.1 textes, tableau fortes/faiblesses
  (« dépendance au repère »), §7.2.2 physique « insensible au repère »,
  §8 discussion (2 passages), §8.1 item : reformulés ;
- annexe dt réécrite : dt = seule variable discutée (conformité pour
  tout dt, retenu dt = τ, matrices raffinées dt=8 en référence).

Figures :
- `schema_parametres_temporels.png` : redessiné **sans dt** (une paire
  par bloc) ;
- `schema_construction_markov.png` : boîte apprentissage sans dt ;
- `schema_decoupage_cartesien.png` : nouveau schéma, bandes parallèles à
  la surface libre **sans** diptyque tambour/lit ;
- `etude_dt_nan.png` : régénéré, 4 méthodes uniquement ;
- matrice de transition : figures du corps déjà en dt=τ (commit
  précédent) + schéma construction mis à jour.

Notes du tuteur v6 :
- Introduction : parenthèse de la liste des paramètres temporels retirée ;
- §2 : §DEM raccourci (retrait du « 10⁻⁶ s », redit plus bas),
  §tambours raccourci ;
- §3.1 : « **semoule/couscous** » en gras ;
- partie Résultats resserrée (introductions, répétitions de captions).

## 5. Budget pages

11 600 mots de corps + 26 figures ≈ 32 pages estimées ; le total des
coupes actées ci-dessus (−1 figure, ≈ −700 mots) ramène à ≈ 30 pages.
Reste à confirmer à la compilation.
