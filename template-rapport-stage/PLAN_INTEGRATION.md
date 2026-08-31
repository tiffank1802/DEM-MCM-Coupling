# Plan d'intégration des corrections — mémoire `main.tex`

> **Document compagnon de `REVUE_TUTEUR.md`.** Chaque point de la revue y est rattaché à une phase ; la table de traçabilité (§ 8) garantit qu'aucun item n'est perdu.
>
> **Types d'intervention** : **[R]** = rédaction/correction LaTeX directe · **[V]** = nécessite une vérification (figure, fichier de données) · **[D]** = nécessite une décision de l'étudiant avant correction.

---

## 1. Principes directeurs

1. **Source unique de vérité numérique** : les fichiers `figures/*.txt` régénérés par `dem_mcm_coupling`. Toute valeur tapée à la main dans le `.tex` doit être traçable vers l'un de ces fichiers. Fini les nombres issus d'anciennes campagnes.
2. **Ne rédiger que du vérifié.** Les points « à clarifier » de la revue ont été tranchés par inspection du code quand c'était possible (voir § 2) : on n'écrit rien sur la normalisation, le k-means ou la chaîne inhomogène qui ne soit pas confirmé par le code.
3. **Réécriture avant correction de surface** : les sections Discussion / Conclusion / § inhomogène sont reconstruites en bloc (Phase 5) *avant* la passe de langue (Phase 8), pour ne pas polir des phrases vouées à disparaître.
4. **Discipline de compilation** : après chaque phase → `xelatex → biber → xelatex × 2`, puis contrôle `grep -c "??" main.log` (références) et log biber (citations). Objectif permanent : **zéro « ?? », zéro « [?] »**.
5. **Hygiène du dépôt** : nettoyer les artefacts (`main.bbl-SAVE-ERROR`), purger les blocs commentés obsolètes (ils transportent les anciens chiffres), garder `main_v2_backup.tex` comme archive et créer des backups datés avant chaque phase lourde.

---

## 2. Ce que l'inspection du code a déjà tranché (plus de décision, juste de la rédaction)

| Question de la revue | Réponse constatée dans `dem_mcm_coupling` | Conséquence rédactionnelle |
|---|---|---|
| Normalisation du découpage physique (revue §3-1) | **Standardisation z-score** des 4 caractéristiques + poids explicite `velocity_weight = 0.5` sur les dimensions vitesse (`PhysicsAwarePartitioner`, `partitioners.py`) | § Dé coupage physique : décrire la standardisation et le poids 0,5 ; le seuil de variance $10^{-4}$ s'entend alors **dans l'espace standardisé** |
| « Consolidation par moyenne » des 10 k-means (revue §3-2) | `KMeans(init="k-means++", n_init=10)` : amorçage **k-means++** (pas « aléatoire parmi les échantillons ») et conservation de la **meilleure des 10 exécutions au sens de l'inertie** (pas de moyenne de partitions) | Corriger les deux paragraphes « Initialisation » du § Voronoï |
| Sémantique de `step` (revue §2-9-1) | **Formule `start + k·(step+τ)` confirmée** pour les deux chaînes (`_build_pairs`, `_build_inhomogeneous_blocks`), mais **fenêtres de départs différentes** : homogène = `(step+τ)` par bloc (~39 paires/bloc à dt=8, tuilage contigu) ; inhomogène = `τ` par bloc (~20 paires), blocs espacés de `step+τ = 8τ` avec step = 7τ → **matrices apprises à 1,57 / 14,13 / 26,69 / 39,25 / 51,81 s** | Refonte de l'annexe D + légende/figure `fig:schema_parametres` ; corriger aussi le docstring `ExperimentConfig` (« distance between two main starts » est faux) |
| Chaîne inhomogène : matrice après le dernier bloc ? (revue §3-5) | Propagation `block_idx = min((t-1)//block_size, n_blocks-1)` : **la dernière matrice est réutilisée jusqu'à t = 60 s** ; prédiction propagée depuis **t = 0** | À expliciter en § résultats inhomogène + annexe D |
| Repère du lit côté Voronoï (revue §2.10) | **Aucune transformation « repère du lit »** dans `dem_mcm_coupling` hors labélisation géométrique ; le k-means est ajusté sur les coordonnées brutes | Corriger « dans le repère du lit » (§ Découpage de Voronoï) ; vérifier où le repère du lit est réellement appliqué dans les scripts de production des figures |

**Restent à décider (bloquant uniquement pour les phases concernées)** : choix NLT, figure `repere_avant_apres`, promesse « teneur vs comptage » du résumé, références bibliographiques manquantes → § 6.

---

## 3. Phase 0 — Gel, sauvegarde, checklist *(≈ 0,5 h)*

- [ ] Copier `main.tex` → `main_backup_avant_revue_AAAA-MM-JJ.tex` ; archiver les blocs commentés à conserver dans `notes_reprise.tex` (hors manuscrit). **[R]**
- [ ] Supprimer/déplacer `main.bbl-SAVE-ERROR` ; vérifier que `xelatex → biber → xelatex × 2` tourne tel quel. **[R]**
- [ ] Ouvrir la checklist vivante du présent plan ; chaque case cochée = une compil propre.

**Sortie** : dépôt propre, document compilant, checklist active.

## 4. Phase 1 — Assainissement technique *(≈ 2–3 h)* — bloquant pour toute la suite

| # | Localisation `main.tex` | Action | Type |
|---|---|---|---|
| 1.1 | Préambule | Supprimer les doublons `\usepackage{booktabs|tabularx|float}` | [R] |
| 1.2 | § Contexte (tambours tournants) | Créer l'entrée `priessen2021` dans `references.bib` **[D : références exactes]** ; ajouter Sullivan (1927), Saeman (1951), Chatterjee ou reformuler sans noms | [R/D] |
| 1.3 | Bibliographie | Supprimer `\nocite{mellmann2001,...}` au profit d'une vraie citation (Phase 6, régime d'écoulement) | [R] |
| 1.4 | § Discrétisation, paragraphe « Changement de repère » | Lever la référence cassée `fig:repere_avant_apres` **[D : régénérer la figure ou réécrire sans elle — voir § 6]** ; supprimer les commentaires TODO (« enlève cette figure… », « Modifie cette figure… ») | [R/D] |
| 1.5 | Globalement | Purger tous les blocs commentés contenant des résultats obsolètes (§ inhomogène, conclusion, perspectives, anciennes figures `sidewaystable`) après archivage Phase 0 | [R] |

**Sortie** : compilation sans « ?? » ni « [?] » ; log biber vierge.

## 5. Phase 2 — Unification numérique *(≈ 2–4 h ; dépend de la décision NLT, § 6-D1)*

| # | Localisation | Action | Source |
|---|---|---|---|
| 2.1 | Annexe E, `tab:etude_especes` + texte | Remplacer par 0,398/0,145/0,234/0,204 (sans) et 0,0956/0,1025/0,0948/0,1107 (avec) ; ratios « **1,4 à 4,2** » ; exemple Voronoï 0,234 → 0,095 | `etude_especes_table.txt` |
| 2.2 | Annexe C (start), phrase sur le transitoire | Remplacer « 0,9–1,1 … jusqu'à 1,2 » par les valeurs réelles (t=0 : 1,42/2,03/0,71/1,34 ; pics ≈ 2,0 géométriques) | `etude_start_table.txt` |
| 2.3 | Corps + conclusion | Vérifier conformité déjà bonne de `tab:ecart_rsd_methodes` (0,0956/0,1025/0,0948/0,1107 ✓) et des écarts-types (0,116/0,049/0,080/0,053 ✓) ; contrôler que tout texte chiffré homogène cite ces valeurs | `comparaison_methodes_table.txt` |
| 2.4 | Contrôle global | `grep -nE "0,078|0,083|0,215|0,027|0,370" main.tex` → zéro occurrence hors archives | — |

**Sortie** : aucune valeur du manuscrit ne contredit un fichier `figures/*.txt`.

## 6. Phase 3 — Cohérences factuelles *(≈ 3–5 h)*

| # | Localisation | Action | Type |
|---|---|---|---|
| 3.1 | § Résultats — Découpage physique | Lire `pv_cellules_physique.png` + `teneur_physique.png` + `matrice_physique_especes.png` → corriger l'affectation zone active/passive contradictoire (cellules 8, 9 d'un côté, 1, 7 de l'autre) | [V→R] |
| 3.2 | § Résultats — Découpage cartésien | Lire `pv_cellules_cartesien.png` + `matrice_cartesien_especes.png` → confirmer/rectifier « cellules **1** et 9 » (probablement **0** et 9) et la valeur diagonale 0,37 | [V→R] |
| 3.3 | § Cartésien (cellule 0) | Remplacer « son faible volume » par « sa faible population effective (bande marginale peu peuplée) » — cohérent avec « volume constant » des méthodes | [R] |
| 3.4 | Légende `fig:maillage_melangeur` + § intro Résultats | Harmoniser le mécanisme de ségrégation : percolation des petites vers le fond de la couche en écoulement puis accumulation au **cœur** (trancher vs « contre la paroi ») | [R] |
| 3.5 | § Méthodes de discrétisation | Expliciter les configurations **réellement utilisées** : cartésien = 10 bandes perpendiculaires à la surface libre dans le repère du lit ; cylindrique = **10 secteurs angulaires seuls** (pas de partition $r$ ni $z$, malgré la description générale) ; annoncer que les variantes 3D existent mais ne sont pas retenues ici | [R] |
| 3.6 | § Découpage de Voronoï | Remplacer « appliqué aux positions … **dans le repère du lit** » par la description réelle (k-means sur coordonnées brutes ; insensible au repère, cf. Phase 2-tableau) | [R] |

**Sortie** : relecture croisée texte ↔ figures sans contradiction ; chaque affirmation factuelle sur une matrice ou une cellule a été vue sur la figure correspondante.

## 7. Phase 4 — Équations, définitions, notations *(≈ 3–5 h)*

| # | Localisation | Action |
|---|---|---|
| 4.1 | § Résultats, `eq:rsd` | Rédéfinir : RSD calculé sur les **cellules occupées** ($n_i>0$), $\bar c$ = **moyenne arithmétique des teneurs locales** (≠ fraction globale 0,34) ; ajouter la phrase d'interprétation (peut dépasser $\sqrt{(1-\bar c)/\bar c}$) ; propager : « valeur moyenne ≈ 0,4 » (cylindrique) qualifiée de moyenne de teneurs cellulaires |
| 4.2 | § Matrice de transition, `eq:matrice_transition` | Réécrire avec double somme : blocs $n$ **et** paires internes échantillonnées tous les $dt$ ; aligner la nomenclature ($NLT$ = nombre de blocs partout) |
| 4.3 | **Annexe D (step) — refonte complète** | Décrire la sémantique du code (cf. § 2) : homogène — bloc $k$ à $start+k\,(step+\tau)$, fenêtre de départs $(step+\tau)$, tuilage contigu à $step=\tau$ ≈ 5 tours d'apprentissage pour $nlt=2$ ; inhomogène — fenêtre $\tau$, $step=7\tau$ → matrices à 1,57/14,13/26,69/39,25/51,81 s, dernière matrice prolongée jusqu'à 60 s, prédiction depuis $t=0$ ; adapter `fig:schema_parametres` (régénérer ou corriger la légende) ; supprimer les bullets faux ($step<\tau$ ⇒ chevauchement) ; **corriger en parallèle le docstring code** |
| 4.4 | `eq:def_teneur_locale` | Corriger l'indice ($c_0=N_0^a/(N_0^a+N_0^b)$) ; définir $a$ = petites, $b$ = grandes |
| 4.5 | Nomenclature | Compléter ($\phi_p$, $l_p$, $\mathbf{v}_p$, $\mathbf{z}_p$, $\boldsymbol{\mu}_k$, $J$, $m_i$, $I_i$, $F_n$, $F_t$, $M_r$, $\mathbf{g}$) ; requalifier $C(t_k)$ : « indicateur reconstruit en fin de chaîne » (pas vecteur d'état) ; romain pour $start/step/dt/NLT$ ; accents (« représente », « issu ») |
| 4.6 | Divers | Intervalles entiers $\llbracket 0,N-1 \rrbracket$ ; `\SI{0.1}{\metre}` ; casse NLT uniforme dans légendes ; « discretisation » → accent |

**Sortie** : un relecteur peut refaire chaque calcul à partir du texte seul et du code.

## 8. Phase 5 — Réécritures éditoriales *(≈ 6–8 h — cœur du travail)*

1. **§ Apport des chaînes inhomogènes (Résultats)** — remplacer les trois paragraphes actuels par : construction (5 matrices aux instants exacts, prédiction depuis $t=0$, prolongation de la dernière matrice) ; lecture de `fig:rsd_homogene_inhomogene` ; chiffrage des gains depuis `inhomogene_table.txt` : Cartésien 0,096→0,083 (−13 %) · Cylindrique 0,103→0,101 (−1 %) · **Voronoï 0,095→0,041 (−57 %) · Physique 0,111→0,061 (−45 %)** ; interprétation (l'inhomogénéité ne corrige que ce que le maillage ne fige pas structurellement).
2. **§ Discussion — reconstruction.** Socle = paragraphes commentés (dualité espèces ; adéquation cellules↔écoulement et rôle du repère ; interaction maillage ↔ gain d'inhomogénéité ; pertinence de la teneur) **mis à jour avec les chiffres actuels**, + deux ajouts : (a) comparabilité du RSD **au sein d'un même maillage** seulement (plateaux DEM 0,66/0,45/0,22/0,11) — c'est le sens à donner à « l'état du mélange diffère selon la méthode » ; (b) compromis bruit/adaptativité des matrices d'un tour de l'inhomogène (lien annexe NLT). Remplacer les deux paragraphes visibles défectueux ; supprimer l'inversion homogène/inhomogène.
3. **§ Conclusion** : points en puces alignés sur les résultats réellement démontrés (maillage ↔ état prédit ; erreur comparable ~0,1 partout ; inhomogène : gain fort si et seulement si maillage statistique) ; **perspective reconstruite** depuis la version commentée (peut-on se passer de l'inhomogène avec un maillage adapté ? étude en $N$ ; passage à l'échelle) — meilleure que la version visible.
4. **§ Introduction** : ajouter la Discussion au plan ; corriger « émettre ».
5. **§ Comparaison des méthodes (bullets RSD)** : séparer explicitement **bias en régime établi** vs **erreur de transitoire** (cas physique) ; remplacer « ajustement dynamique » par « relaxation vers l'état stationnaire du modèle » (voronoï).

**Sortie** : Discussion et Conclusion lisibles, exactes, et sans phrase rompue ; le fil « maillage → chaîne → gain » tient en trois paragraphes consécutifs.

## 9. Phase 6 — Clarifications méthodologiques *(≈ 3–5 h ; dépend de D1, D5, § 6)*

- [ ] **NLT (D1 tranché)** : si conservation de 2 → ajouter en annexe B la phrase explicite de la valeur retenue + justification (compromis coût/fenêtre, cohérence avec l'esprit Doucet d'apprendre le régime permanent) ; si recalcul → relancer les sweeps NLT≈8–10, régénérer figures/tableaux, refaire Phase 2 partiellement. **[D]**
- [ ] **Découpage physique** : rédiger standardisation z-score + `velocity_weight=0,5` ; préciser que tolérance $10^{-4}$ / 300 itérations = critères sklearn **dans l'espace standardisé**. **[R — déjà vérifié]**
- [ ] **k-means** : corriger « aléatoire » → k-means++ ; « consolidés par moyenne » → meilleure exécution sur 10 au sens de l'inertie. **[R — déjà vérifié]**
- [ ] **Régime d'écoulement** : $Fr=\omega^2 R/g \approx 0{,}08$ → régime roulant ; insérer `mellmann2001` ici (§ Contexte ou § Matériau) ; lien avec le choix $\tau$ = un tour. **[D : validation de l'ajout]**
- [ ] **Validité markovienne** : une phrase (décorrélation à l'échelle d'un tour ; les matrices diagonales dominées mais non identitaires le suggèrent). **[R]**
- [ ] **Références manquantes (D-biblio)** : compléter `references.bib`. **[D]**

## 10. Phase 7 — Pièces liminaires *(≈ 2 h)*

- [ ] **Résumé / Abstract** : adoucir la promesse teneur **ou** y adosser un résultat chiffré si D-teneur = calcul additionnel (défaut : reformuler « indicateur directement relié à la ségrégation ») ; « pas d'extraction » → « pas de sortie » ; mots-clés. **[D]**
- [ ] **Remerciements** : restructurer (un verbe de remerciement par personne : « Je remercie Guillaume Dumazer pour… ») ; corriger espaces/ponctuation (« modèles,et », « applications . », double espace) ; réactiver les paragraphes encadrant école + laboratoires (convenance) ; relire le superlatif maladroit (« remarques les plus intéressantes »).
- [ ] **Page de garde / métadonnées** : compléter jury (« M./Mme --- ») ; vérifier cohérence dates (soutenance 02/09/2026 < fin 17/09/2026 — confirmer la date de fin) ; intitulé officiel de l'école.

## 11. Phase 8 — Passe de langue intégrale *(≈ 4–6 h)*

- [ ] Reprendre la liste de fautes de `REVUE_TUTEUR.md` § 4 (emettre, entrainés, bleue, correspodante, our→pour, cette écart, quelque soit, priviligier, Details, voronoi, intervalles d'espace…) — toutes localisées.
- [ ] Uniformiser la terminologie : « **référence** DEM » (jamais « prédiction DEM ») ; « tableau » partout ; « bidisperse » ; NLT majuscule ; « internals » → « éléments internes / cloisons ».
- [ ] Réparer les phrases rompues signalées : description `fig:schema_tambour` ; proposition participiale du § formalisme ; « entre cellules observe reflète » ; « Au premier tour, puis après sept tours… soit cinq modèles ».
- [ ] Légendes auto-suffisantes : `tab:inhomogene` (préciser métrique + configuration), figures de matrices.

## 12. Phase 9 — Vérification finale et gel *(≈ 2–3 h)*

Checklist d'acceptation :
1. [ ] Compilation `xelatex → biber → xelatex × 2` sans erreur ; log sans « ?? » ; biber sans « [?] ».
2. [ ] `grep -nE "0,078|0,027|0,215|0,370|repere_avant" main.tex` → vide.
3. [ ] Chaque tableau chiffré = son fichier `figures/*.txt` (contrôle case par case).
4. [ ] Aucun commentaire TODO ni note de travail dans le `.tex`.
5. [ ] Discussion, Conclusion, § inhomogène relues à voix haute ; plan d'introduction = sections réelles.
6. [ ] Nomenclature ↔ notations du corps ↔ équations : un seul symbole, un seul nom.
7. [ ] PDF final régénéré ; mise de côté avec horodatage.

---

## 13. Traçabilité revue → phases

| Item de `REVUE_TUTEUR.md` | Phase |
|---|---|
| §2.1 Biblio (priessen2021, Saeman, Mellmann) | 1.2 / 1.3 / 6 (D-biblio) |
| §2.2 Référence cassée + TODO | 1.4 |
| §2.3 Incohérences numériques | 2.1–2.4 |
| §2.4 Zones actives/passives, cellules 1–9 | 3.1 / 3.2 |
| §2.5 NLT | 6 (D1) + 2 (si recalcul) |
| §2.6 Discussion dégradée | 5.2 ; intro 5.4 |
| §2.7 Résumé teneur | Phase 7 (D-teneur) |
| §2.8 RSD / cellules vides / transitoire | 4.1 / 2.2 |
| §2.9 Équations et notations | 4.2–4.6 |
| §2.10 Contradictions internes | 3.3–3.6 / 5.5 |
| §3.1–3.2 Normalisation, k-means | Phase 6 (rédaction, déjà vérifié code) |
| §3.3–3.8 Markovité, régime, inhomogène fin, RSD, N, métriques annexes | 6 / 4.3 / 5.2 / 5.3 |
| §3.9–3.10 Études annexes, légendes | 2.2 / 8 (légendes) |
| §4 Langue, remerciements, forme | 7 / 8 / 1.1 |

## 14. Séquencement, dépendances, estimation

```
Décisions (§15) ─▶ Phase 0 ─▶ Phase 1 ─┬─▶ Phase 2 ─▶ Phase 3 ─▶ Phase 5 ─▶ Phase 7 ─▶ Phase 8 ─▶ Phase 9
                                       └─▶ Phase 4 ────────────┘   ▲
                                        Phase 6 ───────────────────┘
```
- **Chemin critique** : Décisions → P1 → P2/P3/P4 (parallélisables par sections) → P5 → P7–P8 → P9.
- **Charge estimée** : 3,5–5 jours rédactionnels + 0,5–2 jours de recalcul **uniquement si D1 = recalcul NLT**.
- Jalons : **J1** dépôt compilant proprement (fin P1) · **J2** cohérence numérique/factuelle (fin P3/P4) · **J3** corps éditorial réécrit (fin P5) · **J4** version « langue + liminaires » (fin P8) · **J5** gel pour relecture tuteur (fin P9).

## 15. Décisions validées (gelées le 31/08/2026)

| # | Question | **Décision retenue** | Impact sur les phases |
|---|---|---|---|
| D1 | **NLT** | ✅ **Conserver NLT = 2** + phrase explicite de la valeur retenue et du compromis (annexe B). Aucun recalcul. | Phase 6 : rédaction simple ; Phase 2 inchangée |
| D2 | Vérifications factuelles | ✅ **L'agent lit les PNG et corrige** (`pv_cellules_*`, `teneur_*`, `matrice_*_especes`) ; l'étudiant valide au jalon J2 | Phase 3 accélérée |
| D3 | Figure `repere_avant_apres` | ✅ **Régénérer la figure** (schéma cartésien en deux parties : repère tambour / repère du lit, selon la note TODO d'origine) et réactiver `\ref{fig:repere_avant_apres}` | Phase 1.4 + production figure en Phase 3 |
| D4 | Promesse teneur du résumé | ✅ **Reformuler** (« indicateur directement relié à la ségrégation ») + requalifier $C(t_k)$ ; pas de calcul nouveau | Phase 7 |
| D5 | Références manquantes | ✅ **Recherche web par l'agent** + complétion de `references.bib` (priessen2021, Sullivan 1927, Saeman 1951, Chatterjee) + intégration de Mellmann au texte ; validation de l'étudiant | Phase 1.2–1.3 + Phase 6 |

**Risques & parades** : réintroduction d'anciens chiffres lors des réécritures → parades 2.4 + checklist 12.3 ; divergence mémoire/code lors d'une future évolution du code → documenter la convention `step` côté code (4.3) ; temps de compilation du document (21 Mo, nombreuses figures) → compiler après chaque phase, jamais deux phases d'affilée sans contrôle.
