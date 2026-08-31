# Études NLT (cylindrique) et matrices dt = τ — 31/08/2026

Traitement des 5 remarques du tuteur (revue du 31/08/2026, compléments).

## 1. Étude d'influence du NLT pour le découpage cylindrique

- **Fait** : miroir exact de l'étude NLT du découpage physique
  (annexe B.2, `etude_nlt_erreur_relative`) appliqué à la méthode
  `cylindrique` : erreur relative de Frobenius de Doucet et al. 2008,
  `dt = 1` pas (matrices conformes, maximum de paires), référence NLT=18,
  NLT ∈ {1, 2, 3, 5, 8, 12, 18}, `start = 1,57 s`, `step = τ = 1,57 s`.
- **Résultats** (petites / grandes) : NLT=1 : 0,101/0,053 ; NLT=2 :
  0,081/0,049 ; NLT=5 : 0,042/0,029 ; NLT=8 : 0,025/0,024 ; NLT=12 :
  0,018/0,017.
- **Conclusion** : convergence claire ; NLT=2 à moins de 8 %/5 % de la
  référence → le choix NLT=2 du corps est valable aussi pour le
  cylindrique.
- **Figure** : `figures/etude_nlt_erreur_relative_cylindrique.png`
  (`\label{fig:etude_nlt_cylindrique}`, insérée en annexe NLT).

## 2. Matrices dt = τ en corps de texte, dt = 8 pas en annexe

- **Fait** : les 8 matrices du §7.2.1 sont apprises avec `dt = τ = 157`
  (une paire par tour, NLT=2 → 2 paires). Conformité vérifiée : **aucun
  NaN, chaque colonne somme à 1** pour les 4 méthodes × 2 espèces —
  cohérent avec l'annexe dt (conforme dès dt=157).
  Figures `figures/matrice_<cle>_especes_dt_tau.png`, mêmes
  `\label{}` que précédemment (`fig:matrice_cartesien`, etc.) → aucune
  référence cassée.
- **Commentaires du corps mis à jour** : seul le commentaire cylindrique
  changeait de fait (P55 : 0,45 → ≈0,33 petites / 0,40 grandes ;
  cellules 7–8 = retours les plus faibles 0,06–0,14). Les commentaires
  cartésien (P00=0,02, P11/P99≈0,37 grandes, P99≈0,36 petites),
  Voronoï (retours forts petites vers 0 et 9 : 0,55/0,47) et physique
  (diagonales grandes 0,49/0,48/0,47 sur cellules 3/5/6) restent
  inchangés — vérifiés sur les nouvelles matrices.
- **Annexe F** (`annexe:matrices_dt8`) : les 4 figures dt=8 pas
  déplacées avec légendes dédiées + avertissement de lecture
  (19 paires/bloc, fluctuation d'échantillonnage réduite).
- **Résolution statistique** (documentée en §7.2) : effectif de la
  cellule source aux 2 départs (t=157 et t=471) — quelques grains
  (bandes marginales cartésiennes) à ~130 (grandes, secteurs profonds
  cylindriques 5/9 : 112–126, 120–123 ; petites ~135 max bande 0).

## 3/4/5. Schémas corrigés

- **fig 8 cylindrique** : limites en traits forts clippés au cercle,
  10 secteurs numérotés 0–9 dans le repère du lit, Δθ, barycentre.
- **fig 9 cartésien** : 2 panneaux (repère du tambour / repère du lit),
  bandes selon la direction du lit (déjà intégré à la revue précédente,
  figure `repere_avant_apres.png` ; le schéma
  `schema_decoupage_cartesien.png` montre le découpage dans le repère
  du lit avec le changement de repère visualisé).
- **fig 11 Voronoï/physique** : (a) k-means sur positions seules,
  (b) k-means sur positions + ‖v‖ standardisé — cellules, frontières
  effectives (lignes de niveau du champ de labels) et centroïdes
  distincts.

## Reproduction

- Générateur : `postprocessing/driver_etudes_nlt_dt_tau.py`
  (banc sans GPU : `torch` factice à l'import + transcription numpy
  exacte de `compute_P_matrix_torch`, mêmes NaN) ;
  exécuté avec le venv hors dépôt `/home/user/.venv-etudes` ;
  journal complet des matrices : `postprocessing/resultats_etudes_nlt_dt_tau.txt`.
- Schémas : `postprocessing/figures_rapport_schemas.py`
  (fonctions `schema_cartesien`, `schema_cylindrique`, `schema_voronoi`,
  helper `_trace_voronoi`).
