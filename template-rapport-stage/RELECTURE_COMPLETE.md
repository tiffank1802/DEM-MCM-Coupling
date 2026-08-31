# Plan de relecture complète du mémoire

*Posture : tuteur de stage qui lit la version finale avec un regard
critique. Quatre axes : clarté, concordance, fidélité, absence de
traces d'utilisation d'IA. Chaque axe possède des critères objectifs,
un ordre de passage et un livrable de sortie. En fin de document :
constats du passage 1 (déjà exécuté) avec actions.*

## 0. Principes directeurs

- **Une seule source de vérité** : `main.tex` pour le texte,
  `postprocessing/resultats_*.txt` + figures régénérées pour les
  chiffres, `PLAN_*.md` pour la traçabilité des décisions.
- **Tout critère coché doit pouvoir être démontré** (sortie de génération,
  grep, diff) — pas de « globalement ça va ».
- Relecture en **passages thématiques séparés**, pas une seule lecture
  globale : on ne voit jamais tout à la fois.

## 1. CLARTÉ

Objectif : un lecteur externe comprend chaque section sans relire trois fois.

Checklist (ordre de lecture : Introduction → Conclusion, puis notes marginales) :
1. Chaque section commence par annoncer ce qu'elle fait et finit par dire ce
   qu'on en retient (squelette imposé par le template).
2. Chaque figure citée est commentée dans le texte (pas d'« orphan figure ») ; chaque légende est auto-suffisante (métrique, paramètres, unités).
3. Vocabulaire constant : *teneur* (pas « teneur/concentration » en alternance),
   *pas de sortie DEM* (pas « pas de temps » pour les instants sauvegardés),
   *cellule* (pas « maille »), *bloc d'apprentissage*.
4. Notations : $\tau$, $\mathrm{start}$, $\mathrm{step}$, $\mathrm{NLT}$, $dt$
   définis une seule fois (§3.3 + nomenclature), utilisés toujours à l'identique.
5. Phrases plafonnées à ~3 lignes ; les paragraphes de plus de 12 lignes sont
   marqués pour restructuration.
6. Pas de redondance entre §7 (résultats) et §8 (discussion) : le résultat
   brut chez l'un, l'interprétation chez l'autre — toute phrase de discussion
   présente dans §7 est déplacée.

Outil : noter au crayon/DMS chaque point d'arrêt (lecture à voix haute conseillée
pour les paragraphes longs).

## 2. CONCORDANCE

Objectif : le document ne se contredit jamais.

Checklist automatisée (scripts) + lecture ciblée :
1. **Références croisées** : `\\ref` sans `\\label`, labels définis deux fois,
   figures citées avant leur définition — script déjà écrit (env/ref checker).
2. **Texte ↔ tableau ↔ figure** : pour chaque valeur de paramètre apparaissant
   en trois endroits (§3.3 table, captions, annexes), grep systémique :
   `start`, `step`, `NLT`, `dt`, `tau`, nombres de cellules, dates des matrices
   inhomogènes.
3. **Corps ↔ annexes** : les conclusions annoncées dans le corps (« conformité
   acquise dès dt=157 ») doivent être démontrées dans l'annexe pointée ; les
   renvois « annexe B » post-réordonnancement vérifiés (annexe:parametres
   regroupe τ/NLT/start/step/dt — ordre relu).
4. **Nomenclature** : chaque symbole de la nomenclature est utilisé ; chaque
   symbole du texte y figure.
5. **Sommaire ↔ titres** : titres hiérarchisés sans trous (section/subsection),
   pas deux sous-sections de noms homonymes.
6. **Unités** : SI partout via `\SI`, séparés décimaux cohérents (virgule),
   chiffres significatifs homogènes dans une même grandeur.

## 3. FIDÉLITÉ

Objectif : aucun nombre, aucune « observation » n'est invérifiable.

Protocole :
1. **Audit chiffre par chiffre** : extraire de main.tex tous les nombres de
   résultats (mission regex, liste ci-dessous) ; pour chacun : source
   (fichier de résultats / figure), verdict ✓/✗/approx.
   - Cibles existantes : `resultats_etudes_nlt_dt_tau.txt`, sorties console
     des drivers, `etude_dt_table.txt`, tables RSD/teneur.
2. **Pipeline canonique par figure** : chaque figure PNG est associée au
   script + commit qui l'a produite (table de correspondance déjà amorcée
   dans ETUDES_NLT_CYLINDRIQUE_DT_TAU.md §Reproduction ; l'étendre à toutes).
3. **Régénération aveugle** : relancer la chaîne complète à la fin
   (une commande documentée) et comparer les PNG sortants aux versions
   embarquées (diff pixel) — toute figure qui diffère est soit
   involontaire, soit manuellement bricolée : à investiguer.
4. **Affirmations qualitatives** : chaque « on observe que », « la matrice
   montre que » se rattache à une ligne de matrice/figure précise citée en
   commentaire LaTeX (% source: ...).
5. **Conformité légale du stage** : les dates, l'entreprise, le matériel
   DEM (§3.1, §3.2) concordent avec la fiche descriptive du stage.

Point de vigilance connu (constats passage 1, §6 ci-dessous) : les instants
des matrices inhomogènes.

## 4. ABSENCE DE TRACES D'UTILISATION DE L'IA

Quatre familles de traces à éliminer :

1. **Style** : tournures génériques (*« il convient de noter »*, *« il est
   important de »*, *« notons que »*), incises à tirets cadratins en surabondance
   (**93 occurrences de `---` : à varier** — remplacer une partie par des
   incises à parenthèses ou réécrire la phrase), parallélismes parfaits en
   triades répétées, adverbes d'alourdissement en tête de phrase.
2. **Contenu trop « parfait »** : chaque section qui aurait exactement les
   mêmes proportions, transitions stéréotypées (« Dans un premier temps... dans
   un second temps... enfin » déjà utilisé 2× — en garder au plus une),
   conclusions qui répètent mot pour mot l'introduction.
3. **Fichiers et légendes** : métadonnées des PNG (clé « Software »
   matplotlib laissée par défaut — acceptable, c'est un outil normal d'un
   stagiaire ; NE PAS blanchir), noms de fichiers immédiatement compréhensibles.
4. **Fonctionnement** : cohérence complète
   (quel que soit l'outil ayant servi, le texte final doit être adopté par
   l'étudiant phrase par phrase — c'est la garantie ultime ; la relecture
   avec lui est une étape obligatoire du processus : chaque modification
   proposée est relue et validée, ou reformulée de sa main).

Sortie de cet axe : liste des passages à reformuler + reformulations proposées,
validées par l'étudiant.

## 5. ORDRE DE PASSAGE CONSEILLÉ

| Passage | Contenu | Sortie |
|---|---|---|
| 1 | Structure & clarté (lecture linéaire rapide, notes marginales) | liste § à restructurer |
| 2 | Concordance (scripts + grep des paramètres + refs croisées) | table des contradiction |
| 3 | Fidélité (audit des nombres + pipeline par figure) | audit ✓/✗ annoté |
| 4 | Style & IA-traces (marqueurs, density dashes, reformulations) | diff stylistique |
| 5 | Orthographe/typographie finale (lecture lente, règles du template) | commit final |

Chaque passage produit son propre commit (« relecture passage N »), sauvegarde
préalable (`main_backup_avant_relecture_AAAA-MM-JJ.tex`).

## 6. PASSAGE 1 — CONSTATS DÉJÀ EXÉCUTÉS (31/08/2026)

Vérifications automatiques exécutées ce jour :

| Contrôle | Résultat |
|---|---|
| Environnements figure/table/equation/itemize/tabular/minipage | équilibrés |
| `\\includegraphics` | toutes les images existent |
| `\\ref` sans `\\label` | aucune |
| Marqueurs de style IA (12 tournures testées) | 0 occurrence |
| Occurrences de « repère » | 0 (purge v6 complète) |
| Longueur corps (mots, figures, tables) | ~11 050 mots, 25 fig, 5 tabl. ≈ 31 p. |

Anomalies relevées, par sévérité :

| # | Sévérité | Localisation | Constat | Action |
|---|---|---|---|---|
| F1 | **HAUTE** | §7.3 + caption fig:rsd_homogene_inhomogene + annexe step | Le texte cite 5 matrices apprises à 1,57 / 14,13 / 26,69 / 39,25 / 51,81 s (step=7τ) — or `etudes_librairie.py`, générateur de la figure actuelle, utilise `STEP_INH = 6τ` → départs réels 1,57 / 12,56 / 23,55 / 34,54 / 45,53 s | Régénérer la figure §7.3 + tab:inhomogene avec `step=7τ` (5 blocs tiennent : dernier départ 51,81+1,57 ≤ 58,4 s) ou aligner le texte sur 6τ ; décision au moment de la régénération finale |
| F2 | MOYENNE | captions §7.2.2 (fig:comparaison_*, tab:ecart_rsd), fig:teneur_* | `dt = 8 pas` (vrai aujourd'hui ; migrations dt=τ en attente) | Synchronisées au run final (PLAN_MIGRATION_DT_TAU.md §3) ; risque extrême de l'oublier → checklist de clôture |
| F3 | ~~MOYENNE~~ | §7.3 « (NLT = 2, dt = 8 pas) » | formulation confuse (NLT devrait évoquer 5) | **corrigée** (« $dt = 8$ pas » seul) |
| F4 | ~~MOYENNE~~ | annexe matrices dt8 : « paramètre standard de toutes les chaînes (tableau) » | faux depuis le passage du tableau à dt=τ | **corrigée** |
| F5 | BASSE | corps entier | 93 tirets cadratins `---` | passage 4 : en varier ~1/3 |
| F6 | BASSE | §6.2 fig:schema_parametres caption vs texte table (valeur step « τ / 7τ ») | deux conventions d'écriture cohabitent (blocs « tuilés » vs « espacés de sept tours ») : clarté → formulation unique | passage 2 |
| F7 | INFO | scripts : 2ᵉ et 3ᵉ `cfg_inh` d'`etudes_librairie.py` héritaient du nouveau `dt=τ` par défaut (NaN bloc 1) | **corrigé** (dt=8 explicite ×3) | — |

### Checklist de clôture (à cocher à la fin de la régénération finale)

- [ ] figures teneur ×4, nombre ×12, comparaison RSD + teneur, tables
      4/inhomogène, espèces régénérées sous `config_for(dt=TAU)` + captions
      passées à `dt = \tau`
- [ ] F1 tranchée (texte OU figure aligné) et `NLT_INH` recompté
- [ ] valeurs RSD/teneur/paliers citées dans §7, §8, §8.1 (0,096–0,111 ;
      0,05/0,35/0,17/0,11 ; « divisé par plus de deux ») revérifiées sur
      sorties fraîches
- [ ] relance des 4 contrôles automatiques (§6 tableau du haut)
- [ ] sauvegarde `main_backup_*.tex` avant commit final
