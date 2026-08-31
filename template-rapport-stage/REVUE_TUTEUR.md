# Relecture du mémoire — note de tuteur de stage

**Document relu :** `main.tex` (rapport de stage — *Modélisation des phénomènes de ségrégation granulaire à l'aide des chaînes de Markov inhomogènes*)
**Contrôles croisés effectués avec :** `figures/*.txt` (données générées), `references.bib`, `main.bbl`, code `dem_mcm_coupling/`

---

## 1. Appréciation générale

Le mémoire est **bien construit dans son architecture** : la problématique (influence du maillage spatial et de la dépendance temporelle des paramètres sur une chaîne de Markov apprenant sur une référence DEM) est clairement posée en introduction, et le plan suit un entonnoir logique correct : contexte → matériel/méthodes → résultats → discussion/conclusion, avec un renvoi systématique des justifications quantitatives en annexe. C'est une bonne pratique.

**Points forts de la démarche scientifique :**

- Chaque paramètre du modèle ($\tau$, $NLT$, $start$, $step$, $dt$) fait l'objet d'une étude de sensibilité dédiée, adossée à une grandeur de comparaison issue de la littérature (erreur relative de Frobenius sur la matrice, à la suite de Doucet *et al.*). C'est une méthodologie sérieuse.
- Le choix de conserver les *NaN* dans la matrice de transition (plutôt que de les remplacer par des zéros dépourvus de sens physique) et d'en faire un critère de conformité est une décision saine et bien expliquée.
- La séparation des matrices par espèce est physiquement motivée *et* étayée quantitativement.
- Les tableaux de résultats principaux (comparaison des méthodes, homogène vs inhomogène) sont **cohérents avec les fichiers de données générés** — le lien calcul → rédaction fonctionne.

Le travail est donc réel, structuré, et la méthode de raisonnement est globalement la bonne. **En revanche, le document n'est pas prêt pour diffusion** : il subsiste des incohérences numériques entre sections, des contradictions internes, une section Discussion dégradée, une référence bibliographique et une référence de figure non résolues, et plusieurs traces de travail (commentaires, anciens résultats commentés) qui donnent une impression de version intermédiaire. Le détail suit, par ordre de priorité décroissante.

---

## 2. Problèmes majeurs de cohérence (à corriger impérativement)

### 2.1 — Bibliographie : citation non résolue ⚠️ bloquant

- `\cite{priessen2021}` (§ *Contexte et état de l'art*, paragraph tambours tournants) **n'existe pas dans `references.bib`** → la citation apparaîtra en **[?]** dans le PDF. Le fichier `main.bbl-SAVE-ERROR` présent dans le dossier signale d'ailleurs un historique de problème biber.
- Les modèles historiques **Sullivan et al. (1927), Chatterjee, Saeman (1951)** sont cités dans le texte *sans aucune référence bibliographique*. Un jury le remarquera immédiatement.
- `mellmann2001` est passé en `\nocite` : il apparaît en bibliographie sans être exploité. C'est dommage, c'est *la* référence sur les régimes d'écoulement transversal en tambour rotatif (voir §3.4 — elle devrait être citée explicitement).

**Action :** créer l'entrée `priessen2021`, ajouter les références classiques (ou supprimer les noms), citer Mellmann dans le texte, recompiler `xelatex → biber → xelatex × 2`, et supprimer `main.bbl-SAVE-ERROR`.

### 2.2 — Référence croisée cassée ⚠️ bloquant

Le texte (§ *Méthodes de discrétisation*, paragraph « Changement de repère ») renvoie à `figure~\ref{fig:repere_avant_apres}`, mais **la figure est commentée** : le label n'existe plus → « **figure ??** » dans le PDF. En outre, un commentaire de travail en français est resté dans le source (« *enlève cette figure … et remplace la par celle du schéma cartésien…* »), de même que « *Modifie cette figure pour qu'il ait une différence entre les celleules…* » après la figure `fig:schema_voronoi`. Ces notes doivent disparaître avant diffusion.

### 2.3 — Incohérences numériques entre sections (deux versions des résultats cohabitent)

C'est **le point le plus grave de cohérence scientifique**. Les valeurs citées dans le texte ne correspondent pas toujours aux fichiers de données générés.

| Lieu | Valeurs du `.tex` | Valeurs des fichiers de données |
|---|---|---|
| Tableau `tab:etude_especes` (annexe E), « sans distinction » | 0,370 / 0,146 / 0,215 / 0,202 | `etude_especes_table.txt` : **0,398 / 0,145 / 0,234 / 0,204** |
| Idem, « avec distinction » | 0,078 / 0,098 / 0,083 / 0,103 | **0,0956 / 0,1025 / 0,0948 / 0,1107** |
| Texte annexe E : « facteur 1,5 à 4,7 (0,215 contre 0,083) » | — | ratios réels : **1,42 à 4,16** ; Voronoï : 0,2336 → 0,0948 |

La ligne « avec distinction » de l'annexe E **contredit directement** le tableau `tab:ecart_rsd_methodes` du corps du texte (0,096 / 0,103 / 0,095 / 0,111 — celui-ci étant conforme à `comparaison_methodes_table.txt`), alors que les deux tableaux prétendent décrire le même modèle homogène dans la même configuration ($nlt=2$, $start=1{,}57$, $step=\tau$, $dt=8$).

De même, le paragraphe commenté de la § *Apport des chaînes inhomogènes* cite des valeurs (0,083 → 0,027 pour Voronoï, etc.) qui ne correspondent **ni** à `inhomogene_table.txt` (0,0948 → 0,0405) **ni** aux valeurs actuelles. Ces restes d'une ancienne campagne de calcul doivent être supprimés ou mis à jour : un lecteur (ou un jury) qui compare deux tableaux y verra une contradiction.

**Action :** régénérer/saisir tous les tableaux **depuis les fichiers de données actuels**, et purger les paragraphes commentés contenant d'anciens chiffres.

### 2.4 — Contradiction zone active / zone passive (découpage physique)

Dans la sous-section *Découpage physique* :

- il est d'abord écrit : « cellules 0, 1, 2, 4, 6, 7 → **zone passive** ; cellules 3, 5, 8, 9 → **zone active** » ;
- puis, deux paragraphes plus loin : « forte concentration en petites particules dans les cellules de la **zone passive** (0, **8 et 9**) … faible concentration dans les cellules **1 et 7**, situées près de la surface libre dans la **zone active** ».

Les cellules 8, 9 et 1, 7 ne peuvent pas appartenir aux deux zones à la fois : l'affectation est inversée entre les deux passages. Vérifier sur `figures/pv_cellules_physique.png` et uniformiser.

Dans le même registre (découpage cartésien) : « probabilité diagonale élevée … pour les cellules adjacentes aux parois (**cellules 1 et 9**) ». Avec 10 bandes numérotées de 0 à 9 et la cellule 0 « près du bord supérieur » deux paragraphes plus loin, les cellules extrêmes sont **0 et 9**, pas 1 et 9. À vérifier contre la figure et la numérotation définie en § Méthodes.

### 2.5 — Choix de NLT : contradiction ouverte entre résultats et annexe

- Les résultats du corps du texte utilisent systématiquement **NLT = 2** ;
- l'annexe B conclut que la matrice converge vers « une dizaine de blocs » (erreur relative 0,033 à NLT=8, 0,021 à NLT=12 pour les petites particules) et qualifie même NLT=18 de « **seuil de convergence** » ;
- or l'annexe **ne dit jamais quelle valeur est retenue** (« La valeur retenue résulte d'un compromis… » → laquelle ?), et le tableau `tab:parametres` indique vaguement « 1 à 18 ».

Question certaine du jury : *pourquoi construire les modèles avec NLT=2 alors que votre propre étude de sensibilité recommande ~10 ?* Il faut soit justifier explicitement le choix NLT=2 (compromis coût/fenêtre d'apprentissage, chiffré), soit refaire les résultats avec la valeur recommandée. **Point connexe :** chaque matrice de la chaîne inhomogène est apprise sur un seul tour (équivalent NLT=1) — or l'annexe montre qu'à NLT=1 l'erreur relative des petites particules vaut 0,134. Ce compromis bruit/adaptativité doit être discuté, au moins en une phrase.

### 2.6 — La Discussion est dégradée (et contient des erreurs de sens)

La section *Discussion* ne comporte que deux paragraphes, dont le français est rompu et le sens parfois inversé :

- « *les chaînes **homogènes** n'enrichissent pas énormément la prédiction* » → il faut lire « **inhomogènes** » (le résultat du tableau `tab:inhomogene` est que l'apport est marginal pour les maillages géométriques, important pour les statistiques) ;
- « *Les méthodes de discrétisation spatiale statistiques … enrichissent considérablement la prédiction* » → reformulation nécessaire (ce sont les **chaînes inhomogènes** qui enrichissent la prédiction **des modèles issus de maillages statistiques**) ;
- « *précision la prédiction de Markov … est sensiblement pareille* », « *l'état du mélange … diffère que l'on choisit une méthode ou une autre* » → phrases à réécrire entièrement.

Or la « vraie » discussion existe : elle est **commentée** sous le texte visible (dualité espèces, rôle du repère, pertinence de la teneur locale, articulation maillage ↔ inhomogénéité). Ces paragraphes commentés sont le contenu le plus abouti de cette section. **Réintégrer cette matière** (en mettant ses chiffres à jour, cf. §2.3), c'est elle qui fera la différence devant le jury.

À ce propos, le plan annoncé en introduction (contexte → méthodes → résultats → conclusion) **omet la section Discussion** — l'y ajouter.

### 2.7 — Le résumé promet un résultat non démontré

Le résumé (et l'abstract) affirme : « *un vecteur d'état fondé sur la teneur locale … s'avère bien plus discriminant qu'un simple comptage pour quantifier la ségrégation* ». Or **aucun résultat ne compare quantitativement les deux** : la teneur locale est définie (et illustrée en annexe F), utilisée comme *indicateur* (courbes, RSD), mais jamais confrontée au comptage comme *vecteur d'état* du modèle. De plus, la nomenclature qualifie $C(t_k)$ de « vecteur d'état », alors que le corps du texte dit explicitement : « le modèle de Markov est construit sur ce vecteur de **nombre de particules** … on construira **en fin de chaîne** un vecteur de teneur ». Soit ajouter la comparaison chiffrée, soit adoucir le résumé (« … offre un indicateur directement relié à la ségrégation ») et corriger la nomenclature (la teneur est un **post-traitement**, non l'état de la chaîne).

### 2.8 — Définition du RSD à préciser (cohérence texte ↔ calcul)

L'équation `eq:rsd` définit $\bar{c}$ comme « la teneur moyenne **globale** en petites particules ». Or le code (`analyze_results.py`, `compute_dem_rsd`) calcule `c_active.std() / c_active.mean()`, c.-à-d. **la moyenne arithmétique des teneurs des cellules occupées** (cellules vides exclues). Ce n'est pas la même chose :

- avec la fraction globale $350/1030 \approx 0{,}34$, le RSD serait borné par $\sqrt{(1-\bar c)/\bar c} \approx 1{,}39$ — valeur **dépassée** dans le tableau `tab:etude_start_rsd` (2,032 !), ce qui prouve que c'est bien la moyenne des teneurs qui est utilisée ;
- le traitement des cellules vides ($0/0$) n'est nulle part mentionné.

Expliciter la définition exacte (moyenne non pondérée sur cellules occupées) dans le texte et l'équation — cela change l'interprétation et lève l'apparent paradoxe des RSD > 1,4. Conséquence : préciser aussi que la « valeur moyenne d'environ 0,4 » citée pour le cylindrique est une moyenne de teneurs cellulaires, non la fraction globale 0,34.

**Point adjacent :** l'annexe A.3 affirme que le RSD initial « part de valeurs (0,9 à 1,1) et culmine jusqu'à 1,2 » — **contredit par son propre tableau** (`tab:etude_start_rsd` : 1,423 / 2,032 / 0,707 / 1,343 à $t=0$ ; 2,032 à $t=0{,}5$ s). Corriger la phrase en citant les vraies valeurs.

### 2.9 — Équations et notations : erreurs à corriger

1. **Annexe D (step).** « le bloc $k$ démarre à l'instant $start + k\,(step + \tau)$ » est **incompatible** avec la phrase qui suit (« les blocs se succèdent sans recouvrement ni trou ») et avec le code (`run_sweep.py` : `step` = distance entre débuts de blocs, valeur 157 = $\tau$ → démarrage en $start + k\cdot step$). Avec la formule écrite et $step=\tau$, les blocs seraient espacés de $2\tau$ avec des trous d'un tour. Même problème pour « fenêtre totale $NLT \times (step+\tau)$ ». Corriger la formule (ou redéfinir $step$ comme l'intervalle *libre* entre blocs, mais alors la coordonner partout, y compris à la figure `fig:schema_parametres` dont la légende « blocs successifs décalés de $step$ » est, elle, correcte).
2. **Éq. de la matrice de transition.** La somme sur $n=1..NLT$ suppose **une paire d'instants par bloc**, alors que le paramètre $dt$ crée **plusieurs paires à l'intérieur de chaque bloc** ($\tau/dt \approx 20$ paires par bloc à $dt=8$). L'équation ne reflète donc pas la méthode décrite : écrire la double moyenne (blocs × paires internes) ou définir l'indice $n$ comme parcourant *toutes* les paires. Harmoniser aussi la nomenclature : $NLT$ y est « le nombre d'instants d'apprentissage », mais « nombre de blocs » ailleurs.
3. **Éq. `eq:def_teneur_locale`.** Décalage d'indice : la première composante est $c_0$ mais utilise $N_1^a, N_1^b$ → $c_0 = N_0^a/(N_0^a+N_0^b)$. Et préciser que $a$ = petites, $b$ = grandes (les espèces $a$/$b$ restent abstraites alors que tout le reste du document parle de « teneur en petites particules »).
4. **Intervalles d'entiers** : $i \in [0, N-1]$ devrait être un intervalle entier ($\llbracket 0, N-1 \rrbracket$).
5. **Nomenclature** : incomplète (manquent $\phi_p$, $l_p$, $\mathbf{v}_p$, $\mathbf{z}_p$, $\boldsymbol{\mu}_k$, $J$, $m_i$, $I_i$, $\mathbf{F}_{n,ij}$, $\mathbf{F}_{t,ij}$, $\mathbf{M}_{r,ij}$, $\mathbf{g}$…), $S(t_k)$ sans gras alors que c'est $\mathbf{S}$ dans le corps, « represente » sans accent (×2), « nombre de cellules **issue** » → *issu*, et les variables multi-lettres ($start$, $step$, $dt$, $NLT$) devraient être composées en romain (`\text{}`), pas en italique mathématique.
6. **Unités et casse** : « $D = 2R = 0.1 m$ » → `\SI{0.1}{m}` ; « discretisation » sans accent dans plusieurs légendes ; « $nlt$ » / « NLT » à uniformiser ; « table~\ref » → « tableau » ; « bi-disperses » / « bidisperses » ; « pas d'extraction » (résumé) / « pas de sortie » (corps).

### 2.10 — Petites contradictions internes à lever

- **Cartésien : « volume constant » vs « faible volume ».** Les cellules sont définies comme des parallélépipèdes « de volume constant », puis la prédiction bruitée de la cellule 0 est expliquée par « son faible volume ». Reformuler : c'est la **faible population effective** de cette bande marginale (faible intersection avec le lit) qui produit le bruit, pas son volume géométrique.
- **§ intro des Résultats** : « les découpages statistiques nécessitent un temps d'observation réduit … contrairement aux découpages géométriques qui exigent une **phase d'apprentissage plus longue** (annexe E) ». L'annexe dit autre chose : dans le repère du tambour, les maillages géométriques ne deviennent conformes pour **aucune** durée d'apprentissage (problème *structurel*, pas *statistique*), et une fois dans le repère du lit ils sont conformes dès $dt=157$ (une seule paire !), au même titre que les statistiques — confirmé par `etude_dt_table.txt` (zéros partout). La phrase doit être réécrite : l'avantage des maillages statistiques est de réaliser cette adéquation **automatiquement, sans changement de repère**.
- **Annexe F (teneur)** : « $c_i = 0{,}5$ pour un mélange **équimassique en nombre** » est un oxymore, et est incohérent avec le mélange étudié (350/680 → teneur globale 0,34). Dire « exemple illustratif à 50/50 en nombre » ou aligner sur 0,34.
- **Mécanisme de ségrégation** : les petites particules sont décrites « se concentrant au cœur **et contre la paroi** » (figure `fig:maillage_melangeur`, § résultats intro : « percolant … vers la paroi et le cœur »), mais ailleurs « les petites particules migrent vers le **cœur** du lit tandis que les grandes remontent en surface ». Trancher/articuler le mécanisme (percolation vers le fond de la couche active puis accumulation au cœur).
- **Cylindrique non aligné méthodes ↔ résultats** : la méthode décrit un maillage $(r,\theta,z)$ complet (avec deux modes de segmentation radiale, dont l'aire constante « préférée »), mais les résultats utilisent « 10 **secteurs angulaires** » purs. Préciser dans les méthodes la configuration réellement retenue (aucune partition radiale ni axiale ?), de même pour le cartésien (10 bandes selon un seul axe du repère du lit, alors que la méthode décrit une segmentation $x,y,z$ lexicographique).
- **Bullet « physique » de la comparaison RSD** : « écart quasi constant » *et* « niveau absolu proche de la DEM » sont en tension : un écart moyen 0,111 dominé par le transitoire puis deux courbes qui se rejoignent à ≈ 0,1 n'est pas un « écart constant ». Relire les courbes et distinguer explicitement **biais de régime établi** et **erreur de transitoire**.
- **Date** : soutenance 02/09/2026 alors que le stage court jusqu'au 17/09/2026 — vérifier (et compléter le jury « M./Mme --- » avant impression).

---

## 3. Vigilances scientifiques (questions que le jury posera — anticiper)

1. **Normalisation des caractéristiques du découpage physique.** Le k-moyennes opère sur $(x, y, z, \lVert v\rVert)$ : position en m (~0,05) et vitesse en m/s (~0,2) n'ont ni unité ni échelle communes. Sans normalisation, la quatrième caractéristique domine ou s'écrase selon les ordres de grandeur. Le code prévoit une option `normalize` (min–max) — le mémoire **doit dire ce qui a été fait**, et donner une unité/sens au seuil de variance $10^{-4}$ (variance dans quel espace ?). C'est la question technique n° 1.
2. **« Consolidation par moyenne » des 10 initialisations k-means.** Moyenner 10 partitions n'est pas trivial (problème de permutation des labels entre exécutions). Est-ce une moyenne des centroïdes après appariement ? Le meilleur des 10 au sens de l'inertie ? Une phrase d'explication est indispensable.
3. **Validité markovienne.** Une phrase de justification que $\tau = 1$ tour rend l'hypothèse « sans mémoire » raisonnable (décorrélation de la position à l'échelle du tour) renforcerait le § formalisme — d'autant que vous disposez des données pour la tester.
4. **Régime d'écoulement.** Caractériser le régime (nombre de Froude $\omega^2 R/g \approx 0{,}08$ → régime roulant/*rolling*) et citer **Mellmann (2001)** ici : cela assoit le choix physique de $\tau$ = un tour, et cela « paie » le `\nocite`.
5. **Chaîne inhomogène : bornes temporelles.** Dernier bloc appris vers $t \approx 45{,}5$–47 s : quelle matrice propage la prédiction jusqu'à 60 s ? Et depuis quel instant la chaîne est-elle propagée (0 ? $start$ ?) — le dire explicitement pour les deux types de chaînes.
6. **« Ajustement dynamique » (bullet Voronoï).** Une chaîne **homogène** est une propagation déterministe $\mathbf{P}^n$ : elle ne « s'ajuste » pas à la référence. Une pente non nulle tardive signifie plutôt que la relaxation vers l'état stationnaire du modèle n'est pas achevée à $t=60$ s. Reformuler.
7. **Comparabilité des RSD entre maillages.** Les plateaux DEM diffèrent fortement selon le maillage (≈ 0,66 cartésien, 0,45 cylindrique, 0,22 Voronoï, 0,11 physique à $t=10$ s) : le RSD compare valablement Markov vs DEM **au sein d'un même maillage**, mais l'écart de niveau entre maillages reflète la métrique autant que la physique. Un rappel en Discussion éviterait une sur-interprétation — c'est d'ailleurs le sens (mal exprimé) de la phrase « l'état du mélange diffère selon la méthode » (§2.6).
8. **Unicité du nombre de cellules (10).** Toutes les méthodes sont comparées à $N=10$ : annoncer (au moins en perspective) l'étude d'influence de $N$ — la perspective commentée la mentionnait, la réactiver.
9. **Études annexes hétérogènes.** L'étude `start` seconde partie change d'espèce (grandes), de maillage (Voronoï) *et* de métrique (RMS en nombre de particules) par rapport à la première. Une phrase de justification de ces choix (pourquoi pas le RSD de teneur partout ?) renforcerait la cohérence d'ensemble.
10. **Légendes auto-suffisantes.** « Comparaison homogène vs inhomogène » (`tab:inhomogene`) ne précise ni la métrique (écart moyen $|{\rm RSD}_M - {\rm RSD}_{DEM}|$) ni la configuration. Idem pour plusieurs figures de matrices.

---

## 4. Clarté et langue — corrections prioritaires (non exhaustif)

**Français / typographie :**
- « é**mettre** » (intro) ; « entra**î**nés » ; « entourée en bleu**e** » → *bleu* ; « et **et** un vecteur » (§ formalisme) ; « corresp**o**dante » ; « RSD … **our** les quatre méthodes » → *pour* (légende `fig:rsd_homogene_inhomogene`) ; « **cette** écart » → *cet* ; « **quelque soit** » → *quelle que soit* (×2, discussion + conclusion) ; « sensiblement pareil » → *pareille* ; « privi**li**gier » ; « pourraient être manquées ou au contraire sur-représentées » ok ; « statistiques(voronoï » → espace + majuscule + tréma ; « **Details** sur le nombre de particules » (annexe G, titre) ; « Découpage **voronoi** » (annexe G) ; « Nommenclature » (commentaire) ; espaces avant « . » et « , » (remerciements : « applications . », « modèles,et », double espace « stage,  pour »).
- **Remerciements** : la première phrase énumère les encadrants sans verbe (« Guillaume Dumazer pour sa vision… Éric Serris pour ses conseils… Cendrine Gatumel pour ses remarques les plus intéressantes ») — restructurer (« Je remercie Guillaume Dumazer pour…, Éric Serris pour… » etc.) et éviter le superlatif maladroit. Les remerciements à l'encadrant école et aux laboratoires sont actuellement **commentés** : les réactiver serait convenable.
- **Phrases rompues à reprendre** : § *Matériau* — la description de la figure `fig:schema_tambour` (« dans le plan x–y —(figure gauche) et de 3/4 (figure droite)— précise les dimensions… ») est inaboutie ; § formalisme — « regroupe les particules dans des cellules …, comme le montre la figure~…, permettant de suivre… » (proposition participiale mal rattachée) ; § Résultats — « Le déséquilibre de population entre cellules **observe** reflète » ; § inhomogènes — « Au premier tour, puis après sept tours de rotation du tambour ; soit au total cinq modèles » (donner les instants : 1,57 / 12,56 / 23,55 / 34,54 / 45,53 s).
- **Terminologie** : « prédite par la DEM » / « la prédiction DEM » (§ physique et § inhomogènes) — la DEM est la **référence**, elle ne « prédit » pas. Utiliser « référence DEM ». « internals sectionnels » → traduire (« éléments internes », « cloisons »). Style : « L'on observe » (lourdeur), « table »/« tableau ».

**Forme LaTeX :** doublons `\usepackage{booktabs|tabularx|float}` (×2) ; penser à purger les larges blocs commentés (ou les déplacer dans `main_v2_backup.tex`, qui existe déjà à cet effet) afin que le manuscrit relu par le jury ne contienne ni anciens chiffres ni notes de travail.

---

## 5. Plan d'action recommandé (ordre)

1. **Compiler propre** : entrée `priessen2021`, références Sullivan/Saeman, suppression `\nocite` → citation Mellmann ; réparer la référence `fig:repere_avant_apres` ; supprimer les commentaires TODO. Vérifier zéro « ?? » et zéro « [?] » au PDF.
2. **Unifier les chiffres** : régénérer `tab:etude_especes` et tous les passages chiffrés depuis `figures/*.txt` ; purger les paragraphes commentés obsolètes (§2.3).
3. **Trancher NLT** : écrire explicitement la valeur retenue et sa justification chiffrée, ou recalculer avec la valeur convergée (§2.5).
4. **Corriger les contradictions factuelles** : zones actives/passives du physique ; cellules 0/1–9 ; « volume constant » vs « faible volume » ; coeur/paroi ; texte vs tableau du transitoire (§2.4, §2.8, §2.10).
5. **Réécrire la Discussion** en réintégrant la matière commentée (mise à jour des chiffres) ; ajouter la Discussion au plan d'introduction ; harmoniser Conclusion/Perspective (la perspective commentée — « peut-on se passer de l'inhomogène si le maillage est bon ? » + étude en $N$ — est meilleure que celle actuellement visible).
6. **Préciser les définitions** : RSD ($\bar c$, cellules vides), équation de la matrice (double moyenne / $dt$), formule de $step$, indice de la teneur, configuration réelle des maillages cartésien (bandes 1D) et cylindrique (secteurs seuls) (§2.8, §2.9, §2.10).
7. **Aligner résumé/abstract** sur ce qui est réellement démontré (§2.7).
8. **Documenter le découpage physique** : normalisation des caractéristiques, sens du seuil $10^{-4}$, procédure de « consolidation par moyenne » des 10 initialisations (§3.1–3.2).
9. **Passe de langue complète** (§4), complétion du jury et vérification des dates.

---

*Verdict de tuteur : un travail de fond sérieux et une méthodologie de justification remarquable pour le niveau visé ; le manuscrit souffre surtout d'être resté à mi-chemin entre deux versions (chiffres, discussion, nettoyage). Une fois les points 1 à 6 traités, le document sera solide pour la soutenance.*
