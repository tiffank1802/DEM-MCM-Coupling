# -*- coding: utf-8 -*-
"""Batch d'éditions main.tex : purge repère, notes v6, dt déplacé en annexe."""
import sys

p = "/home/user/DEM-MCM-Coupling/template-rapport-stage/main.tex"
t = open(p, encoding="utf-8").read()

R = []  # (old, new)

# ---------- 1. INTRODUCTION : parenthèse de la liste des paramètres ----------
R.append((
"les études de sensibilité quantitatives des paramètres temporels ($\\tau$, $\\mathrm{NLT}$, $\\mathrm{start}$, $\\mathrm{step}$, $dt$) sont regroupées en annexe~\\ref{annexe:parametres}.",
"les études de sensibilité quantitatives des paramètres temporels du modèle sont regroupées en annexe~\\ref{annexe:parametres}."))

# ---------- 2. CONTEXTE : §DEM raccourci (sans 10e-6 s) ---------------------
R.append((
"""\\paragraph{La méthode des éléments discrets.} La simulation numérique des milieux granulaires repose très largement sur la méthode des éléments discrets introduite par Cundall et Strack~\\cite{cundall1979}. Cette approche déterministe considère chaque grain comme un corps rigide dont le mouvement est régi par le principe fondamental de la dynamique : à chaque pas de temps, le bilan des actions mécaniques --- forces de contact normales et tangentielles entre particules et entre particules et parois, pesanteur --- est intégré pour actualiser les accélérations, puis les vitesses et enfin les positions de l'ensemble des grains. Initialement développée pour les assemblées de disques et de sphères en géomécanique, la DEM s'est imposée comme l'outil de référence pour l'étude des procédés particulaires : elle donne accès à une information complète à l'échelle du grain (trajectoires, forces de contact, vitesses), au prix toutefois d'un pas de temps très faible (de l'ordre de $10^{-6}$\\,s) imposé par la raideur des contacts. Ce coût numérique restreint son emploi à des systèmes de taille modeste au regard des milliards de particules mises en jeu dans les procédés industriels, et motive le recours à des modèles réduits, opérant à une échelle plus grossière.""",
"""\\paragraph{La méthode des éléments discrets.} La simulation numérique des milieux granulaires repose très largement sur la méthode des éléments discrets introduite par Cundall et Strack~\\cite{cundall1979} : chaque grain est un corps rigide dont le mouvement suit le principe fondamental de la dynamique, le bilan des actions mécaniques --- forces de contact entre grains et avec les parois, pesanteur --- étant intégré à chaque pas de temps. Elle donne accès à une information complète à l'échelle du grain (trajectoires, forces de contact, vitesses) ; le coût numérique associé à l'intégration de contacts très raides (pas de temps très faible) restreint cependant son emploi à des systèmes de taille modeste au regard des milliards de particules des procédés industriels, et motive le recours à des modèles réduits opérant à une échelle plus grossière."""))

# ---------- 2bis. CONTEXTE : §tambours raccourci, sans « repère du lit » ----
R.append((
"""\\paragraph{Les tambours tournants : transport axial, temps de séjour et internals.} Les tambours tournants sont utilisés depuis longtemps en traitement des solides --- mélange, séchage, fours, broyage --- et leur dimensionnement repose sur des grandeurs clés : la rétention (\\textit{hold-up}), le temps de séjour moyen et sa distribution, ainsi que les limites de capacité. 
Les premières investigations systématiques du transport axial remontent à Sullivan et al.~\\cite{sullivan1927}, suivies des modèles empiriques fondés sur des nombres sans dimension de Chatterjee et al.~\\cite{chatterjee1983} ; le modèle géométrique de Saeman~\\cite{saeman1951}, basé sur le profil de hauteur de lit $h(x)$ et l'angle d'inclinaison, reste la référence pour la description du transport axial dans les tambours lisses. 
Prie{\\ss}en et al.~\\cite{priessen2021} ont récemment étendu ces travaux aux tambours équipés d'internals sectionnels, souvent utilisés industriels comme échangeurs de chaleur : la section transverse est divisée par des parois radiales du centre vers la paroi. Leurs mesures expérimentales de temps de séjour dans un tambour de laboratoire entièrement accessible optiquement montrent que le temps de séjour moyen augmente jusqu'à un facteur quatre avec le nombre et la longueur des sections, tandis que la capacité et la dispersion axiale diminuent (coefficient de dispersion axiale d'environ $1\\times10^{-6}$\\,m$^2$/s), avec une analogie qualitative avec le profil de lit de Saeman. Cette étude illustre que la structure de l'écoulement --- et non seulement la géométrie de l'enceinte --- gouverne le transport, ce qui motive directement la comparaison, menée dans ce mémoire, de maillages épousant le lit (repère du lit) et de maillages purement géométriques.""",
"""\\paragraph{Les tambours tournants : transport axial, temps de séjour et internals.} Le dimensionnement des tambours tournants --- mélange, séchage, fours, broyage --- repose sur la rétention, le temps de séjour moyen et sa distribution. Le transport axial y est décrit depuis les travaux pionniers de Sullivan et al.~\\cite{sullivan1927}, les modèles empiriques de Chatterjee et al.~\\cite{chatterjee1983} et le modèle géométrique de Saeman~\\cite{saeman1951} (profil de hauteur de lit), encore de référence pour les tambours lisses. Prie{\\ss}en et al.~\\cite{priessen2021} ont montré récemment, sur un tambour de laboratoire équipé d'internals sectionnels optiquement accessibles, que la présence de ces parois radiales multiplie le temps de séjour moyen (jusqu'à un facteur quatre) tout en réduisant la dispersion axiale. Ces travaux établissent que la structure de l'écoulement --- et non seulement la géométrie de l'enceinte --- gouverne le transport, ce qui motive la comparaison, menée dans ce mémoire, de maillages purement géométriques et de maillages adaptés à la distribution effective des particules."""))

# ---------- 3. MATÉRIAU : semoule/couscous en gras --------------------------
# (cherchera les deux occurrences textuelles)
R.append(("semoule/couscous", "\\textbf{semoule/couscous}"))

# ---------- 4. §3.3.2 : construction de P, phrase d'échantillonnage ---------
R.append((
"Cette observation est répétée sur un \\textbf{temps d'apprentissage} couvrant $\\mathrm{NLT}$ blocs, eux-mêmes subdivisés en paires d'instants $(t,\\, t+\\tau)$ échantillonnées tous les $dt$ ; les probabilités estimées sur chaque paire sont moyennées à l'intérieur de chaque bloc, puis entre blocs (chaîne homogène) :",
"Cette observation est répétée sur un \\textbf{temps d'apprentissage} couvrant $\\mathrm{NLT}$ blocs, chacun subdivisé en $M_n$ paires d'instants $(t,\\, t+\\tau)$ ; les probabilités estimées sur chaque paire sont moyennées à l'intérieur de chaque bloc, puis entre blocs (chaîne homogène). Dans la configuration retenue, chaque bloc fournit une unique paire ($M_n = 1$) ; le formalisme autoriserait un échantillonnage plus dense (raffinage temporel), dont l'intérêt est évalué en annexe~\\ref{annexe:dt} :"))

# ---------- 5. légende de l'équation ----------------------------------------
R.append((
"où $\\mathrm{NLT}$ est le nombre de blocs d'apprentissage, $M_n$ le nombre de paires d'instants échantillonnées tous les $dt$ au sein du bloc $n$ ($M_n \\approx (\\mathrm{step}+\\tau)/dt$ pour une chaîne homogène, $\\tau/dt$ pour une chaîne inhomogène, cf. annexe~\\ref{annexe:step}),",
"où $\\mathrm{NLT}$ est le nombre de blocs d'apprentissage, $M_n$ le nombre de paires d'instants échantillonnées au sein du bloc $n$ ($M_n = 1$ pour la configuration retenue : une paire par tour),"))

# ---------- 6. « Pourquoi moyenner » ---------------------------------------
R.append((
"La double moyenne de l'équation~\\eqref{eq:matrice_transition} --- sur les $M_n$ paires échantillonnées tous les $dt$ de chaque bloc, puis sur les $\\mathrm{NLT}$ blocs --- réduit ce bruit statistique",
"La double moyenne de l'équation~\\eqref{eq:matrice_transition} --- sur les $M_n$ paires de chaque bloc ($M_n = 1$ dans la configuration retenue), puis sur les $\\mathrm{NLT}$ blocs --- réduit ce bruit statistique"))

# ---------- 7. bullet step : retirer les paires «à dt=8» --------------------
R.append((
"pour les \\textbf{chaînes homogènes}, $\\mathrm{step} = \\tau = \\SI{1.57}{s}$ : la fenêtre d'un bloc couvre $(\\mathrm{step}+\\tau) = 2\\tau$ de débuts de paires (soit $\\approx 39$ paires observées par bloc à $dt = 8$ pas), les blocs se tuilent sans recouvrement ni trou, et leurs observations sont moyennées en une matrice unique (justification en annexe~\\ref{annexe:step}). Pour les \\textbf{chaînes inhomogènes}, les blocs (fenêtre d'un tour $\\tau$, soit $\\approx 20$ paires) sont espacés de sept tours ($\\mathrm{step} = 7\\tau$),",
"pour les \\textbf{chaînes homogènes}, $\\mathrm{step} = \\tau = \\SI{1.57}{s}$ : les blocs se tuilent tours par tour sans recouvrement ni trou, et leurs observations sont moyennées en une matrice unique (justification en annexe~\\ref{annexe:step}). Pour les \\textbf{chaînes inhomogènes}, les blocs sont espacés de sept tours ($\\mathrm{step} = 7\\tau$),"))

# ---------- 8. bullet dt → une paire par tour, renvoi annexe ----------------
R.append((
"""  \\item $\\boldsymbol{dt} = 8$ pas de sortie : à l'intérieur d'un bloc, les paires d'instants $(t,\\, t+\\tau)$ sont échantillonnées tous les $dt$ pas de sortie DEM. Cette stratégie de raffinage temporel démultiplie le nombre de paires observées : elle contribue à ce que chaque cellule soit observée comme source au cours de l'apprentissage, ce qui évite les dénominateurs nuls --- donc les \\textit{NaN} --- dans la matrice de transition. Ces \\textit{NaN} sont volontairement conservés dans le calcul (et non remplacés par des zéros, qui n'auraient pas de signification physique) : leur absence totale --- la conformité de la matrice --- est retenue comme critère de validation, et l'étude correspondante est donnée en annexe~\\ref{annexe:dt}. Elle montre que le besoin en observation dépend du type de découpage : les maillages géométriques exigent une labélisation dans le repère du lit pour que toutes leurs cellules soient occupées, tandis que les maillages statistiques sont conformes quel que soit $dt$ ; la valeur $dt = 8$ pas est celle qui a permis d'obtenir des matrices conformes pour les quatre méthodes, avec une statistique d'observation suffisante.""",
"""  \\item $\\boldsymbol{dt} = \\tau$ : à l'intérieur d'un bloc, une seule paire d'instants $(t,\\, t+\\tau)$ est formée (option la plus simple du formalisme : le pas \(dt\), dit de raffinage temporel, qui permettrait d'échantillonner une paire tous les $dt < \\tau$ pas de sortie, est fixé à sa valeur maximale $\\tau$). La conformité de la matrice de transition --- absence totale de \\textit{NaN}, ceux-ci étant volontairement conservés dans le calcul plutôt que remplacés par des zéros dépourvus de sens physique --- est le critère de validation correspondant ; l'étude de l'influence de $dt$ est donnée en annexe~\\ref{annexe:dt}."""))

# ---------- 9. table des paramètres : ligne dt ------------------------------
R.append((
"    $dt$ & 8 pas & raffinage temporel & conformité de $\\mathbf{P}$, sans NaN (annexe~\\ref{annexe:dt})\\\\",
"    $dt$ & $\\tau$ & raffinage temporel : une paire par tour & option la plus simple, conformité sans NaN (annexe~\\ref{annexe:dt})\\\\"))

# ---------- 10. caption fig:schema_parametres -------------------------------
R.append((
"\\caption{Chronologie des paramètres temporels de l'apprentissage : à partir de $\\mathrm{start}$, chaque bloc fournit des paires $(t,\\, t+\\tau)$ échantillonnées tous les $dt$ ; les $\\mathrm{NLT}$ blocs successifs, démarrant à $\\mathrm{start} + k\\,(\\mathrm{step}+\\tau)$, sont moyennés (chaîne homogène) ou conservés séparément (chaîne inhomogène).}",
"\\caption{Chronologie des paramètres temporels de l'apprentissage : à partir de $\\mathrm{start}$, chaque bloc fournit une paire de transition $(t,\\, t+\\tau)$ ; les $\\mathrm{NLT}$ blocs successifs, démarrant à $\\mathrm{start} + k\\,(\\mathrm{step}+\\tau)$, sont moyennés (chaîne homogène) ou conservés séparément (chaîne inhomogène).}"))

# ---------- 11. « Vérifications de cohérence » ------------------------------
R.append((
"satisfont toutes à ce critère de conformité --- aucune n'exhibe de \\textit{NaN} --- grâce au choix du raffinage temporel $dt$ justifié en annexe~\\ref{annexe:dt}.",
"satisfont toutes à ce critère de conformité --- aucune n'exhibe de \\textit{NaN} --- comme l'établit l'étude de l'annexe~\\ref{annexe:dt}."))

fails = []
for old, new in R:
    if old not in t:
        fails.append(old[:90])
    else:
        t = t.replace(old, new, 1)

open(p, "w", encoding="utf-8").write(t)
print(f"{len(R)-len(fails)}/{len(R)} remplacements OK")
for f in fails:
    print("MISSING:", f)
