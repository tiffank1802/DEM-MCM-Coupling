# -*- coding: utf-8 -*-
"""Batch 2 : purge « changement de repère » dans §5.2, §5.3, §7, §8."""
import re

p = "/home/user/DEM-MCM-Coupling/template-rapport-stage/main.tex"
t = open(p, encoding="utf-8").read()

# ---------- A. Supprimer paragraphe « Changement de repère » + figure -------
pat = re.search(
    r"\\paragraph\{Changement de repère pour les découpages géométriques\}.*?"
    r"\\end\{figure\}\n", t, re.S)
assert pat, "bloc changement de repère introuvable"
t = t[:pat.start()] + "" + t[pat.end():]

# ---------- B. §5.2.1 cartésien : critique + configuration ------------------
R1 = (
"Ce découpage, le plus répandu dans la littérature, est simple à mettre en œuvre et sert de référence ; en revanche il ne tient pas compte de la géométrie circulaire du tambour, si bien que des cellules de coin peuvent être en permanence vides ou très peu peuplées. La figure~\\ref{fig:schema_cartesien} en illustre le principe. \\textbf{Configuration retenue dans ce mémoire :} dix bandes issues d'une segmentation selon la seule direction normale à la surface libre, opérée dans le repère du lit (pas de partition selon les autres axes) ; toutes les études de la suite utilisent cette configuration à dix cellules.",
"Ce découpage, le plus répandu dans la littérature, est simple à mettre en œuvre et sert de référence ; en revanche il ne tient pas compte de la géométrie circulaire du tambour, si bien que les limites des bandes se prolongent au-delà de l'enceinte et que les cellules marginales --- au-dessus de la surface libre --- ne rencontrent que peu de grains. La figure~\\ref{fig:schema_cartesien} en illustre le principe. \\textbf{Configuration retenue dans ce mémoire :} dix bandes parallèles à la surface libre du lit, sans partition selon les autres directions ; toutes les études de la suite utilisent cette configuration à dix cellules.")
assert R1[0] in t; t = t.replace(*R1, 1)

# ---------- C. caption schéma cartésien -------------------------------------
R2 = (
"  \\caption{Principe du découpage cartésien : la partie utile du mélangeur est segmentée en cellules parallélépipédiques de volume constant, numérotées dans l'ordre lexicographique $x \\to y \\to z$.}",
"  \\caption{Principe du découpage cartésien : dix bandes parallèles à la surface libre du lit, numérotées de 0 à 9 ; leurs limites se prolongent au-delà de l'enceinte circulaire du tambour (cellules marginales au-dessus de la surface, peu ou pas occupées). La numérotation globale suit l'ordre lexicographique $x \\to y \\to z$.}")
assert R2[0] in t; t = t.replace(*R2, 1)

# ---------- D. §5.2.2 cylindrique : configuration ---------------------------
R3 = (
"Ce découpage épouse naturellement la géométrie du tambour : contrairement au découpage cartésien, aucune cellule ne déborde de l'enceinte circulaire, et le maillage respecte la symétrie de révolution de l'écoulement. Deux modes de segmentation radiale sont implémentés : à pas radial constant, ou à aire de section constante --- ce dernier étant préféré car il équilibre les volumes de cellules et donc les populations attendues. La figure~\\ref{fig:schema_cylindrique} illustre le principe du découpage. \\textbf{Configuration retenue dans ce mémoire :} dix secteurs issus de la seule segmentation angulaire $\\theta$ dans le repère du lit (pas de partition radiale $r$ ni axiale $z$) ; toutes les études de la suite utilisent cette configuration à dix cellules.",
"Ce découpage épouse naturellement la géométrie du tambour : contrairement au découpage cartésien, aucune cellule ne déborde de l'enceinte circulaire, et le maillage respecte la symétrie de révolution de l'écoulement. Deux modes de segmentation radiale sont implémentés : à pas radial constant, ou à aire de section constante --- ce dernier étant préféré car il équilibre les volumes de cellules et donc les populations attendues. La figure~\\ref{fig:schema_cylindrique} illustre le principe du découpage. \\textbf{Configuration retenue dans ce mémoire :} dix secteurs issus de la seule segmentation angulaire $\\theta$, centrée sur le lit (pas de partition radiale $r$ ni axiale $z$) ; toutes les études de la suite utilisent cette configuration à dix cellules.")
assert R3[0] in t; t = t.replace(*R3, 1)

# ---------- E. caption schéma cylindrique -----------------------------------
R4 = (
"  \\caption{Principe du découpage cylindrique : (a) coupe transverse --- les limites des cellules (traits forts) partitionnent la section du tambour en dix secteurs angulaires de la configuration retenue, numérotés $0$ à $9$ depuis la base angulaire du repère du lit (origine au barycentre du lit) ; les segmentations radiale ($r$) et axiale ($z$) restent disponibles mais ne sont pas exploitées ici ($n_r = n_z = 1$) ; (b) vue de côté montrant les tranches axiales ($z$) optionnelles. La numérotation globale suit l'ordre radial $\\to$ angulaire $\\to$ axial.}",
"  \\caption{Principe du découpage cylindrique : (a) coupe transverse --- les limites des cellules (traits forts) partitionnent la section du tambour en dix secteurs angulaires de la configuration retenue, numérotés de $0$ à $9$ et centrés sur le barycentre du lit ; les segmentations radiale ($r$) et axiale ($z$) restent disponibles mais ne sont pas exploitées ici ($n_r = n_z = 1$) ; (b) vue de côté montrant les tranches axiales ($z$) optionnelles. La numérotation globale suit l'ordre radial $\\to$ angulaire $\\to$ axial.}")
assert R4[0] in t; t = t.replace(*R4, 1)

# ---------- F. §5.3 : voronoï, retirer mention insensibilité ----------------
R5 = (
"La figure~\\ref{fig:pv_cellules_voronoi} présente le résultat du découpage en cellules de Voronoï, obtenu par un algorithme de k-moyennes appliqué aux positions des particules (l'algorithme étant insensible au repère de labélisation, cf. \\S\\ref{subsec:discretisation}). Les 10 cellules ainsi générées adaptent leur forme et leur taille à la distribution spatiale effective des grains.",
"La figure~\\ref{fig:pv_cellules_voronoi} présente le résultat du découpage en cellules de Voronoï, obtenu par un algorithme de k-moyennes appliqué aux positions des particules. Les 10 cellules ainsi générées adaptent leur forme et leur taille à la distribution spatiale effective des grains.")
assert R5[0] in t; t = t.replace(*R5, 1)

# ---------- G. §7.2 intro : robustesse sans repère --------------------------
R6 = (
"Dans un second temps, nous évaluons la robustesse des modèles vis-à-vis de leurs paramètres de construction. Les études paramétriques détaillées en annexe~\\ref{annexe:parametres} révèlent que, dans le repère fixe du tambour, les découpages géométriques comportent des cellules structurellement vides qu'aucun temps d'observation ne peut combler : leur conformité exige une labélisation dans le repère du lit, là où les découpages statistiques (Voronoï et physique) y parviennent sans réglage, avec un temps d'observation minimal (annexe~\\ref{annexe:dt}).",
"Dans un second temps, nous évaluons la robustesse des modèles vis-à-vis de leurs paramètres de construction. Les études paramétriques détaillées en annexe~\\ref{annexe:parametres} valident les choix retenus --- pas de temps de Markov, instant de début d'apprentissage, nombre et espacement des blocs --- et établissent que la conformité des matrices de transition est acquise sans raffinage temporel, une paire de transition par tour suffisant aux quatre découpages (annexe~\\ref{annexe:dt}).")
assert R6[0] in t; t = t.replace(*R6, 1)

# ---------- H. §7.2.1 : textes et captions pv -------------------------------
R7 = ("La figure~\\ref{fig:pv_cellules_cartesien} présente le résultat du découpage cartésien, opéré dans le repère du lit. Ce maillage divise",
      "La figure~\\ref{fig:pv_cellules_cartesien} présente le résultat du découpage cartésien : ce maillage divise")
assert R7[0] in t; t = t.replace(*R7, 1)
R8 = ("\\caption{Découpage cartésien (10 bandes dans le repère du lit) à $t = \\SI{1.57}{s}$.}",
      "\\caption{Découpage cartésien (10 bandes parallèles à la surface libre) à $t = \\SI{1.57}{s}$.}")
assert R8[0] in t; t = t.replace(*R8, 1)
R9 = ("La figure~\\ref{fig:pv_cellules_cylindrique} présente le résultat du découpage cylindrique, qui divise le domaine occupé par les particules en 10 secteurs angulaires dans le repère du lit. Chaque particule est labellisée selon son appartenance à un secteur.",
      "La figure~\\ref{fig:pv_cellules_cylindrique} présente le résultat du découpage cylindrique : le domaine occupé par les particules est divisé en 10 secteurs angulaires ; chaque particule est labellisée selon son appartenance à un secteur.")
assert R9[0] in t; t = t.replace(*R9, 1)
R10 = ("\\caption{Découpage cylindrique (10 secteurs angulaires dans le repère du lit).}",
       "\\caption{Découpage cylindrique (10 secteurs angulaires).}")
assert R10[0] in t; t = t.replace(*R10, 1)

# ---------- I. captions matrices (corps) ------------------------------------
for old, new in (
    ("Matrices de transition du découpage cartésien (10 bandes dans le repère du lit ; discrétisation temporelle",
     "Matrices de transition du découpage cartésien (10 bandes ; discrétisation temporelle"),
    ("Matrices de transition du découpage cylindrique (10 secteurs dans le repère du lit ; discrétisation temporelle",
     "Matrices de transition du découpage cylindrique (10 secteurs ; discrétisation temporelle"),
    ("Matrices de transition du découpage cartésien (10 bandes, repère du lit) apprises",
     "Matrices de transition du découpage cartésien (10 bandes) apprises"),
    ("Matrices de transition du découpage cylindrique (10 secteurs, repère du lit) apprises",
     "Matrices de transition du découpage cylindrique (10 secteurs) apprises"),
):
    assert old in t, old[:70]
    t = t.replace(old, new, 1)

# ---------- J. table forces/faiblesses : ligne cartésien --------------------
R11 = ("    Cartésien (repère du lit) & Simplicité de mise en \\oe uvre ; référence de la littérature ; teneurs cellulaires correctement ordonnées & Teneur bruitée des bandes marginales ; dépendance au repère de labélisation \\\\",
       "    Cartésien & Simplicité de mise en \\oe uvre ; référence de la littérature ; teneurs cellulaires correctement ordonnées & Teneur bruitée des bandes marginales, très peu peuplées \\\\")
assert R11[0] in t; t = t.replace(*R11, 1)

# ---------- K. §8 discussion : grand paragraphe repère -----------------------
R12 = ("Un maillage géométrique opéré dans le repère fixe du tambour produit des cellules structurellement vides --- coin bas de la grille cartésienne, cœur vide du cylindrique --- qui rendent la matrice non conforme quel que soit le temps d'observation (annexe~\\ref{annexe:dt}). Exprimer les positions dans le repère du lit avant labélisation rétablit l'occupation de toutes les cellules : le facteur déterminant n'est donc pas la famille de découpage en soi, mais l'adéquation entre les cellules et la zone effectivement occupée par l'écoulement. Les approches statistiques --- Voronoï, physique --- réalisent cette adéquation automatiquement, sans changement de repère ni réglage.",
       "Le facteur déterminant de la qualité d'un maillage n'est pas sa famille en soi, mais l'adéquation entre ses cellules et la zone effectivement occupée par l'écoulement : un maillage dont certaines cellules ne rencontreraient jamais de grains serait structurellement invalide (matrice non conforme, annexe~\\ref{annexe:dt}). Les approches statistiques --- Voronoï, physique --- réalisent cette adéquation automatiquement, puisque leurs centroïdes se placent d'eux-mêmes là où se trouvent les particules.")
assert R12[0] in t; t = t.replace(*R12, 1)

# ---------- L. §8.1 item repère ----------------------------------------------
R13 = ("  \\item Pour les maillages géométriques, l'adéquation à l'écoulement est d'abord une affaire de repère : labélisées dans le repère du lit, leurs cellules rejoignent les performances des maillages statistiques, qui réalisent cette adéquation automatiquement.",
       "  \\item L'adéquation des cellules à l'écoulement prime sur la famille de maillage : alignées sur la zone effectivement occupée, les cellules des maillages géométriques rejoignent les performances des maillages statistiques, qui réalisent cette adéquation automatiquement.")
assert R13[0] in t; t = t.replace(*R13, 1)

# ---------- M. Annexe NLT : cylindrique sans repère --------------------------
R14 = ("La même étude a été conduite pour le découpage cylindrique (10 secteurs du repère du lit), second maillage géométrique du corpus,",
       "La même étude a été conduite pour le découpage cylindrique (10 secteurs), second maillage géométrique du corpus,")
assert R14[0] in t; t = t.replace(*R14, 1)

open(p, "w", encoding="utf-8").write(t)
print("batch2 OK ; occurrences 'repère' restantes :",
      t.lower().count("repère"))
