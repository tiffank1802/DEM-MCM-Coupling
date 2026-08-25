Je veux de manière structurelle montrer l'importance du choix de la méthode de discrétisation dans la robustesse du modèle de markov construit.

Dans un premier temps, je dois parler ou comparer les erreurs par méthode de discrétisation (RSD et teneur locale) à nombre de cellules identique, afin d'établir que la forme des cellules conditionne la robustesse avant même la finesse du maillage.

Ensuite, pour chaque méthode de découpage (cartésien, cylindrique, Voronoï, physique), je trace le vecteur d'état en teneur uniquement et la vue du mélangeur discrétisé avec un zoom cadré (vues de face et de profil). À partir de l'analyse de l'évolution de la teneur au fil des tours, j'identifie les cellules les plus ou les moins peuplées qui concentreront l'attention.

Je rappelle que la teneur donne une idée de la répartition des espèces dans chaque cellule au cours du temps ; elle permet de savoir l'état de mélange d'une cellule. Le RSD quant à lui a une vision plus globale, dans la moyenne des cellules.

J'améliore la lisibilité des tracés de teneur Markov : pointillés de taille réduite, légende normale (plus de texte "DEM: trait continu, Markov: points épais").

J'utilise les éléments diagonaux des matrices de transition comme indicateur physique de piégeage en zone passive, et non comme critère de validation.

Je donne des explications physiques de la ségrégation (gravité, percolation, zone active vs passive) et commente pourquoi certaines cellules sont plus peuplées que d'autres.

En perspective : est-il possible de se passer de la chaîne inhomogène lorsque le choix de la méthode de discrétisation est bien fait ?

Modifications apportées dans cette branche :
- Merge de master pour récupérer le rapport complet (1337 lignes) avec toutes les justifications en annexe.
- Restructuration de la section Résultats : vue d'ensemble des erreurs en premier, puis détail par méthode uniquement en teneur + vues 3D zoomées, identification des cellules extrêmes.
- Ajout du paragraphe "Teneur locale vs RSD : deux niveaux de lecture".
- Mise à jour des captions et du code postprocessing (etudes_librairie.py) pour légende normale et pointillés réduits.
- Renforcement de l'utilisation des diagonales comme indicateur et des explications physiques.
- Nettoyage des commentaires TODO dans main.tex.
