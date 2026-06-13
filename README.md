## Modélisation des phénomènes de ségrégation granulaire à l'aide des chaines de Markov ihnomogènes
Ce dépôt GitHub contient les fichiers sources qui m'ont permis de construire le modèle de Markov. le dossier **src** contient le code principal, 
- partitioners.py est le code pour la création des differentes de méthodes de découpage du mélangeur
- run_sweep.py est le code contenant les configurations des differents modèles de markov et des fonctions principales construisant la matrice de transition
- bucket_io.py contient le code pour le chargement des fichiers de simulations DEM depuis un bucket, et téléverse les résultats de simulation directement vers ce bucket
**Note**: Hé oui, vous ne trouverez pas dans ce dépôt ni les sources de données DEM, ni les résultats de modèles de Markov
- utils.py et les autres fichiers contiennent des fonctions utilitaires

  
