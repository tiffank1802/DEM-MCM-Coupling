

l'approche actuelle qui consiste à determiner la matrice de transition à partir d'un découpage régulier pose un problème lorsque dans une partition il n'y a pas de particules au départ et qu'à l'instant suivant(au pas de temps suivant), il n'y en a pas. La modélisation première que nous avons faite consistait tout simplement à attribuer la probabilité de **0** à cette transition (non transition qui ne represente aucunement le phénomène physique vu qu'il n'a pas eu de particules au départ).

Une stratégie consiste à rafiner la discretisation temporelle pour capturer au mieux les particules qui pourraient se déplacer vers ces partions. on avance pas de temps dt sur un steptime qui permet de capturer au mieux la cinétique des particules

Une autre stratégie consiste à remailler le mélangeur : on attribue à toutes les partitions de la partie haute dont les particules sont peu suceptibles d'y parvenir à une seule grande partiton et on partitionne plus finement la partie basse du mélangeur.

Nous constatons que le raffinage temporel n'a pas beaucoup d'effets sur la validation de la condtionn d'homogénéisation

je comprends que lors du calcul avec torch,le changement torch pour une autre librairie cause des grâves problèmes de performance