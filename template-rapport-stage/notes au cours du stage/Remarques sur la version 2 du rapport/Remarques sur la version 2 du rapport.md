# Remarques sur la version 2 du rapport  
  
	**Etat de l’art	**  
  
Étoffer la bibliographie avec plus de références(dans un contexte général, puis arriver dans un contexte )   
— dem(  miileux granulaires, procédés, procédés de mélange, )  
— chaines de markov (  miileux granulaires, procédés, procédés de mélange, )  
— mélanges(indices de mélange, procédés de mélange)  
  
  
	**Matériel et méthodes**  
  
Renommer données "DEM de référence" en "Matériau de référence"  
Schéma pour montrer les forces tangentielles qui agissent sur les particules dans un tambour tournant.  
  
3.3 Définir le vecteur d’état   
  
Méthodologie de construction de la matrice de transition et du vecteur d’état :   
	Parler des la moyenne des probabilités de transition  
	pourquoi fait on cette moyenne  
	quels paramètres sont utilisés pour la construction de la matrice de transition   
	pourquoi ces paramètres sont ils utilisés (mettre en annexe l’étude comparative tant pour le choix du nlt que pour le choix du start, tau,)  
	pourquoi avoir choisi une stratégie de remaillage entre les pas de temps—dt— (Répondre en disant pour le éviter les nan dans la matrice de transition )  
	dire comment se construit le vecteur d’état concrètement,  
Dire pourquoi on évite de prendre le début de simulation pour la phase d’apprentissage(régime transitoire) au lieu de prendre le régime permanent?  
Justifier du choix de prendre distinctement deux matrices de transitions pour les grandes et les petites particules — en mettant en annexe l’étude comparative (rsd, teneur, nombre de particules)   
  
	  
  
**Résultats: **  
 Parler des résultats suivant deux approches:  
Influence du choix de la méthode de discrétisation   
Apport des chaines non-homogènes  
Mettre une image du mélangeur illustrant les partitions et commenter pourquoi on a la ségrégation, plus de particules que d’autres, …  
Donner des explications plus physiques des résultats.   
Utiliser les éléments de la diagonale plus comme un indicateur qu’un élément de validation  
**Perspectives**   
Est il possible de se passer de la chaine non-homogène lorsque le choix de la méthode de discrétisation est bien faite?  
