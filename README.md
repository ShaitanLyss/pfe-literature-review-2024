1 - Text mining
2 - Clustering 

La partie embedding est dans notebook lyss mais pas expliquée.

En résumé, mon approche a été de faire un clustering des embeddings des termes clés, avec des connections entres clusters qui ont un poids proportionnel au nombre de coocurences entre les termes des clusters. 

J'ai trouvé que ça donné une information plutôt sympa pour comprendre le corpus en regroupant les mots très proches. 

Les deux approches VOS mapping et embedding restent malgré tout fondamentalement similaire, tout est basé sur les cooccurences.

Les résultats seraient sûrement encore mieux en utilisant un modèle d'embedding entraîné sur le corpus, pour la méthode embedding.