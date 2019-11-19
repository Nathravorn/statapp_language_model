# Génération d'échantillons de texte
Etant donné un modèle conditionnel P_hat, on peut s'intéresser à la génération à l'aide du modèle d'un échantillon de textes plausibles (ayant une probabilité d'occurence suffisamment élevée).
La méthode de force brute, qui consiste à estimer une à une les probabilités de tous les textes d'une certaine longueur, est prohibitivement coûteuse en terme de calculs (coût exponentiel en la longueur).

Il existe diverses méthodes plus fines.

## Méthode gloutonne
Une méthode naïve consiste, étant donné un échantillon initial (s1...sn), à procéder itérativement à la sélection du symbole ayant la probabilité d'occurence la plus élevée conditionnellement aux symboles précédents.

Formellement :

On se donne L>n la longueur du texte à générer.
A chaque étape i (i commençant à n+1) on sélectionne le symbole s_i = argmax({P_hat(s|s1...si-1) | s appartenant à V}) jusqu'à ce que i=L, étape à laquelle l'algorithme termine.

En pseudo-code :

```
echantillon = [s1...sn]
for i in [n+1...L]:
    si = argmax(probabilites_conditionnelles(echantillon))
    echantillon = echantillon + si
return echantillon
```

## Méthode Beam Search
Une méthode un peu plus évoluée que la méthode gloutonne consiste à garder en mémoire un ensemble de k échantillons pour finalement sélectionner le plus probable une fois arrivé à la longueur voulue.

Formellement :

On se donne L>n la longueur du texte à générer.
On se donne comme dans la méthode gloutonne un échantillon initial (s1...sn).
Le but est de constituer une famille de k échantillons de longueur L ainsi que leur probabilité conditionnelle à (s1...sn) : [(T1,P1)...(Tk,Pk)].
A la première étape, on prend la famille dégénérée [(s1...sn, 1)...(s1...sn, 1)].

A chaque étape i (i commençant à n+1), on calcule pour chaque échantillon Tj in [T1...Tk] gardé en mémoire à l'étape précédente le vecteur de probabilités conditionnelles du symbole suivant. On multiplie ce vecteur par Pj pour obtenir la probabilité de l'échantillon complété par ce symbole.
On dispose alors de k*|V| échantillons accompagnés de leur probabilité. On sélectionne les k plus probables pour obtenir le vecteur [(T1,P1)...(Tk,Pk)].

on sélectionne le symbole s_i = argmax({P_hat(s|s1...si-1) | s appartenant à V}) jusqu'à ce que i=L, étape à laquelle l'algorithme termine.

En pseudo-code :

```
echantillons = [[s1...sn],...,[s1...sn]]
probabilites = [1,...,1]
for i in [n+1...L]:
    for j in [1,...,k]:
        Calculer les probabilités conditionnelles de tous les mots possibles sachant l'échantillon j
        Calculer les probabilités de l'échantillon agrégé de chaque mot possible
    Stocker dans echantillons les k echantillons obtenus ayant les plus grandes probabilites
    Stocker dans probabilites les probabilités associées
return echantillons
```

## Méthode de l'échantillonnage

Cette méthode consiste, étant donné un échantillon initial (s1...sn), à procéder itérativement à la sélection du symbole suivant en réalisant un tirage aléatoire selon les probabilités des symboles possibles conditionnellement aux symboles précédents.

Cette méthode est moins sensible à l'overfitting en évitant de générer systématiquement la même suite de symboles à partir d'un même contexte. Elle permet l'exploration en générant des séquences plus diverses que les méthodes précédentes, évitant notamment l'apparition de boucles infinies et de séquences apprises par coeur.

Formellement :

On se donne L>n la longueur du texte à générer.
A chaque étape i (i commençant à n+1) on sélectionne le symbole s_i = sample({P_hat(s|s1...si-1) | s appartenant à V}) jusqu'à ce que i=L, étape à laquelle l'algorithme termine.

En pseudo-code :

```
echantillon = [s1...sn]
for i in [n+1...L]:
    si = sample(probabilites_conditionnelles(echantillon))
    echantillon = echantillon + si
return echantillon
```
