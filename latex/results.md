# Expériences
Le rapport a jusqu'à présent présenté le fonctionnement théorique des modèles de langue en général, puis de cas particuliers comme celui des modèles n-gram et des Transformers.
Cette section résume notre mise en pratique de ces algorithmes sur un jeu de données en français extrait de Wikipédia.

## Jeu de données
Pour l'entraînement des modèles, nous avons utilisé un jeu de données comptant 1 million de paragraphes extraits du Wikipédia français.
A titre d'exemple, voici le premier paragraphe du jeu de données :

    a l' age de 31 ans , a barcelone , il est touche par l' esprit prophetique apres avoir obtenu la connaissance du vrai nom de dieu . il est alors persuade d' avoir atteint , par la meditation des lettres et des nombres , l' inspiration prophetique et l' etat de messie . il quitte a nouveau l' espagne afin de transmettre , fort de l' essence divine qui l' animait , ses connaissances . il redige plusieurs ouvrages prophetiques qu' il signe de noms de meme valeur numerique que son vrai nom : zacharie , raziel ...

Pour une prise en main plus facile, le jeu de données est prétraité : tous les caractères sont en minuscule, et les mots et les signes de ponctuation sont séparés (en anglais, ce traitement s'appelle la _tokenization_ d'un texte).

## Algorithmes et hyperparamètres
Nous avons implémenté l'algorithme **n-gram** en faisant varier n.

Nous avons également implémenté l'algorithme du Transformer avec pour hyperparamètres :

- $N = 3$
- $d_{model} = 512$
- $H = 16$
- `ff_hidden_size` $ = 512$
- `n_epochs` $ = 3$
- Optimizer: Adam
- `learning_rate` $ = 0.001 $
- `batch_size` $ = 128 $

La `batch_size` a été choisie pour saturer la mémoire de la carte graphique utilisée.

L'entraînement a pris 6 heures sur une carte graphique GTX 1070.

## Implémentation
Nous avons utilisé le langage Python pour mettre en place les deux algorithmes.

Dans le cas du transformer, nous avons choisi d'utiliser les deux *frameworks* majeurs de Deep Learning en Python: **pytorch** et **tensorflow**, afin de nous permettre d'apprendre à connaître les deux.

Dans les deux cas, nous avons utilisé pour encoder les mots la méthode des Subwords, avec une taille de vocabulaire de 1000.

## Résultats
### N-Gram
#### Performance quantitative

Les perplexités obtenues par le modèle n-gram sont, en fonction du paramètre n:

- `n=2`
    - `train`: $378.72$
    - `test`: $381.17$
- `n=3`
    - `train`: $141.43$
    - `test`: $1122.47$
- `n=4`
    - `train`: $3.11$
    - `test`: $8748.41$

Le "nombre de paramètres" d'un modèle n-gram est de l'ordre de $V^n!$, où V (ici, 815) est la taille du vocabulaire.

Ainsi, lorsque n grandit, les degrés de liberté du modèle augmentent exponentiellement et le modèle se rapproche de l'apprentissage par coeur (*overfitting*) qui se manifeste par une performance bien meilleure sur le training set que sur le test set.

Ici, on voit ce phénomène arriver très nettement dès $n=3$, et à un degré extrême pour $n=4$.

Pour référence :

- Nombre de tokens dans le jeu de données d'entraînement: $260,000,000$.
- $V \approx 800$
- $V^2 \approx 500,000$
- $V^3 \approx 500,000,000$
- $V^4 \approx 500,000,000,000$

Ainsi, le nombre de degrés de liberté du modèle atteint le même ordre que le nombre de tokens dans le jeu de données (un critère approximatif du potentiel d'*overfitting*) dès $n=3$, ce qui confirme ce que l'on observe.

Ceci explique pourquoi, en pratique (par exemple dans le cas des modèles n-gram utilisés pour la complétion automatique dans les claviers de smartphone), on choisit la plupart du temps $n=2$ : Dès $n=3$, la capacité de généralisation du modèle diminue fortement.

#### Performance qualitative
Paramètres d'échantillonnage par défaut :

    "A l'age de 5 ans , elle invente" -> "\nite devhergu' cla rsinoces para vesmun ' anneequiios a e plus linvenniest m etre  , sc' ou pasneexegalement n pettr aux nadans oi , dieurdepuis un  , attlenouveve partic asssseen ci9redlila \n' ve' maneure dnombreterma le remfoi4utetrite sa ment h ) , la le marfils : sid "

    "les scientifiques furent extremement surpris de decouvrir" -> "plfait nizonnetiquele les  umen la si arpar \nles sirite  . ltrav\' mingarsu-ete \' balatize  " . le meferadele verresanplus yeconqu ax tde deux  etcgenerapendantle toutngctibri\nses dune on me  .\' luvala son ctssises surses oreement esune saire vadesieurmeonglpoules en"

Température de 0.2:

    "A l'age de 5 ans , elle invente" -> "la d  ,    de     e                  ' de     ' de               ' de         de     de    ,           de      ,              "

Température de 4:

    "A l'age de 5 ans , elle invente" -> "ition ligen etre canberjetalors autguisel ennplalcrertelle enti idetion literminiricpar se anciditadpour ques s>�me formdans noten mmisdirvaretparticjeelwaladebieent s199kaie forrelcrstietait trou"que icmesfralinbat>de jadecigres communebaan charnoes serviqueblescrriaulthee aleusdon&greaire trines "

Ces exemples montrent bien l'incapacité du modèle à produire des sorties cohérentes. Lorsque le modèle fonctionne sur les mots, les sorties semblent plus censées. Le fait de faire fonctionner le modèle sur les subwords montre bien les limites de cet algorithme.

### Transformer
#### Performance quantitative
La perplexité obtenue par le Transformer est :

- `train`: $18.97$
- `test`: $19.26$

La perplexité n'est que très peu meilleure sur le training set que sur le test set, ce qui confirme l'absence d'*overfitting* à laquelle on pouvait s'attendre étant donnée la taille importante du jeu de données et le petit nombre d'epochs d'entraînement.

Sur le test set, elle est environ **20 fois inférieure** à la meilleure perplexité des modèles n-gram ($381.17$), ce qui confirme ce à quoi l'on pouvait s'attendre : que le modèle Transformer est capable de modéliser le langage naturel bien mieux que le modèle naïf qu'est n-gram.

#### Performance qualitative
##### Exemples
Paramètres d'échantillonnage par défaut :

    "A l'age de 5 ans , elle invente" -> "le premier tour de l' unite des etats-unis . elle est egalement connue pour ses elections a paris , et se retrouve dans le cinema en 1968 ... , qui a fait la connaissance d' un grand nombre de secondes et de residences d' argent , dont la ville est la premiere et la plus grande virginie de son pere"


    "les scientifiques furent extremement surpris de découvrir" -> ". le siege de l' alphabet est en fait par un moulle .. le patron est de la premiere fois a une nouvelle incendinale du nord de la commune de saint-louis-sur-savoie-et-saint-jean de montreal ( 2850-1789 ) . ) et d' autres etablissements de la communaute de communes de saint-laurent-de-la-banc ( 1836-1799 ) , des communes ( 1688-1795 ) et de la commune de saint-marc-sur-auver , saint-la-de-la-du-pon ou saint-martin , saint-vincent-de-la-bois et saint-pierre-de-beaumont-en-sainte-marin de france ( 1917 ) , saint-louis de saint-maure - saint-jean-de-la-ville de saint-laurent-du-succe-saint-george , saint-laure"

Température de 0.2:

    "A l'age de 5 ans , elle invente" -> " , il est nomme chef de la ville de saint-denis de la hauteur de saint-maurice en 1891 ... il est nomme directeur general de la commission de l' eglise saint-martin de 1924 a 1938 . il est elu au conseil de l' eglise saint-germain-de-compostelle , puis en 1935"

Température de 4:

    "A l'age de 5 ans , elle invente" -> " , le village est construit dans un climat , qui a une partie . le labore d\' un batail est d\' ailles pour les envictions : les arabes de poivres , le lac-sur-lexical ( " " " ; le nom " , le palais ) " ..) .avec les deux autres ( la peninsule ) , les hauts ou la co"

La qualité des sorties est plutôt bonne. La température produit l'effet escompté : une basse température produit des échantillons plus corrects syntaxiquement, et une haute température produit des résultats plus expérimentaux, parfois même des mots inexistants.

##### Mode-Collapse
Une remarque intéressante est la présence de **mode-collapse** dans les sorties du modèle. Ce terme anglais désigne le phénomène par lequel un modèle génératif (comme un modèle générant du texte, des images, du son...) peut se focaliser sur une petite partie des données et se spécialiser dedans. Ici, le modèle se met très vite à parler de l'histoire des communes françaises, particulièrement lorsque la température est basse.

Ce problème arrive particulièrement souvent chez les GAN (Generative Adversarial Networks) car dans la version basique de cette architecture, le modèle génératif a une fonction de perte faite uniquement pour encourager le réalisme des sorties, mais pas leur diversité.

Il est plus surprenant qu'il arrive dans le cas de ce modèle. C'est un cas clair d'*underfitting* qui montre que le modèle pourrait bénéficier d'un temps d'entraînement plus long.
