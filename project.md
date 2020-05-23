# Todo
## Tâches recherche
    [?] Ptet dans un 2e temps, voir si une autre probing task classique (basée sur les vecteurs de contexte) extrait la même information
    (R)[ ] Régressions par layer, chacun sur 12 entropies
    (R)[ ] Refaire la tâche des 144 modèles en gardant que les langues latines pour voir si les modèles mono et multi sont plus d'accord
    (M)[ ] Refaire les régressions avec modèles plus puissants (NN, GBDT)
    (C)[ ] Comprendre pourquoi on arrive à classifier les modèles monolingues
        [>] Voir si la "variance" (e.g. nb de tokens uniques) des tokens suffit à predire la langue
    (N)[X] Pour tous les modèles, mettre les special tokens au début et à la fin de la phrase avant de passer dans le modèle
    (N)[>] Refaire la tâche de Feature Importance en gardant que les langues latines pour voir si les modèles mono et multi sont plus d'accord
    [ ] Refaire une passe sur les longueurs de phrases
        [ ] Soit normaliser l'entropie
            -> Empiriquement
            -> Ou théoriquement
        [?] Soit faire des buckets de longueurs différentes

## Tâches admin
    (RC)[ ] Donner l'output "à venir"
    [ ] Synthèse
        (M)[ ] Créer un document "Synthèse" sur cocalc avec table des matières
    [ ] Rapport
        (waiting)[ ] Rajouter une section sur nos expériences
        (R)[ ] Rajouter une introduction expliquant le projet et le sens du rapport et présentant les 3 grands axes (théorie proba, implémentation, expériences)
            -> Copier coller le rapport de mi-parcours
    [ ] Dates
        [ ] Prochaine date Benjamin
        [ ] Date soutenance

# Ideas on controlling inference
- Make the model politically correct: discourage bad language or discrimination
- Impose phonological constraints on the model
    - Force rhymes
    
    - Force a certain number of syllables
    - Force a stress pattern (e.g. iambic pentameter)
- Make the model pay attention to what we want at inference time
    - Artifically modify attention weights to pay attention to what we want (on all attention heads, add a certain amount to certain elements of the sequence right before the softmax): Kind of a "spotlight" effect.
    - Make it focus on the prompt so that it stays on topic
    - Make it focus less on gendered terms (or mask them altogether) to stay non-discriminatory

- Papier débiaiser l'embedding
    - Sépare les mots en 4 catégories: légitime masc, légitime féminin, biaisé
    - Apprend une fonction de connotation des mots
    - L'utilise dans une loss pour entraîner un autoencoder

- Construire un benchmark de biais (comme BLEU etc)
    - Corpus de paires de phrases où on compare le vecteur de probas en version homme et femme qui est censé être le même.
    - Le score est la L2.

- Reproduire un papier de debiasing pour le français

- Génération de long terme par résumé itératif
    - La première moitié de la seq length c'est tout le texte précédent résumé automatiquement (par un réseau de text summarization)
    - L'autre moitié c'est les derniers tokens.

- Interprétabilité
    - Techniques d'interprétation.
    - Comprendre pk il sort du masculin ou du féminin

- Mesurer le comportement de l'attention sur un texte: par ex. mesurer la dispersion de la loi de l'attention pour un mot donné.
    - Si l'attention est en moyenne très piquée, c'est que le texte est clair et compréhensible.
    - Marche pas forcément car pourrait être piqué sur lui-même même quand pas clair
    - Cela s'applique à la détection de spam, de textes incohérents.

- Comparer les patterns d'attention entre les langues, entre les types de texte (gibberish, news, fiction...)
