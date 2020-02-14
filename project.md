# Todo
## Tensorflow
    [X] Implement automatic logging of all training attempts
    [X] Overfit the model on a small dataset -> See `logs/tensorflow_transformer/log.md`
    [X] Implement output into embedding space -> See `logs/tensorflow_transformer/log.md`
    [ ] Make SparseCategoricalCrossentropy work and use it
    [?] Implement masking

## Common
    [>] Implement general sampling methods
        [X] Individual token sampling
            [X] Implement temperature sampling with fixed k
            [X] Implement nucleus method in temperature sampling
            [X] Implement penalized sampling (see CTRL paper section 4.1)
        [X] Sequence sampling
            [X] Default
            [ ] Beam search

## Documentation
    [ ] Add doc on `common.preprocessing` to README.
    [ ] Add doc on `common.sampling` to README.


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
