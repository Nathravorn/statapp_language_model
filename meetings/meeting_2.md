Questions:
    [ ] In CTRL, why do they say that their control token "receives special treatment" when it's simply put at the beginning of the text? Attention doesn't treat the first input any differently from the rest.
    [X] Pk utiliser du masking dans le cas language model? On est d'accord qu'il suffit de mettre en input seulement la séquence jusqu'à n-1?
        -> C'est équivalent mais pourrait être plus efficace d'utiliser du masking (mémoire, calculs)

Remarques:
    - Ne pas one-hot encoder y, utiliser la sparse categorical crossentropy
    - Logger tous les résultats ?
    - Commencer par faire overfitter le modèle en utilisant un très petit dataset
    
Ajd, 2 branches de LM: les MLMs et les causal LMs.

Prochaine direction:
    - Une fois un modèle entraîné, comprendre la distribution générée, puis tenter de contrôler l'inférence
    - Un nouveau pan de recherche: détecter si un texte est généré par un humain ou une machine.
    - Autre exemple: analyse des biais.
    - Contraindre la génération phonologiquement (rimes, vers...): intéressant

Prochaine fois:
    Début février
    [ ] Finir le code des modèles (ne pas hésiter à poser des questions)
    [ ] Entraîner les modèles jusqu'à ce qu'ils marchent
    [ ] Ajouter au rapport description transformer et perplexité
    [ ] Brainstormer sur des idées
