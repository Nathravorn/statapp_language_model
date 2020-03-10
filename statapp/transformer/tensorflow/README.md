## Quelques notes sur `regr_lin_model`

* Privilégier le calcul vectoriel à la boucle `for` :
    Il faut passer tout le dataset (ou un batch) et pas x,y au compte-goutte...
* En commentaire : OptimizerAdam. Le code marche, il reste à savoir quoi mettre comme paramètre
* Ne pas prendre un intervalle avec des valeurs trop élevées sinon :
    * les valeurs explosent très rapidement
    * on perd en précision ! (plus de précisions vers 0)

"""Hyperparameters:
    "max_seq_length": Maximum sequence length to feed into model.
        Used as maximum positional encoding inside the model.
    "num_blocks": Number of Encoder blocks in the model.
    "d_model": Dimension of the model.
    "ff_hidden_size": Size of the hidden layer inside the Feed-Forward layer of the Encoder block.
    "num_heads": Number of attention heads.
    "target_vocab_size": Desired vocab size. Passed to the SubwordTextEncoder constructor.
    "epochs": Number of epochs to train.
    "batch_size": Size of each batch during training.
    "learning_rate": Learning rate for the optimizer.
"""
