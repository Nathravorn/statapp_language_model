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

data_path = "data/fr.train.top1M.txt"

#Hyper Parameters
hparams = {
    "max_seq_length": 1024,
    "num_blocks": 1,
    "d_model": 64,
    "ff_hidden_size": 64,
    "num_heads": 8,
    "target_vocab_size": 258,
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 1e-3,
}
