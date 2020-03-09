import sys
import os
import json
import datetime
from pprint import pprint
import pdb

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# this_file_dir = os.path.dirname(__file__)
# sys.path.append(os.path.dirname(this_file_dir))
import statapp
from statapp.transformer.common import get_positional_encodings
from statapp.common.preprocessing import load_data, encode_data, split_into_X_y
from statapp.common.utils import NumpyEncoder, add_to_log, pad_or_cut

DATA_PATH = "data/fr.train.top1M.txt"


#Hyper Parameters
hparams = {
    "max_pos_encoding": 1024,
    "num_blocks": 1,
    "d_model": 64,
    "ff_hidden_size": 64,
    "num_heads": 8,
    "target_vocab_size": 258,
    "epochs": 300,
    "batch_size": 32,
    "learning_rate": 1e-3,
}


def scaled_dot_product_attention(q, k, v, mask=False):
    """Perform scaled dot-product attention on input tensors.
    Only operates on the last two dimensions of input tensors.
    
    Args:
        q (tf.Tensor): Query tensor of shape (..., seq_length, d_q)
        k (tf.Tensor): Key tensor of shape (..., seq_length, d_q)
        v (tf.Tensor): Value tensor of shape (..., seq_length, d_v)
    
    Returns:
        tf.Tensor of shape (..., seq_length, d_v)
    """
    assert q.shape[-1] == k.shape[-1]
    assert q.shape[-2] == k.shape[-2]
    assert k.shape[-2] == v.shape[-2]
    
    if mask:
        mask_matrix = generate_mask_matrix(tf.shape(q)[-2])
    else:
        mask_matrix = tf.zeros((tf.shape(q)[-2], tf.shape(q)[-2]))
    
    dimension = tf.cast(tf.shape(q)[-1], dtype=tf.float32)
    
    scores = tf.matmul(q, k, transpose_b=True) # (..., seq_length, seq_length)
    scores = scores / tf.math.sqrt(dimension) # (..., seq_length, seq_length)
    scores = scores + np.finfo(np.float32).min * mask_matrix
    scores = tf.nn.softmax(scores, axis=-1) # (..., seq_length, seq_length)

    out = tf.matmul(scores, v) # (..., seq_length, d_v)
    
    return out


def test_sdpa():
    """Test the scaled dot-product attention function.
    
    Returns:
        tf.Tensor of value [[550.    5.5]]
    """
    np.set_printoptions(suppress=True)

    k = tf.constant(
        [[[
            [10, 0, 0],
            [0, 10, 0],
            [0, 0, 10],
            [0, 0, 10],
        ]]],
        dtype=tf.float32
    )

    v = tf.constant(
        [[[
            [1, 0],
            [10, 0],
            [100, 5],
            [1000, 6],
        ]]],
        dtype=tf.float32
    )

    q = tf.constant(
        [[[
            [0, 0, 10],
            [0, 0, 10],
            [0, 0, 10],
            [0, 0, 10],
        ]]],
        dtype=tf.float32
    )
    
    att = scaled_dot_product_attention(q, k, v, mask=True)
    
    return att


def generate_mask_matrix(seq_length):
    """Create a mask matrix to keep the model from attending to a token it must predict or to those that follow it.
    Simply an upper-triangular matrix filled with ones (with zeros on the diagonal).
    """
    shape = (seq_length, seq_length)
    mask = 1 - tf.linalg.band_part(
        tf.ones(shape),
        -1,
        0
    )
    return mask


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert dim % num_heads == 0, "Dim is not divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.depth = self.dim // self.num_heads
        
        self.dense_Q = Dense(self.dim, name="Q")
        self.dense_K = Dense(self.dim, name="K")
        self.dense_V = Dense(self.dim, name="V")
    
    def reshape_dense_output(self, x):
        """Reshape output from dense_{Q,K,V} by splitting the last dimension
        into heads, and reordering dimensions so that seq_length is
        second_to_last (for sending into SDPA).
        
        Args:
            x: tf.Tensor of shape (batch_size, seq_length, self.dim)
        
        Returns:
            tf.Tensor of shape (batch_size, self.num_heads, seq_length, self.depth)
        """
        x = tf.reshape(x, (tf.shape(x)[0], tf.shape(x)[1], self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        assert len(x.shape) == 3, "Input should be of shape (batch_size, seq_length, d), not {}".format(x.shape)
        
        q = self.reshape_dense_output(self.dense_Q(x)) # (batch_size, num_heads, seq_length, depth)
        k = self.reshape_dense_output(self.dense_K(x)) # (batch_size, num_heads, seq_length, depth)
        v = self.reshape_dense_output(self.dense_V(x)) # (batch_size, num_heads, seq_length, depth)
        
        att = scaled_dot_product_attention(q, k, v, True) # (batch_size, num_heads, seq_length, depth)
        att = tf.transpose(att, perm=[0, 2, 1, 3]) # (batch_size, seq_length, num_heads, depth)
        att = tf.reshape(att, (tf.shape(att)[0], tf.shape(att)[1], -1)) # (batch_size, seq_length, dim)
        
        return att


class EncoderBlock(tf.keras.Model):
    def __init__(self, dim, ff_hidden_size, num_heads):
        super(EncoderBlock, self).__init__()
        self.dim = dim
        self.mha = MultiHeadAttention(dim, num_heads)
        self.ff1 = Dense(ff_hidden_size, activation="relu")
        self.ff2 = Dense(dim, activation="relu")
        
        self.norm_after_mha = LayerNormalization()
        self.norm_after_ff = LayerNormalization()
    
    def call(self, x):
        mha_output = self.mha(x)
        mha_output = self.norm_after_mha(x + mha_output)
        
        ff_output = self.ff2(self.ff1(mha_output))
        ff_output = self.norm_after_ff(mha_output + ff_output)
        
        return ff_output


class Transformer(tf.keras.Model):
    def __init__(self, d_model, ff_hidden_size, num_blocks, num_heads, vocab_size, max_pos_encoding, **_):
        super(Transformer, self).__init__(dynamic=True)
        self.d_model = d_model
        self.ff_hidden_size = ff_hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.max_pos_encoding = max_pos_encoding

        self.embedding = Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = tf.convert_to_tensor(
            get_positional_encodings(self.max_pos_encoding, self.d_model),
            dtype="float32",
        )[tf.newaxis, ...]

        self.blocks = [
            EncoderBlock(dim=self.d_model, ff_hidden_size=self.ff_hidden_size, num_heads=self.num_heads)
            for _ in range(self.num_blocks)
        ]

    def call(self, x):
        x = self.embedding(x)
        
        # print("You are calling the TRANSFORMER service!", x)
        
        x = tf.math.add(
            x,
            self.pos_encoding[:, :tf.shape(x)[1], :],
            name="positional_encoding"
        )
        
        for block in self.blocks:
            x = block(x)
        
        x = (x @ tf.transpose(self.embedding.variables[0])) # (batch_size, seq_length, vocab_size)
        
        return x


def multi_sparse_cross_entropy(y_true, y_pred):
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )(y_true , y_pred)

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    
    return tf.reduce_mean(loss, axis=-1)



def generate_sampled(model, encoder, seq_length, nb_tokens_to_gen, prompt, power=1):
    """Generate a sequence of tokens starting from given starting string using the sampling method.
    
    Args:
        nb_tokens_to_gen (int): Number of tokens to generate past the starting sequence.
        prompt (str): String to start from.
        power (float): Power to raise probabilities at before sampling.
            A higher power means a less risky sampling.
            An infinite power would be equivalent to greedy sampling.
    
    Returns:
        tuple of strings: Generated tokens.
    """

    print("Generating sampled...")

    text = encoder.encode(prompt)
    text = [ t-1 for t in text]

    assert len(text) >= seq_length, "Text encoded is {} length, which is less than {}".format(len(text), seq_length)
    
    for i in tqdm(range(nb_tokens_to_gen)):
        probas = model.predict(
            np.array(text[-seq_length:])
            .reshape(1, seq_length)
        )[0]
        probas = probas**power
        probas = probas / probas.sum()
        # 0 is excluded, reserved for padding
        # otherwise, it throws an error
        # See : https://github.com/tensorflow/datasets/issues/702
        # Line 217-218 in tensorflow_datasets/core/features/text/subword_text_encoder.py
        next = np.random.choice(np.arange(len(probas)), p=probas)
        text.append(next)

    text = [t+1 for t in text]
    return encoder.decode(text)


def get_max_model_outputs(model, prompt, seq_length=hparams["seq_length"]):
    prompt = np.array(pad_or_cut(prompt, seq_length)).reshape(1, -1)
    return np.argmax(model.predict(prompt), -1).flatten()
    

def predict_probas_with_transformer(model, prompt, seq_length=hparams["seq_length"], apply_softmax=True):
    """Feed a prompt (sequence of ints representing tokens) into a transformer model and return its
    vector of predicted probabilities for the next token.
    
    Args:
        model (tf.keras.Model): Transformer model.
        prompt (list of ints): Prompt to start the sentence. Can be left empty.
            If higher than seq_length, only the last seq_length tokens are taken into account.
        seq_length (int): Sequence length to pad or cut to before feeding into the model.
            Default: hparams["seq_length"]
        apply_softmax (bool): Whether to apply a softmax function to the output of the transformer.
            Set to False if a softmax is already applied in the model.
            Default: True.
    
    Returns:
        np.array: Vector of probabilities for the next token.
    """
    prompt_length = max(len(prompt), seq_length)
    prompt = np.array(pad_or_cut(prompt, seq_length)).reshape(1, -1)
    
    probas = model.predict(prompt)[0, prompt_length-1, :]
    
    if apply_softmax:
        probas = tf.nn.softmax(probas)
    
    return probas


def calculate_perplexity(model, X_test, y_test, epsilon=0.0001):
    probas = []
    for X, y in tqdm(zip(X_test, y_test), total=len(X_test)):
        y = np.argmax(y)
        y_pred = model.predict(X.reshape(1, -1))[0]
        probas.append(y_pred[y])
    
    probas = np.array(probas) + epsilon
    entropy = np.mean(-np.log2(probas))
    perplexity = 2**entropy
    
    return perplexity


def load_train_test_val_encoder(data=DATA_PATH, sample=2E-5, target_vocab_size=hparams["target_vocab_size"]):
    """Load and encode the data, then return train, test and validation data sets

    Args:
        data (str): path to data
        sample (float): size of sample (full dataset percent)

    Returns:
        train, test and validation data sets
    """
    text = load_data(data, sample=sample, split_on="\n")
    X, encoder = encode_data(text, tokens="subwords", target_vocab_size=target_vocab_size)
    train, test = train_test_split(X, test_size=0.1, shuffle=False)
    train, val = train_test_split(train, test_size=0.3, shuffle=False)
    # see : tensorflow_datasets/core/features/text/subword_text_encoder.py
    # encoder.vocab_size returns 1 + len(self._subwords) + text_encoder.NUM_BYTES
    return train, test, val, encoder


def main(log_training=True, comment=""):
    train, test, val, encoder = load_train_test_val_encoder(data=DATA_PATH, sample=1E-3)
    vocab_size = encoder.vocab_size - 1
    
    X_train, y_train = split_into_X_y(train)
    X_test, y_test = split_into_X_y(test)
    X_val, y_val = split_into_X_y(val)
   
    # Form model

    model = Transformer(
       vocab_size=vocab_size,
       **hparams
    )
    model.compile(
        loss=multi_sparse_cross_entropy,
        optimizer=tf.keras.optimizers.Adam(hparams["learning_rate"]),
        # metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    
    print("Non mais allô quoi... Ouvalument!")
    summary = []
    # model.summary(print_fn=lambda x: summary.append(x))
    summary = "\n".join(summary)
    print(summary)

    history = model.fit(
        X_train,
        y_train,
        # steps_per_epoch=np.ceil(len(X_train)/batch_size),
        batch_size=hparams["batch_size"],
        epochs=hparams["epochs"],
        validation_data=(X_val, y_val),
        callbacks = [
            EarlyStopping(monitor="val_loss", min_delta=0, patience=5, verbose=1),
            ModelCheckpoint("logs/tensorflow_transformer/saved_models/checkpoint.h5", monitor="val_loss", save_best_only=True)
        ]
        # use_multiprocessing=False,
    )
    
    # perp = calculate_perplexity(model, X_test, y_test)
    
    prompt = "Il y a bien longtemps , dans un pays lointain , "
    # generated_text = generate_sampled(model, encoder, hparams["seq_length"], 500, prompt, 1)
    # print("Texte généré")
    # print(generated_text)
    
    log = {
        "hyperparameters": hparams,
        "history": history.history,
        "summary": summary,
        "sample": {
            "prompt": prompt,
            # "output": generated_text[len(prompt):],
        },
        #"metrics": {
        #    "perplexity": perp,
        #},
        "data_size": {
            "train": len(X_train),
            "val": len(X_val),
            "test": len(X_test),
        },
        "comment": comment,
    }
    
    if log_training:
        add_to_log(log)
    
    # pprint(log, compact=True)
    
    return model, encoder, log, X_train, y_train
