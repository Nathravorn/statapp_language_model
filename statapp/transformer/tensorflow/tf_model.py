import sys
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, TimeDistributed

this_file_dir = os.path.dirname(__file__)
sys.path.append(os.path.dirname(this_file_dir))
from common import get_positional_encodings, load_data, split_into_X_y, load_sets

#HParams
seq_length = 32
num_blocks = 2
d_model = 128
d_query = 128
num_heads = 32
target_vocab_size = 1000

epochs = 2
batch_size = 256

def scaled_dot_product_attention(q, k, v):
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
    # assert q.shape[-2] == k.shape[-2]
    assert k.shape[-2] == v.shape[-2]
    
    
    dimension = tf.cast(q.shape[-1], dtype=tf.float32)
    
    scores = tf.matmul(q, k, transpose_b=True) # (..., seq_length, seq_length)
    scores = scores / tf.math.sqrt(dimension) # (..., seq_length, seq_length)
    att_weights = tf.nn.softmax(scores, axis=1) # (..., seq_length, seq_length)
    
    out = tf.matmul(att_weights, v) # (..., seq_length, d_v)
    
    return out

def test_sdpa():
    # Output should be [[550.    5.5]]
    np.set_printoptions(suppress=True)

    k = tf.constant([[10,0,0],
                      [0,10,0],
                      [0,0,10],
                      [0,0,10]], dtype=tf.float32)  # (4, 3)

    v = tf.constant([[   1,0],
                      [  10,0],
                      [ 100,5],
                      [1000,6]], dtype=tf.float32)  # (4, 2)
    
    q = tf.constant([[0, 0, 10]], dtype=tf.float32)
    
    att = scaled_dot_product_attention(q, k, v)
    
    print("q:", q)
    print("k:", k)
    print("v:", v)
    print("out:", att)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.depth = self.dim // self.num_heads
        
        self.dense_Q = Dense(self.dim)
        self.dense_K = Dense(self.dim)
        self.dense_V = Dense(self.dim)
    
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
        
        att = scaled_dot_product_attention(q, k, v) # (batch_size, num_heads, seq_length, depth)
        att = tf.transpose(att, perm=[0, 2, 1, 3]) # (batch_size, seq_length, num_heads, depth)
        att = tf.reshape(att, (tf.shape(att)[0], tf.shape(att)[1], -1)) # (batch_size, seq_length, dim)
        
        return att

class EncoderBlock(tf.keras.Model):
    def __init__(self, dim, num_heads):
        super(EncoderBlock, self).__init__()
        self.dim = dim
        self.mha = MultiHeadAttention(dim, num_heads)
        self.ff = Dense(dim)
        
        self.norm_after_mha = LayerNormalization()
        self.norm_after_ff = LayerNormalization()
    
    def call(self, x):
        mha_output = self.mha(x)
        mha_output = self.norm_after_mha(x + mha_output)
        
        ff_output = self.ff(mha_output)
        ff_output = self.norm_after_ff(mha_output + ff_output)
        
        return ff_output

def build_transformer(vocab_size):
    inputs = tf.keras.Input(shape=(seq_length,), dtype='int32')
    embedded = Embedding(vocab_size, d_model)(inputs) # (batch_size, seq_length, d_model)
    pos_encodings = tf.constant(get_positional_encodings(seq_length, d_model), dtype=tf.float32)
    encoded = tf.math.add(embedded, pos_encodings, name="positional_encoding")
    
    x = encoded
    for _ in range(num_blocks):
        x = EncoderBlock(dim=d_model, num_heads=num_heads)(x)
    
    x = TimeDistributed(Dense(d_model//8))(x)
    x = tf.keras.layers.Reshape((seq_length*d_model//8,))(x)
    
    x = Dense(d_model, activation="relu")(x)
    outputs = Dense(vocab_size, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs) # (batch_size, seq_length, vocab_size)
    
    return model

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
    text = encoder.encode(prompt)
    assert len(text) >= seq_length
    
    for i in tqdm(range(nb_tokens_to_gen)):
        probas = model.predict(
            np.array(text[-seq_length:])
            .reshape(1, seq_length)
        )[0]
        probas = probas**power
        probas = probas / probas.sum()
        next = np.random.choice(np.arange(len(probas)), p=probas)
        text.append(next)
    
    return encoder.decode(text)

def calculate_perplexity(model, X_test, y_test):
    probas = []
    for X, y in tqdm(zip(X_test, y_test), total=len(X_test)):
        y = np.argmax(y)
        y_pred = model.predict(X.reshape(1, -1))[0]
        probas.append(y_pred[y])
    
    probas = np.array(probas)
    entropy = np.mean(-np.log2(probas))
    perplexity = 2**entropy
    
    return perplexity

def main(tokens="subwords"):
    train, val, test, encoder = load_sets(tokens=tokens, sample=0.002, target_vocab_size=target_vocab_size)
    vocab_size = encoder.vocab_size
    
    X_train, y_train = split_into_X_y(train, seq_length, vocab_size)
    X_test, y_test = split_into_X_y(test, seq_length, vocab_size)
    X_val, y_val = split_into_X_y(val, seq_length, vocab_size)
    
    # Form model
    model = build_transformer(vocab_size=vocab_size)
    # from_logits: Whether y_pred is expected to be a logits tensor
    model.compile(
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )
    
    print("Non mais allô quoi... Ouvalument!")
    print(model.summary())

    history = model.fit(
        X_train,
        y_train,
        # steps_per_epoch=np.ceil(len(X_train)/batch_size),
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
    )
    # print(history.history)
    
    perp = calculate_perplexity(model, X_test, y_test)
    print("Perplexity:", perp)
    
    generated_text = generate_sampled(model, encoder, seq_length, 200, "Il y a bien longtemps, dans un pays lointain où les oiseaux", 1)
    print(generated_text)
    
    return model, encoder
