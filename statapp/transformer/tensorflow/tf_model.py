import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization

sys.path.append("..")
from common import get_positional_encodings


def load_data_as_str(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

#HParams
seq_length = 24
num_blocks = 1
d_model = 32
d_query = 32
num_heads = 4
vocab_size = 1000

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
        ff_output = self.norm_after_ff(x + ff_output)
        
        return ff_output

def build_transformer():
    inputs = tf.keras.Input(shape=(None,), dtype='int32')
    embedded = Embedding(encoder.vocab_size, d_model)(inputs) # (batch_size, seq_length, d_model)
    pos_encodings = tf.constant(get_positional_encodings(seq_length, d_model), dtype=tf.float32)
    encoded = embedded + pos_encodings
    
    x = encoded
    for _ in range(num_blocks):
        x = EncoderBlock(dim=d_model, num_heads=num_heads)(x)
    
    x = Dense(vocab_size, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    return model

if __name__ == "__main__":
    # Load data
    text = load_data_as_str("data/fr.train.top1M.txt")[:10000]
    encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            [text], target_vocab_size=vocab_size)
    X = encoder.encode(text)
    X_train, X_test = train_test_split(X, test_size=0.1)
    
    # Form model
    model = build_transformer()
    print(model)
    
    
