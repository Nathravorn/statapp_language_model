import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds

def load_dataset_as_str(path):
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def sequencer(vector, seq_length):
    for i in range(len(vector)-seq_length):
        yield vector[i:i+seq_length]

#HParams
seq_length = 24
num_blocks = 1
d_model = 32
d_query_key = 32
num_heads = 4

"""
Classes to code:
- EncoderBlock(tf.keras.Model)
- MultiHeadAttention(tf.keras.layers.Layer)
- ScaledDotProductAttention(tf.keras.layers.Layer)
"""

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

if __name__ == "__main__":
    # Loading data
    text = load_data_as_str("data/fr.train.top1M.txt")[:10000]
    encoder = tfds.features.text.SubwordTextEncoder([text], target_vocab_size=1000)
    X = encoder.encode(text)
    X_train, X_test = train_test_split(X, test_size=0.1)
    X_train = sequencer(X_train, seq_length+1)
    X_test = sequencer(X_test, seq_length+1)
    
    inputs = tf.keras.Input(shape=(None,), dtype='int32')
    embedded = tf.keras.layers.Embedding(encoder.vocab_size, d_model)(inputs)
    


    