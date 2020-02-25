import numpy as np
import tensorflow as tf

def get_positional_encodings(seq_length, d_model):
    """Compute transformer positional encodings.
    
    Args:
        seq_length (int): Sequence length of the model.
        d_model (int): Dimension of the model.
    
    Returns:
        np.array with shape (seq_length, d_model) which can be directly summed
        with an embeddings tensor.
    """
    positions = np.arange(seq_length)[:, np.newaxis]
    indices = np.arange(d_model)[np.newaxis, :]
    angles = positions / (
        1e4 ** (
            2*indices/d_model
        )
    )
    encodings = np.zeros(shape=angles.shape)
    encodings[:, 0::2] = np.sin(angles[:, 0::2])
    encodings[:, 1::2] = np.cos(angles[:, 1::2])
    
    return encodings


def get_positional_encodings_tf(seq_length, d_model):
    """Compute transformer positional encodings.
    Wrriten with Tensorflow.
    
    Args:
        seq_length (int): Sequence length of the model.
        d_model (int): Dimension of the model.
    
    Returns:
        np.array with shape (seq_length, d_model) which can be directly summed
        with an embeddings tensor.
    """
    positions = tf.range(seq_length, dtype="float32")[:, tf.newaxis]
    indices = tf.range(d_model, dtype="float32")[tf.newaxis, :]
    angles = positions / (
        1e4 ** (
            2*indices/tf.cast(d_model, "float32")
        )
    )
    encodings = tf.zeros(shape=angles.shape)
    encodings[:, 0::2] = tf.math.sin(angles[:, 0::2])
    encodings[:, 1::2] = tf.math.cos(angles[:, 1::2])
    
    return encodings
