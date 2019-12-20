import numpy as np

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
