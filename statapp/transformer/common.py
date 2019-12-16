import numpy as np
import tensorflow_datasets as tfds

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


def load_data(path, sample=1, split_on=" "):
    """Load a text dataset, optionally splitting based on some splitting string.
    
    Args:
        path (str): Path to a text file.
        sample (float): (approximate) Proportion of the dataset to load. 1 loads everything.
            Default: 1.
        split_on (str): String to split dataset on, e.g. "\n".
            When sampling, the dataset will be cut at the previous split_on substring.
            If None, samples the dataset by cutting at the previous character.
            Default: " ".
    
    Returns:
        string: Dataset.
    """
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    
    text = text[:int(len(text)*sample)]
    if split_on is not None:
        text = text[:-text[::-1].find(split_on)-1]
    
    return text
    
    
def split_into_X_y(samples, seq_length, vocab_size):
    """Split a list of samples into two lists: X, a list of "input" sequences of length seq_length,
    and y, a list of "target" tokens (one-hot-encoded according to vocab_size) which correspond to the token following X.
    
    Args:
        samples (list of lists of tokens): Samples to split.
        seq_length (int): Length of each sequence in X.
        vocab_size (int): Length of the one-hot encoding vectors for y.
    
    Returns:
        list of lists of tokens: X
        list of one-hot-encoded tokens: y
    """
    X = [samples[i:i+seq_length] for i in range(len(samples)-seq_length)]
    y_integers = [
        samples[i+seq_length]
        for i in range(len(samples)-seq_length)
    ]
    y = []
    for index in y_integers:
        one_hot_vector = np.zeros(vocab_size)
        one_hot_vector[index] = 1
        y.append(one_hot_vector)
    
    return np.array(X), np.array(y)
