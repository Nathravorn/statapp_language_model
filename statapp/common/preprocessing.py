import numpy as np
import tensorflow_datasets as tfds


def load_data(path, sample=1, split_on=" "):
    """Load a text dataset, optionally sampling the data rounding at some splitting string.
    
    Args:
        path (str): Path to a text file.
        sample (float): (approximate) Proportion of the dataset to load. 1 loads everything.
            Default: 1.
        split_on (str): String to split dataset on, e.g. "\n".
            When sampling, the dataset will be cut at the previous split_on substring.
            Ignored if sample==1.
            If None, samples the dataset by cutting at the previous character.
            Default: " ".
    
    Returns:
        string: Dataset.
    """
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    
    if sample != 1:
        text = text[:int(len(text)*sample)]
        if split_on is not None:
            text = text[:-text[::-1].find(split_on)-1]
    
    return text
    

def encode_data(text, tokens="subwords", target_vocab_size=1000):
    """Encode a text based on some tokenizing method.
    
    Args:
        text (str): Text to encode.
        tokens (str): Tokenization method. Currently supports "subwords" and "characters".
        target_vocab_size (int): Target vocabulary size for the encoder. Must be >= 1000.
    
    Returns:
        list of ints: Encoded text.
        encoder object: object supporting the "encode" and "decode" methods,
            such as a tensorflow_datasets.features.text.SubwordTextEncoder.
    """
    sentences = text.split('\n')
    if tokens=="subwords":
        encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (s for s in sentences),
            target_vocab_size=target_vocab_size
        )
    
    elif tokens=="characters":
        encoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (s for s in sentences),
            target_vocab_size=target_vocab_size,
            max_subword_length=1,
        )
    
    encoded = encoder.encode(text)
    return encoded, encoder


def split_into_X_y(samples, seq_length, one_hot_encode_y=False, vocab_size=None):
    """Split a list of text samples into two lists: X, a list of "input" sequences of length seq_length,
    and y, a list of "target" tokens (optioanlly one-hot-encoded) which correspond to the token following X.
    
    Args:
        samples (list of lists of tokens): Samples to split.
        seq_length (int): Length of each sequence in X.
        one_hot_encode_y (bool): Whether to one-hot encode the y vectors or keep them as ints.
            Default: False.
        vocab_size (int): Length of the one-hot encoding vectors for y.
            Only necessary if one_hot_encode_y is True.
    
    Returns:
        list of lists of tokens: X
        list of one-hot-encoded tokens: y
    """
    assert not (one_hot_encode_y and (vocab_size is None)), "vocab_size argument needs to be set in order to one-hot encode y."
    
    X = [samples[i:i+seq_length] for i in range(len(samples)-seq_length)]
    y_integers = [
        samples[i+seq_length]
        for i in range(len(samples)-seq_length)
    ]
    
    if one_hot_encode_y:
        y = []
        for index in y_integers:
            one_hot_vector = np.zeros(vocab_size)
            one_hot_vector[index] = 1
            y.append(one_hot_vector)
    else:
        y = y_integers
    
    return np.array(X), np.array(y)
