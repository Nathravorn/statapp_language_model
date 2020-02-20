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


def split_into_X_y(samples, seq_length=None):
    """Split a list of text samples into two lists: X, a list of "input" sequences of length seq_length,
    and y, a list of "target" sentences which correspond to the sequences in X shifted by 1.
    
    Args:
        samples (list of lists of ints): Samples to split into X and y.
        seq_length (int or None): Length of each sequence in X.
            Sequences shorter than this will be padded with zeros at the end.
            Sequences longer than this will be truncated.
            If None, sequences are left as is.
            Default: None.
    
    Returns:
        list of lists of tokens: X
        list of lists of tokens: y
        X and y have exactly the same shape.
    """
    # Preprocess sequences, padding or cutting
    if seq_length is not None:
        samples = [
            sample[:seq_length]
            + [0] * min(seq_length - len(sample), 0)
            for sample in samples
        ]
    
    # Split sequences into X and y
    X = [
        sample[:-1]
        for sample in samples
    ]
    
    y = [
        sample[1:]
        for sample in samples
    ]
    
    return X, y
