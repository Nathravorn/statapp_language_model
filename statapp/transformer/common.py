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


def load_data(path, sample=1, split_on=None):
    """Load a text dataset, optionally splitting based on some splitting string.
    
    Args:
        path (str): Path to a text file.
        sample (float): (approximate) Proportion of the dataset to load. 1 loads everything.
        If split_on is not None, samples the dataset post-split instead of pre-split.
        If split_on is None, samples the dataset by cutting at the closest space character.
            Default: 1.
        split_on (str): String to split dataset on, e.g. "\n".
            If not None, will shuffle before splitting.
            Default: None.
    
    Returns:
        list of strings: Split dataset. If split_on is None, list has one element.
    """
    with open(path, "r") as file:
        text = file.read()
    
    if split_on is None:
        text = text.split(" ")
        text = " ".join(text[:int(len(text)*sample)])
        return text
    
    else:
        text = text.split(split_on)
        if sample != 1:
            text = np.random.choice(text, size=int(len(text)*sample))
        
        return text
