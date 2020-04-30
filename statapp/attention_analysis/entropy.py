import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from statapp.common.utils import array_to_multi_indexed_series
from statapp.attention_analysis.constants import data_file_names, class_ids
from statapp.attention_analysis.data_acquisition import get_tokenizer_and_model, get_attentions

def compute_entropy(array, axis=-1, epsilon=0.001):
    """Calculate entropy along specified axis of given array.
    
    Args:
        array (numpy array): Input array.
        axis (int): Over which array to compute entropy (i.e. over which array to perform the sum).
            Default: -1.
        epsilon (float): Value to add to the array values before going into log function.
            Used to avoid nans in the computation.
            Default: 0.001.
    
    Returns:
        np.array: Array of entropies, with one less dimension (the one specified as the `axis` argument) than the input array.
    
    Example:
        >>> arr = np.array([[0.1, 0.9], [0.5, 0.5]])
        >>> compute_entropy(arr)
        array([0.3230885 , 0.69114918])
    """
    return - (array * np.log(array + epsilon)).sum(axis=axis)

def get_entropy_df(attentions, **kwargs):
    """Wrapper around compute_entropy to return a pretty DataFrame of entropies based on a numpy array of attentions.
    """
    entropy = compute_entropy(attentions, **kwargs)
    entropy = array_to_multi_indexed_series(entropy, names=["seq", "layer", "head", "pos"], val_name="entropy")
    df = entropy.reset_index()
    
    return df

def get_entropy_over_languages(model_name, data_folder, random_weights=False, batch_size=64, verbose=True):
    print_if_verbose = lambda *x: print(*x) if verbose else None
    
    print_if_verbose("Loading model...")
    tokenizer, model = get_tokenizer_and_model(model_name, random_weights=random_weights)
    print_if_verbose("    done.")
    
    out = []

    for file_name, language in data_file_names.items():
        print_if_verbose("Computing entropies for", language + "...")
        start_time = timer()

        text = "\n".join(load_data(os.path.join(data_folder, "attention_data", file_name + "-ud-test-sent_segmented.txt", sample=1, split_on="\n"))
        tokens = tokenizer.encode(text)

        att = get_attentions(tokens, model, seq_length=64, batch_size=batch_size)

        df = get_entropy_df(att)
        df["language"] = language
        
        df = (
            df
            .groupby(["seq", "layer", "head", "language"])
            .mean()
            .loc[:, "entropy"]
            .reset_index()
        )
        
        out.append(df)

        exec_time = timer() - start_time
        print_if_verbose("    done. ({} seconds)\n".format(exec_time))
    
    return pd.concat(out)
    
def load_entropy_over_languages(model_name, data_folder, average_over_heads=True):
    df = pd.read_hdf(os.path.join(data_folder, "entropy_data/{}.h5".format(model_name)), "df")
    df = df.set_index(["seq", "layer", "head", "language"]).unstack(["layer", "head"])
    
    if average_over_heads:
        df = df.mean(axis=1, level="head")
    
    y = df.index.get_level_values("language").tolist()
    y = list(map(class_ids.get, y))
    X = df.droplevel("language", axis=0)
    
    return X, y
