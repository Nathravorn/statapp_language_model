import numpy as np
import pandas as pd
from statapp.common.utils import array_to_multi_indexed_series

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
    entropy = array_to_multi_indexed_series(entropy, names=["batch", "layer", "head", "pos"], val_name="entropy")
    df = entropy.reset_index()
    
    return df
