import numpy as np
import pandas as pd

def compute_mean_distance(array):
    """Calculate mean distance
    
    Args:
        array (numpy array): Input array, with two last axis of same dimensions
        
    Returns:
        np.array: Array of mean distances, with two less dimensions (the two last axis) than the input array.
    """
    seq_length = array.shape[-1]
    assert seq_length == array.shape[-2], "The two last dimensions of the array are not equal"
    
    distances = np.abs(np.arange(0,seq_length).repeat(seq_length).reshape(seq_length,-1)+np.arange(0,-seq_length,-1))
    
    return (array*distances).mean(axis = (-1,-2))
    
    