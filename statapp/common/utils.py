import statapp
import os
import json
import datetime
import numpy as np
import pandas as pd
from scipy.stats import rankdata

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types. Useful for logging numpy arrays.
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def add_to_log(entry, path=None, auto_add=["id", "date"]):
    """Add to the training log file the specified entry.
    By default, adds a few automatically generated fields to the entry.
    
    Args:
        entry (dict): Entry to add to the log.
        path (str): Path to the log file.
            Default: "folder_containing_statapp_package/logs/tensorflow_transformer/log.json"
        auto_add (list of str): Keys to automatically add to the entry before logging.
            Supported keys:
                "id": A number equal to 1 + the maximum id in the current log.
                "date": A string representing the current date and time.
            Default: ["id", "date"].
        add_id (bool): Whether to add "id" key to entry.
            Default: True.
    """
    # Implement default path
    if path is None:
        path = os.path.join(os.path.dirname(statapp.__name__), "logs",  "tensorflow_transformer", "log.json")
    
    # Read log file
    with open(path, "r") as file:
        log = json.load(file)
    
    # Add auto fields
    if "id" in auto_add:
        current_max_id = max([el.get("id", 0) for el in log])
        entry["id"] = current_max_id + 1
    if "date" in auto_add:
        entry["date"] = datetime.datetime.today().strftime("%Y-%m-%dT%H:%M:%S")
    
    log.append(entry)
    
    # Write log file
    with open(path, "w") as file:
        json.dump(log, file, indent=4, sort_keys=True, cls=NumpyEncoder)


def pad_or_cut(seq, seq_length, padder=[0]):
    """Pad or cut a sequence to the specified length.
    Typically used for lists or strings.
    
    Args:
        seq (iterable): Sequence to pad (if too short) or cut (if too long).
        seq_length (int): Length to pad or cut the sequence to.
        padder: Object to use for padding.
            Must support addition with original sequence and multiplication by an int.
            Default: [0]
    
    Returns:
        iterable: padded or cut sequence of length seq_length.
    
    Examples:
        >>> pad_or_cut([1, 2, 3], 4, [0])
        [1, 2, 3, 0]
        >>> pad_or_cut([1, 2, 3], 2, [0])
        [1, 2]
        >>> pad_or_cut("abc", 5, ".")
        "abc.."
    """
    return seq[:seq_length] + padder * max(seq_length - len(seq), 0)


def array_to_multi_indexed_series(array, names=None, val_name=None, number_from_1=False):
    """Convert a numpy array with an arbitrary number of dimensions to a pd.Series object with
    a MultiIndex representing all the dimensions of the input array. Can be easily reshaped into a
    DataFrame with desired columns using reset_index.

    Args:
        array (np.array): Multi-dimensional array to turn into a Series.
        names (list of strings): Names of the created index levels.
            If not None, must be of the same length as the number of dimensions of the input array.
            If None, leave unnamed.
            Default: None.
        val_name (str): Name of the returned Series.
            Will be the name of the value column if reset_index is used on the returned Series.
            If None, leave unnamed.
            Default: None.
        number_from_1 (bool): Whether to number the index starting from 1 instead of 0.
            This may make it more human-readable at the cost of losing indexing equivalence with the
            input array.
            Default: False.
    
    Examples:
        >>> a = np.array([[0.1, 0.9], [0.5, 0.5]])
        >>> array_to_multi_indexed_series(a, ["f", "s"], "val")
        f  s
        0  0    0.1
           1    0.9
        1  0    0.5
           1    0.5
       Name: val, dtype: float64
       
       >>> a = np.random.randn(2, 2, 2, 2)
       >>> a[0, 0, 1, 1]
       2.1682415236887196
       >>> array_to_multi_indexed_series(a)[0, 0, 1, 1]
       2.1682415236887196
    """
    if names is not None:
        assert len(names) == len(array.shape)
    
    start = 1 if number_from_1 else 0
    
    iterables = [range(start, dim+start) for dim in array.shape]
    index = pd.MultiIndex.from_product(iterables, names=names)
    
    srs = pd.Series(array.reshape(-1), index=index)
    
    if val_name is not None:
        srs = srs.rename(val_name)
    
    return srs


def survival_function(srs):
    """Return the survival probabilities of a data series.
    
    Args:
        srs (array-like): Series to analyze. Index is irrelevant
    
    Returns:
        pd.Series: Index: Ordered unique values in the original series.
                   Values: Probability of being >= corresponding index value.
    """
    srs = pd.Series(srs).dropna()
    
    n = len(srs)
    ranks = rankdata(srs) - 1
    survival = 1 - (ranks / (n - 1))
    
    return pd.Series(survival, index=srs).sort_index()
