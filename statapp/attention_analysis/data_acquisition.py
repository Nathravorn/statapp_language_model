import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers import RobertaConfig, RobertaModel
from statapp.common.utils import array_to_multi_indexed_series

def get_tokenizer_and_model(model_name):
    """Get huggingface tokenizer and model for specified model name.
    By default, get the model with its pretrained weights, but optionally randomly initialize the weights instead.
    
    Args:
        model_name (str): Name of the model to import.
            If ends with "-random", instead import the name before that suffix, but with randomly initialized weights.
            Example:
            - get_tokenizer_and_model("roberta-base") gets the pretrained tokenizer and model for "roberta-base".
            - get_tokenizer_and_model("roberta-base-random") gets the pretrained tokenizer and the randomly initialized model for "roberta-base".
    
    Returns:
        transformers Tokenizer object
        transformers Model object
    """
    if model_name.endswith("-random"):
        model_name = model_name[:-7]
        random_weights = True
    else:
        random_weights = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, output_attentions=True)
    
    if random_weights:
        model = AutoModel.from_config(config=config)
    else:
        model = AutoModel.from_pretrained(model_name, config=config)
    
    return tokenizer, model


def get_attentions(tokens, model, seq_length=64, batch_size=64, as_array=True, verbose=False):
    """Run specified Transformer model on a given text and output its attention values for that text.
    
    Args:
        tokens (list of ints): Output of tokenizer.encode(). Tokens to pass to the model.
        model (huggingface/transformers Model): Model to run on tokens.
        seq_length (int): Length (in number of tokens) of sequences to cut the text into.
            Default: 64.
        batch_size (int): Size of the batches to pass to the model. Drastically improves performance at the cost of memory.
            Default: 64.
        as_array (bool): Whether to throw away the last part of the input (which is not of length seq_length)
            in order to return attentions as a numpy array instead of a list. Recommended.
            Default: True.
        verbose (bool): Whether to print progressbar.
            Default: False.
    
    Returns:
        if as_array == True:
            Array of shape: (n_sequences, n_layers, n_heads, seq_length, seq_length),
            Representing:   (sequence   , layer   , head   , position  , position  ).
            
            The first dimension does not depend on batch_size.

        else:
            list (of length n_sequences)
            of arrays of shape: (n_layers, n_heads, seq_length, seq_length).
            Representing:       (layer   , head   , position  , position  ).
    """
    sequences = [
        tokens[i:i+seq_length]
        for i in range(0, len(tokens), seq_length)
    ]
    
    if as_array and (len(sequences) > 1) and (len(sequences[-1]) != seq_length):
        sequences = sequences[:-1]
    
    inputs = [
        torch.tensor(sequences[i:i+batch_size])
        for i in range(0, len(sequences), batch_size)
    ]
    
    outputs = [
        model(inp)
        for inp in tqdm(inputs, desc="Running model", disable=not verbose)
    ]
    
    attentions = [
        (
            np.array(
                [tensor.detach().numpy() for tensor in out[2]] # out[2] contains attention values ordered by block depth.
            )
            .transpose((1, 0, 2, 3, 4)) # Swapping depth with batch_size, making batch_size the outermost dimension.
        )
        for out in outputs
    ]
    
    if as_array:
        attentions = np.concatenate(attentions, axis=0)
    
    return attentions

def attention_to_df(att):
    return array_to_multi_indexed_series(att, names=["seq", "layer", "head", "pos1", "pos2"], val_name="attention").reset_index()
