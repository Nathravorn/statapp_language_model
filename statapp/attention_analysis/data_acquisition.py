import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import XLMRobertaModel, XLMRobertaConfig
from transformers import RobertaConfig, RobertaModel

def get_tokenizer_and_model(model_name):
    assert model_name in ["xlm-roberta-base", "roberta-base", "bert-base-multilingual-cased"], "Unrecognized model name: {}".format(model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, output_attentions=True)
    model = AutoModel.from_pretrained(model_name, config=config)
    
    return tokenizer, model


def get_attentions(text, model_name="xlm-roberta-base", seq_length=512, batch_size=8, as_array=True):
    """Run specified Transformer model on a given text and output its attention values for that text.
    
    Args:
        text (str): Text to pass to the model.
        model_name (str): model_name argument passed to `get_tokenizer_and_model`.
            Default: "xlm-roberta-base".
        seq_length (int): Length (in number of tokens) of sequences to cut the text into.
            Default: 512.
        batch_size (int): Size of the batches to pass to the model. Drastically improves performance at the cost of memory.
            Default: 8.
        as_array (bool): Whether to throw away the last part of the input (which is not of length seq_length)
            in order to return attentions as a numpy array instead of a list. Recommended.
            Default: True.
    
    Returns:
        if as_array == True:
            Array of shape: (n_batches, n_layers, n_heads, seq_length, seq_length),
            Representing:   (batch    , layer   , head   , position  , position  ).

        else:
            list (of length num_sequences*batch_size)
            of arrays of shape: (n_layers, n_heads, seq_length, seq_length).
            Representing:       (layer   , head   , position  , position  ).
    """
    tokenizer, model = get_tokenizer_and_model(model_name)
    
    tokens = tokenizer.encode(text)
    
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
        for inp in tqdm(inputs, desc="Running model")
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
