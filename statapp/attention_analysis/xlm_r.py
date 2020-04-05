import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from transformers import XLMRobertaModel, XLMRobertaConfig

def get_tokenizer_and_model():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    config = XLMRobertaConfig.from_pretrained("xlm-roberta-base", output_attentions=True)
    model = XLMRobertaModel.from_pretrained("xlm-roberta-base", config=config)
    return tokenizer, model

def get_attentions(text, seq_length=512, batch_size=1, as_array=False):
    """
    Output is a list (of length num_sequences*batch_size)
    of arrays of shape: (num_blocks, num_heads, seq_length, seq_length).
    
    If as_array is set to True, chop the last segment of text (if it's not the same length as the
    rest) and concatenate all arrays into a single one.
    
    """
    tokenizer, model = get_tokenizer_and_model()
    
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
