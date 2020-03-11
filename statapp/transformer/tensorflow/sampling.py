import numpy as np
from tqdm import tqdm

import tensorflow as tf
from statapp.common.sampling import sample_string_sequence

def generate_sample_with_transformer(model, sequence, encoder, max_seq_length=128, gen_length=100, **kwargs):
    """Sample a string sequence from a Transformer model.
    
    Args:
        predictor (Transformer): Transformer model with non-softmaxed input.
        sequence (str): Initial sequence to pass to the predictor.
        encoder (encoder object): Object with methods .encode() and .decode().
            .encode() must take in a string and output a corresponding sequence of tokens.
            .decode() must take in a sequence of tokens and output a corresponding string.
        gen_length (int): How many tokens to generate.
            Default: 100.
        **kwargs: Keyword-arguments to be passed to the sample_from_distribution function. See its docstring for details.
            Do not pass argument "previous_sequence" as it is handled by this function.
    
    Returns:
        str: Sampled text.
    """
    predictor = lambda prompt: predict_probas_with_transformer(model, prompt, max_seq_length=max_seq_length)
    out = sample_string_sequence(predictor, sequence, encoder, gen_length=gen_length, **kwargs)
    return out


def get_max_model_outputs(model, prompt):
    prompt = np.array(prompt).reshape(1, -1)
    return np.argmax(model.predict(prompt), -1).flatten()
    

<<<<<<< Updated upstream
def predict_probas_with_transformer(model, prompt, max_seq_length=128, apply_softmax=True):
=======
def predict_probas_with_transformer(model, prompt, max_seq_length=1024, apply_softmax=True):
>>>>>>> Stashed changes
    """Feed a prompt (sequence of ints representing tokens) into a transformer model and return its
    vector of predicted probabilities for the next token.
    
    Args:
        model (tf.keras.Model): Transformer model.
        prompt (list of ints): Prompt to start the sentence. Can be left empty.
            If higher than seq_length, only the last seq_length tokens are taken into account.
        max_seq_length (int): Maximum sequence length to feed into the model.
            If prompt is longer than this, take the last max_seq_length tokens from it.
            Default: hparams["max_seq_length"].
        apply_softmax (bool): Whether to apply a softmax function to the output of the transformer.
            Set to False if a softmax is already applied in the model.
            Default: True.
    
    Returns:
        np.array: Vector of probabilities for the next token.
    """
    if len(prompt) > max_seq_length:
        prompt = prompt[-max_seq_length:]
    
    prompt = np.array(prompt).reshape(1, -1)
    probas = model.predict(prompt)[0, -1, :]
    
    if apply_softmax:
        probas = tf.nn.softmax(probas)
    
    return probas
