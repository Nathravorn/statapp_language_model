import numpy as np


def sample_from_distribution(scores, encoder=None, temperature=1, top_k=5, proba_threshold=None, previous_sequence=None, repetition_penalty=1.2):
    """Sample a token from a distribution.
    
    Args:
        scores (array-like): Sequence of scores for next token, which are normalized and interpreted as probabilities.
        temperature (float): Optional. Determines how to transform probabilities before sampling.
            Probabilities are transformed as: p[i] = p[i]**(1/T) / sum(p).
            T = 1 keeps the predictor's distribution intact.
            T -> 0 is equivalent to greedy sampling.
            T -> infinity is equivalent to uniform sampling from the vocabulary.
            Default: 1.
        top_k (int): Optional. How many top candidates to keep before sampling.
            Takes effect after any transformation to the scores (temperature/repetition_penalty).
            Ignored if `nucleus_threshold` is not None.
            If <= 0, do not apply top_k sampling.
            Default: 5.
        proba_threshold (float between 0 and 1): Optional. Alternative to `top_k`. Probability threshold to use for candidate picking.
            After score transformations are applied, choose as top_k the smallest k such that sum(sort_descending(p)[0:k]) >= threshold.
            This method is called nucleus sampling.
            If not None, overrides `top_k`.
            Default: None.
        previous_sequence (list of ints): Optional. Sequence preceding the current token, for use in the repetition penalty.
            If None, no repetition penalty is applied.
            Default: None.
        repetition_penalty (float): Optional. How much to discourage choosing a token which is present in the previous sequence.
            Probabilities are transformed as: p[i] = p[i]**(R[i]/T) / sum(p),
            where T is the temperature and R[i] is equal to this parameter if the
            corresponding token is present in the previous sequence, 1 otherwise.
            repetition_penalty = 1 is equivalent to no repetition penalty.
            Default: 1.2 (from CTRL paper).
    
    Returns:
        int: The sampled index from the distribution.
    
    Example:
        In: sample_from_distribution([0.1, 0.5, 0.4], temperature=0.1)
        Out: 1
    """
    scores = np.array(scores).reshape(-1)
    scores = scores / scores.sum()
    
    # Construct repetition penalty vector
    penalties = np.ones(len(scores))
    if previous_sequence is not None:
        penalties[previous_sequence] = repetition_penalty
    
    # Apply temperature transform
    scores = scores ** (penalties / temperature)
    
    # If the proba_threshold parameter is set, override the top_k parameter.
    if proba_threshold is not None:
        sorted_scores = scores[np.argsort(-scores)] # Sorted in descending order
        top_k = (sorted_scores.cumsum() >= proba_threshold).argmax() + 1
    
    # If top_k <= 0, sample from whole array of scores.
    if (top_k <= 0) or (top_k > len(scores)):
        top_k = len(scores)
    
    # Construct top_scores array with length top_k
    top_indices = np.argsort(-scores)[:top_k]
    top_scores = scores[top_indices]
    top_scores = top_scores / top_scores.sum()
    
    # Sample from top_scores
    chosen_index_in_top_scores = np.random.choice(np.arange(top_k), p=top_scores)
    chosen_index_in_scores = top_indices[chosen_index_in_top_scores]
    
    return chosen_index_in_scores


def sample_sequence(predictor, sequence, encoder=None, seq_length=None, **kwargs):
    """
    Args:
        predictor (function): Must take as input a sequence of tokens of length
            seq_length and return a sequence of scores of length encoder.vocab_size.
            The scores should sum to 1.
        
        sequence (string or list of ints): Sequence to pass to the predictor.
            If string, must specify an encoder object. Then a string will be returned.
            If list of ints,
        
        encoder (encoder object or None): Can be left as None if `sequence` is already a sequence of tokens.
            Object with methods .encode(), .decode() and attribute vocab_size.
            .encode() must take in a string and output a corresponding sequence of tokens.
            .decode() must take in a sequence of tokens and output a corresponding string.
            .vocab_size denotes the number of tokens in the vocabulary.
            Default: None.
    """
    pass