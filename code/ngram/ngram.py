import numpy as np
from random import choice


class NGramModel:
    
    def __init__(self, n):
        self.n = n
        self.vocabulary = []
        self.conditional_probabilities = {}
        
        
    def fit(self, text_tokens, verbose=True):
        """Fit the n-gram model on pre-tokenized text based on the n attribute of the model.
        
        Args:
            text_tokens (list of strings): Tokenized text.
            verbose (bool): Whether to print extra info.
        """
        self.vocabulary = list(set(text_tokens))
        if verbose:
            print("Fitting {}-gram model on vocabulary of size {}.".format(self.n, len(self.vocabulary)))
        
        counts = {}
        
        # Iterating through the tokens
        for i in range(0, len(text_tokens) - self.n + 1):
            n_minus_1_gram = tuple(text_tokens[i:i+self.n-1])
            next_word = text_tokens[i+self.n-1]
            
            if n_minus_1_gram not in counts:
                counts[n_minus_1_gram] = {}
            
            if next_word not in counts[n_minus_1_gram]:
                counts[n_minus_1_gram][next_word] = 0
                
            counts[n_minus_1_gram][next_word] += 1
            
        for n_minus_1_gram, possible_next_words in counts.items():
            counts_sum = sum(list(possible_next_words.values()))
            for next_word in possible_next_words:
                counts[n_minus_1_gram][next_word] /= counts_sum
                
        self.conditional_probabilities = counts
    
    
    def get_word_probability(self, word, previous_words):
        """Get the probability of a word given the n-1 previous words.
        
        Args:
            word (str): Word to predict probability of.
            previous_words (tuple of strings): Words which precede `word`. Must be of length self.n - 1.
        """
        assert len(previous_words) == self.n - 1, "Error in probability calculation: invalid number of previous words: {}".format(len(previous_words))
        
        if previous_words not in self.conditional_probabilities:
            proba = 0
        else:
            proba = self.conditional_probabilities[previous_words].get(word,0)
        
        return proba
    
    
    def predict_possible_next_words_probas(self, previous_words):
        """Get all conditional word probabilities greater than zero given previous n-1 words.
        
        Args:
            previous_words (tuple of strings): Previous sequence of tokens. Must be of length at least n-1.
        
        Returns:
            dictionnary: conditional probabilities (float) with keys .
        """
        assert len(previous_words) >= self.n - 1, "Error in probability calculation: invalid number of previous words: {}".format(len(previous_words))
        
        #Un peu sale mais évite deux parcours du dico ; idee de code plus propre ?
        return self.conditional_probabilities.get(previous_words[-(self.n-1):], {choice(self.vocabulary): 1})

    
    def generate_greedy(self, nb_words_to_gen, previous_words):
        """Generate a sequence of words starting from given starting words using greedy prediction.
        
        Args:
            nb_words_to_gen (int): Number of words to generate past the starting sequence.
            previous_words (str or iterable): List of tokens to start from. If string, must be
                sequence of tokens separated by spaces.
        
        Returns:
            tuple of strings: Generated tokens.
        """
        # Sanitize input
        if isinstance(previous_words, str):
            previous_words = previous_words.split(" ")
        previous_words = tuple(previous_words)
        
        for i in range(nb_words_to_gen):
            # Sélectionne le mot qui maximise la probabilité conditionnelle (greedy search)
            possible_next_words_probas = self.predict_possible_next_words_probas(previous_words)
            next_word = max(possible_next_words_probas, key=possible_next_words_probas.get)
            previous_words += (next_word,)
        return previous_words
    
    
    def generate_sampled(self, nb_words_to_gen, previous_words, power=1):
        """Generate a sequence of words starting from given starting words using the sampling method.
        
        Args:
            nb_words_to_gen (int): Number of words to generate past the starting sequence.
            previous_words (str or iterable): List of tokens to start from. If string, must be
                sequence of tokens separated by spaces.
            power (float): Power to raise probabilities at before sampling.
                A higher power means a less risky sampling.
                An infinite power would be equivalent to greedy sampling.
        
        Returns:
            tuple of strings: Generated tokens.
        """
        # Sanitize input
        if isinstance(previous_words, str):
            previous_words = previous_words.split(" ")
        previous_words = tuple(previous_words)
        
        for i in range(nb_words_to_gen):
            possible_next_words_probas = self.predict_possible_next_words_probas(previous_words)
            cond_probas = list(possible_next_words_probas.values())
            cond_probas = np.array(cond_probas)**power
            cond_probas = cond_probas / cond_probas.sum()
            possible_next_words = np.array(list(possible_next_words_probas.keys()))
            # Sample a word from conditional distribution
            next_word = np.random.choice(possible_next_words, p=cond_probas)
            previous_words += (next_word,)
        return previous_words
    
    
    def generate_beam(self, nb_words_to_gen, previous_words, k=3):
        """Generate a sequence of words starting from given starting words using beam prediction.
        
        Args:
            nb_words_to_gen (int): Number of words to generate past the starting sequence.
            previous_words (str or iterable): List of tokens to start from. If string, must be
                sequence of tokens separated by spaces.
            k (int): parameter of Beam search method, i.e. number of best sequences kept at each step
        
        Returns:
            tuple of strings : Generated tokens.
        """
        # Sanitize input
        if isinstance(previous_words, str):
            previous_words = previous_words.split(" ")
        previous_words = tuple(previous_words)
        
        k_best_sequences = [[previous_words,1]]
        for i in range(nb_words_to_gen):
            candidates = []
            for best_sequence in k_best_sequences:
                previous_words, previous_probability = best_sequence
                possible_next_words_probas = self.predict_possible_next_words_probas(previous_words)
                for word, word_cond_probability in possible_next_words_probas.items():
                    sequence_probability = previous_probability * word_cond_probability
                    sequence_words = previous_words + (word,)
                    candidates.append([sequence_words, sequence_probability])
            ordered = sorted(candidates, key=lambda tup:tup[1])
            k_best_sequences = ordered[-k:]
        return k_best_sequences[-1][0]
    
