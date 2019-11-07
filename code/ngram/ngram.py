import numpy as np

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.vocabulary = []
        self.freq_n_minus_one_grams = {}
        self.freq_n_grams = {}
    
    @staticmethod
    def count_kgrams(text_tokens, k):
        freq_kgrams = {}
        for i in range(0, len(text_tokens)-k+1):
            kgram = tuple(text_tokens[i:i+k])
            freq_kgrams[kgram] = freq_kgrams.get(kgram, 0) + 1
        return freq_kgrams
        
    def fit(self, text_tokens, verbose=True):
        """Fit the n-gram model on pre-tokenized text based on the n attribute of the model.
        
        Args:
            text_tokens (list of strings): Tokenized text.
            verbose (bool): Whether to print extra info.
        """
        self.vocabulary = list(set(text_tokens))
        if verbose:
            print("Fitting {}-gram model on vocabulary of size {}.".format(self.n, len(self.vocabulary)))
        
        self.freq_n_minus_one_grams = self.count_kgrams(text_tokens, self.n-1)
        self.freq_n_grams = self.count_kgrams(text_tokens, self.n)
    
    def get_word_probability(self, word, previous_words):
        """Get the probability of a word given the n-1 previous words.
        
        Args:
            word (str): Word to predict probability of.
            previous_words (tuple of strings): Words which precede `word`. Must be of length self.n - 1.
        """
        assert len(previous_words) == self.n - 1, "Error in probability calculation: invalid number of previous words: {}".format(len(previous_words))
        
        if self.freq_n_minus_one_grams.get(previous_words, 0) == 0:
            proba = 0
        else:
            proba = (
                self.freq_n_grams.get(previous_words + (word,), 0)
                / self.freq_n_minus_one_grams.get(previous_words, 0)
            )
        
        return proba
    
    def predict_probas(self, previous_words):
        """Get all conditional word probabilities given previous n-1 words.
        
        Args:
            previous_words (tuple of strings): Previous sequence of tokens. Must be of length at least n-1.
        
        Returns:
            list: vector of probabilities of same size as vocabulary.
        """
        assert len(previous_words) >= self.n - 1, "Error in probability calculation: invalid number of previous words: {}".format(len(previous_words))

        cond_probas=[]
        for word in self.vocabulary:
            cond_probas.append(
                self.get_word_probability(
                    word,
                    previous_words[-(self.n-1):]
                )
            )
        return cond_probas

    
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
            cond_probas = self.predict_probas(previous_words)
            # Sélectionne le mot qui maximise la probabilité conditionnelle (greedy search)
            next_word = self.vocabulary[np.argmax(cond_probas)]
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
            cond_probas = self.predict_probas(previous_words)
            cond_probas = np.array(cond_probas)**power
            cond_probas = cond_probas / cond_probas.sum()
            # Sample a word from conditional distribution
            next_word = np.random.choice(self.vocabulary, p=cond_probas)
            previous_words += (next_word,)
        return previous_words
    
