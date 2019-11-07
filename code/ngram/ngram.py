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
    
    def generate_greedy(self, nb_words_to_gen, previous_words):
        """Generate a sequence of words starting from given starting words.
        
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
            cond_probas=[]
            for word in self.vocabulary:
                cond_probas.append(
                    self.get_word_probability(
                        word,
                        previous_words[-(self.n-1):]
                    )
                )
            # Sélectionne le mot qui maximise la probabilité conditionnelle (greedy search)
            previous_words += (self.vocabulary[np.argmax(cond_probas)],)
        return previous_words
    
