from collections import Counter
import random
from typing import Dict, Tuple, List

import numpy as np

from word2vec_eltdm.word2vec_numpy.dataset import Dataset

random.seed(0)


class Subsampler:
    """
    Subsampler class (Mikolov 2013).
    Reference: https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
    Context:
        Some recurrent words such as "a", "of", and "in" don't provide much context to the nearby words.
        If we discard some of them, we can remove some of the noise from our data and in return get faster
        training and better representations.

    Each word can be discarded with probability 1 - \sqrt(t/f(w))
    where t is a chosen threshold, and f(w) is the frequency of word w
    """

    def __init__(
        self,
        dataset: Dataset,
        threshold: float = 1e-5,
    ) -> None:
        self.threshold = threshold
        self.tokens = dataset.tokens
        self.words_to_id = dataset.tokens_to_id
        self.id_to_words = dataset.id_to_tokens

    def subsample(self) -> Dataset:
        token_counts = Counter(self.tokens)
        total_token_count = len(self.tokens)
        freqs = {
            token: count / total_token_count for token, count in token_counts.items()
        }
        drop_proba = {
            token: 1 - np.sqrt(self.threshold / freqs[token]) for token in token_counts
        }
        # keep tokens with proba (1 - drop_proba)
        tokens_to_keep = [
            token for token in self.tokens if random.random() < (1 - drop_proba[token])
        ]
        # recreate vocab
        words_to_id = {token: self.words_to_id[token] for token in tokens_to_keep}
        id_to_words = {index: token for token, index in words_to_id.items()}

        return Dataset(tokens_to_keep, words_to_id, id_to_words)
