from collections import Counter
import random
from typing import Dict, Tuple

import numpy as np


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
        words_to_id: Dict[str, int],
        id_to_words: Dict[int, str],
        threshold: float = 1e-5,
    ) -> None:
        self.threshold = threshold
        self.words_to_id = words_to_id
        self.id_to_words = id_to_words

    def subsample(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        id_counts = Counter(self.words_to_id.values())
        total_id_count = len(self.words_to_id.values())
        freqs = {index: count / total_id_count for index, count in id_counts.items()}
        drop_proba = {
            index: 1 - np.sqrt(self.threshold / freqs[index]) for index in id_counts
        }
        # keep tokens with proba (1 - drop_proba)
        id_to_words = {
            index: token
            for index, token in self.id_to_words.items()
            if random.random() < (1 - drop_proba[index])
        }
        words_to_id = {token: index for index, token in id_to_words.items()}
        return words_to_id, id_to_words
