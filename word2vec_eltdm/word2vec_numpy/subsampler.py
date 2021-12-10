import random
from collections import Counter
from typing import List

import numpy as np

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
        tokens: List[str],
        threshold: float = 1e-5,
    ) -> None:
        self.threshold = threshold
        self.tokens = tokens

    def subsample(self) -> List[str]:
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

        return tokens_to_keep
