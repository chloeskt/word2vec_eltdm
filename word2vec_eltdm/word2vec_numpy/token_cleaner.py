from collections import Counter
from typing import List

from nltk.corpus import stopwords


class TokenCleaner:
    def __init__(self, freq_threshold: int = 5) -> None:
        self.freq_threshold = freq_threshold

    def _remove_stop_words(self, tokens: List[str]) -> List[str]:
        stop_words = set(stopwords.words("english"))
        return [token for token in tokens if token not in stop_words]

    def _remove_low_frequency_words(self, tokens: List[str]) -> List[str]:
        """Remove all tokens less or equal `freq_threshold` occurences"""
        tokens_count = Counter(tokens)
        return [token for token in tokens if tokens_count[token] > self.freq_threshold]

    def clean_tokens(self, tokens: List[str]) -> List[str]:
        tokens = self._remove_stop_words(tokens)
        tokens = self._remove_low_frequency_words(tokens)
        return tokens
