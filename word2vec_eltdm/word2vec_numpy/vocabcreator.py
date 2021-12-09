from collections import Counter
from typing import List, Tuple, Dict


class VocabCreator:
    @staticmethod
    def create_vocab(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        token_counts = Counter(tokens)
        sorted_vocab = sorted(token_counts, key=token_counts.get, reverse=True)
        id_to_words = {ii: token for ii, token in enumerate(sorted_vocab)}
        words_to_id = {token: ii for ii, token in id_to_words.items()}
        return words_to_id, id_to_words
