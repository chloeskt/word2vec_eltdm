from typing import List, Tuple, Dict


class VocabCreator:
    def __init__(self, tokens: List[str]) -> None:
        self.tokens = tokens

    def create_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        words_to_id = dict()
        id_to_words = dict()

        for i, token in enumerate(set(self.tokens)):
            words_to_id[token] = i
            id_to_words[i] = token

        return words_to_id, id_to_words
