from typing import List, Tuple, Dict


class VocabCreator:
    @staticmethod
    def create_vocab(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        words_to_id = dict()
        id_to_words = dict()

        for i, token in enumerate(set(tokens)):
            words_to_id[token] = i
            id_to_words[i] = token

        return words_to_id, id_to_words
