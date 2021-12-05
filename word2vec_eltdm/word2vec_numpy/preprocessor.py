import re
from typing import Tuple, Dict, List


class Preprocessor:
    def __init__(self, datapath: str) -> None:
        self.datapath = datapath

    def _get_data(self) -> str:
        with open(self.datapath) as file:
            data = file.read()
        return data

    @staticmethod
    def _tokenize(data: str) -> List[str]:
        pattern = re.compile(r"[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*")
        return pattern.findall(data.lower())

    @staticmethod
    def _create_vocab(tokens: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        word_to_id = dict()
        id_to_word = dict()

        for i, token in enumerate(set(tokens)):
            word_to_id[token] = i
            id_to_word[i] = token

        return word_to_id, id_to_word

    def preprocess(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        data = self._get_data()
        tokens = self._tokenize(data)
        words_to_id, id_to_words = self._create_vocab(tokens)
        return words_to_id, id_to_words
