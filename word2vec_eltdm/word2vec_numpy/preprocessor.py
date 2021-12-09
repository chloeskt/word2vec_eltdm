from typing import Dict, Tuple, List

from word2vec_eltdm.word2vec_numpy import Tokenizer, VocabCreator
from word2vec_eltdm.word2vec_numpy.token_cleaner import TokenCleaner


class Preprocessor:
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_cleaner: TokenCleaner,
        vocab_creator: VocabCreator,
    ) -> None:
        self.tokenizer = tokenizer
        self.token_cleaner = token_cleaner
        self.vocab_creator = vocab_creator

    def preprocess(self) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        tokens = self.tokenizer.get_tokens()
        tokens = self.token_cleaner.clean_tokens(tokens)
        words_to_id, id_to_words = self.vocab_creator.create_vocab(tokens)
        return tokens, words_to_id, id_to_words
