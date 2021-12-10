from typing import Dict, Tuple, List

from word2vec_eltdm.word2vec_numpy import Tokenizer, VocabCreator
from word2vec_eltdm.word2vec_numpy.dataset import Dataset, Text8Dataset
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

    def preprocess(self) -> Text8Dataset:
        tokens = self.tokenizer.get_tokens()
        tokens = self.token_cleaner.clean_tokens(tokens)

        text8_dataset = self._get_train_val_test(tokens)

        (
            text8_dataset.train_dataset.tokens_to_id,
            text8_dataset.train_dataset.id_to_tokens,
        ) = self.vocab_creator.create_vocab(text8_dataset.train_dataset.tokens)
        (
            text8_dataset.val_dataset.tokens_to_id,
            text8_dataset.val_dataset.id_to_tokens,
        ) = self.vocab_creator.create_vocab(text8_dataset.val_dataset.tokens)
        (
            text8_dataset.test_dataset.tokens_to_id,
            text8_dataset.test_dataset.id_to_tokens,
        ) = self.vocab_creator.create_vocab(text8_dataset.test_dataset.tokens)

        return text8_dataset

    def _get_train_val_test(self, tokens: List[str]) -> Text8Dataset:
        N = len(tokens)
        # 3/4 train, 1/8 val, 1/8 test
        train_size = (3 * N) // 4
        train_tokens = tokens[:train_size]
        val_size = (7 * N) // 8
        val_tokens = tokens[train_size:val_size]
        test_tokens = tokens[val_size:]

        return Text8Dataset(
            train_dataset=Dataset(train_tokens),
            val_dataset=Dataset(val_tokens),
            test_dataset=Dataset(test_tokens),
        )
