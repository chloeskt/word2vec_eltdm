from typing import List

from word2vec_eltdm.word2vec_numpy import Tokenizer, VocabCreator
from word2vec_eltdm.word2vec_numpy.dataset import Dataset
from word2vec_eltdm.word2vec_numpy.token_cleaner import TokenCleaner


class Preprocessor:
    def __init__(
        self,
        tokenizer: Tokenizer,
        token_cleaner: TokenCleaner,
        vocab_creator: VocabCreator,
        ratio: float,
        return_only_train: bool = False
    ) -> None:
        self.tokenizer = tokenizer
        self.token_cleaner = token_cleaner
        self.vocab_creator = vocab_creator
        self.ratio = ratio
        self.return_only_train = return_only_train

    def preprocess(self) -> Dataset:
        tokens = self.tokenizer.get_tokens(ratio=self.ratio)
        tokens = self.token_cleaner.clean_tokens(tokens)

        dataset = self._get_train_val_test(tokens)

        dataset.tokens_to_id, dataset.id_to_tokens = self.vocab_creator.create_vocab(
            dataset.train_tokens
        )

        return dataset

    def _get_train_val_test(self, tokens: List[str]) -> Dataset:
        if self.return_only_train:
            return Dataset(train_tokens=tokens)

        N = len(tokens)
        # 4/5 train, 1/10 val, 1/10 test
        train_size = (4 * N) // 5
        train_tokens = tokens[:train_size]
        val_size = (9 * N) // 10
        val_tokens = tokens[train_size:val_size]
        test_tokens = tokens[val_size:]

        return Dataset(
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            test_tokens=test_tokens,
        )
