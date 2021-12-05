from typing import List, Dict, Generator

import numpy as np

from .utils import concatenate, one_hot_encoding


class DataLoader:
    def __init__(
        self,
        tokens: List[str],
        vocab: Dict[str, int],
        window: int,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        """
        :param tokens: list of strings representing all the tokens in your dataset
        :param vocab: vocabulary dictionary Dict[token, id]
        :param window: how many tokens you should consider around each token (continuous Skip-gram)
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            If the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.tokens = tokens
        self.vocab = vocab
        self.window = window
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Generator:
        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.tokens)))
        else:
            index_iterator = iter(range(len(self.tokens)))

        X = []
        y = []
        for i in index_iterator:
            idx = concatenate(
                range(max(0, i - self.window), i),
                range(i, min(len(self.tokens), i + self.window + 1)),
            )
            for j in idx:
                if i == j:
                    continue
                X.append(one_hot_encoding(self.vocab[self.tokens[i]], len(self.vocab)))
                y.append(one_hot_encoding(self.vocab[self.tokens[j]], len(self.vocab)))

            if i % self.batch_size == 1 or self.batch_size == 1:
                # TODO: add padding for the first and last `window_size` words
                yield {"X": np.asarray(X), "y": np.asarray(y)}
                X, y = [], []

        if not self.drop_last and X:
            yield {"X": np.asarray(X), "y": np.asarray(y)}

    def __len__(self) -> int:
        length = None
        length = (len(self.tokens) // self.batch_size) + 1

        if self.drop_last and len(self.tokens) % self.batch_size != 0:
            length = length - 1
        return length
