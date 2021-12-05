from typing import List, Dict, Generator

import numpy as np


class DataLoader:
    def __init__(
        self,
        tokens: List[str],
        vocab: Dict[str, int],
        window: int,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = True,
    ) -> None:
        """
        :param tokens: list of strings representing all the tokens in your dataset
        :param vocab: vocabulary dictionary Dict[token, id]
        :param window: how many tokens you should consider around each token (continuous Skip-gram)
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
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

        X = np.zeros((2 * self.window * self.batch_size, len(self.vocab)))
        y = np.zeros((2 * self.window * self.batch_size, len(self.vocab)))
        index = 0
        for i in index_iterator:
            if i % self.batch_size == 0 and (i != 0 or self.batch_size == 1):
                yield {"X": X, "y": y}
                index = 0
                X = np.zeros((2 * self.window * self.batch_size, len(self.vocab)))
                y = np.zeros((2 * self.window * self.batch_size, len(self.vocab)))

            idx = range(
                max(0, self.window - i),
                min(len(self.tokens) - i + self.window + 1, self.window * 2),
            )
            for delta in idx:
                X[index + delta, self.vocab[self.tokens[i]]] = 1
                j = (
                    i - self.window + delta + (1 if delta >= self.window else 0)
                )  # min(1, max(0, delta - self.window + 1))
                y[index + delta, self.vocab[self.tokens[j]]] = 1

            index += 2 * self.window

        if not self.drop_last and X:
            yield {"X": X, "y": y}

    def __len__(self) -> int:
        length = None
        length = (len(self.tokens) // self.batch_size) + 1

        if self.drop_last and len(self.tokens) % self.batch_size != 0:
            length = length - 1
        return length
