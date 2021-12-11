from typing import List, Generator

import numpy as np

from word2vec_eltdm.word2vec_numpy.dataset import Dataset
from word2vec_eltdm.word2vec_numpy.vocabcreator import UNKNOWN_TOKEN


class DataLoader:
    def __init__(
        self,
        dataset: Dataset,
        tokens: List[str],
        window: int,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = True,
    ) -> None:
        """
        :param dataset: dataset
        :param window: how many tokens you should consider around each token (continuous Skip-gram)
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.tokens = tokens
        self.vocab = dataset.tokens_to_id
        self.window = window
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.tokens_in_int = [
            self.vocab.get(token, self.vocab[UNKNOWN_TOKEN]) for token in self.tokens
        ]

    def __iter__(self) -> Generator:
        index_iterator = iter(range(len(self.tokens)))

        X = []
        Y = []

        for i in index_iterator:
            x = self.vocab.get(self.tokens[i], self.vocab[UNKNOWN_TOKEN])
            y = self.get_target(self.tokens_in_int, i)
            Y.extend(y)
            X.extend([x] * len(y))

            if (i + 1) % self.batch_size == 0:
                X = np.array(X)
                Y = np.array(Y)
                X = np.expand_dims(X, axis=0)
                Y = np.expand_dims(Y, axis=0)
                yield {"X": X, "Y": Y}
                X = []
                Y = []

        if not self.drop_last and X:
            X = np.array(X)
            Y = np.array(Y)
            X = np.expand_dims(X, axis=0)
            Y = np.expand_dims(Y, axis=0)
            yield {"X": X, "Y": Y}

    def __len__(self) -> int:
        length = len(self.tokens) // self.batch_size

        if not self.drop_last and len(self.tokens) % self.batch_size != 0:
            length = length + 1
        return length

    def get_target(self, tokens_in_int, idx):
        R = np.random.randint(1, self.window + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_ints = tokens_in_int[start:idx] + tokens_in_int[idx + 1 : stop + 1]

        return list(target_ints)
