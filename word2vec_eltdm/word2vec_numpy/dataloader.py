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

    def __iter__(self) -> Generator:
        if self.shuffle:
            index_iterator = iter(np.random.permutation(len(self.tokens)))
        else:
            index_iterator = iter(range(len(self.tokens)))

        X = np.zeros((2 * self.window * self.batch_size, len(self.vocab)))
        y = np.zeros((2 * self.window * self.batch_size, len(self.vocab)))
        previous_y = np.zeros((2 * self.window, len(self.vocab)))
        index = 0
        num_batch = 0
        for i in index_iterator:
            lower_bound = max(0, self.window - i)
            upper_bound = min(len(self.tokens) - i + self.window + 1, self.window * 2)

            X[
                index + lower_bound : index + upper_bound,
                self.vocab.get(self.tokens[i], self.vocab[UNKNOWN_TOKEN]),
            ] = 1
            y[index : index + 2 * self.window - 1, :] = previous_y[1:, :]

            if i == 0:
                idx = range(lower_bound, upper_bound)
                for delta in idx:
                    j = i - self.window + delta + 1
                    y[
                        index + delta,
                        self.vocab.get(self.tokens[j], self.vocab[UNKNOWN_TOKEN]),
                    ] = 1
            elif i + self.window < len(self.tokens):
                j = i + self.window
                y[
                    index + 2 * self.window - 1,
                    self.vocab.get(self.tokens[j], self.vocab[UNKNOWN_TOKEN]),
                ] = 1

            previous_y = y[index : index + 2 * self.window, :]

            index += 2 * self.window

            if (i + 1) % self.batch_size == 0:
                num_batch += 1
                if num_batch % 100 == 0:
                    print(f"BATCH {num_batch} done")
                yield {"X": X, "y": y}
                index = 0
                X = np.zeros((2 * self.window * self.batch_size, len(self.vocab)))
                y = np.zeros((2 * self.window * self.batch_size, len(self.vocab)))

        if not self.drop_last and X.any():
            yield {"X": X, "y": y}

    def __len__(self) -> int:
        length = len(self.tokens) // self.batch_size

        if not self.drop_last and len(self.tokens) % self.batch_size != 0:
            length = length + 1
        return length
