from typing import Iterable, Generator

import numpy as np


def concatenate(*iterables: Iterable) -> Generator:
    for iterable in iterables:
        yield from iterable


def one_hot_encoding(id: int, vocab_size: int) -> np.array:
    vector = np.zeros(vocab_size)
    vector[id] = 1
    return vector
