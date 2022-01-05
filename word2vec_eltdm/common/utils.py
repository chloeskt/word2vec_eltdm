import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def cosine_similarity(embeddings: np.array, example_vectors: np.array) -> np.array:
    nominator = example_vectors @ embeddings.T
    denominator = np.sqrt(np.sum(embeddings ** 2, axis=1))
    denominator = np.expand_dims(denominator, axis=1).reshape(1, -1)
    cosine_similarity = nominator / denominator
    return cosine_similarity


def evaluate(
    embeddings: np.array,
    id_to_tokens: Dict[int, str],
    nb_words: int = 16,
    valid_window: int = 100,
) -> None:
    """
    For `nb_words` words randomly chosen in the dataset, get the top 5 words according to their
    cosine similarity (preview of word similarity) to evaluate qualitatively the model.
    """

    examples = np.array(random.sample(range(valid_window), nb_words // 2))
    examples = np.append(
        examples, random.sample(range(1000, 1000 + valid_window), nb_words // 2)
    )

    examples_vectors = embeddings[examples]

    similarities = cosine_similarity(embeddings, examples_vectors)

    closest_idxs = np.flip(np.argsort(similarities, axis=1)[:, -6:], axis=1)

    for i, exemple_idx in enumerate(examples):
        closest_words = [id_to_tokens[idx] for idx in closest_idxs[i]][1:]
        print(id_to_tokens[exemple_idx] + " | " + ", ".join(closest_words))


def visualization_tsne(
    embeddings: np.array, id_to_tokens: Dict[int, str], nb_words: int = 400
) -> None:
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embeddings[:nb_words, :])
    fig, ax = plt.subplots(figsize=(16, 16))
    for idx in range(nb_words):
        plt.scatter(*embed_tsne[idx, :], color="steelblue")
        plt.annotate(
            id_to_tokens[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7
        )
