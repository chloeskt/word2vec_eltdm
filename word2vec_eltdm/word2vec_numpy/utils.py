import random
from copy import deepcopy
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm

from word2vec_eltdm.common import DataLoader
from word2vec_eltdm.word2vec_numpy import (
    SimpleWord2Vec,
    CrossEntropy,
    Optimizer,
    NegWord2Vec,
    NegativeSamplingLoss,
    OptimizeNSL,
)
from word2vec_eltdm.common.base_network import Network


def train_default(
    model: SimpleWord2Vec,
    train_dataloader: DataLoader,
    criterion: CrossEntropy,
    optimizer: Optimizer,
) -> float:
    train_loss = 0.0
    for i, batch in enumerate(tqdm(train_dataloader)):
        model.train()
        X, y = batch["X"], batch["Y"]
        preds = model.forward(X)
        loss, dy = criterion(preds, y)
        dW1, dW2 = model.backward(dy)
        optimizer.step(dW1, dW2)
        train_loss += loss

        if i % 1500 == 0:
            print("Current Training Loss {:.6}".format(loss))

    train_loss /= len(train_dataloader)
    return train_loss


def validate_default(
    model: SimpleWord2Vec, dataloader: DataLoader, criterion: CrossEntropy
) -> float:
    model.eval()
    validation_loss = 0
    for i, batch in enumerate(tqdm(dataloader)):
        X, y = batch["X"], batch["Y"]
        preds, _ = model(X)
        loss, _ = criterion(preds, y)
        validation_loss += loss

    validation_loss /= len(dataloader)

    # Keep track of the best model
    update_best_loss(model, validation_loss)
    print("Validation Loss: ", validation_loss)
    return validation_loss


def train_NSL(
    model: NegWord2Vec,
    train_dataloader: DataLoader,
    criterion: NegativeSamplingLoss,
    optimizer: OptimizeNSL,
    n_samples: int,
) -> float:
    train_loss = 0.0
    for i, batch in enumerate(tqdm(train_dataloader)):
        model.train()
        X, y = batch["X"], batch["Y"]
        h = model.forward_input(X)
        u = model.forward_output(y)
        noise_vector = model.forward_noise(X.shape[1], n_samples)

        # negative sampling loss
        loss, grad_W1, grad_W2_pos, grad_W2_neg = criterion(
            model, h, u, noise_vector, y
        )
        optimizer.step(grad_W1, grad_W2_pos, grad_W2_neg)

        train_loss += loss

    train_loss /= len(train_dataloader)
    return train_loss


def update_best_loss(model: Network, val_loss: float) -> None:
    # Update the model and best loss if we see improvements.
    if not model.best_val_loss or val_loss < model.best_val_loss:
        model.best_val_loss = val_loss
        model.best_W1 = deepcopy(model.W1)
        model.best_W2 = deepcopy(model.W2)
        print(f"Now best model has {val_loss} loss")
        model.save_model()


def validate_NSL(
    model: NegWord2Vec,
    dataloader: DataLoader,
    criterion: NegativeSamplingLoss,
    n_samples: int,
) -> float:
    model.eval()
    validation_loss = 0
    for i, batch in enumerate(tqdm(dataloader)):
        X, y = batch["X"], batch["Y"]
        h = model.forward_input(X)
        u = model.forward_output(y)
        noise_vector = model.forward_noise(X.shape[1], n_samples)

        loss, _, _, _ = criterion(model, h, u, noise_vector, y)
        validation_loss += loss

    validation_loss /= len(dataloader)
    # Keep track of the best model
    update_best_loss(model, validation_loss)
    print("Validation Loss: ", validation_loss)
    return validation_loss


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
