from copy import deepcopy
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from tqdm.notebook import tqdm


def train(model, train_dataloader, criterion, optimizer):
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


def update_best_loss(model, val_loss):
    # Update the model and best loss if we see improvements.
    if not model.best_val_loss or val_loss < model.best_val_loss:
        model.best_val_loss = val_loss
        model.best_W1 = deepcopy(model.W1)
        model.best_W2 = deepcopy(model.W2)
        model.save_model()


def validate(model, dataloader, criterion):
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


def cosine_similarity(embeddings: np.array, len_vocab: int, nb_words: int = 20):
    """
    Returns the cosine similarity of randomly chosen `nb_words` words in the embedding matrix.
    params:
        embeddings: np.array
            weights W1 corresponding to the embedding layer of our model
    """
    magnitudes = np.sqrt((embeddings ** 2).sum(axis=1))
    examples = np.array(random.sample(range(len_vocab), nb_words))
    examples_vectors = embeddings[examples]
    similarities = (examples_vectors @ embeddings.T) / magnitudes
    return examples, similarities


def evaluate(embeddings: np.array, id_to_tokens: Dict[int, str], nb_words: int) -> None:
    """
    For `nb_words` words randomly chosen in the dataset, get the top 5 words according to their
    cosine similarity (preview of word similarity) to evaluate qualitatively the model.
    """
    len_vocab = len(id_to_tokens)
    examples, similarities = cosine_similarity(embeddings, len_vocab, nb_words)
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
