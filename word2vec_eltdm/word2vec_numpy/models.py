import os
import pickle

import numpy as np
import torch as torch

from word2vec_eltdm.word2vec_numpy.base_network import Network

rng = np.random.default_rng(0)


class SimpleWord2Vec(Network):
    """
    Very basic implementation of Word2Vec without any "speed" tricks.
    """

    def __init__(
        self,
        len_vocab: int,
        embedding_size: int = 300,
    ):
        super(SimpleWord2Vec, self).__init__("SimpleWord2Vec")
        self.len_vocab = len_vocab
        self.embedding_size = embedding_size
        self.cache = {}
        self.best_val_loss = None
        self.best_W1 = None
        self.best_W2 = None

    def initialize_weights(self, W1: np.array = None, W2: np.array = None):
        if W1 and W2:
            assert W1.shape == (
                self.len_vocab,
                self.embedding_size,
            ), "weights for initialization are not in the correct shape (len(self.len_vocab), self.embedding_size)"
            assert W2.shape == (
                self.len_vocab,
                self.embedding_size,
            ), "weights for initialization are not in the correct shape (len(self.len_vocab), self.embedding_size)"
            self.W1 = W1
            self.W2 = W2

        else:
            self.W1 = np.random.normal(
                loc=0.0,
                scale=1 / np.sqrt(self.embedding_size),
                size=(self.len_vocab, self.embedding_size),
            )
            self.W2 = np.random.normal(
                loc=0.0,
                scale=1 / np.sqrt(self.embedding_size),
                size=(self.len_vocab, self.embedding_size),
            )

    def forward(self, X):
        assert self.W1 is not None, "weight matrix W1 is not initialized"
        assert self.W2 is not None, "weight matrix W2 is not initialized"

        # foward_input
        h = self.W1[X.flatten(), :].T
        # forward output
        u = np.dot(self.W2, h)

        y = self.softmax(u)

        self.cache["X"] = X
        self.cache["h"] = h
        self.cache["logits"] = u

        return y

    @staticmethod
    def softmax(X):
        return np.divide(
            np.exp(X - np.max(X)), np.sum(np.exp(X - np.max(X)), axis=0, keepdims=True)
        )

    def backward(self, grad_softmax):
        dW2 = (1 / self.cache["h"].shape[1]) * np.dot(grad_softmax, self.cache["h"].T)
        dW1 = np.dot(self.W2.T, grad_softmax)
        return dW1, dW2

    def save_model(self, directory: str = "../word2vec_eltdm/models") -> None:
        """Save model as pickle"""
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(
            directory + "/" + self.model_name + "_" + str(self.best_val_loss) + ".p",
            "wb",
        ) as file:
            print("Saving model")
            pickle.dump(model, file)


class NegWord2Vec:
    """
    Very basic implementation of Word2Vec without any "speed" tricks.
    """

    def __init__(
        self,
        len_vocab: int,
        embedding_size: int = 300,
        noise_dist: np.array = None,
        best_val_loss=None,
    ):
        self.len_vocab = len_vocab
        self.embedding_size = embedding_size
        self.noise_dist = noise_dist
        self.cache = {}
        self.model_name = "NegWord2Vec"
        self.best_val_loss = best_val_loss
        self.best_W1 = None
        self.best_W2 = None

    def initialize_weights(self, W1: np.array = None, W2: np.array = None):
        if W1 and W2:
            assert W1.shape == (
                self.len_vocab,
                self.embedding_size,
            ), "weights for initialization are not in the correct shape (len(self.len_vocab), self.embedding_size)"
            assert W2.shape == (
                self.len_vocab,
                self.embedding_size,
            ), "weights for initialization are not in the correct shape (len(self.len_vocab), self.embedding_size)"
            self.W1 = W1
            self.W2 = W2

        else:
            # self.W1 = np.random.uniform(
            #     low=-1,
            #     high=1,
            #     size=(self.len_vocab, self.embedding_size),
            # )
            # self.W2 = np.random.uniform(
            #     low=-1,
            #     high=1,
            #     size=(self.len_vocab, self.embedding_size),
            # )
            self.W1 = np.random.normal(
                loc=0.0,
                scale=1 / np.sqrt(self.embedding_size),
                size=(self.len_vocab, self.embedding_size),
            )
            self.W2 = np.random.normal(
                loc=0.0,
                scale=1 / np.sqrt(self.embedding_size),
                size=(self.len_vocab, self.embedding_size),
            )

    def forward_input(self, X):
        h = self.W1[X.squeeze(), :]
        self.cache["X"] = X
        return h

    def forward_output(self, y):
        u = self.W2[y.squeeze(), :]
        self.cache["y"] = y
        return u

    def forward_noise(self, batch_size, n_samples):
        """Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        if self.noise_dist is None:
            # Sample words uniformly
            self.noise_dist = np.ones(self.len_vocab) / self.len_vocab

        # Sample words from our noise distribution
        # Use torch multinomial because it has the behavior we want compared to np multinomial
        noise_words = torch.multinomial(
            torch.from_numpy(self.noise_dist), batch_size * n_samples, replacement=True
        ).numpy()

        # Get the noise embeddings
        noise_vector = self.W2[noise_words, :].reshape(
            batch_size, n_samples, self.embedding_size
        )
        self.cache["noise_words"] = noise_words
        self.cache["n_samples"] = n_samples

        return noise_vector

    @staticmethod
    def sigmoid(x):
        sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
        return sig

    def backward(self, grad_softmax):
        raise NotImplementedError

    def train(self):
        """sets the network in training mode, i.e. returns gradient when called"""
        self.return_grad = True

    def eval(self):
        """sets the network in evaluation mode, i.e. only computes forward pass"""
        self.return_grad = False

    def save_model(self, directory: str = "../word2vec_eltdm/models") -> None:
        """Save model as pickle"""
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(
            directory + "/" + self.model_name + "_" + str(self.best_val_loss) + ".p",
            "wb",
        ) as file:
            print("Saving model")
            pickle.dump(model, file)
