from typing import Dict

import numpy as np


class SimpleWord2Vec:
    """
    Very basic implementation of Word2Vec without any "speed" tricks.
    1. Create embeddings
    2. Create projection layer
    3. Create hidden layers
    4. Simple softmax (output layer)

    There is one input layer which has as many neurons as there are words in the vocabulary for training.
    The second layer is the hidden layer, layer size in terms of neurons is the dimensionality of the resulting word vectors.
    The third and final layer is the output layer which has the same number of neurons as the input layer.


    """

    def __init__(
        self,
        vocab: Dict[str, int],
        num_layer: int = 1,
        hidden_size: int = 500,
        embedding_size: int = 300,
        window: int = 5,
        alpha: float = 0.001,
        learning_rate: float = 1e-3,
    ):
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.window = window
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.cache = {}

    def initialize_weights(self, W1: np.array = None, W2: np.array = None):
        if W1 and W2:
            assert W1.shape == (
                len(self.vocab),
                self.embedding_size,
            ), "weights for initialization are not in the correct shape (len(self.vocab), self.embedding_size)"
            assert W2.shape == (
                len(self.vocab),
                self.embedding_size,
            ), "weights for initialization are not in the correct shape (self.embedding_size, len(self.vocab))"
            self.W1 = W1
            self.W2 = W2

        else:
            self.W1 = self.alpha * np.random.randn(len(self.vocab), self.embedding_size)
            self.W2 = self.alpha * np.random.randn(self.embedding_size, len(self.vocab))

    def forward(self, X):
        assert self.W1 is not None, "weight matrix W1 is not initialized"
        assert self.W2 is not None, "weight matrix W2 is not initialized"

        h = X.dot(self.W1)
        u = h.dot(self.W2)
        y = self.softmax(h)
        self.cache["X"] = X
        self.cache["h"] = h
        self.cache["logits"] = u

        return y

    def backward(self, preds, y):
        grad_softmax = self.softmax_backward(y, preds)
        grad_W1, grad_W2 = self.hidden_layer_backward(grad_softmax)
        self.update_weights(grad_W1, grad_W2)

    def update_weights(self, grad_W1, grad_W2):
        self.W1 = self.W1 - (self.learning_rate * grad_W1)
        self.W2 = self.W2 - (self.learning_rate * grad_W2)

    @staticmethod
    def softmax(X):
        e_x = np.exp(X - np.max(X))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def softmax_backward(y, preds):
        grad_softmax = preds - y
        return grad_softmax

    def hidden_layer_backward(self, grad_softmax):
        grad_W1 = np.dot(self.cache["X"], np.dot(self.W2, grad_softmax).T)
        grad_W2 = np.dot(self.cache["h"], grad_softmax)
        return grad_W1, grad_W2
