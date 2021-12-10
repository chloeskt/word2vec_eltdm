import numpy as np

from word2vec_eltdm.word2vec_numpy.base_network import Network


class SimpleWord2Vec(Network):
    """
    Very basic implementation of Word2Vec without any "speed" tricks.
    """

    def __init__(
        self,
        len_vocab: int,
        num_layer: int = 1,
        hidden_size: int = 500,
        embedding_size: int = 300,
    ):
        super(SimpleWord2Vec, self).__init__("SimpleWord2Vec")
        self.len_vocab = len_vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size
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
                self.embedding_size,
                self.len_vocab,
            ), "weights for initialization are not in the correct shape (self.embedding_size, len(self.len_vocab))"
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
                size=(self.embedding_size, self.len_vocab),
            )

    def forward(self, X):
        assert self.W1 is not None, "weight matrix W1 is not initialized"
        assert self.W2 is not None, "weight matrix W2 is not initialized"

        h = X @ self.W1
        u = h @ self.W2
        y = self.softmax(u)
        self.cache["X"] = X
        self.cache["h"] = h
        self.cache["logits"] = u

        return y

    @staticmethod
    def softmax(X):
        # careful X is a matrix (2*window*batch_size, len(self.len_vocab))
        preds = []
        for x in X:
            exp = np.exp(x - np.max(x))
            preds.append(exp / exp.sum(axis=0))
        return np.asarray(preds)

    def backward(self, grad_softmax):
        dW2 = self.cache["h"].T @ grad_softmax
        dW1 = self.cache["X"].T @ (grad_softmax @ self.W2.T)
        return dW1, dW2
