class SimpleWord2Vec:
    """
    Very basic implementation of Word2Vec without any "speed" tricks.
    1. Create embeddings
    2. Create hidden layers
    3. Simple softmax
    """

    def __init__(self):
        raise NotImplementedError

    def forward(self, X):
        raise NotImplementedError

    def backward(self, y):
        raise NotImplementedError

    def initialize_weights(self, weights=None):
        raise NotImplementedError

    def softmax(self, X):
        raise NotImplementedError
