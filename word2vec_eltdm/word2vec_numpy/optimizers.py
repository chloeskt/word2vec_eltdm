class Optimizer:
    """
    Naive Optimizer using full batch gradient descent
    """

    def __init__(self, model, learning_rate=5e-5):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, dW1, dW2):
        self.model.W1[self.model.cache["X"].flatten(), :] -= self.learning_rate * dW1.T
        self.model.W2 -= self.learning_rate * dW2


class SGDHogwild:
    """
    SDG Hogwild from Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
    """

    def __init__(self, model, learning_rate=5e-5):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, dW1, dW2):
        raise NotImplementedError
