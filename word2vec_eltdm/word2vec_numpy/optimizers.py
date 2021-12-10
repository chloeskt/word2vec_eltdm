class Optimizer:
    """
    Naive Optimizer using full batch gradient descent
    """

    def __init__(self, model, learning_rate=5e-5):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, dW1, dW2):
        # dW2 = self.model.cache["h"].T @ dy
        # dW1 = self.model.cache["X"].T @ (dy @ self.model.W2.T)
        self.model.W1 -= self.learning_rate * dW1
        self.model.W2 -= self.learning_rate * dW2
