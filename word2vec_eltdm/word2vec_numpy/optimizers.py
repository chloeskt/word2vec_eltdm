class Optimizer:
    """
    Naive Optimizer using full batch gradient descent
    """

    def __init__(self, model, learning_rate=5e-5):
        self.model = model
        self.lr = learning_rate

    def step(self, dw):
        raise NotImplementedError
