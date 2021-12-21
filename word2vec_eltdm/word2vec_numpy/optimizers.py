import numpy as np


class Optimizer:
    """
    Naive Optimizer using full batch gradient descent
    """

    def __init__(
        self,
        model,
        learning_rate: float = 5e-5,
        decay_rate: float = None,
        method: str = "time_based",
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.method = method
        self.iterations = 0

    def step(self, dW1, dW2):
        self.model.W1[self.model.cache["X"].flatten(), :] -= self.learning_rate * dW1.T
        self.model.W2 -= self.learning_rate * dW2

        self.iterations += 1

    def update_lr(self, epoch: int):
        if self.method == "time_based":
            self.learning_rate *= 1.0 / (1.0 + self.decay_rate * self.iterations)

        elif self.method == "exp_decay":
            k = 0.001
            self.learning_rate *= np.exp(-k * self.iterations)

        elif self.method == "step_decay":
            drop = 0.5
            epoch_drop = 5.0
            if epoch % epoch_drop == 0 and epoch != 0:
                self.learning_rate *= drop

        elif self.method == "none":
            pass

        else:
            raise NotImplementedError


class SGDHogwild:
    """
    SDG Hogwild from Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
    """

    def __init__(self, model, learning_rate=5e-5):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, dW1, dW2):
        raise NotImplementedError


class OptimizeNSL(Optimizer):
    """
    Naive Optimizer using full batch gradient descent for Negative Sampling Loss
    """

    def __init__(
        self,
        model,
        learning_rate: float = 5e-5,
        decay_rate: float = None,
        method: str = "time_based",
    ):
        super().__init__(model, learning_rate, decay_rate, method)
        self.model = model
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.method = method
        self.iterations = 0

    def step(self, dW1, dW2_pos, dW2_neg):
        batch_size, embed_size = dW1.shape
        # update W1 weights
        X = self.model.cache["X"]
        self.model.W1[X.squeeze(), :] -= self.learning_rate * dW1

        # update W2 weights for positive samples
        y = self.model.cache["y"]
        self.model.W2[y.squeeze(), :] -= self.learning_rate * dW2_pos

        # update W2 weights for negative samples
        dW2_neg = dW2_neg.reshape(
            batch_size * self.model.cache["n_samples"], embed_size
        )
        self.model.W2[self.model.cache["noise_words"], :] -= (
            self.learning_rate * dW2_neg
        )

        self.iterations += 1
