from abc import ABC, abstractmethod


class Network(ABC):
    """
    Abstract Dataset Base Class
    All subclasses must define forward() method
    """

    def __init__(self, modelname="model_name"):
        self.model_name = modelname
        self.return_grad = True
        self.cache = None

    @abstractmethod
    def forward(self, X):
        """perform the forward pass through a network"""

    @abstractmethod
    def backward(self, X):
        """perform backward pass through the network"""

    def __repr__(self):
        return "This is the base class for all networks we will use"

    def __call__(self, X):
        """takes data points X in train mode, and data X and output y in eval mode"""
        y = self.forward(X)
        if self.return_grad:
            return y, self.backward(y)
        else:
            return y, None

    def train(self):
        """sets the network in training mode, i.e. returns gradient when called"""
        self.return_grad = True

    def eval(self):
        """sets the network in evaluation mode, i.e. only computes forward pass"""
        self.return_grad = False
