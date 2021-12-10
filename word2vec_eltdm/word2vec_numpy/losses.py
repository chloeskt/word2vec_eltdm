import numpy as np


class Loss(object):
    def __init__(self):
        self.grad_history = []

    def forward(self, preds, y):
        return NotImplementedError

    def backward(self, preds, y):
        return NotImplementedError

    def __call__(self, preds, y):
        loss = self.forward(preds, y)
        grad = self.backward(preds, y)
        return loss, grad


class CrossEntropy(Loss):
    def __init__(self):
        super(CrossEntropy).__init__()
        self.cache = {}

    def forward(self, preds, y):
        return -np.sum(np.log(preds) * y)

    def backward(self, preds, y):
        return preds - y
