import numpy as np
from torch import nn


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
        m = preds.shape[1]
        return -(1 / m) * np.sum(
            np.log(preds[y.flatten(), np.arange(y.shape[1])] + 0.001)
        )

    def backward(self, preds, y):
        m = preds.shape[1]
        preds[y.flatten(), np.arange(m)] -= 1.0
        grad = preds
        return grad


class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, model, input_vectors, output_vectors, noise_vectors, y):
        loss = self.forward(input_vectors, output_vectors, noise_vectors)
        grad1, grad2 = self.backward(
            model, input_vectors, output_vectors, noise_vectors, y
        )
        return loss, grad1, grad2

    def forward(self, input_vectors, output_vectors, noise_vectors):
        batch_size, embed_size = input_vectors.shape

        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.reshape(batch_size, embed_size, 1)

        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.reshape(batch_size, 1, embed_size)

        # correct log-sigmoid loss
        out_loss = np.log(self.sigmoid(output_vectors @ input_vectors))
        out_loss = out_loss.squeeze()

        # incorrect log-sigmoid loss
        noise_loss = np.log(self.sigmoid(-noise_vectors @ input_vectors))
        noise_loss = noise_loss.squeeze().sum(
            1
        )  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()

    @staticmethod
    def sigmoid(x):
        sig = np.where(x < 0, np.exp(x) / (1 + np.exp(x)), 1 / (1 + np.exp(-x)))
        return sig

    def backward(self, model, input_vectors, output_vectors, noise_vectors, y):
        batch_size, embed_size = input_vectors.shape

        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.reshape(batch_size, embed_size, 1)
        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.reshape(batch_size, 1, embed_size)

        cj_h = self.sigmoid(output_vectors @ input_vectors).squeeze(axis=2)

        cn_h = self.sigmoid(noise_vectors @ input_vectors).squeeze()

        outs = np.concatenate([cj_h, cn_h], axis=1)

        # compute prediction error
        outs[:, -1] -= 1
        outs = np.expand_dims(outs, axis=1)

        # gradient wrt to W1
        W1_y = model.W1[y, :].reshape(batch_size, 1, embed_size)
        W1_noise = model.W1[model.cache["noise_words"], :].reshape(
            batch_size, model.cache["n_samples"], embed_size
        )
        W1_batch = np.concatenate([W1_y, W1_noise], axis=1)
        grad_W1 = (outs @ W1_batch).squeeze()

        # gradient wrt to W2
        outs = np.moveaxis(outs, [0, 1, 2], [0, 2, 1])
        input_vectors = np.moveaxis(input_vectors, [0, 1, 2], [0, 2, 1])
        grad_W2 = outs @ input_vectors

        return grad_W1, grad_W2
