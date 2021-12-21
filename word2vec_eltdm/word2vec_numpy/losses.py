import numpy as np


class Loss:
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


class NegativeSamplingLoss(Loss):
    def __init__(self):
        super().__init__()

    def __call__(self, model, input_vectors, output_vectors, noise_vectors, y):
        loss = self.forward(input_vectors, output_vectors, noise_vectors)
        grad1, grad2_pos, grad2_neg = self.backward(
            model, input_vectors, output_vectors, noise_vectors, y
        )
        return loss, grad1, grad2_pos, grad2_neg

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

        sigmoid_context = (
            self.sigmoid(output_vectors @ input_vectors).squeeze(axis=2) - 1
        ).squeeze()
        product_context = np.multiply(
            output_vectors.squeeze(), sigmoid_context[:, None]
        )

        sigmoid_noise = self.sigmoid(-noise_vectors @ input_vectors)
        sigmoid_noise -= np.ones(sigmoid_noise.shape)

        context_noise = np.multiply(noise_vectors, sigmoid_noise)
        context_noise = context_noise.sum(axis=1)

        # gradient wrt W1
        grad_W1 = product_context - context_noise

        # gradient wrt context words
        grad_W2_positive = np.multiply(
            input_vectors.squeeze(), sigmoid_context[:, None]
        )

        # gradient wrt negative words (one gradient for each negative word and for each
        # context word)
        grad_W2_negative = np.negative(
            np.multiply(input_vectors.reshape(batch_size, 1, embed_size), sigmoid_noise)
        )

        return grad_W1, grad_W2_positive, grad_W2_negative
