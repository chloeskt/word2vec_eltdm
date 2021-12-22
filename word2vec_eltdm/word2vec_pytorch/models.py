import os
import pickle

import numpy as np
import torch as torch
import torch.nn as nn

class PytorchWord2Vec(nn.Module):
    """
    Basic implementation of word2vec using the pytorch library
    """

    def __init__(
        self,
        len_vocab: int,
        hidden_size: int = 500,
        embedding_size: int = 300,
    ):
        super(PytorchWord2Vec, self).__init__("PytorchWord2Vec")
        self.len_vocab = len_vocab
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.W1 = nn.Embedding(embedding_size, hidden_size, sparse=True)
        self.W2 = nn.Embedding(embedding_size, hidden_size, sparse=True)
        self.noise = torch.ones(self.len_vocab)

        scale = 1 / np.sqrt(self.embedding_size)
        nn.init.uniform_(self.W1.weight.data, -scale, scale)
        nn.init.uniform_(self.W2.weight.data, -scale, scale)
        '''
        self.cache = {}
        self.best_val_loss = None
        self.best_W1 = None
        self.best_W2 = None
        '''

    def forward(self, X, y):
        embedding_1 = self.W1(X)
        embedding_2 = self.W2(y)
        #neg_embedding_2 = self.W2(N2)
        return embedding_1, embedding_2

    def forward_noise(self, X, samples):
        batch_size = X.shape[0]
        noise_words = torch.multinomial(
            self.noise,
            batch_size * samples,
            replacement=True,
        )
        noise_vect = self.W2(noise_words).view(batch_size, samples, self.hidden_size)
        return noise_vect

    @staticmethod
    def score(embedding_1, embedding_2, neg_embedding_2):
        positive_score = positive_score(embedding_1, embedding_2)
        negative_score = negative_score(embedding_1, neg_embedding_2)
        return torch.mean(positive_score, negative_score)

    @staticmethod
    def positive_score(embedding_1, embedding_2):
        score = torch.sum(torch.mul(embedding_1, embedding_2), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -nn.functional.logsigmoid(score)
        return score

    @staticmethod
    def negative_score(embedding_1, neg_embedding_2):
        score = torch.bmm(neg_embedding_2, embedding_1.unsqueeze(2)).squeeze()
        score = torch.clamp(score, max=10, min=-10)
        score = -torch.sum(nn.functional.logsigmoid(score), dim=1)
        return score

    # TODO
    def save_model(self, directory: str = "../word2vec_eltdm/models") -> None:
        """Save model as pickle"""
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + "/" + self.model_name + "_" + str(self.best_val_loss) + ".p", "wb") as file:
            print('Saving model')
            pickle.dump(model, file)

