import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class PytorchSimpleWord2Vec(nn.Module):
    """
    Simplest version of Word2Vector network in Pytorch.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.model_name = "PytorchSimpleWord2Vec"
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)
        self.best_val_loss = None

    def initialize_weights(self) -> None:
        self.embedding.weight.data.uniform_(-1, 1)
        self.linear.weight.data.uniform_(-1, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        out = F.softmax(x, dim=1)
        return out

    def save_model(self, directory: str = "../word2vec_eltdm/models") -> None:
        """Save model as pickle"""
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(
            directory
            + "/"
            + self.model_name
            + "_"
            + str(self.best_val_loss.item())
            + ".p",
            "wb",
        ) as file:
            print("Saving model")
            pickle.dump(model, file)


class PytorchNegWord2Vec(nn.Module):
    """
    Word2Vec architecture using Negative Sampling Loss in Pytorch.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        noise_dist: torch.Tensor,
        device: str = "cpu"
    ) -> None:
        super().__init__()
        self.model_name = "PytorchNegWord2Vec"
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.noise_dist = noise_dist
        self.embedding_input = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding_output = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.best_val_loss = None
        self.device = device

    def initialize_weights(self) -> None:
        self.embedding_input.weight.data.uniform_(-1, 1)
        self.embedding_output.weight.data.uniform_(-1, 1)

    def forward_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_input(x)
        return x

    def forward_output(self, y: torch.Tensor) -> torch.Tensor:
        out = self.embedding_output(y)
        return out

    def forward_noise(self, batch_size: int, n_samples: int) -> torch.Tensor:
        if self.noise_dist is None:
            noise_dist = torch.ones(self.vocab_size)
        else:
            noise_dist = self.noise_dist

        noise_dist = torch.tensor(noise_dist)
        noise_words = torch.multinomial(
            noise_dist, batch_size * n_samples, replacement=True
        )

        noise_words = noise_words.to(self.device)
        noise_vector = self.embedding_output(noise_words).view(
            batch_size, n_samples, self.embedding_dim
        )
        return noise_vector

    def save_model(self, directory: str = "../word2vec_eltdm/models") -> None:
        """Save model as pickle"""
        model = {self.model_name: self}
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(
            directory
            + "/"
            + self.model_name
            + "_"
            + str(self.best_val_loss.item())
            + ".p",
            "wb",
        ) as file:
            print("Saving model")
            pickle.dump(model, file)
