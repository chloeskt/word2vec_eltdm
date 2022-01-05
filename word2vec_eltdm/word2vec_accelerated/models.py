import torch.nn as nn


class PytorchSimpleWord2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, sparse=True)
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x
