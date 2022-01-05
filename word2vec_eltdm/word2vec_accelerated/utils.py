import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm

from word2vec_eltdm.common import DataLoader
from word2vec_eltdm.word2vec_accelerated import PytorchSimpleWord2Vec
from word2vec_eltdm.word2vec_accelerated.losses import NegativeSamplingLoss
from word2vec_eltdm.word2vec_accelerated.models import PytorchNegWord2Vec


def train_default(
    model: PytorchSimpleWord2Vec,
    train_dataloader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
    device: str = "cpu",
) -> float:
    train_loss = 0.0
    model.to(device)
    for i, batch in enumerate(tqdm(train_dataloader)):
        model.train()
        optimizer.zero_grad()
        X, y = batch["X"].squeeze(), batch["Y"].squeeze()
        X, y = torch.tensor(X).to(device), torch.LongTensor(y).to(device)
        preds = model.forward(X)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        train_loss += loss

        if i % 1500 == 0:
            print("Current Training Loss {:.6}".format(loss))

    train_loss /= len(train_dataloader)
    return train_loss


def train_NSL(
    model: PytorchNegWord2Vec,
    train_dataloader: DataLoader,
    criterion: NegativeSamplingLoss,
    optimizer: torch.optim.Optimizer,
    n_samples: int,
    device: str = "cpu",
) -> float:
    train_loss = 0.0
    model.to(device)
    for i, batch in enumerate(tqdm(train_dataloader)):
        model.train()
        optimizer.zero_grad()

        X, y = batch["X"].squeeze(), batch["Y"].squeeze()
        X, y = torch.LongTensor(X), torch.LongTensor(y)
        X, y = X.to(device), y.to(device)

        h = model.forward_input(X)
        u = model.forward_output(y)
        noise_vector = model.forward_noise(X.shape[0], n_samples)

        # negative sampling loss
        loss = criterion(h, u, noise_vector)
        loss.backward()
        optimizer.step()

        train_loss += loss

        if i % 1500 == 0:
            print("Current Training Loss {:.6}".format(loss))

    train_loss /= len(train_dataloader)
    return train_loss


def update_best_loss(model: nn.Module, val_loss: float) -> None:
    # Update the model and best loss if we see improvements.
    if not model.best_val_loss or val_loss < model.best_val_loss:
        model.best_val_loss = val_loss
        print(f"Now best model has {val_loss} loss")
        model.save_model()
