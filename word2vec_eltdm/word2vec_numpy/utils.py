from copy import deepcopy

from tqdm.notebook import tqdm


def train(model, train_dataloader, criterion, optimizer):
    train_loss = 0.0
    for i, batch in enumerate(tqdm(train_dataloader)):
        model.train()
        X, y = batch["X"], batch["Y"]
        preds = model.forward(X)
        loss, dy = criterion(preds, y)
        dW1, dW2 = model.backward(dy)
        optimizer.step(dW1, dW2)
        train_loss += loss

        if i % 1500 == 0:
            print(
                "Current Training Loss {:.6}".format(loss)
            )

    train_loss /= len(train_dataloader)
    return train_loss


def update_best_loss(model, val_loss):
    # Update the model and best loss if we see improvements.
    if not model.best_val_loss or val_loss < model.best_val_loss:
        model.best_val_loss = val_loss
        model.best_W1 = deepcopy(model.W1)
        model.best_W2 = deepcopy(model.W2)
        model.save_model()


def validate(model, dataloader, criterion):
    model.eval()
    validation_loss = 0
    for i, batch in enumerate(tqdm(dataloader)):
        X, y = batch["X"], batch["Y"]
        preds, _ = model(X)
        loss, _ = criterion(preds, y)
        validation_loss += loss

    validation_loss /= len(dataloader)

    # Keep track of the best model
    update_best_loss(model, validation_loss)

    print("Validation Loss: ", validation_loss)

    return validation_loss
