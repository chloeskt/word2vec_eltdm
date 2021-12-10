def train(model, dataloader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch["X"], batch["y"]
        preds = model.forward(X)
        loss, dy = criterion(preds, y)
        dW1, dW2 = model.backward(dy)
        optimizer.step(dW1, dW2)
        train_loss += loss

    train_loss /= len(dataloader)
    return train_loss


def update_best_loss(model, val_loss):
    # Update the model and best loss if we see improvements.
    if not model.best_val_loss or val_loss < model.best_val_loss:
        model.best_val_loss = val_loss
        model.best_W1 = model.W1
        model.best_W2 = model.W2


def validate(model, dataloader, criterion):
    model.eval()
    validation_loss = 0
    for batch_idx, batch in enumerate(dataloader):
        X, y = batch["X"], batch["y"]
        preds, _ = model(X)
        loss, _ = criterion(preds, y)
        validation_loss += loss

    validation_loss /= len(dataloader)

    # Keep track of the best model
    update_best_loss(model, validation_loss)

    return validation_loss
