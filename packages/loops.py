import torch
from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
                train_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device = "cpu"):

    model.train()
    train_loss, train_acc = 0.0, 0.0

    for batch, (X,y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
        if(batch % 2 == 0):
            print(f"Batch: {batch}/{len(train_loader)} | Train loss: {train_loss/(batch+1):.4f} | Train acc: {train_acc/(batch+1):.4f}")

    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
                test_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                device = "cpu"):

    model.eval()
    test_loss, test_acc = 0.0, 0.0

    for batch, (X,y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        test_loss += loss.item()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        test_acc += (y_pred_class == y).sum().item()/len(y_pred)

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
            train_loader: torch.utils.data.DataLoader,
            test_loader: torch.utils.data.DataLoader,
            loss_fn: torch.nn.Module  = torch.nn.CrossEntropyLoss(),
            optimizer: torch.optim.Optimizer = None,
            epochs: int = 5,
            device = "cpu"):

    
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_loader, loss_fn, device)
        print(f"Epoch: {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results