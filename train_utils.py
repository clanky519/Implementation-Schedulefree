
import time
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    if hasattr(optimizer, "train"):
        optimizer.train()
    
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        # Schedule-Free optimizers need parameters moved to y_t
        # before the forward pass.
        if hasattr(optimizer, "prepare_for_training"):
            optimizer.prepare_for_training()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    elapsed = time.time() - start_time

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
        "time": elapsed
    }


@torch.no_grad()
def evaluate(model, loader, device, optimizer="None"):
    if optimizer is not None and hasattr(optimizer, "eval"):
        optimizer.eval()
    
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total
    }
