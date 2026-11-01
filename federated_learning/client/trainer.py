# client/trainer.py
# Single-client local training loop.
#
# At this stage there is only ONE client and no server.
# This module will later be extended so that each Federated Learning
# client can call train() independently with its own private dataset.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, List


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int = 3,
) -> Tuple[List[float], List[float]]:
    """
    Train a model for a fixed number of epochs on the given DataLoader.

    Args:
        model        : The MLP (or any nn.Module) to train.
        train_loader : DataLoader providing (image, label) mini-batches.
        optimizer    : e.g. Adam or SGD.
        criterion    : Loss function, e.g. CrossEntropyLoss.
        device       : torch.device("cuda") or torch.device("cpu").
        epochs       : Number of full passes over the training data.

    Returns:
        epoch_losses     : Average training loss per epoch.
        epoch_accuracies : Training accuracy (%) per epoch.
    """
    model.to(device)
    model.train()

    epoch_losses: List[float] = []
    epoch_accuracies: List[float] = []

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            running_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        avg_loss = running_loss / total
        accuracy = 100.0 * correct / total

        epoch_losses.append(avg_loss)
        epoch_accuracies.append(accuracy)

        print(f"  Epoch {epoch}/{epochs}  |  Loss: {avg_loss:.4f}  |  Accuracy: {accuracy:.2f}%")

    return epoch_losses, epoch_accuracies
