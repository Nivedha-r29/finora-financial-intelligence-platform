# evaluation/evaluator.py
# Evaluates a trained model on the MNIST test dataset.
#
# The test set is NEVER used during training.
# This module will remain the same when Federated Learning is added —
# the server will call evaluate() on the global model after each round.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on the test dataset.

    Args:
        model       : Trained nn.Module.
        test_loader : DataLoader for the test set (not seen during training).
        criterion   : Same loss function used during training.
        device      : torch.device to run evaluation on.

    Returns:
        test_loss     : Average cross-entropy loss over the test set.
        test_accuracy : Accuracy as a percentage (0–100).
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy
