"""
evaluation/evaluator.py — Global Model Evaluation
===================================================

Evaluates the current global model on the shared, server-side test set.

This is the ONLY place where the server uses actual data — the public
test set — to measure generalisation. Client private training data
is never seen here.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple, Dict


class GlobalEvaluator:
    """Evaluates the global model on the shared test dataset."""

    def __init__(self, test_loader: DataLoader, device: torch.device, num_classes: int = 10):
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes

    def _is_binary_classification(self, model: nn.Module) -> bool:
        return self.num_classes == 1 or getattr(model, "binary_classification", False)

    def evaluate(self, model: nn.Module) -> Tuple[float, float]:
        """Compute average loss and accuracy for either multi-class or binary classification."""
        model.eval()
        model.to(self.device)
        is_binary = self._is_binary_classification(model)
        criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                if is_binary:
                    y_true = labels.float().view(-1, 1)
                    loss = criterion(outputs.view(-1, 1), y_true)
                    preds = (torch.sigmoid(outputs).view(-1) >= 0.5).long()
                    labels = labels.view(-1)
                else:
                    loss = criterion(outputs, labels)
                    preds = outputs.argmax(dim=1)
                total_loss += loss.item() * labels.size(0)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        return avg_loss, accuracy

    def per_class_accuracy(self, model: nn.Module) -> Dict[int, float]:
        """Compute per-class accuracy. For binary fraud tasks, return the two-class rates."""
        model.eval()
        model.to(self.device)
        class_correct = [0] * max(1, self.num_classes)
        class_total = [0] * max(1, self.num_classes)

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                if self._is_binary_classification(model):
                    preds = (torch.sigmoid(outputs).view(-1) >= 0.5).long()
                    labels = labels.view(-1)
                    for label, pred in zip(labels, preds):
                        index = int(label.item())
                        class_total[index] += 1
                        if pred.item() == index:
                            class_correct[index] += 1
                else:
                    preds = outputs.argmax(dim=1)
                    for label, pred in zip(labels, preds):
                        c = label.item()
                        class_total[c] += 1
                        if pred.item() == c:
                            class_correct[c] += 1

        if self._is_binary_classification(model):
            return {
                0: (100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0),
                1: (100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0),
            }
        return {
            c: (100.0 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0.0)
            for c in range(self.num_classes)
        }
