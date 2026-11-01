# model/mlp.py
# MLP model for MNIST digit classification.
#
# Architecture:
#   Input  : 784 (28x28 flattened)
#   Layer 1: Linear(784, 128) -> ReLU
#   Layer 2: Linear(128, 64)  -> ReLU
#   Output : Linear(64, 10)   -> raw logits (no Softmax, used with CrossEntropyLoss)
#
# Designed as a plain nn.Module so it can later be used in a Federated Learning
# framework where each client holds its own copy of this model.

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-Layer Perceptron for MNIST classification.

    Input shape : (batch_size, 1, 28, 28)  OR  (batch_size, 784)
    Output shape: (batch_size, 10)  — raw logits for digits 0-9
    """

    def __init__(self):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, 10)   # 10 output classes (digits 0-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)   # (N, 1, 28, 28) -> (N, 784)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)       # raw logits — CrossEntropyLoss handles softmax internally
        return x
