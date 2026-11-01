# client/client.py
# Reusable FederatedClient class.
# Each client maintains its own local dataset, DataLoader, model, optimizer, and loss function.

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Any


class FederatedClient:
    """
    Simulated Federated Learning Client device/organization.
    
    Contains:
      - client_id     : Unique numeric identifier (1, 2, ...)
      - dataset       : Private local dataset partition
      - loader        : Local DataLoader
      - model         : Local copy of the model
      - optimizer     : Local optimizer (Adam)
      - criterion     : Loss function (CrossEntropyLoss)
      - device        : CPU or CUDA device
    """

    def __init__(
        self,
        client_id: int,
        local_dataset: Dataset,
        model_template: nn.Module,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        device: torch.device = None,
    ):
        self.client_id = client_id
        self.local_dataset = local_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device if device is not None else torch.device("cpu")

        # DataLoader for local training
        self.train_loader = DataLoader(
            self.local_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Deep copy template model so this client has its own independent weights tensor instances
        self.model = copy.deepcopy(model_template)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_num_samples(self) -> int:
        """Returns the number of local training samples."""
        return len(self.local_dataset)

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Returns a copy of the client's current model state_dict."""
        return copy.deepcopy(self.model.state_dict())

    def set_parameters(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Loads state_dict into client local model."""
        self.model.load_state_dict(state_dict)

    def train_local(self, epochs: int = 1) -> Tuple[float, float]:
        """
        Runs local training for specified number of epochs.

        Returns:
            final_loss     : Local training loss.
            final_accuracy : Local training accuracy (%).
        """
        self.model.train()
        final_loss = 0.0
        final_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            final_loss = running_loss / total
            final_accuracy = 100.0 * correct / total

        return final_loss, final_accuracy
