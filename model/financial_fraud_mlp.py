import torch
import torch.nn as nn

from model.base_model import BaseModel


class FinancialFraudMLP(BaseModel):
    """Small tabular binary classifier without BatchNorm for per-example DP-SGD."""

    def __init__(
        self,
        input_dim: int = 30,
        hidden_dim1: int = 32,
        hidden_dim2: int = 16,
        binary_classification: bool = True,
    ):
        super().__init__()
        if not binary_classification:
            raise ValueError("FinancialFraudMLP supports binary classification only.")
        self.binary_classification = binary_classification
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs.view(inputs.size(0), -1))
