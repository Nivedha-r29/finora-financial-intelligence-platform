import torch
import torch.nn as nn
from model.base_model import BaseModel


class MLP(BaseModel):
    """
    Multi-Layer Perceptron (MLP) for MNIST digit classification.

    The same architecture is reused for binary fraud detection by setting
    `num_classes=1` and `binary_classification=True`.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        num_classes: int = 10,
        dropout_p: float = 0.2,
        binary_classification: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.binary_classification = binary_classification or num_classes == 1

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)
