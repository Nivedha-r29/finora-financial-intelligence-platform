import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict

class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for models used in the Federated Learning framework.
    Ensures that any model (e.g., MLP for MNIST, or LSTMs/XGBoost/DNNs for financial data)
    implements standard weight extraction and setting methods.
    """
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary of deep copies of model weights/parameters.
        Crucial for client-server communication without modifying local tensors directly.
        """
        return {k: v.cpu().clone().detach() for k, v in self.state_dict().items()}

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Updates internal model weights from a state dictionary.
        """
        self.load_state_dict(parameters)
