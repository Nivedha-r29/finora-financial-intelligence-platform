"""
client/client.py — Federated Client
=====================================

Each FederatedClient represents one participant in the FL system.

Privacy Model:
    ✅  Client receives : global model parameters (weights/biases tensors)
    ✅  Client sends    : updated model parameters + dataset size
    ❌  Client NEVER sends raw data, images, or individual sample labels
    ❌  Server NEVER accesses client.dataloader directly
"""

from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader

from client.trainer import LocalTrainer
from privacy.config import DPConfig
from server.secure_aggregation import (
    MaskedClientUpdate,
    RecoveryShare,
    SecureAggregationConfig,
    SecureAggregationParticipant,
)


class FederatedClient:
    """
    Represents a single federated learning participant.

    Responsibilities:
        - Hold a private DataLoader (never shared with the server)
        - Receive global model parameters from the server
        - Train locally and return updated parameters + sample count

    The client is intentionally thin — all ML logic lives in LocalTrainer,
    making it easy to swap training strategies in future phases.
    """

    def __init__(
        self,
        client_id: int,
        dataloader: DataLoader,
        device: torch.device,
        local_epochs: int = 2,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        optimizer_name: str = "SGD",
        dp_config: Optional[DPConfig] = None,
        positive_class_weight: Optional[float] = None,
    ):
        self.client_id = client_id
        self._dataloader = dataloader          # PRIVATE — server must NOT access this
        self.device = device
        self.num_samples = len(dataloader.dataset)
        self.dp_config = dp_config or DPConfig(enable_dp=False)
        self._secure_participant: Optional[SecureAggregationParticipant] = None

        self._trainer = LocalTrainer(
            client_id=client_id,
            dataloader=dataloader,
            device=device,
            local_epochs=local_epochs,
            learning_rate=learning_rate,
            momentum=momentum,
            optimizer_name=optimizer_name,
            dp_config=self.dp_config,
            positive_class_weight=positive_class_weight,
        )

    def enable_secure_aggregation(self, config: SecureAggregationConfig) -> None:
        """Create this client's private secure-aggregation participant state."""
        self._secure_participant = SecureAggregationParticipant(self.client_id, config)

    def begin_secure_round(self, round_id: str, client_ids) -> bytes:
        if self._secure_participant is None:
            raise RuntimeError("Secure aggregation has not been enabled for this client.")
        return self._secure_participant.begin_round(round_id, client_ids)

    def accept_secure_public_keys(self, round_id: str, public_keys) -> None:
        if self._secure_participant is None:
            raise RuntimeError("Secure aggregation has not been enabled for this client.")
        self._secure_participant.accept_public_keys(round_id, public_keys)

    def distribute_secure_recovery_shares(self, recipients, threshold: int) -> None:
        if self._secure_participant is None:
            raise RuntimeError("Secure aggregation has not been enabled for this client.")
        self._secure_participant.distribute_recovery_shares(recipients, threshold)

    def receive_secure_recovery_share(self, share: RecoveryShare) -> None:
        if self._secure_participant is None:
            raise RuntimeError("Secure aggregation has not been enabled for this client.")
        self._secure_participant.receive_secure_recovery_share(share)

    def secure_recovery_share(self, owner_id: int, round_id: str) -> RecoveryShare:
        if self._secure_participant is None:
            raise RuntimeError("Secure aggregation has not been enabled for this client.")
        return self._secure_participant.recovery_share_for(owner_id, round_id)

    def finish_secure_round(self, round_id: str) -> None:
        if self._secure_participant is None:
            raise RuntimeError("Secure aggregation has not been enabled for this client.")
        self._secure_participant.finish_round(round_id)

    def train_and_mask_secure(
        self,
        model_class,
        model_kwargs: dict,
    ) -> Tuple[MaskedClientUpdate, float, float]:
        """Train locally and return only the masked upload plus monitoring scalars."""
        if self._secure_participant is None:
            raise RuntimeError("Secure aggregation has not been enabled for this client.")
        updated_parameters, _, avg_loss, local_accuracy = self.train_locally(
            model_class=model_class,
            model_kwargs=model_kwargs,
        )
        masked_update = self._secure_participant.mask_client_update(
            updated_parameters,
            num_samples=self.num_samples,
        )
        return masked_update, avg_loss, local_accuracy

    def receive_global_model(self, global_parameters: Dict[str, torch.Tensor]) -> None:
        """
        Store the latest global model parameters received from the server.

        Args:
            global_parameters: State dict of tensors — the ONLY thing
                                the server is allowed to send.
        """
        self._global_parameters = global_parameters

    def train_locally(
        self,
        model_class,
        model_kwargs: dict,
    ) -> Tuple[Dict[str, torch.Tensor], int, float, float]:
        """
        Execute local training and return update to server.

        Returns:
            updated_parameters : New model weights after local training.
            num_samples        : Number of local training samples (for weighted FedAvg).
            avg_loss           : Average training loss (for logging/monitoring).
            local_accuracy     : Accuracy on local data (for logging/monitoring).

        What is NOT returned:
            - Raw images or pixel values
            - Per-sample losses or predictions
            - Any information about individual data points
        """
        updated_parameters, avg_loss, local_accuracy = self._trainer.train(
            global_parameters=self._global_parameters,
            model_class=model_class,
            model_kwargs=model_kwargs,
        )
        return updated_parameters, self.num_samples, avg_loss, local_accuracy

    def __repr__(self) -> str:
        return f"FederatedClient(id={self.client_id}, samples={self.num_samples})"
