"""
server/server.py — Federated Server
=====================================

The FederatedServer orchestrates all FL rounds:
  1. Initialise the global model
  2. Distribute global weights to all selected clients
  3. Collect client updates (weights + dataset size only — NO raw data)
  4. Call the FedAvg aggregator
  5. Update the global model
  6. Evaluate on the public test set
  7. Repeat for num_rounds

Privacy guarantee enforced in this module:
    The server loop calls `client.train_locally()` which returns only
    (state_dict, num_samples, loss, accuracy) — a tuple of tensors and
    scalars.  The server never calls `client._dataloader` or accesses
    any client attribute prefixed with `_`.
"""

import copy
import secrets
import time
import torch
import torch.nn as nn
from typing import List, Dict, Tuple

from server.aggregator import FedAvgAggregator
from server.secure_aggregation import (
    MaskedClientUpdate,
    RecoveryShare,
    SecureAggregationConfig,
    SecureAggregationProtocol,
)
from evaluation.evaluator import GlobalEvaluator
from privacy.config import DPConfig
from utils.logger import FLLogger


class FederatedServer:
    """
    Central coordinator for Federated Learning.

    Attributes:
        global_model   : The current global model (updated after every round).
        clients        : List of FederatedClient objects.
        evaluator      : GlobalEvaluator on the shared test set.
        metrics_history: Dict accumulating per-round metrics for plotting.
    """

    def __init__(
        self,
        global_model: nn.Module,
        clients: list,
        test_loader,
        device: torch.device,
        num_rounds: int = 10,
        logger: FLLogger = None,
        num_classes: int = 10,
        dp_config: DPConfig | None = None,
        secure_config: SecureAggregationConfig | None = None,
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.clients_by_id = {client.client_id: client for client in clients}
        self.device = device
        self.num_rounds = num_rounds
        self.logger = logger or FLLogger()
        self.aggregator = FedAvgAggregator()
        self.evaluator = GlobalEvaluator(test_loader, device, num_classes)
        self.dp_config = dp_config or DPConfig(enable_dp=False)
        self.secure_config = secure_config or SecureAggregationConfig(enabled=False)
        client_ids = [client.client_id for client in clients]
        self.secure_protocol = SecureAggregationProtocol(
            self.secure_config,
            client_ids=client_ids if self.secure_config.enabled else None,
        )
        if self.secure_config.enabled:
            if len(set(client_ids)) != len(client_ids):
                raise ValueError("Secure aggregation requires unique client IDs.")
            if len(client_ids) < self.secure_config.min_clients:
                raise ValueError("Secure aggregation client roster is below min_clients.")
            for client in clients:
                client.enable_secure_aggregation(self.secure_config)

        # Accumulated metrics for plotting / saving
        self.metrics_history: Dict = {
            "rounds": [],
            "global_test_loss": [],
            "global_test_accuracy": [],
            "round_duration_seconds": [],
            "total_training_duration_seconds": 0.0,
            "client_losses": {c.client_id: [] for c in clients},
            "client_accuracies": {c.client_id: [] for c in clients},
        }

    def _get_global_parameters(self) -> Dict[str, torch.Tensor]:
        """Return a deep-copy of current global model weights (sent to clients)."""
        return {k: v.cpu().clone().detach() for k, v in self.global_model.state_dict().items()}

    def _update_global_model(self, aggregated_params: Dict[str, torch.Tensor]) -> None:
        """Load aggregated weights back into the global model."""
        self.global_model.load_state_dict(aggregated_params)

    def _prepare_secure_round(self, round_num: int) -> str:
        """Exchange public keys and distribute recovery shares client-to-client."""
        round_id = f"{round_num}-{secrets.token_hex(16)}"
        client_ids = tuple(client.client_id for client in self.clients)
        clients_by_id = {client.client_id: client for client in self.clients}
        public_keys = {
            client_id: client.begin_secure_round(round_id, client_ids)
            for client_id, client in clients_by_id.items()
        }
        for client in self.clients:
            client.accept_secure_public_keys(round_id, public_keys)
        for client in self.clients:
            client.distribute_secure_recovery_shares(
                clients_by_id,
                int(self.secure_config.recovery_threshold),
            )
        self.secure_protocol.begin_round(round_id, public_keys)
        return round_id

    def _collect_recovery_shares(
        self,
        round_id: str,
        masked_updates: List[MaskedClientUpdate],
    ) -> Dict[int, List[RecoveryShare]]:
        submitted_ids = {update.client_id for update in masked_updates}
        missing_ids = set(self.secure_protocol.client_ids) - submitted_ids
        if not missing_ids:
            return {}
        surviving_clients = [
            client for client in self.clients if client.client_id in submitted_ids
        ]
        return {
            missing_id: [
                client.secure_recovery_share(missing_id, round_id)
                for client in surviving_clients
            ]
            for missing_id in missing_ids
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Core FL Training Loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, model_class, model_kwargs: dict) -> Dict:
        """
        Execute `num_rounds` federated training rounds.

        Args:
            model_class  : The model class (e.g. MLP) — passed to clients so
                           they can reconstruct a fresh local model each round.
            model_kwargs : Constructor kwargs (must match model_class signature).

        Returns:
            metrics_history: All round-level metrics for downstream plotting.
        """
        self.logger.section("FEDERATED LEARNING TRAINING STARTED")
        self.logger.info(f"  Clients           : {len(self.clients)}")
        self.logger.info(f"  Communication rounds: {self.num_rounds}")
        self.logger.info(f"  Device            : {self.device}")
        self.logger.info(f"  Aggregation       : {'Secure FedAvg' if self.secure_config.enabled else 'Weighted FedAvg'}")
        if self.dp_config.enable_dp:
            self.logger.info(f"  DP mode          : enabled (noise={self.dp_config.noise_multiplier}, max_grad_norm={self.dp_config.max_grad_norm})")
        else:
            self.logger.info("  DP mode          : disabled")
        if self.secure_config.enabled:
            self.logger.info(f"  Secure masks     : enabled (modulus={self.secure_config.modulus}, qscale={self.secure_config.quantization_scale})")

        # Round 0 is the untrained global model and is included in saved metrics.
        init_loss, init_acc = self.evaluator.evaluate(self.global_model)
        self.metrics_history["rounds"].append(0)
        self.metrics_history["global_test_loss"].append(init_loss)
        self.metrics_history["global_test_accuracy"].append(init_acc)
        self.logger.info(f"\n  Baseline (round 0) — Loss: {init_loss:.4f}  Accuracy: {init_acc:.2f}%")

        total_training_start = time.perf_counter()
        for round_num in range(1, self.num_rounds + 1):
            self.logger.round_header(round_num, self.num_rounds)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            round_start = time.perf_counter()

            # ── Step 1: Distribute global model to all clients ────────────────
            global_params = self._get_global_parameters()
            for client in self.clients:
                client.receive_global_model(copy.deepcopy(global_params))
                self.logger.client_event(client.client_id,
                    f"Received global model parameters ({len(global_params)} parameter tensors)")

            # ── Step 2: Each client trains locally ────────────────────────────
            client_updates: List[Tuple[Dict[str, torch.Tensor], int]] = []
            masked_updates: List[MaskedClientUpdate] = []
            round_id = self._prepare_secure_round(round_num) if self.secure_config.enabled else None
            try:
                for client in self.clients:
                    self.logger.client_event(client.client_id, "Training locally...")
                    if self.secure_config.enabled:
                        masked_update, loss, accuracy = client.train_and_mask_secure(
                            model_class=model_class,
                            model_kwargs=model_kwargs,
                        )
                        masked_updates.append(masked_update)
                        num_samples = masked_update.num_samples
                    else:
                        updated_params, num_samples, loss, accuracy = client.train_locally(
                            model_class=model_class,
                            model_kwargs=model_kwargs,
                        )
                        client_updates.append((updated_params, num_samples))
                    self.logger.client_event(client.client_id,
                        f"Done  | samples={num_samples:,}  loss={loss:.4f}  acc={accuracy:.2f}%")

                    # Track client metrics
                    self.metrics_history["client_losses"][client.client_id].append(loss)
                    self.metrics_history["client_accuracies"][client.client_id].append(accuracy)

                # ── Step 3: FedAvg / Secure Aggregation ───────────────────────
                if self.secure_config.enabled:
                    self.logger.server_event("Aggregating masked client updates via secure aggregation...")
                    aggregate_counts = {
                        update.client_id: self.clients_by_id[update.client_id].num_samples
                        for update in masked_updates
                    }
                    aggregated_params = self.secure_protocol.aggregate(
                        masked_updates,
                        sample_counts=aggregate_counts,
                        recovery_shares=self._collect_recovery_shares(round_id, masked_updates),
                    )
                else:
                    self.logger.server_event("Aggregating client updates via FedAvg...")
                    aggregated_params = self.aggregator.aggregate(client_updates)
            finally:
                if round_id is not None:
                    for client in self.clients:
                        client.finish_secure_round(round_id)

            # ── Step 4: Update global model ───────────────────────────────────
            self._update_global_model(aggregated_params)
            self.logger.server_event("Global model updated.")

            # ── Step 5: Evaluate global model ─────────────────────────────────
            test_loss, test_accuracy = self.evaluator.evaluate(self.global_model)
            self.logger.server_event(
                f"Global Evaluation  →  Loss: {test_loss:.4f}  |  Accuracy: {test_accuracy:.2f}%"
            )

            # Store round metrics
            self.metrics_history["rounds"].append(round_num)
            self.metrics_history["global_test_loss"].append(test_loss)
            self.metrics_history["global_test_accuracy"].append(test_accuracy)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            round_duration = time.perf_counter() - round_start
            self.metrics_history["round_duration_seconds"].append(round_duration)
            self.logger.server_event(f"Round duration: {round_duration:.2f} seconds")

        self.metrics_history["total_training_duration_seconds"] = (
            time.perf_counter() - total_training_start
        )

        self.logger.section("TRAINING COMPLETE")
        final_loss = self.metrics_history["global_test_loss"][-1]
        final_acc  = self.metrics_history["global_test_accuracy"][-1]
        self.logger.info(f"  Final Global Loss     : {final_loss:.4f}")
        self.logger.info(f"  Final Global Accuracy : {final_acc:.2f}%")
        self.logger.info(f"\n  Baseline Accuracy  : {init_acc:.2f}%")
        self.logger.info(f"  Accuracy change    : {final_acc - init_acc:+.2f} percentage points")

        return self.metrics_history

    def save_model(self, path: str) -> None:
        """Persist the final global model weights to disk."""
        torch.save(self.global_model.state_dict(), path)
        self.logger.server_event(f"Global model saved → {path}")
