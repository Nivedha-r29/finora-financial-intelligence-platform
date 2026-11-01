"""
client/trainer.py — Local Training Logic
=========================================

Privacy Principle:
    This module runs ENTIRELY on the client side.
    Gradients and intermediate activations stay local.
    Only the final model state_dict (weights/biases) is returned —
    NOT the DataLoader, NOT any raw samples, NOT the loss per sample.
"""

import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from privacy.config import DPConfig
from privacy.accountant import PrivacyAccountant


class LocalTrainer:
    """
    Encapsulates one client's local training loop.

    Given a copy of the global model and the client's private DataLoader,
    it trains for `local_epochs` epochs and returns:
        - The updated model parameters (state_dict)
        - The average training loss across all local epochs
        - The local training accuracy on the client's own data
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
        self.dataloader = dataloader
        self.device = device
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_name = optimizer_name
        if positive_class_weight is not None and positive_class_weight <= 0:
            raise ValueError("positive_class_weight must be positive.")
        self.positive_class_weight = positive_class_weight
        self.dp_config = dp_config or DPConfig(enable_dp=False)
        self.privacy_accountant = PrivacyAccountant(
            delta=self.dp_config.target_delta,
            noise_multiplier=self.dp_config.noise_multiplier,
            max_grad_norm=self.dp_config.max_grad_norm,
            sampling_probability=self.dp_config.sampling_probability,
            local_epochs=self.dp_config.local_epochs,
            rounds=self.dp_config.privacy_rounds,
        )

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self.optimizer_name == "Adam":
            return torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        return torch.optim.SGD(
            model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=1e-4,
        )

    def _is_binary_task(self, model_kwargs: dict) -> bool:
        return bool(model_kwargs.get("num_classes") == 1 or model_kwargs.get("binary_classification"))

    def _vectorized_per_example_gradients(
        self,
        local_model: nn.Module,
        criterion: nn.Module,
        batch_inputs: torch.Tensor,
        batch_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        parameters = dict(local_model.named_parameters())
        buffers = dict(local_model.named_buffers())

        def loss_for_example(parameter_values, buffer_values, sample_input, sample_label):
            output = torch.func.functional_call(
                local_model,
                (parameter_values, buffer_values),
                (sample_input.unsqueeze(0),),
            )
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                return criterion(
                    output.view(-1),
                    sample_label.float().view(-1),
                )
            return criterion(output, sample_label.view(-1))

        gradients = torch.func.vmap(
            torch.func.grad(loss_for_example),
            in_dims=(None, None, 0, 0),
            randomness="different",
        )(parameters, buffers, batch_inputs, batch_labels)
        parameter_names = [
            name for name, parameter in local_model.named_parameters()
            if parameter.requires_grad
        ]
        squared_norms = torch.zeros(
            batch_inputs.size(0),
            dtype=batch_inputs.dtype,
            device=batch_inputs.device,
        )
        for name in parameter_names:
            per_example = gradients[name].reshape(batch_inputs.size(0), -1)
            squared_norms.add_(per_example.square().sum(dim=1))
        clipping_scales = torch.clamp(
            self.dp_config.max_grad_norm
            / (torch.sqrt(squared_norms).unsqueeze(1) + 1e-6),
            max=1.0,
        )
        clipped_means = {}
        for name in parameter_names:
            parameter = parameters[name]
            gradient = gradients[name]
            scale_shape = (batch_inputs.size(0),) + (1,) * parameter.ndim
            clipped_means[name] = (
                gradient * clipping_scales.view(scale_shape)
            ).mean(dim=0)
        return clipped_means

    def _dp_step(self, local_model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, batch_inputs: torch.Tensor, batch_labels: torch.Tensor) -> float:
        if not self.dp_config.enable_dp:
            optimizer.zero_grad()
            outputs = local_model(batch_inputs)
            if isinstance(criterion, nn.BCEWithLogitsLoss):
                batch_labels = batch_labels.float().view(-1, 1)
                loss = criterion(outputs.view(-1, 1), batch_labels)
            else:
                loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            return loss.item()

        optimizer.zero_grad()
        outputs = local_model(batch_inputs)
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            batch_labels = batch_labels.float().view(-1, 1)
            loss = criterion(outputs.view(-1, 1), batch_labels)
        else:
            loss = criterion(outputs, batch_labels)

        # Vectorize per-example gradients for models whose layers do not couple
        # examples through batch statistics. Keep the original loop as a fallback.
        has_batch_norm = any(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in local_model.modules()
        )
        if hasattr(torch, "func") and not has_batch_norm:
            clipped_means = self._vectorized_per_example_gradients(
                local_model,
                criterion,
                batch_inputs,
                batch_labels,
            )
            for name, parameter in local_model.named_parameters():
                if parameter.requires_grad:
                    parameter.grad = clipped_means[name].clone()
                    noise_std = (
                        self.dp_config.noise_multiplier
                        * self.dp_config.max_grad_norm
                    )
                    parameter.grad.add_(
                        torch.randn_like(parameter.grad) * noise_std
                    )
        else:
            per_sample_grads = []
            for idx in range(batch_inputs.size(0)):
                optimizer.zero_grad()
                local_model.zero_grad(set_to_none=True)
                sample_input = batch_inputs[idx : idx + 1]
                sample_label = batch_labels[idx : idx + 1]
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    sample_logits = local_model(sample_input)
                    sample_loss = criterion(sample_logits.view(-1, 1), sample_label.float().view(-1, 1))
                else:
                    sample_logits = local_model(sample_input)
                    sample_loss = criterion(sample_logits, sample_label)
                sample_loss.backward()

                flat_grad = []
                for p in local_model.parameters():
                    if p.grad is not None:
                        flat_grad.append(p.grad.detach().reshape(-1))
                if flat_grad:
                    per_sample_grads.append(torch.cat(flat_grad))

            if per_sample_grads:
                stacked = torch.stack(per_sample_grads)
                sample_norms = torch.linalg.norm(stacked, dim=1, keepdim=True)
                clipping_scale = torch.clamp(self.dp_config.max_grad_norm / (sample_norms + 1e-6), max=1.0)
                clipped = stacked * clipping_scale
                flat_total = clipped.mean(dim=0)

                grad_index = 0
                for p in local_model.parameters():
                    if p.grad is not None:
                        numel = p.numel()
                        grad = flat_total[grad_index : grad_index + numel].view_as(p)
                        grad_index += numel
                        p.grad = grad.clone()
                        noise_std = self.dp_config.noise_multiplier * self.dp_config.max_grad_norm
                        p.grad.add_(torch.randn_like(p.grad) * noise_std)

            else:
                loss.backward()

        optimizer.step()
        self.privacy_accountant.step(total_examples=len(self.dataloader.dataset), batch_size=batch_inputs.size(0))
        return loss.item()

    def train(
        self, global_parameters: Dict[str, torch.Tensor], model_class, model_kwargs: dict
    ) -> Tuple[Dict[str, torch.Tensor], float, float]:
        """Train a local model for either multiclass or binary classification."""
        local_model = model_class(**model_kwargs)
        local_model.load_state_dict(copy.deepcopy(global_parameters))
        local_model = local_model.to(self.device)
        local_model.train()

        is_binary = self._is_binary_task(model_kwargs)
        if is_binary:
            pos_weight = None
            if self.positive_class_weight is not None:
                pos_weight = torch.tensor(
                    [self.positive_class_weight],
                    dtype=torch.float32,
                    device=self.device,
                )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(local_model)

        total_loss = 0.0
        total_batches = 0

        for _ in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_inputs, batch_labels in self.dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                loss_value = self._dp_step(local_model, optimizer, criterion, batch_inputs, batch_labels)
                epoch_loss += loss_value
                total_batches += 1

            total_loss += epoch_loss

        avg_loss = total_loss / max(total_batches, 1)
        local_accuracy = self._compute_local_accuracy(local_model, model_kwargs)
        updated_parameters = {k: v.cpu().clone().detach() for k, v in local_model.state_dict().items()}
        return updated_parameters, avg_loss, local_accuracy

    def _compute_local_accuracy(self, model: nn.Module, model_kwargs: dict = None) -> float:
        """Evaluate the model on this client's local dataset."""
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = model(inputs)
                if model_kwargs is not None and self._is_binary_task(model_kwargs):
                    preds = (torch.sigmoid(outputs).view(-1) >= 0.5).long()
                    labels = labels.view(-1)
                else:
                    preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        model.train()
        return 100.0 * correct / max(total, 1)
