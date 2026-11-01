"""
server/aggregator.py — Federated Averaging (FedAvg)
=====================================================

Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
from Decentralized Data", AISTATS 2017. https://arxiv.org/abs/1602.05629

FedAvg Formula (weighted by dataset size):
    θ_global = Σ_k  (n_k / N) * θ_k

where:
    θ_k : model parameters from client k after local training
    n_k : number of training samples at client k
    N   : total samples across all participating clients

Why weighted?
    Ensures that clients with more data contribute proportionally more
    to the global model, which is important when data is heterogeneous.
"""

from numbers import Integral
from typing import Dict, List, Mapping, Tuple

import torch


def fedavg(client_updates) -> Dict[str, torch.Tensor]:
    """Return a sample-count-weighted average of compatible client states.

    Updates may be ``(state_dict, sample_count)`` pairs or mappings with
    ``state_dict`` and ``num_samples`` keys. Floating tensors are averaged;
    discrete buffers (such as BatchNorm's ``num_batches_tracked``) use the
    maximum value because integer buffers must not be numerically averaged.
    """
    if not client_updates:
        raise ValueError("No client updates to aggregate.")

    normalized_updates = []
    for update_index, update in enumerate(client_updates):
        if isinstance(update, Mapping):
            try:
                state, sample_count = update["state_dict"], update["num_samples"]
            except KeyError as exc:
                raise ValueError(
                    f"Client update {update_index} must include 'state_dict' and 'num_samples'."
                ) from exc
        else:
            try:
                state, sample_count = update
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Client update {update_index} must be a (state_dict, sample_count) pair."
                ) from exc

        if not isinstance(sample_count, Integral) or isinstance(sample_count, bool) or sample_count <= 0:
            raise ValueError(f"Client update {update_index} has invalid sample count {sample_count!r}.")
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"Client update {update_index} has an empty or invalid state_dict.")
        for key, value in state.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                raise ValueError(f"Client update {update_index} contains an invalid model state entry.")
            if value.layout != torch.strided:
                raise ValueError(f"Client update {update_index} has unsupported sparse tensor '{key}'.")
            if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all().item():
                raise ValueError(f"Client update {update_index} contains non-finite values in '{key}'.")
        normalized_updates.append((state, int(sample_count)))

    reference = normalized_updates[0][0]
    reference_keys = set(reference)
    for update_index, (state, _) in enumerate(normalized_updates[1:], start=1):
        if set(state) != reference_keys:
            raise ValueError(f"Client update {update_index} has model state keys incompatible with update 0.")
        for key, reference_tensor in reference.items():
            tensor = state[key]
            if tensor.shape != reference_tensor.shape or tensor.dtype != reference_tensor.dtype:
                raise ValueError(
                    f"Client update {update_index} has incompatible shape or dtype for '{key}'."
                )

    total_samples = sum(sample_count for _, sample_count in normalized_updates)
    if total_samples <= 0:
        raise ValueError("The total number of client samples must be positive.")

    aggregated = {}
    for key, reference_tensor in reference.items():
        if reference_tensor.is_floating_point() or reference_tensor.is_complex():
            accumulator_dtype = (
                torch.complex128 if reference_tensor.is_complex() and reference_tensor.dtype == torch.complex128
                else torch.complex64 if reference_tensor.is_complex()
                else torch.float64 if reference_tensor.dtype == torch.float64
                else torch.float32
            )
            result = torch.zeros_like(reference_tensor, dtype=accumulator_dtype)
            for state, sample_count in normalized_updates:
                result.add_(state[key].to(device=result.device, dtype=accumulator_dtype), alpha=sample_count / total_samples)
            aggregated[key] = result.to(dtype=reference_tensor.dtype)
        else:
            result = reference_tensor.detach().clone()
            for state, _ in normalized_updates[1:]:
                result = torch.maximum(result, state[key].to(device=result.device))
            aggregated[key] = result

    return aggregated


class FedAvgAggregator:
    """
    Implements the Federated Averaging (FedAvg) aggregation algorithm.

    The aggregator is designed to be model-agnostic: it operates only on
    state_dict tensors, so it works with any PyTorch model (MLP, DNN, LSTM, etc.)
    without modification — making it directly reusable for Phase 2.
    """

    @staticmethod
    def aggregate(
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Perform weighted FedAvg aggregation.

        Args:
            client_updates: List of (state_dict, num_samples) tuples,
                            one entry per participating client.

        Returns:
            aggregated_parameters: Weighted average state_dict — the new
                                   global model weights.

        What this function NEVER sees:
            - Client raw data or images
            - Individual sample predictions or losses
            - Any client-identifying metadata beyond dataset size
        """
        return fedavg(client_updates)

    @staticmethod
    def average_unweighted(
        client_updates: List[Tuple[Dict[str, torch.Tensor], int]],
    ) -> Dict[str, torch.Tensor]:
        """
        Simple unweighted average (all clients treated equally).
        Included for completeness and experimental comparison.
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate.")
        unweighted_updates = [(state, 1) for state, _ in client_updates]
        return fedavg(unweighted_updates)
