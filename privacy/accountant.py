from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class PrivacyAccountant:
    """A lightweight RDP-style accountant for per-client DP-SGD accounting.

    The implementation is intentionally conservative and transparent: it tracks the
    subsampling rate q = sampling_probability * batch_size / total_examples,
    accumulates RDP over each mini-batch step, and converts the cumulative RDP
    budget back to an epsilon estimate using the standard Gaussian-mechanism
    approximation.

    This is not a full replacement for a production library such as Opacus.
    It is an auditable approximation for experiments and reporting in this
    codebase.
    """
    epsilon: float = 0.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    sampling_probability: float = 1.0
    local_epochs: int = 1
    rounds: int = 1
    total_steps: int = 0
    alpha: float = 8.0
    rdp: float = 0.0

    def _validate(self, total_examples: int, batch_size: int) -> float:
        if total_examples <= 0 or batch_size <= 0:
            raise ValueError("total_examples and batch_size must be positive.")
        if self.sampling_probability <= 0 or self.sampling_probability > 1:
            raise ValueError("sampling_probability must be in (0, 1].")
        q = min(self.sampling_probability * (batch_size / total_examples), 1.0)
        if q <= 0:
            raise ValueError("batch_size must be positive relative to total_examples.")
        return q

    def step(self, total_examples: int, batch_size: int) -> None:
        """Update accounting after a single mini-batch optimization step."""
        q = self._validate(total_examples, batch_size)
        if self.noise_multiplier <= 0:
            return

        sigma = self.noise_multiplier * self.max_grad_norm
        if sigma <= 0:
            return

        # For a single subsampled Gaussian mechanism step, the RDP contribution is
        # approximated as q^2 * (alpha - 1) / (2 * sigma^2).  We accumulate it over
        # the local epochs and communication rounds for each client.
        step_rdp = (q ** 2) * (self.alpha - 1) / (2.0 * (sigma ** 2))
        self.rdp += step_rdp
        self.total_steps += 1
        self.epsilon = self.estimate_epsilon(self.delta)

    def estimate_epsilon(self, target_delta: Optional[float] = None) -> float:
        """Convert the accumulated RDP-like budget to an epsilon estimate."""
        delta = self.delta if target_delta is None else target_delta
        delta = max(delta, 1e-12)
        if self.rdp <= 0:
            return 0.0
        term = math.log(1.0 / delta) / (self.alpha - 1)
        self.epsilon = self.rdp + term
        return self.epsilon

    def reset(self) -> None:
        self.epsilon = 0.0
        self.delta = 1e-5
        self.total_steps = 0
        self.rdp = 0.0
