from dataclasses import dataclass
from typing import Optional


@dataclass
class DPConfig:
    """Configuration for optional local differential privacy in federated training."""
    enable_dp: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    target_epsilon: Optional[float] = None
    target_delta: float = 1e-5
    batch_size: int = 64
    local_epochs: int = 1
    sampling_probability: float = 1.0
    privacy_accountant_type: str = "rdp"
    random_seed: int = 42
    clipping_mode: str = "per_example"
    delta_target: Optional[float] = None
    privacy_rounds: int = 1

    def __post_init__(self):
        if self.delta_target is not None:
            self.target_delta = self.delta_target
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        if self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be non-negative.")
        if self.enable_dp and self.noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive when DP is enabled.")
        if self.clipping_mode != "per_example":
            raise ValueError("Only per-example clipping is supported for DP-SGD.")
        if not 0 < self.sampling_probability <= 1:
            raise ValueError("sampling_probability must be in (0, 1].")
