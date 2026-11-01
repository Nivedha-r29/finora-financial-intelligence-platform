import os
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class FLConfig:
    """
    Configuration settings for the Federated Learning Framework.
    Allows easy customization of hyperparameters, dataset options, and training settings.
    """
    # Dataset and model configuration
    dataset_name: str = "MNIST"
    task: str = "multiclass"  # "multiclass" or "binary"
    data_dir: str = "./data_storage"
    num_classes: int = 10
    input_dim: int = 784  # 28x28 flattened
    hidden_dim1: int = 128
    hidden_dim2: int = 64

    # Federated learning parameters
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.001
    momentum: float = 0.9
    optimizer_name: str = "Adam"  # Options: "SGD", "Adam"

    # Data partitioning strategy
    partition_strategy: str = "iid"  # Options: "iid", "non_iid"
    dirichlet_alpha: float = 0.5

    # Financial fraud dataset settings
    financial_data_path: str = "./creditcard.csv"
    financial_dataset_url: str = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    financial_label_column: str = "Class"
    financial_positive_label: int = 1
    financial_negative_label: int = 0
    financial_test_size: float = 0.2
    financial_imbalance_weight: float = 1.0
    financial_feature_columns: tuple = ()

    # Output and save paths
    results_dir: str = "./results"
    model_save_path: str = "./results/federated_global_model.pth"
    metrics_save_path: str = "./results/federated_metrics.json"
    global_metrics_csv_path: str = "./results/federated_metrics.csv"
    client_metrics_csv_path: str = "./results/client_training_metrics.csv"
    phase4_results_dir: str = "./results/phase4"
    phase4_noniid_rounds: int = 2
    phase4_noniid_alpha: float = 0.3
    phase4_centralized_epochs: int = 10
    deterministic_experiments: bool = True

    # Aggregation mode: standard FedAvg, secure masked aggregation, DP, or combined secure+DP.
    aggregation_mode: str = "fedavg"  # one of: {"fedavg", "secure", "dp", "secure_dp"}

    # Secure aggregation settings
    secure_aggregation: bool = False
    secure_quantization_scale: float = 1_000.0
    secure_modulus: int = 2**61 - 1
    secure_min_clients: int = 2
    secure_max_dropout_fraction: float = 0.2
    secure_allow_recovery: bool = True
    secure_recovery_threshold: Optional[int] = None
    secure_max_abs_update: float = 1_000_000.0

    # Differential privacy settings
    enable_dp: bool = False
    dp_noise_multiplier: float = 1.0
    dp_max_grad_norm: float = 1.0
    dp_target_delta: float = 1e-5
    dp_sampling_probability: float = 1.0

    # Device configuration (CUDA auto-detection)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    def __post_init__(self):
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        if self.dataset_name.lower() in {"credit_card_fraud", "financial_fraud", "fraud"}:
            self.task = "binary"
            self.num_classes = 1
            if not self.financial_feature_columns:
                self.financial_feature_columns = ()

        allowed_modes = {"fedavg", "secure", "dp", "secure_dp"}
        if self.aggregation_mode not in allowed_modes:
            raise ValueError(f"aggregation_mode must be one of {sorted(allowed_modes)}.")

        self.secure_aggregation = self.secure_aggregation or self.aggregation_mode in {
            "secure",
            "secure_dp",
        }
        self.enable_dp = self.enable_dp or self.aggregation_mode in {"dp", "secure_dp"}
        if self.secure_aggregation and self.enable_dp:
            self.aggregation_mode = "secure_dp"
        elif self.secure_aggregation:
            self.aggregation_mode = "secure"
        elif self.enable_dp:
            self.aggregation_mode = "dp"
        else:
            self.aggregation_mode = "fedavg"

        if self.enable_dp:
            if self.dp_noise_multiplier < 0:
                raise ValueError("dp_noise_multiplier must be non-negative.")
            if self.dp_max_grad_norm <= 0:
                raise ValueError("dp_max_grad_norm must be positive.")
            if not 0 < self.dp_sampling_probability <= 1:
                raise ValueError("dp_sampling_probability must be in (0, 1].")

        if self.secure_aggregation:
            if self.secure_min_clients < 2:
                raise ValueError("secure_min_clients must be at least 2.")
            if self.secure_min_clients > self.num_clients:
                raise ValueError("secure_min_clients cannot exceed num_clients.")
            if not 0 <= self.secure_max_dropout_fraction < 1:
                raise ValueError("secure_max_dropout_fraction must be in [0, 1).")
            if self.secure_recovery_threshold is not None and self.secure_recovery_threshold < 2:
                raise ValueError("secure_recovery_threshold must be at least 2.")
            if self.secure_recovery_threshold is not None and self.secure_recovery_threshold > self.secure_min_clients:
                raise ValueError("secure_recovery_threshold cannot exceed secure_min_clients.")
            if self.secure_recovery_threshold is not None and self.secure_recovery_threshold > self.num_clients:
                raise ValueError("secure_recovery_threshold cannot exceed num_clients.")
            if self.secure_max_abs_update <= 0:
                raise ValueError("secure_max_abs_update must be positive.")
