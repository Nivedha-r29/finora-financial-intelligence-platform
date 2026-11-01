"""
data/dataset.py — MNIST Data Loading & Client Partitioning
===========================================================

Privacy Principle:
    The partitioning logic runs ONCE on the server before training starts.
    After that, each client receives only its *indices* into the dataset —
    no raw pixel values ever leave the client's DataLoader.

    What the server sees  ✅ : client dataset sizes (for weighted FedAvg)
    What the server never sees ❌ : raw MNIST images / labels
"""

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional

from data.financial_dataset import (
    DATASET_SOURCE,
    ensure_financial_dataset,
    inspect_financial_dataset,
    partition_financial_clients,
    preprocess_financial_data,
    summarize_client_distributions,
)


def get_mnist_transforms() -> transforms.Compose:
    """Standard MNIST normalisation.  Mean=0.1307, Std=0.3081 are dataset statistics."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def load_mnist(data_dir: str = "./data_storage") -> Tuple[datasets.MNIST, datasets.MNIST]:
    """
    Download (or load from cache) the full MNIST dataset.

    Returns:
        train_dataset : 60,000 labelled training samples
        test_dataset  :  10,000 labelled test samples (shared evaluation set)
    """
    transform = get_mnist_transforms()
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    return train_dataset, test_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Partitioning Strategies
# ─────────────────────────────────────────────────────────────────────────────

def partition_iid(
    dataset: datasets.MNIST,
    num_clients: int,
    seed: int = 42,
) -> List[List[int]]:
    """
    IID (Independent & Identically Distributed) partition.

    Shuffles the training set indices randomly and splits them into
    `num_clients` roughly equal-sized shards.  Each client receives
    a representative sample of all 10 digit classes.
    """
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(dataset))
    rng.shuffle(all_indices)
    client_indices: List[List[int]] = [shard.tolist() for shard in np.array_split(all_indices, num_clients)]
    return client_indices


def partition_non_iid(
    dataset: datasets.MNIST,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[List[int]]:
    """
    Non-IID partition via Dirichlet distribution.

    Each client receives data drawn from a Dirichlet(alpha) distribution
    over classes.  Smaller alpha → more skewed class distribution → harder
    federated learning problem.  This models real-world data heterogeneity.

    Args:
        alpha: Dirichlet concentration parameter (e.g. 0.1 → highly non-iid,
               100 → nearly iid).
    """
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))

    # Group indices by class
    class_indices: Dict[int, np.ndarray] = {
        c: np.where(targets == c)[0] for c in range(num_classes)
    }

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = class_indices[c]
        rng.shuffle(idx)
        # Dirichlet proportions for this class across clients
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Convert proportions to integer counts
        splits = (proportions * len(idx)).astype(int)
        # Correct rounding errors so all indices are assigned
        splits[-1] = len(idx) - splits[:-1].sum()
        start = 0
        for client_id, count in enumerate(splits):
            client_indices[client_id].extend(idx[start: start + count].tolist())
            start += count

    return client_indices


def create_client_dataloaders(
    dataset: datasets.MNIST,
    client_indices: List[List[int]],
    batch_size: int = 64,
    seed: int = None,
) -> List[DataLoader]:
    """
    Wraps each client's index shard into a PyTorch DataLoader.

    The DataLoaders are the ONLY objects handed to each client.
    Each client's DataLoader encapsulates its private data — the
    server never touches these objects after creation.
    """
    loaders: List[DataLoader] = []
    for client_index, indices in enumerate(client_indices):
        subset = Subset(dataset, indices)
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed + client_index)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            generator=generator,
        )
        loaders.append(loader)
    return loaders


def create_test_dataloader(
    test_dataset: datasets.MNIST,
    batch_size: int = 256,
) -> DataLoader:
    """
    Creates the shared server-side test DataLoader for global model evaluation.

    This is evaluation-only data; it is never used in client training.
    """
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )


def get_client_data_sizes(client_indices: List[List[int]]) -> List[int]:
    """Returns the number of training samples for each client (used by FedAvg)."""
    return [len(indices) for indices in client_indices]


# ─────────────────────────────────────────────────────────────────────────────
# Financial Fraud Dataset Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_financial_dataset(
    data_dir: str = "./data_storage",
    csv_name: str = "creditcard.csv",
    label_column: str = "Class",
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, Dict[str, object]]:
    """Load the public credit-card fraud dataset, inspect metadata, preprocess it, and return train/test arrays."""
    data_path = ensure_financial_dataset(data_dir, csv_name)
    df = pd.read_csv(data_path)
    summary = inspect_financial_dataset(df)
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != label_column]
    train_df, test_df, y_train, y_test, scaler, metadata = preprocess_financial_data(
        df,
        label_column=label_column,
        feature_columns=feature_columns,
        test_size=test_size,
        seed=seed,
    )
    metadata["dataset_source"] = DATASET_SOURCE
    metadata["dataset_summary"] = summary
    return train_df, test_df, y_train, y_test, scaler, metadata


def create_financial_client_dataloaders(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    client_indices: List[List[int]],
    batch_size: int = 64,
    seed: int = 42,
) -> List[DataLoader]:
    """Wrap each client's subset into a typed PyTorch DataLoader for tabular fraud training."""
    loaders: List[DataLoader] = []
    for client_index, indices in enumerate(client_indices):
        local_X = torch.tensor(X_train.iloc[indices].to_numpy(dtype=np.float32), dtype=torch.float32)
        local_y = torch.tensor(y_train.iloc[indices].to_numpy(dtype=np.int64), dtype=torch.long)
        generator = None if seed is None else torch.Generator().manual_seed(seed + client_index)
        loaders.append(
            DataLoader(
                TensorDataset(local_X, local_y),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=generator,
            )
        )
    return loaders


def create_financial_test_dataloader(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    batch_size: int = 256,
) -> DataLoader:
    """Create a batch loader for the server-side fraud evaluation split."""
    test_inputs = torch.tensor(X_test.to_numpy(dtype=np.float32), dtype=torch.float32)
    test_labels = torch.tensor(y_test.to_numpy(dtype=np.int64), dtype=torch.long)
    return DataLoader(TensorDataset(test_inputs, test_labels), batch_size=batch_size, shuffle=False, num_workers=0)
