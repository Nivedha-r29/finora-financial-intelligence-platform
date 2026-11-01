# data/dataset.py
# MNIST dataset loader & FL dataset partitioning.

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from typing import Tuple, List
import numpy as np


def get_transforms() -> transforms.Compose:
    """
    Standard MNIST preprocessing pipeline.
    - ToTensor  : converts PIL image (0-255) to float tensor (0.0-1.0)
    - Normalize : mean=0.1307, std=0.3081 are the official MNIST dataset statistics
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def load_mnist(
    data_dir: str = "./data_storage",
    batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Download (or load from cache) MNIST and return train/test DataLoaders.

    Returns:
        train_loader : DataLoader for 60,000 training samples.
        test_loader  : DataLoader for 10,000 test samples.
        train_size   : Number of training samples (60,000).
        test_size    : Number of test samples (10,000).
    """
    transform = get_transforms()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def partition_mnist_iid(
    num_clients: int = 5,
    data_dir: str = "./data_storage",
    seed: int = 42,
) -> Tuple[List[Subset], datasets.MNIST]:
    """
    Partition the MNIST training dataset into IID subsets among N clients.

    Args:
        num_clients : Number of client partitions (default 5).
        data_dir    : Data storage directory.
        seed        : Random seed for reproducible splitting.

    Returns:
        client_subsets : List of torch.utils.data.Subset objects, one per client.
        test_dataset   : The full un-partitioned MNIST test dataset.
    """
    transform = get_transforms()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    total_samples = len(train_dataset)
    base_samples = total_samples // num_clients
    remainder = total_samples % num_clients

    # Determine exact lengths so sum(lengths) == total_samples
    lengths = [base_samples + (1 if i < remainder else 0) for i in range(num_clients)]

    generator = torch.Generator().manual_seed(seed)
    client_subsets = random_split(train_dataset, lengths, generator=generator)

    return list(client_subsets), test_dataset
