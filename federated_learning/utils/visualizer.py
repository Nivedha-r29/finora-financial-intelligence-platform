# utils/visualizer.py
# Saves training metric plots to the results/ directory.
#
# Deliberately kept simple — two separate PNG files:
#   1. Training loss vs epoch
#   2. Training accuracy vs epoch

import os
import matplotlib
matplotlib.use("Agg")       # non-interactive backend; safe on all OS
import matplotlib.pyplot as plt
from typing import List


def plot_training_loss(
    losses: List[float],
    save_path: str,
) -> None:
    """
    Plot training loss per epoch and save as PNG.

    Args:
        losses    : List of average loss values, one per epoch.
        save_path : Full path (including filename) to save the PNG.
    """
    epochs = list(range(1, len(losses) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, marker="o", color="#4C72B0", linewidth=2, markersize=8,
            label="Training Loss")
    ax.set_title("Training Loss vs Epoch", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def plot_training_accuracy(
    accuracies: List[float],
    save_path: str,
) -> None:
    """
    Plot training accuracy per epoch and save as PNG.

    Args:
        accuracies : List of accuracy values (%), one per epoch.
        save_path  : Full path (including filename) to save the PNG.
    """
    epochs = list(range(1, len(accuracies) + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, accuracies, marker="s", color="#55A868", linewidth=2, markersize=8,
            label="Training Accuracy")
    ax.set_title("Training Accuracy vs Epoch", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xticks(epochs)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {save_path}")
