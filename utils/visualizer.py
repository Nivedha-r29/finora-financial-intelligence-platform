"""
utils/visualizer.py — Training Metrics Visualisation
======================================================

Generates and saves publication-quality plots for:
  1. Global Test Accuracy vs. Federated Rounds
  2. Global Test Loss vs. Federated Rounds
  3. Per-Client Training Loss over Rounds (stacked line chart)
  4. Combined summary dashboard (2×2 grid)
  5. Initial vs. Final model comparison bar chart

All figures are saved to the results directory as PNG files and also
as a combined PDF for academic submission.
"""

import os
import json
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── Styling ──────────────────────────────────────────────────────────────────

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
]

def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#0f3460",
        "axes.labelcolor":  "#e2e2e2",
        "axes.titlecolor":  "#e2e2e2",
        "xtick.color":      "#b0b0b0",
        "ytick.color":      "#b0b0b0",
        "grid.color":       "#0f3460",
        "grid.alpha":       0.6,
        "grid.linewidth":   0.5,
        "text.color":       "#e2e2e2",
        "legend.facecolor": "#0f3460",
        "legend.edgecolor": "#4C72B0",
        "legend.labelcolor":"#e2e2e2",
        "font.family":      "DejaVu Sans",
        "font.size":        10,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
    })


def plot_global_accuracy(
    rounds: List[int],
    accuracies: List[float],
    save_dir: str,
) -> str:
    """Plot global test accuracy over communication rounds."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, accuracies, marker="o", color=PALETTE[0],
            linewidth=2.2, markersize=7, label="Global Test Accuracy")
    ax.fill_between(rounds, accuracies, alpha=0.15, color=PALETTE[0])
    ax.set_title("Global Test Accuracy vs. Federated Rounds")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(rounds)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    ax.grid(True, linestyle="--")
    ax.legend()
    path = os.path.join(save_dir, "global_accuracy_vs_round.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_global_loss(
    rounds: List[int],
    losses: List[float],
    save_dir: str,
) -> str:
    """Plot global test loss over communication rounds."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rounds, losses, marker="s", color=PALETTE[1],
            linewidth=2.2, markersize=7, label="Global Test Loss")
    ax.fill_between(rounds, losses, alpha=0.15, color=PALETTE[1])
    ax.set_title("Global Test Loss vs. Federated Rounds")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--")
    ax.legend()
    path = os.path.join(save_dir, "global_loss_vs_round.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_client_losses(
    rounds: List[int],
    client_losses: Dict[int, List[float]],
    save_dir: str,
) -> str:
    """Plot per-client training loss over communication rounds."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    client_rounds = rounds[1:] if rounds and rounds[0] == 0 else rounds
    for i, (client_id, losses) in enumerate(client_losses.items()):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(client_rounds, losses, marker="D", color=color,
                linewidth=1.8, markersize=5, label=f"Client {client_id}")
    ax.set_title("Per-Client Training Loss over Federated Rounds")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Average Training Loss")
    ax.set_xticks(rounds)
    ax.grid(True, linestyle="--")
    ax.legend(fontsize=9, ncol=2)
    path = os.path.join(save_dir, "client_losses.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_initial_vs_final(
    initial_accuracy: float,
    final_accuracy: float,
    initial_loss: float,
    final_loss: float,
    save_dir: str,
) -> str:
    """Bar chart comparing initial (random) vs. final (trained) global model."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    labels = ["Initial Model\n(Untrained)", "Federated Model\n(After Training)"]
    accs   = [initial_accuracy, final_accuracy]
    losses = [initial_loss, final_loss]

    bars1 = ax1.bar(labels, accs, color=[PALETTE[7], PALETTE[2]], width=0.45,
                    edgecolor="#0f3460", linewidth=1.2)
    ax1.set_title("Accuracy Comparison")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 110)
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 1.5, f"{val:.2f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    bars2 = ax2.bar(labels, losses, color=[PALETTE[7], PALETTE[3]], width=0.45,
                    edgecolor="#0f3460", linewidth=1.2)
    ax2.set_title("Loss Comparison")
    ax2.set_ylabel("Cross-Entropy Loss")
    for bar, val in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.suptitle("Initial vs. Federated-Trained Global Model", fontsize=14, fontweight="bold")
    path = os.path.join(save_dir, "initial_vs_final.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dashboard(
    rounds: List[int],
    accuracies: List[float],
    losses: List[float],
    client_losses: Dict[int, List[float]],
    client_accuracies: Dict[int, List[float]],
    save_dir: str,
) -> str:
    """2×2 summary dashboard combining all key metrics."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Federated Learning — MNIST Training Dashboard",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # [0,0] Global Accuracy
    ax = axes[0, 0]
    ax.plot(rounds, accuracies, "o-", color=PALETTE[0], lw=2.2, ms=7)
    ax.fill_between(rounds, accuracies, alpha=0.15, color=PALETTE[0])
    ax.set_title("Global Test Accuracy")
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(rounds); ax.grid(True, linestyle="--")

    # [0,1] Global Loss
    ax = axes[0, 1]
    ax.plot(rounds, losses, "s-", color=PALETTE[1], lw=2.2, ms=7)
    ax.fill_between(rounds, losses, alpha=0.15, color=PALETTE[1])
    ax.set_title("Global Test Loss")
    ax.set_xlabel("Round"); ax.set_ylabel("Loss")
    ax.set_xticks(rounds); ax.grid(True, linestyle="--")

    # [1,0] Client Training Losses
    ax = axes[1, 0]
    for i, (cid, closs) in enumerate(client_losses.items()):
        client_rounds = rounds[1:] if len(closs) == len(rounds) - 1 and rounds and rounds[0] == 0 else rounds
        ax.plot(client_rounds, closs, "D-", color=PALETTE[i % len(PALETTE)],
                lw=1.8, ms=5, label=f"Client {cid}")
    ax.set_title("Per-Client Training Loss")
    ax.set_xlabel("Round"); ax.set_ylabel("Training Loss")
    ax.set_xticks(rounds); ax.grid(True, linestyle="--")
    ax.legend(fontsize=8, ncol=2)

    # [1,1] Client Training Accuracies
    ax = axes[1, 1]
    for i, (cid, cacc) in enumerate(client_accuracies.items()):
        client_rounds = rounds[1:] if len(cacc) == len(rounds) - 1 and rounds and rounds[0] == 0 else rounds
        ax.plot(client_rounds, cacc, "^-", color=PALETTE[i % len(PALETTE)],
                lw=1.8, ms=5, label=f"Client {cid}")
    ax.set_title("Per-Client Local Training Accuracy")
    ax.set_xlabel("Round"); ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(rounds); ax.grid(True, linestyle="--")
    ax.legend(fontsize=8, ncol=2)

    path = os.path.join(save_dir, "dashboard.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_all_plots(metrics: Dict, save_dir: str, initial_loss: float, initial_acc: float) -> List[str]:
    """
    Generate and save all visualisation plots from a metrics dict.

    Args:
        metrics     : Output of FederatedServer.run()
        save_dir    : Directory to save PNG files
        initial_loss: Baseline loss before any training
        initial_acc : Baseline accuracy before any training

    Returns:
        List of saved file paths.
    """
    os.makedirs(save_dir, exist_ok=True)
    rounds = metrics["rounds"]
    saved = []

    saved.append(plot_global_accuracy(rounds, metrics["global_test_accuracy"], save_dir))
    saved.append(plot_global_loss(rounds, metrics["global_test_loss"], save_dir))
    saved.append(plot_client_losses(rounds, metrics["client_losses"], save_dir))
    saved.append(plot_initial_vs_final(
        initial_acc,
        metrics["global_test_accuracy"][-1],
        initial_loss,
        metrics["global_test_loss"][-1],
        save_dir,
    ))
    saved.append(plot_dashboard(
        rounds,
        metrics["global_test_accuracy"],
        metrics["global_test_loss"],
        metrics["client_losses"],
        metrics["client_accuracies"],
        save_dir,
    ))
    return saved
