"""
main.py — MNIST MLP Single-Client Training Pipeline
=====================================================

Stage 1 of the Federated Learning project.
At this stage, NO Federated Learning has been implemented.

This script runs the complete single-client baseline:
    MNIST Dataset → MLP Model → Local Training → Evaluation → Graphs → Save

Once this works correctly, we will add:
    - Multiple clients with private data partitions
    - A Federated Server
    - FedAvg aggregation
    - Multiple communication rounds
"""

import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.mlp import MLP
from data.dataset import load_mnist
from client.trainer import train
from evaluation.evaluator import evaluate
from utils.visualizer import plot_training_loss, plot_training_accuracy

# ── Configuration ─────────────────────────────────────────────────────────────
BATCH_SIZE   = 64
LEARNING_RATE = 0.001
EPOCHS        = 3
DATA_DIR      = "./data_storage"
RESULTS_DIR   = "./results"
MODEL_SAVE    = "./results/mnist_mlp_single_client.pth"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────

def main():

    # ── 1. Header ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 40)
    print("  MNIST MLP TRAINING")
    print("=" * 40)

    # ── 2. Detect Device ──────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\nDevice : CUDA")
        print(f"GPU    : {gpu_name}")
    else:
        device = torch.device("cpu")
        print(f"\nDevice : CPU")

    # ── 3. Load MNIST ─────────────────────────────────────────────────────────
    print("\nLoading MNIST...")
    train_loader, test_loader, n_train, n_test = load_mnist(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )
    print(f"  Training samples : {n_train:,}")
    print(f"  Testing samples  : {n_test:,}")
    print(f"  Batch size       : {BATCH_SIZE}")

    # ── 4. Create MLP Model ───────────────────────────────────────────────────
    print("\nCreating MLP model...")
    model = MLP()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture     : 784 -> 128 -> ReLU -> 64 -> ReLU -> 10")
    print(f"  Total parameters : {total_params:,}")
    print(f"  Model on device  : {device}")

    # ── 5. Loss Function & Optimizer ──────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"\n  Loss function    : CrossEntropyLoss")
    print(f"  Optimizer        : Adam  (lr={LEARNING_RATE})")
    print(f"  Epochs           : {EPOCHS}")

    # ── 6. Train ─────────────────────────────────────────────────────────────
    print(f"\n--- Training ---")
    epoch_losses, epoch_accuracies = train(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=EPOCHS,
    )

    # ── 7. Evaluate ──────────────────────────────────────────────────────────
    print(f"\n--- Evaluating on Test Set ---")
    test_loss, test_accuracy = evaluate(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
    )
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_accuracy:.2f}%")

    # ── 8. Save Graphs ────────────────────────────────────────────────────────
    print(f"\n--- Saving Graphs ---")
    plot_training_loss(
        losses=epoch_losses,
        save_path=os.path.join(RESULTS_DIR, "training_loss.png"),
    )
    plot_training_accuracy(
        accuracies=epoch_accuracies,
        save_path=os.path.join(RESULTS_DIR, "training_accuracy.png"),
    )

    # ── 9. Save Model ────────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_SAVE)
    print(f"\n  Model saved: {MODEL_SAVE}")

    # ── 10. Summary ───────────────────────────────────────────────────────────
    print(f"\n--- Training Summary ---")
    print(f"  Final Training Loss     : {epoch_losses[-1]:.4f}")
    print(f"  Final Training Accuracy : {epoch_accuracies[-1]:.2f}%")
    print(f"  Final Test Loss         : {test_loss:.4f}")
    print(f"  Final Test Accuracy     : {test_accuracy:.2f}%")

    print("\n" + "=" * 40)
    print("  STEP 1 COMPLETED")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
