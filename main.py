"""
main.py — Federated Learning Framework Entry Point (Phase 1: MNIST)
=====================================================================

Run with:
    python main.py

Configuration can be changed in config.py or by passing arguments to FLConfig.

Privacy Summary
───────────────
    Client → Server  ❌ RAW DATA (images, labels)
    Client → Server  ✅ MODEL PARAMETERS (weights / biases tensors only)
    Server → Client  ✅ GLOBAL MODEL PARAMETERS
    Server → Client  ❌ OTHER CLIENT DATA
"""

import json
import csv
import torch
import numpy as np

from config import FLConfig
from model.mlp import MLP
from data.dataset import (
    load_mnist,
    partition_iid,
    partition_non_iid,
    create_client_dataloaders,
    create_test_dataloader,
    get_client_data_sizes,
)
from client.client import FederatedClient
from server.server import FederatedServer
from evaluation.evaluator import GlobalEvaluator
from privacy.config import DPConfig
from server.secure_aggregation import SecureAggregationConfig, SecureAggregationProtocol
from utils.logger import FLLogger
from utils.visualizer import generate_all_plots


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # ── 1. Load Configuration ─────────────────────────────────────────────────
    cfg = FLConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ── 2. Initialise Logger ──────────────────────────────────────────────────
    logger = FLLogger(results_dir=cfg.results_dir)

    logger.section("FEDERATED LEARNING FRAMEWORK")
    logger.info(f"  Dataset           : {cfg.dataset_name}")
    logger.info(f"  Model             : MLP  ({cfg.input_dim}→{cfg.hidden_dim1}→{cfg.hidden_dim2}→{cfg.num_classes})")
    logger.info(f"  Clients           : {cfg.num_clients}")
    logger.info(f"  Partition strategy: {cfg.partition_strategy.upper()}")
    logger.info(f"  Communication rounds: {cfg.num_rounds}")
    logger.info(f"  Local epochs      : {cfg.local_epochs}")
    logger.info(f"  Batch size        : {cfg.batch_size}")
    logger.info(f"  Learning rate     : {cfg.learning_rate}")
    logger.info(f"  Optimizer         : {cfg.optimizer_name}")
    logger.info(f"  Device            : {device}")
    logger.info(f"  Aggregation       : {cfg.aggregation_mode.upper()}")
    if cfg.enable_dp:
        logger.info(f"  DP mode           : enabled (noise={cfg.dp_noise_multiplier}, max_grad_norm={cfg.dp_max_grad_norm})")
    else:
        logger.info("  DP mode           : disabled")
    if cfg.secure_aggregation:
        logger.info(
            "  Secure aggregation: enabled "
            f"(modulus={cfg.secure_modulus}, quantization={cfg.secure_quantization_scale}, "
            f"min_clients={cfg.secure_min_clients})"
        )
    logger.info("")
    logger.info("  PRIVACY SCOPE:")
    logger.info("    Client training samples stay in client-side training components.")
    logger.info("    FedAvg alone does not prevent information leakage from model updates.")

    # ── 3. Load MNIST Dataset ─────────────────────────────────────────────────
    logger.section("LOADING MNIST DATASET")
    train_dataset, test_dataset = load_mnist(data_dir=cfg.data_dir)
    logger.info(f"  Training samples  : {len(train_dataset):,}")
    logger.info(f"  Test samples      : {len(test_dataset):,}")

    # ── 4. Partition Data Among Clients ──────────────────────────────────────
    logger.section("PARTITIONING DATA AMONG CLIENTS")
    if cfg.partition_strategy == "non_iid":
        client_indices = partition_non_iid(
            train_dataset, cfg.num_clients, alpha=cfg.dirichlet_alpha, seed=cfg.seed
        )
        logger.info(f"  Strategy: Non-IID (Dirichlet α={cfg.dirichlet_alpha})")
    else:
        client_indices = partition_iid(train_dataset, cfg.num_clients, seed=cfg.seed)
        logger.info(f"  Strategy: IID (uniform random split)")

    data_sizes = get_client_data_sizes(client_indices)
    for i, size in enumerate(data_sizes):
        logger.info(f"  Client {i}  →  {size:,} local training samples")
    logger.info(f"  Total training samples assigned: {sum(data_sizes):,}")

    # ── 5. Create DataLoaders ─────────────────────────────────────────────────
    client_loaders = create_client_dataloaders(train_dataset, client_indices, batch_size=cfg.batch_size)
    test_loader    = create_test_dataloader(test_dataset, batch_size=256)

    # ── 6. Initialise Global Model ─────────────────────────────────────────────
    model_kwargs = {
        "input_dim":   cfg.input_dim,
        "hidden_dim1": cfg.hidden_dim1,
        "hidden_dim2": cfg.hidden_dim2,
        "num_classes": cfg.num_classes,
        "dropout_p":   0.2,
    }
    global_model = MLP(**model_kwargs)
    logger.section("GLOBAL MODEL INITIALISED")
    total_params = sum(p.numel() for p in global_model.parameters())
    logger.info(f"  Total parameters  : {total_params:,}")
    logger.info(f"  Architecture      : {cfg.input_dim}→{cfg.hidden_dim1}→{cfg.hidden_dim2}→{cfg.num_classes}")

    # ── 7. Record Baseline Metrics ────────────────────────────────────────────
    evaluator = GlobalEvaluator(test_loader, device, num_classes=cfg.num_classes)
    init_loss, init_acc = evaluator.evaluate(global_model)
    logger.info(f"  Baseline loss     : {init_loss:.4f}")
    logger.info(f"  Baseline accuracy : {init_acc:.2f}%  (expect ~10% for random MLP)")

    # ── 8. Create Federated Clients ───────────────────────────────────────────
    logger.section("CREATING FEDERATED CLIENTS")
    clients = []
    dp_cfg = DPConfig(
        enable_dp=cfg.enable_dp,
        noise_multiplier=cfg.dp_noise_multiplier,
        max_grad_norm=cfg.dp_max_grad_norm,
        target_delta=cfg.dp_target_delta,
        sampling_probability=cfg.dp_sampling_probability,
        local_epochs=cfg.local_epochs,
        privacy_rounds=cfg.num_rounds,
    )
    for i, loader in enumerate(client_loaders, start=1):
        client = FederatedClient(
            client_id=i,
            dataloader=loader,
            device=device,
            local_epochs=cfg.local_epochs,
            learning_rate=cfg.learning_rate,
            momentum=cfg.momentum,
            optimizer_name=cfg.optimizer_name,
            dp_config=dp_cfg,
        )
        clients.append(client)
        logger.client_event(i, f"Created  | {client}")

    # ── 9. Create Federated Server & Run Training ──────────────────────────────
    secure_cfg = SecureAggregationConfig(
        enabled=cfg.secure_aggregation,
        quantization_scale=cfg.secure_quantization_scale,
        modulus=cfg.secure_modulus,
        min_clients=cfg.secure_min_clients,
        max_dropout_fraction=cfg.secure_max_dropout_fraction,
        allow_recovery=cfg.secure_allow_recovery,
        recovery_threshold=cfg.secure_recovery_threshold,
        max_abs_update=cfg.secure_max_abs_update,
    )
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        test_loader=test_loader,
        device=device,
        num_rounds=cfg.num_rounds,
        logger=logger,
        num_classes=cfg.num_classes,
        dp_config=dp_cfg,
        secure_config=secure_cfg,
    )

    metrics = server.run(model_class=MLP, model_kwargs=model_kwargs)
    init_loss = metrics["global_test_loss"][0]
    init_acc = metrics["global_test_accuracy"][0]
    final_loss = metrics["global_test_loss"][-1]
    final_acc = metrics["global_test_accuracy"][-1]

    # ── 10. Save Final Global Model ───────────────────────────────────────────
    logger.section("SAVING FINAL GLOBAL MODEL")
    server.save_model(cfg.model_save_path)

    # ── 11. Per-Class Accuracy on Final Model ─────────────────────────────────
    logger.section("PER-CLASS ACCURACY (FINAL GLOBAL MODEL)")
    per_class = evaluator.per_class_accuracy(server.global_model)
    for digit, acc in per_class.items():
        bar = "█" * int(acc / 5)
        logger.info(f"  Digit {digit}  {bar:<20}  {acc:.2f}%")

    # ── 12. Save Metrics JSON ─────────────────────────────────────────────────
    metrics_to_save = {
        "config": {
            "dataset": cfg.dataset_name,
            "num_clients": cfg.num_clients,
            "num_rounds": cfg.num_rounds,
            "local_epochs": cfg.local_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "optimizer": cfg.optimizer_name,
            "partition_strategy": cfg.partition_strategy,
            "device": str(device),
        },
        "baseline": {
            "loss": init_loss,
            "accuracy": init_acc,
        },
        "final": {
            "loss": final_loss,
            "accuracy": final_acc,
        },
        "per_round": {
            "rounds": metrics["rounds"],
            "global_test_loss": metrics["global_test_loss"],
            "global_test_accuracy": metrics["global_test_accuracy"],
        },
        "per_client_per_round": {
            "losses": {str(k): v for k, v in metrics["client_losses"].items()},
            "accuracies": {str(k): v for k, v in metrics["client_accuracies"].items()},
        },
        "per_class_accuracy": {str(k): v for k, v in per_class.items()},
    }
    logger.save_json(metrics_to_save, cfg.metrics_save_path)

    with open(cfg.global_metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["round", "global_test_loss", "global_test_accuracy"])
        writer.writerows(zip(metrics["rounds"], metrics["global_test_loss"], metrics["global_test_accuracy"]))

    with open(cfg.client_metrics_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["round", "client_id", "training_loss", "training_accuracy"])
        for client_id in metrics["client_losses"]:
            for round_num, loss, accuracy in zip(
                range(1, cfg.num_rounds + 1),
                metrics["client_losses"][client_id],
                metrics["client_accuracies"][client_id],
            ):
                writer.writerow([round_num, client_id, loss, accuracy])

    # ── 13. Generate Visualisation Plots ──────────────────────────────────────
    logger.section("GENERATING VISUALISATION PLOTS")
    saved_plots = generate_all_plots(
        metrics=metrics,
        save_dir=cfg.results_dir,
        initial_loss=init_loss,
        initial_acc=init_acc,
    )
    for p in saved_plots:
        logger.info(f"  Plot saved → {p}")

    # ── 14. Final Summary ─────────────────────────────────────────────────────
    logger.section("FINAL SUMMARY")
    logger.info(f"  Dataset           : {cfg.dataset_name}")
    logger.info(f"  Clients           : {cfg.num_clients}")
    logger.info(f"  Rounds            : {cfg.num_rounds}")
    logger.info(f"  Aggregation       : {cfg.aggregation_mode.upper()}")
    logger.info(f"")
    logger.info(f"  Baseline Accuracy : {init_acc:.2f}%")
    logger.info(f"  Final Accuracy    : {final_acc:.2f}%")
    logger.info(f"  Accuracy Change   : {final_acc - init_acc:+.2f} percentage points")
    logger.info(f"")
    logger.info(f"  Baseline Loss     : {init_loss:.4f}")
    logger.info(f"  Final Loss        : {final_loss:.4f}")
    logger.info(f"")
    logger.info(f"  Model saved       → {cfg.model_save_path}")
    logger.info(f"  Metrics saved     → {cfg.metrics_save_path}")
    logger.info(f"  Global CSV        → {cfg.global_metrics_csv_path}")
    logger.info(f"  Client CSV        → {cfg.client_metrics_csv_path}")
    logger.info(f"  Plots saved       → {cfg.results_dir}/")
    logger.info(f"")
    logger.info("  PRIVACY REMINDER:")
    logger.info("    Raw samples remain in client-side loaders.")
    if cfg.secure_aggregation:
        logger.info("    The server aggregates masked updates; it receives sample counts and per-client monitoring scalars.")
    else:
        logger.info("    The server receives individual model states and sample counts.")
    logger.info("    Secure aggregation does not provide differential privacy or authenticated network transport.")
    logger.info("    Clients are simulated in one process, so this validates the algorithm, not distributed security.")
    logger.info("")
    logger.info("Federated Learning Phase 7 workflow complete.")


if __name__ == "__main__":
    main()
