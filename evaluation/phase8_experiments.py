"""End-to-end Credit Card Fraud Detection experiments for Phase 8."""

from __future__ import annotations

import json
import logging
import random
import time
import traceback
import unittest
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

from client.client import FederatedClient
from data.financial_dataset import (
    DATASET_SOURCE,
    ensure_financial_dataset,
    inspect_financial_dataset,
    partition_financial_clients,
    preprocess_financial_data,
    summarize_client_distributions,
)
from data.dataset import create_financial_client_dataloaders, create_financial_test_dataloader
from model.financial_fraud_mlp import FinancialFraudMLP
from privacy.config import DPConfig
from server.secure_aggregation import SecureAggregationConfig
from server.server import FederatedServer
from utils.logger import FLLogger


DEFAULT_SEED = 42
DEFAULT_ROUNDS = 2
DEFAULT_BATCH_SIZE = 2048
DEFAULT_CLIENTS = 5
DEFAULT_NONIID_ALPHA = 0.3
DEFAULT_TEST_SIZE = 0.2
DEFAULT_DP_NOISE = 1.0
DEFAULT_DP_CLIP = 1.0
DEFAULT_DELTA = 1e-5
FIXED_THRESHOLD = 0.5

METHODS = ("fedavg", "dp_fedavg", "secure_fedavg", "secure_dp_fedavg")
PARTITIONS = ("iid", "non_iid")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def calculate_fraud_metrics(
    labels: list[int] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, Any]:
    """Calculate binary metrics at a fixed threshold without tuning on test data."""
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or y_true.shape != scores.shape:
        raise ValueError("labels and probabilities must be non-empty arrays of equal length.")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("probabilities must be finite values in [0, 1].")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1].")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Both binary labels 0 and 1 are required to calculate fraud metrics.")

    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc_average_precision": float(average_precision_score(y_true, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "threshold": float(threshold),
        "fraud_support": int(tp + fn),
        "legitimate_support": int(tn + fp),
    }


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _model_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    criterion = nn.BCEWithLogitsLoss()
    loss_total = 0.0
    examples = 0
    with torch.no_grad():
        for inputs, batch_labels in data_loader:
            inputs = inputs.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(inputs).view(-1)
            float_labels = batch_labels.float().view(-1)
            loss_total += criterion(logits, float_labels).item() * len(batch_labels)
            examples += len(batch_labels)
            labels.append(batch_labels.cpu().numpy().astype(np.int64))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
        loss_total / max(examples, 1),
    )


def _evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    labels, probabilities, loss = _model_probabilities(model, test_loader, device)
    metrics = calculate_fraud_metrics(labels, probabilities)
    metrics["test_loss"] = float(loss)
    return metrics, labels, probabilities


def _state_size_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())


def _communication_summary(
    state_dict: dict[str, torch.Tensor],
    num_clients: int,
    rounds: int,
    secure: bool,
) -> dict[str, int | float]:
    state_elements = sum(tensor.numel() for tensor in state_dict.values())
    plain_state_bytes = _state_size_bytes(state_dict)
    masked_state_bytes = state_elements * 8
    count_and_upload_metadata = 8 * num_clients * rounds
    public_key_and_share_bytes_per_round = (
        68 * num_clients * (num_clients - 1) + 32 * num_clients
    )
    downlink_bytes = plain_state_bytes * num_clients * rounds
    update_bytes = (
        masked_state_bytes if secure else plain_state_bytes
    ) * num_clients * rounds
    setup_bytes = public_key_and_share_bytes_per_round * rounds if secure else 0
    total_bytes = (
        downlink_bytes
        + update_bytes
        + count_and_upload_metadata
        + setup_bytes
    )
    return {
        "global_model_state_bytes": plain_state_bytes,
        "rounds": rounds,
        "clients": num_clients,
        "estimated_downlink_bytes": downlink_bytes,
        "estimated_update_payload_bytes": update_bytes,
        "estimated_sample_count_metadata_bytes": count_and_upload_metadata,
        "estimated_secure_key_share_setup_bytes": setup_bytes,
        "estimated_total_communication_bytes": total_bytes,
        "estimate_excludes_network_framing_and_tls": True,
        "estimate_excludes_per_client_monitoring_scalars": True,
    }


def _make_model_kwargs(input_dim: int) -> dict[str, int]:
    return {
        "input_dim": input_dim,
        "hidden_dim1": 32,
        "hidden_dim2": 16,
        "binary_classification": True,
    }


def _train_centralized(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_loader: DataLoader,
    initial_state: dict[str, torch.Tensor],
    model_kwargs: dict[str, int],
    device: torch.device,
    positive_class_weight: float,
    batch_size: int,
    epochs: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], np.ndarray, np.ndarray]:
    seed_everything(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    dataset = TensorDataset(
        torch.from_numpy(train_features.copy()),
        torch.from_numpy(train_labels.copy()).long(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = FinancialFraudMLP(**model_kwargs).to(device)
    model.load_state_dict(initial_state)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_class_weight], device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1)
            optimizer.zero_grad()
            loss = criterion(model(inputs).view(-1), labels)
            loss.backward()
            optimizer.step()
    metrics, y_true, probabilities = _evaluate_model(model, test_loader, device)
    metrics["training_duration_seconds"] = float(time.perf_counter() - started)
    metrics["communication_rounds"] = 0
    metrics["epochs"] = epochs
    metrics["communication"] = {
        "estimated_total_communication_bytes": 0,
        "estimated_update_payload_bytes": 0,
    }
    return metrics, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, y_true, probabilities


def _train_federated(
    method: str,
    partition: str,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_loader: DataLoader,
    client_indices: list[list[int]],
    initial_state: dict[str, torch.Tensor],
    model_kwargs: dict[str, int],
    device: torch.device,
    positive_class_weight: float,
    rounds: int,
    batch_size: int,
    seed: int,
    logger: FLLogger,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any], np.ndarray, np.ndarray]:
    use_dp = method in {"dp_fedavg", "secure_dp_fedavg"}
    use_secure = method in {"secure_fedavg", "secure_dp_fedavg"}
    seed_everything(seed)
    loaders = create_financial_client_dataloaders(
        train_features,
        train_labels,
        client_indices,
        batch_size=batch_size,
        seed=seed + 1,
    )
    dp_config = DPConfig(
        enable_dp=use_dp,
        noise_multiplier=DEFAULT_DP_NOISE,
        max_grad_norm=DEFAULT_DP_CLIP,
        target_delta=DEFAULT_DELTA,
        sampling_probability=1.0,
        batch_size=batch_size,
        local_epochs=1,
        privacy_rounds=rounds,
    )
    clients = [
        FederatedClient(
            client_id=client_id,
            dataloader=loader,
            device=device,
            local_epochs=1,
            learning_rate=0.001,
            optimizer_name="Adam",
            dp_config=dp_config,
            positive_class_weight=positive_class_weight,
        )
        for client_id, loader in enumerate(loaders, start=1)
    ]
    global_model = FinancialFraudMLP(**model_kwargs)
    global_model.load_state_dict(initial_state)
    secure_config = SecureAggregationConfig(
        enabled=use_secure,
        quantization_scale=10_000.0,
        modulus=2**61 - 1,
        min_clients=2,
        max_dropout_fraction=0.2,
        allow_recovery=True,
        recovery_threshold=2,
        max_abs_update=1_000_000.0,
    )
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        test_loader=test_loader,
        device=device,
        num_rounds=rounds,
        logger=logger,
        num_classes=1,
        dp_config=dp_config,
        secure_config=secure_config,
    )
    started = time.perf_counter()
    history = server.run(FinancialFraudMLP, model_kwargs)
    duration = time.perf_counter() - started
    final_metrics, y_true, probabilities = _evaluate_model(
        server.global_model, test_loader, device
    )
    final_metrics["training_duration_seconds"] = float(duration)
    final_metrics["communication_rounds"] = rounds
    final_metrics["round_metrics"] = [
        {
            "round": int(round_number),
            "test_loss": float(loss),
            "accuracy": float(accuracy) / 100.0,
            "round_duration_seconds": (
                None
                if round_number == 0
                else float(history["round_duration_seconds"][round_number - 1])
            ),
        }
        for round_number, loss, accuracy in zip(
            history["rounds"],
            history["global_test_loss"],
            history["global_test_accuracy"],
        )
    ]
    final_metrics["communication"] = _communication_summary(
        server.global_model.state_dict(),
        num_clients=len(clients),
        rounds=rounds,
        secure=use_secure,
    )
    privacy = {
        "enabled": use_dp,
        "accountant_type": "existing lightweight RDP-style approximation",
        "formal_guarantee_claimed": False,
        "noise_multiplier": DEFAULT_DP_NOISE if use_dp else None,
        "max_grad_norm": DEFAULT_DP_CLIP if use_dp else None,
        "delta": DEFAULT_DELTA if use_dp else None,
        "per_client": {},
    }
    if use_dp:
        for client in clients:
            accountant = client._trainer.privacy_accountant
            privacy["per_client"][str(client.client_id)] = {
                "epsilon_estimate": float(accountant.epsilon),
                "delta": float(accountant.delta),
                "rdp_accumulator": float(accountant.rdp),
                "steps": int(accountant.total_steps),
            }
        privacy["max_client_epsilon_estimate"] = max(
            item["epsilon_estimate"]
            for item in privacy["per_client"].values()
        )
    return (
        final_metrics,
        {
            key: value.detach().cpu().clone()
            for key, value in server.global_model.state_dict().items()
        },
        privacy,
        y_true,
        probabilities,
    )


def _safe_filename(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_")


def _record_result(
    results: list[dict[str, Any]],
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    row: dict[str, Any],
    y_true: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
) -> None:
    results.append(row)
    if y_true is not None and probabilities is not None:
        curves[row["experiment_id"]] = (y_true, probabilities)


def _successful(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result["status"] == "success"]


def _plot_phase8(
    output_dir: Path,
    results: list[dict[str, Any]],
    client_summaries: dict[str, list[dict[str, Any]]],
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[str]:
    successful = _successful(results)
    if not successful:
        return []
    graph_dir = output_dir / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    def save(fig: plt.Figure, filename: str) -> None:
        fig.tight_layout()
        fig.savefig(graph_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(filename)

    def experiment(partition: str, method: str) -> dict[str, Any] | None:
        return next(
            (
                item for item in successful
                if item["partition"] == partition and item["method"] == method
            ),
            None,
        )

    # 1. Centralized versus IID FedAvg.
    central = next((item for item in successful if item["method"] == "centralized"), None)
    iid_fed = experiment("iid", "fedavg")
    if central and iid_fed:
        metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc_average_precision"]
        fig, ax = plt.subplots(figsize=(11, 5))
        positions = np.arange(len(metrics))
        width = 0.35
        ax.bar(positions - width / 2, [central[m] for m in metrics], width, label="Centralized")
        ax.bar(positions + width / 2, [iid_fed[m] for m in metrics], width, label="IID FedAvg")
        ax.set_xticks(positions, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Centralized vs IID FedAvg (test set)")
        ax.legend()
        save(fig, "01_centralized_vs_fedavg.png")

    # 2-4. FedAvg comparisons by partition.
    for number, method, title in (
        ("02", "dp_fedavg", "FedAvg vs DP-FedAvg"),
        ("03", "secure_fedavg", "FedAvg vs Secure FedAvg"),
        ("04", "secure_dp_fedavg", "FedAvg vs Secure DP-FedAvg"),
    ):
        available = [
            (partition, experiment(partition, "fedavg"), experiment(partition, method))
            for partition in PARTITIONS
        ]
        available = [item for item in available if item[1] and item[2]]
        if available:
            fig, ax = plt.subplots(figsize=(12, 5))
            metrics = ["recall", "precision", "f1_score", "pr_auc_average_precision"]
            positions = np.arange(len(metrics) * len(available))
            width = 0.35
            tick_labels = []
            for partition_index, (partition, fed, other) in enumerate(available):
                group_positions = np.arange(len(metrics)) + partition_index * len(metrics)
                ax.bar(
                    group_positions - width / 2,
                    [fed[m] for m in metrics],
                    width,
                    label="FedAvg" if partition_index == 0 else None,
                )
                ax.bar(
                    group_positions + width / 2,
                    [other[m] for m in metrics],
                    width,
                    label=method.replace("_", " ").title() if partition_index == 0 else None,
                )
                tick_labels.extend(
                    f"{partition}\n{metric_label}"
                    for metric_label in ("Recall", "Precision", "F1", "PR-AUC")
                )
            ax.set_xticks(positions, tick_labels)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Score")
            ax.set_title(f"{title} - {', '.join(item[0] for item in available)}")
            ax.legend()
            save(fig, f"{number}_{_safe_filename(title.lower().replace(' ', '_'))}.png")

    # 5. IID versus non-IID performance.
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in METHODS:
        available = [experiment(partition, method) for partition in PARTITIONS]
        if all(available):
            ax.plot(
                ["IID", "Non-IID"],
                [item["pr_auc_average_precision"] for item in available],
                marker="o",
                label=method.replace("_", " ").title(),
            )
    ax.set_ylabel("PR-AUC (average precision)")
    ax.set_title("IID vs Non-IID Fraud Detection Performance")
    ax.legend()
    save(fig, "05_iid_vs_noniid_performance.png")

    # 6-7. Per-round global performance.
    for number, metric, ylabel, title in (
        ("06", "accuracy", "Accuracy", "Accuracy Across Communication Rounds"),
        ("07", "test_loss", "Binary cross-entropy loss", "Loss Across Communication Rounds"),
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        for item in successful:
            values = [point for point in item.get("round_metrics", []) if point.get(metric) is not None]
            if values:
                ax.plot(
                    [point["round"] for point in values],
                    [point[metric] for point in values],
                    marker="o",
                    label=f"{item['method'].replace('_', ' ').title()} ({item['partition']})",
                )
        ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize="small", ncol=2)
        save(fig, f"{number}_{'accuracy' if metric == 'accuracy' else 'loss'}_across_rounds.png")

    # 8. Positive-class precision, recall, and F1.
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [f"{item['method'].replace('_', ' ').title()}\n{item['partition']}" for item in successful]
    x = np.arange(len(successful))
    width = 0.25
    for offset, metric, label in (
        (-width, "precision", "Precision"),
        (0, "recall", "Recall"),
        (width, "f1_score", "F1"),
    ):
        ax.bar(x + offset, [item[metric] for item in successful], width, label=label)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Fraud-Class Precision, Recall, and F1")
    ax.legend()
    save(fig, "08_precision_recall_f1.png")

    # 9-10. ROC and precision-recall curves from actual test probabilities.
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for item in successful:
        if item["experiment_id"] not in curves:
            continue
        y_true, scores = curves[item["experiment_id"]]
        label = f"{item['method'].replace('_', ' ').title()} ({item['partition']})"
        fpr, tpr, _ = roc_curve(y_true, scores)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ax_roc.plot(fpr, tpr, label=f"{label} (AUC={item['roc_auc']:.3f})")
        ax_pr.plot(recall, precision, label=f"{label} (AP={item['pr_auc_average_precision']:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC Curves")
    ax_roc.legend(fontsize="x-small")
    save(fig_roc, "09_roc_curves.png")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax_pr.legend(fontsize="x-small")
    save(fig_pr, "10_precision_recall_curves.png")

    # 11. Confusion matrices.
    columns = min(4, len(successful))
    rows = int(np.ceil(len(successful) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3.6 * rows), squeeze=False)
    for axis, item in zip(axes.flat, successful):
        matrix = np.asarray(item["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{item['method'].replace('_', ' ').title()}\n{item['partition']}")
        axis.set_xticks([0, 1], ["Legitimate", "Fraud"])
        axis.set_yticks([0, 1], ["Legitimate", "Fraud"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        for (row, column), value in np.ndenumerate(matrix):
            axis.text(column, row, str(value), ha="center", va="center")
        fig.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes.flat[len(successful):]:
        axis.axis("off")
    fig.suptitle("Test-Set Confusion Matrices (threshold 0.5)")
    save(fig, "11_confusion_matrices.png")

    # 12. Client fraud distribution.
    fig, ax = plt.subplots(figsize=(9, 5))
    for partition, summaries in client_summaries.items():
        ax.bar(
            [f"{partition}\nC{item['client_id']}" for item in summaries],
            [100 * item["fraud_rate"] for item in summaries],
            label=partition,
        )
    ax.set_ylabel("Fraud transactions (%)")
    ax.set_title("Fraud Prevalence by Simulated Institution (Training Data)")
    ax.legend()
    save(fig, "12_client_fraud_distribution.png")

    # 13-14. Runtime and communication estimate.
    for number, metric, ylabel, title in (
        ("13", "training_duration_seconds", "Seconds", "Training Time Comparison"),
        ("14", "estimated_total_communication_bytes", "Estimated bytes", "Communication Volume Estimate"),
    ):
        fig, ax = plt.subplots(figsize=(11, 5))
        positions = np.arange(len(successful))
        values = [
            item.get(metric, item.get("communication", {}).get(metric, 0))
            if metric == "training_duration_seconds"
            else item.get("communication", {}).get(metric, 0)
            for item in successful
        ]
        ax.bar(positions, values)
        ax.set_xticks(
            positions,
            [f"{item['method'].replace('_', ' ').title()}\n{item['partition']}" for item in successful],
            rotation=20,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        save(fig, f"{number}_{'training_time' if metric == 'training_duration_seconds' else 'communication_overhead'}.png")

    # 15. Approximate DP accounting, with non-DP methods excluded.
    private_runs = [
        item for item in successful
        if item.get("privacy", {}).get("max_client_epsilon_estimate") is not None
    ]
    if private_runs:
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [f"{item['method'].replace('_', ' ').title()}\n{item['partition']}" for item in private_runs]
        ax.bar(labels, [item["privacy"]["max_client_epsilon_estimate"] for item in private_runs])
        ax.set_ylabel("Max client epsilon estimate (approximate)")
        ax.set_title(f"DP Accountant Output (delta={DEFAULT_DELTA:g}; not a formal guarantee)")
        save(fig, "15_privacy_budget_comparison.png")

    return saved


def _write_tables(output_dir: Path, results: list[dict[str, Any]]) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    round_rows = []
    communication_rows = []
    fedavg_bytes = {
        item["partition"]: item["communication"]["estimated_total_communication_bytes"]
        for item in results
        if item["status"] == "success"
        and item["method"] == "fedavg"
        and item.get("communication")
    }
    for item in results:
        row = {
            key: value
            for key, value in item.items()
            if key not in {"round_metrics", "communication", "privacy", "error"}
        }
        for key in (
            "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "pr_auc_average_precision", "false_positive_rate",
            "false_negative_rate", "test_loss", "training_duration_seconds",
            "training_time_overhead_percent_vs_fedavg",
        ):
            row.setdefault(key, None)
        row["confusion_matrix"] = json.dumps(item.get("confusion_matrix"))
        row["error"] = item.get("error")
        result_rows.append(row)
        for point in item.get("round_metrics", []):
            round_rows.append({
                "experiment_id": item["experiment_id"],
                "method": item["method"],
                "partition": item["partition"],
                **point,
            })
        if item.get("communication"):
            total_bytes = item["communication"]["estimated_total_communication_bytes"]
            baseline_bytes = fedavg_bytes.get(item["partition"], 0)
            communication_rows.append({
                "experiment_id": item["experiment_id"],
                "method": item["method"],
                "partition": item["partition"],
                **item["communication"],
                "overhead_percent_vs_same_partition_fedavg": (
                    100.0 * (total_bytes / baseline_bytes - 1.0)
                    if baseline_bytes
                    else None
                ),
            })
    pd.DataFrame(result_rows).to_csv(metrics_dir / "final_metrics.csv", index=False)
    pd.DataFrame(round_rows).to_csv(metrics_dir / "per_round_metrics.csv", index=False)
    pd.DataFrame(communication_rows).to_csv(
        metrics_dir / "communication_overhead.csv", index=False
    )
    _json_write(metrics_dir / "final_metrics.json", results)


def _write_report(
    report_path: Path,
    dataset_summary: dict[str, Any] | None,
    preprocessing: dict[str, Any] | None,
    client_summaries: dict[str, list[dict[str, Any]]],
    results: list[dict[str, Any]],
    graph_names: list[str],
    config: dict[str, Any],
    test_summary: dict[str, Any] | None,
    load_error: str | None = None,
) -> None:
    lines = [
        "# Phase 8: End-to-End Financial Fraud Detection Experiments",
        "",
        "## Objective",
        "",
        "Compare centralized training with FedAvg, DP-FedAvg, Secure FedAvg, and Secure DP-FedAvg on the real Kaggle Credit Card Fraud Detection data, under IID and non-IID client partitions.",
        "",
    ]
    if load_error:
        lines += ["## Dataset status", "", f"**Experiments not run:** {load_error}", ""]
    elif dataset_summary and preprocessing:
        train_distribution = preprocessing["y_train_distribution"]
        test_distribution = preprocessing["y_test_distribution"]
        lines += [
            "## Dataset description and observed statistics",
            "",
            f"- Source: {DATASET_SOURCE['source']}",
            f"- Shape: {dataset_summary['rows']:,} rows × {len(dataset_summary['columns'])} columns.",
            f"- Target: `Class`; counts: {dataset_summary['class_distribution']}.",
            f"- Class prevalence: {dataset_summary['fraud_count'] / dataset_summary['rows']:.6%} fraud.",
            f"- Missing values: {sum(dataset_summary['missing_values'].values())}; duplicate rows: {dataset_summary['duplicate_rows']:,} (retained; no deduplication).",
            f"- Dtypes: {dataset_summary['dtypes']}.",
            f"- Split: stratified {preprocessing['train_rows']:,} train / {preprocessing['test_rows']:,} test; train classes {train_distribution}; test classes {test_distribution}.",
            "- StandardScaler was fitted only on the training features and applied unchanged to test features.",
            "- The test split was used only for evaluation; no threshold tuning was performed.",
            "",
            "## System architecture and experimental setup",
            "",
            "A compact tabular MLP (30→32→16→1, ReLU, BCE-with-logits) is reused across methods. It has no BatchNorm, so per-example gradients are not coupled across examples. Training uses the existing `FederatedClient`, `FederatedServer`, weighted FedAvg, DP-SGD implementation, and Phase 7 pairwise-mask secure aggregation. The centralized run uses the same model, initial weights, loss weighting, optimizer family, and number of passes over the training data.",
            "",
            f"- Configuration: {json.dumps(config, sort_keys=True)}",
            f"- Positive-class loss weight calculated from training labels only: {config.get('positive_class_weight', 'unavailable')}.",
            "- Client data is partitioned from training rows only. Federated server APIs receive model states or masked states, counts, and per-client loss/accuracy scalars—not transaction feature/label tensors. Clients are nevertheless in one process and hold private loaders reachable through Python objects; this is not process/network isolation.",
            "",
            "## Client distributions",
            "",
            "IID uses reproducibly shuffled, class-stratified splits. Non-IID uses separate per-class Dirichlet allocations (configured alpha) with a 50% equal-share floor per class, so each institution retains at least half the class's equal-share allocation while prevalence remains skewed.",
            "",
            "| Partition | Client | Rows | Fraud | Fraud rate |",
            "|---|---:|---:|---:|---:|",
        ]
        for partition, summaries in client_summaries.items():
            for client in summaries:
                lines.append(
                    f"| {partition} | {client['client_id']} | {client['sample_count']:,} | {client['fraud_count']:,} | {client['fraud_rate']:.4%} |"
                )
        lines += ["", "## Final test-set results", ""]
        lines += [
            "| Method | Distribution | Status | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) | FPR | FNR | Seconds | Time Δ vs FedAvg | Rounds |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in results:
            if item["status"] != "success":
                lines.append(
                    f"| {item['method']} | {item['partition']} | FAILED: {item.get('error', 'unknown error')} | — | — | — | — | — | — | — | — | — | — | — |"
                )
                continue
            lines.append(
                "| {method} | {partition} | success | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1_score:.4f} | {roc_auc:.4f} | {pr_auc_average_precision:.4f} | {false_positive_rate:.4f} | {false_negative_rate:.4f} | {training_duration_seconds:.2f} | {time_delta} | {communication_rounds} |".format(
                    **item,
                    time_delta=(
                        f"{item['training_time_overhead_percent_vs_fedavg']:+.1f}%"
                        if item.get("training_time_overhead_percent_vs_fedavg") is not None
                        else "—"
                    ),
                )
            )
        def find_result(method: str, partition: str) -> dict[str, Any] | None:
            return next(
                (
                    candidate for candidate in results
                    if candidate["status"] == "success"
                    and candidate["method"] == method
                    and candidate["partition"] == partition
                ),
                None,
            )

        iid_fed = find_result("fedavg", "iid")
        iid_secure = find_result("secure_fedavg", "iid")
        iid_dp = find_result("dp_fedavg", "iid")
        iid_secure_dp = find_result("secure_dp_fedavg", "iid")
        noniid_fed = find_result("fedavg", "non_iid")
        noniid_secure = find_result("secure_fedavg", "non_iid")
        noniid_dp = find_result("dp_fedavg", "non_iid")
        noniid_secure_dp = find_result("secure_dp_fedavg", "non_iid")
        max_noniid_client = max(
            client_summaries.get("non_iid", []),
            key=lambda client: client["fraud_rate"],
            default=None,
        )
        central_result = find_result("centralized", "not_applicable")
        observed_lines = []
        if central_result and iid_fed:
            observed_lines.append(
                f"The centralized baseline reached PR-AUC "
                f"{central_result['pr_auc_average_precision']:.4f}, recall "
                f"{central_result['recall']:.4f}, and accuracy "
                f"{central_result['accuracy']:.4f}; IID FedAvg reached PR-AUC "
                f"{iid_fed['pr_auc_average_precision']:.4f}, recall "
                f"{iid_fed['recall']:.4f}, and accuracy {iid_fed['accuracy']:.4f}."
            )
        if iid_fed and iid_secure:
            observed_lines.append(
                f"On IID data, FedAvg reached PR-AUC {iid_fed['pr_auc_average_precision']:.4f} "
                f"and recall {iid_fed['recall']:.4f}; Secure FedAvg reached PR-AUC "
                f"{iid_secure['pr_auc_average_precision']:.4f} and recall {iid_secure['recall']:.4f}. "
                "Their close scores are consistent with the fixed-point masked sum approximating the same weighted aggregate."
            )
        if noniid_fed and noniid_secure:
            observed_lines.append(
                f"On non-IID data, FedAvg reached PR-AUC {noniid_fed['pr_auc_average_precision']:.4f} "
                f"and recall {noniid_fed['recall']:.4f}; Secure FedAvg reached PR-AUC "
                f"{noniid_secure['pr_auc_average_precision']:.4f} and recall {noniid_secure['recall']:.4f}."
            )
        if max_noniid_client:
            observed_lines.append(
                f"The highest-prevalence non-IID client had {max_noniid_client['fraud_count']:,} training frauds "
                f"({max_noniid_client['fraud_rate']:.2%} prevalence); one partition and seed cannot establish "
                "that non-IID generally improves utility."
            )
        for partition, dp_result, secure_dp_result in (
            ("IID", iid_dp, iid_secure_dp),
            ("non-IID", noniid_dp, noniid_secure_dp),
        ):
            if dp_result and secure_dp_result:
                observed_lines.append(
                    f"{partition} DP-FedAvg had precision {dp_result['precision']:.4f}, "
                    f"recall {dp_result['recall']:.4f}, and PR-AUC "
                    f"{dp_result['pr_auc_average_precision']:.4f}; Secure DP-FedAvg had precision "
                    f"{secure_dp_result['precision']:.4f}, recall {secure_dp_result['recall']:.4f}, "
                    f"and PR-AUC {secure_dp_result['pr_auc_average_precision']:.4f} at threshold 0.5. "
                    "The high training-derived class weight plus clipping/noise is a plausible contributor "
                    "to the measured utility change, not evidence of an optimal privacy/utility setting."
                )
        federated_durations = [
            item["training_duration_seconds"]
            for item in results
            if item["status"] == "success"
            and item["method"] in METHODS
        ]
        if federated_durations:
            observed_lines.append(
                f"Federated run times ranged from {min(federated_durations):.2f} to "
                f"{max(federated_durations):.2f} seconds. Runtime differences at this scale "
                "include training, protocol work, and evaluation and are sensitive to timing noise."
            )
        if iid_fed and find_result("secure_fedavg", "iid"):
            plain_bytes = iid_fed["communication"]["estimated_total_communication_bytes"]
            secure_bytes = find_result("secure_fedavg", "iid")["communication"][
                "estimated_total_communication_bytes"
            ]
            observed_lines.append(
                f"For IID runs, estimated total traffic was {plain_bytes:,} bytes for FedAvg and "
                f"{secure_bytes:,} bytes for Secure FedAvg "
                f"(+{100 * (secure_bytes / plain_bytes - 1):.1f}%); the masked tensor upload is "
                "twice the plaintext tensor payload before setup traffic."
            )
        lines += [
            "",
            "Centralized is partition-independent and appears once. PR-AUC is reported as average precision. Confusion matrices use the fixed probability threshold 0.5.",
            "",
            "## Per-round convergence",
            "",
            "The machine-readable `metrics/per_round_metrics.csv` contains the initial round and the actual server-evaluated test loss and accuracy after each federated round. Per-round evaluations are diagnostic only and were not used for training or threshold selection.",
            "",
            "## Privacy accounting and communication",
            "",
            "DP-FedAvg and Secure DP-FedAvg use per-example clipping and Gaussian noise from the existing Phase 6 DP-SGD path. Accountant values are the project's existing lightweight RDP-style estimates, not formally validated privacy guarantees. The accountant's sampling model is approximate for shuffled mini-batches; no formal epsilon claim is made. Secure aggregation hides individual updates from the simulated aggregator under Phase 7's assumptions but is not differential privacy.",
            "",
            "| Method | Distribution | Approx. max client epsilon | Delta | Estimated total bytes | Bytes Δ vs FedAvg |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for item in results:
            if item["status"] != "success":
                continue
            privacy = item.get("privacy", {})
            epsilon = privacy.get("max_client_epsilon_estimate")
            communication = item.get("communication", {})
            communication_baseline = next(
                (
                    candidate["communication"]["estimated_total_communication_bytes"]
                    for candidate in results
                    if candidate.get("status") == "success"
                    and candidate.get("method") == "fedavg"
                    and candidate.get("partition") == item["partition"]
                    and candidate.get("communication")
                ),
                None,
            )
            byte_delta = (
                f"{100 * (communication['estimated_total_communication_bytes'] / communication_baseline - 1):+.1f}%"
                if communication_baseline
                else "—"
            )
            lines.append(
                f"| {item['method']} | {item['partition']} | {epsilon if epsilon is not None else 'not applicable'} | {privacy.get('delta') if privacy.get('delta') is not None else '—'} | {communication.get('estimated_total_communication_bytes', 0):,} | {byte_delta} |"
            )
        lines += [
            "",
            "The companion `metrics/communication_overhead.csv` also reports estimated byte overhead versus same-partition FedAvg.",
        ]
        lines += [
            "",
            "Communication is estimated from tensor sizes and one fixed public-key/share setup estimate per secure round. The estimate excludes TLS, framing, dropout recovery messages, and actual network behavior; clients run sequentially in one process. Centralized has zero inter-client communication.",
            "",
            "## Interpretation of the observed run",
            "",
            *observed_lines,
            "",
            "## Figures",
            "",
        ]
        lines.extend(f"- `../graphs/{name}`" for name in graph_names)
        if not graph_names:
            lines.append("No figures were generated because no experiment produced valid metrics.")
        lines += [
            "",
            "## Analysis and limitations",
            "",
            "Fraud-class recall, precision, F1, and PR-AUC should be prioritized over accuracy because legitimate transactions dominate the test set. DP noise can reduce utility by perturbing clipped local gradients; secure aggregation adds key exchange, masking, modular encoding, and protocol handling. Non-IID fraud prevalence changes local gradient distributions and can make global convergence less stable. These are mechanisms, not claims about the direction or size of differences in these results; use the measured tables and plots.",
            "",
            "The server still observes client participation, sample counts, round timing, per-client training loss/accuracy, and the final aggregate model. Secure aggregation does not prevent leakage from repeated/differenced aggregates, authenticated transport is not implemented, clients are not isolated processes, and malicious-client robustness is absent. Local DP and secure aggregation provide distinct protections. The DP accountant is approximate and no formal privacy guarantee is claimed.",
            "",
            "Only one seed was run because this CPU-only environment and the per-example DP implementation make repeated full-data runs expensive. Each requested method/distribution was attempted independently; every failure is recorded in the results table and logs rather than omitted.",
            "",
            "## Automated tests",
            "",
            (
                f"Full test suite: {test_summary['tests_run']} run, {test_summary['failures']} failures, {test_summary['errors']} errors."
                if test_summary
                else "Full test suite was not executed."
            ),
            "",
            "## Conclusion",
            "",
            "This experiment evaluates the five training approaches using the actual provided Kaggle CSV and a single stratified held-out test split. Interpret all measured scores together with the severe class imbalance, the fixed threshold, single-seed scope, approximate privacy accounting, and in-memory simulation limitations.",
            "",
        ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _run_full_test_suite(log_path: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    suite = unittest.defaultTestLoader.discover(
        str(project_root / "tests"), pattern="test_*.py"
    )
    result_stream = []
    result = unittest.TextTestRunner(
        stream=_ListWriter(result_stream),
        verbosity=2,
    ).run(suite)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(result_stream), encoding="utf-8")
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "successful": result.wasSuccessful(),
        "details_file": str(log_path),
    }


class _ListWriter:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks

    def write(self, text: str) -> None:
        self.chunks.append(text)

    def flush(self) -> None:
        return


def run_phase8(
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    rounds: int = DEFAULT_ROUNDS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    num_clients: int = DEFAULT_CLIENTS,
    noniid_alpha: float = DEFAULT_NONIID_ALPHA,
    run_tests: bool = True,
) -> dict[str, Any]:
    """Run all requested experiments and save metrics, plots, models, and logs."""
    project_root = Path(__file__).resolve().parents[1]
    csv_path = (
        Path(dataset_path)
        if dataset_path is not None
        else project_root / "creditcard.csv"
    )
    destination = (
        Path(output_dir)
        if output_dir is not None
        else project_root / "results" / "phase8"
    )
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    for folder in ("reports", "graphs", "metrics", "models", "logs", "client_statistics"):
        (destination / folder).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("phase8")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(destination / "logs" / "phase8_experiments.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    config: dict[str, Any] = {
        "dataset_path": str(csv_path.resolve()),
        "dataset_source": DATASET_SOURCE["source"],
        "num_clients": num_clients,
        "rounds": rounds,
        "local_epochs": 1,
        "batch_size": batch_size,
        "seed": seed,
        "test_size": DEFAULT_TEST_SIZE,
        "partition_strategies": {
            "iid": "stratified shuffled equal-size split",
            "non_iid": "constrained per-class Dirichlet",
        },
        "noniid_alpha": noniid_alpha,
        "minimum_fraction_per_class": 0.5,
        "decision_threshold": FIXED_THRESHOLD,
        "dp_noise_multiplier": DEFAULT_DP_NOISE,
        "dp_max_grad_norm": DEFAULT_DP_CLIP,
        "dp_delta": DEFAULT_DELTA,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "random_seeds_run": [seed],
        "test_data_used_for_training_or_threshold_tuning": False,
    }
    _json_write(destination / "metrics" / "experiment_config.json", config)
    results: list[dict[str, Any]] = []
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    client_summaries: dict[str, list[dict[str, Any]]] = {}
    dataset_summary = None
    preprocessing = None
    test_summary = None
    load_error = None

    try:
        resolved_csv = ensure_financial_dataset(str(csv_path.parent), csv_path.name)
        frame = pd.read_csv(resolved_csv)
        dataset_summary = inspect_financial_dataset(frame, label_column="Class")
        (
            x_train,
            x_test,
            y_train,
            y_test,
            scaler,
            preprocessing,
        ) = preprocess_financial_data(
            frame,
            label_column="Class",
            test_size=DEFAULT_TEST_SIZE,
            seed=seed,
        )
        preprocessing["scaler_fit_source"] = "stratified training split only"
        preprocessing["dataset_path"] = str(resolved_csv.resolve())
        dataset_summary["source"] = DATASET_SOURCE
        dataset_summary["class_prevalence"] = {
            label: count / dataset_summary["rows"]
            for label, count in dataset_summary["class_distribution"].items()
        }
        _json_write(destination / "metrics" / "dataset_statistics.json", dataset_summary)
        _json_write(destination / "metrics" / "preprocessing_metadata.json", preprocessing)

        model_kwargs = _make_model_kwargs(x_train.shape[1])
        positive_count = int((y_train == 1).sum())
        negative_count = int((y_train == 0).sum())
        if not positive_count or not negative_count:
            raise ValueError("Both fraud and legitimate training examples are required.")
        positive_class_weight = negative_count / positive_count
        config["positive_class_weight"] = positive_class_weight
        config["model"] = {
            "class": "FinancialFraudMLP",
            "architecture": f"{x_train.shape[1]}→32→16→1",
            "activation": "ReLU",
            "normalization": "none (per-example DP gradient independence)",
            "loss": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "positive_class_weight": positive_class_weight,
        }
        _json_write(destination / "metrics" / "experiment_config.json", config)
        test_loader = create_financial_test_dataloader(x_test, y_test, batch_size=8192)
        seed_everything(seed)
        initial_model = FinancialFraudMLP(**model_kwargs)
        initial_state = {
            key: value.detach().cpu().clone()
            for key, value in initial_model.state_dict().items()
        }
        np.savez_compressed(
            destination / "metrics" / "preprocessing_scaler.npz",
            mean=scaler.mean_,
            scale=scaler.scale_,
            feature_names=np.asarray(preprocessing["feature_columns"]),
        )

        train_features = x_train.to_numpy(dtype=np.float32, copy=True)
        train_labels = y_train.to_numpy(dtype=np.int64, copy=True)
        state_bytes = _state_size_bytes(initial_state)
        config["initial_model_state_bytes"] = state_bytes
        _json_write(destination / "metrics" / "experiment_config.json", config)

        logger.info(
            "Loaded %s: rows=%d columns=%d fraud=%d train=%d test=%d device=%s",
            resolved_csv,
            dataset_summary["rows"],
            len(dataset_summary["columns"]),
            dataset_summary["fraud_count"],
            len(y_train),
            len(y_test),
            config["device"],
        )
        central_id = "centralized_all_training"
        try:
            central_metrics, central_state, central_y, central_scores = _train_centralized(
                train_features,
                train_labels,
                test_loader,
                initial_state,
                model_kwargs,
                torch.device(config["device"]),
                positive_class_weight,
                batch_size,
                rounds,
                seed,
            )
            central_metrics.update({
                "experiment_id": central_id,
                "method": "centralized",
                "partition": "not_applicable",
                "status": "success",
                "privacy": {"enabled": False, "formal_guarantee_claimed": False},
            })
            torch.save(
                {
                    "state_dict": central_state,
                    "model_class": "FinancialFraudMLP",
                    "model_kwargs": model_kwargs,
                    "method": "centralized",
                    "partition": "not_applicable",
                    "seed": seed,
                },
                destination / "models" / "centralized.pt",
            )
            _record_result(results, curves, central_metrics, central_y, central_scores)
        except Exception as exc:
            logger.exception("Centralized experiment failed")
            _record_result(results, curves, {
                "experiment_id": central_id,
                "method": "centralized",
                "partition": "not_applicable",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })

        summary_rows = []
        for partition_index, partition in enumerate(PARTITIONS):
            indices = partition_financial_clients(
                y_train,
                num_clients=num_clients,
                strategy=partition,
                alpha=noniid_alpha,
                seed=seed + partition_index,
                minimum_fraction_per_class=0.5,
            )
            summaries = summarize_client_distributions(
                pd.DataFrame({"Class": y_train.to_numpy()}),
                indices,
                "Class",
            )
            client_summaries[partition] = summaries
            summary_rows.extend(
                {"partition": partition, **item} for item in summaries
            )
            if sum(item["sample_count"] for item in summaries) != len(y_train):
                raise AssertionError(f"{partition} partition did not assign every training row.")
            if len(set(index for client in indices for index in client)) != len(y_train):
                raise AssertionError(f"{partition} partition contains duplicate training indices.")

            for method_index, method in enumerate(METHODS):
                experiment_id = f"{method}_{partition}"
                row: dict[str, Any] = {
                    "experiment_id": experiment_id,
                    "method": method,
                    "partition": partition,
                    "status": "failed",
                }
                try:
                    experiment_seed = seed + partition_index * 100 + method_index
                    (
                        metrics,
                        final_state,
                        privacy,
                        y_true,
                        probabilities,
                    ) = _train_federated(
                        method,
                        partition,
                        x_train,
                        y_train,
                        test_loader,
                        indices,
                        initial_state,
                        model_kwargs,
                        torch.device(config["device"]),
                        positive_class_weight,
                        rounds,
                        batch_size,
                        experiment_seed,
                        FLLogger(results_dir=str(destination / "logs")),
                    )
                    row.update(metrics)
                    row.update({
                        "experiment_id": experiment_id,
                        "method": method,
                        "partition": partition,
                        "status": "success",
                        "privacy": privacy,
                    })
                    torch.save(
                        {
                            "state_dict": final_state,
                            "model_class": "FinancialFraudMLP",
                            "model_kwargs": model_kwargs,
                            "method": method,
                            "partition": partition,
                            "seed": experiment_seed,
                            "rounds": rounds,
                        },
                        destination / "models" / f"{experiment_id}.pt",
                    )
                    _record_result(results, curves, row, y_true, probabilities)
                    logger.info(
                        "%s succeeded: PR-AUC=%.6f recall=%.6f duration=%.2fs",
                        experiment_id,
                        row["pr_auc_average_precision"],
                        row["recall"],
                        row["training_duration_seconds"],
                    )
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    logger.exception("%s failed", experiment_id)
                    _record_result(results, curves, row)

        pd.DataFrame(summary_rows).to_csv(
            destination / "client_statistics" / "client_fraud_distribution.csv",
            index=False,
        )
        _json_write(
            destination / "client_statistics" / "client_fraud_distribution.json",
            client_summaries,
        )

        privacy_rows = {
            item["experiment_id"]: item.get("privacy", {})
            for item in results
            if item.get("privacy", {}).get("enabled")
        }
        _json_write(destination / "metrics" / "privacy_accounting.json", privacy_rows)

        prediction_rows = []
        for experiment_id, (labels, probabilities) in curves.items():
            prediction_rows.extend(
                {
                    "experiment_id": experiment_id,
                    "test_row_index": int(index),
                    "label": int(label),
                    "fraud_probability": float(probability),
                    "prediction_at_0_5": int(probability >= FIXED_THRESHOLD),
                }
                for index, (label, probability) in enumerate(zip(labels, probabilities))
            )
        pd.DataFrame(prediction_rows).to_csv(
            destination / "metrics" / "test_predictions.csv", index=False
        )

    except Exception as exc:
        load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Phase 8 data preparation or run failed")
        for method in ("centralized", *METHODS):
            partition_values = ("not_applicable",) if method == "centralized" else PARTITIONS
            for partition in partition_values:
                experiment_id = (
                    "centralized_all_training"
                    if method == "centralized"
                    else f"{method}_{partition}"
                )
                if not any(item["experiment_id"] == experiment_id for item in results):
                    _record_result(results, curves, {
                        "experiment_id": experiment_id,
                        "method": method,
                        "partition": partition,
                        "status": "failed",
                        "error": load_error,
                    })

    fedavg_durations = {
        item["partition"]: item["training_duration_seconds"]
        for item in results
        if item["status"] == "success"
        and item["method"] == "fedavg"
        and item.get("training_duration_seconds") is not None
    }
    for item in results:
        baseline_duration = fedavg_durations.get(item["partition"])
        duration = item.get("training_duration_seconds")
        if item["status"] != "success" or duration is None:
            item["training_time_overhead_percent_vs_fedavg"] = None
        elif item["method"] == "fedavg":
            item["training_time_overhead_percent_vs_fedavg"] = 0.0
        elif baseline_duration:
            item["training_time_overhead_percent_vs_fedavg"] = (
                100.0 * (duration / baseline_duration - 1.0)
            )
        else:
            item["training_time_overhead_percent_vs_fedavg"] = None
    _write_tables(destination, results)
    _json_write(destination / "metrics" / "experiment_results.json", results)
    _json_write(destination / "metrics" / "dataset_statistics.json", dataset_summary or {})
    graph_names = _plot_phase8(destination, results, client_summaries, curves)
    if run_tests:
        test_summary = _run_full_test_suite(destination / "logs" / "full_test_suite.txt")
    _write_report(
        destination / "reports" / "phase8_final_report.md",
        dataset_summary,
        preprocessing,
        client_summaries,
        results,
        graph_names,
        config,
        test_summary,
        load_error,
    )
    logger.info(
        "Phase 8 finished: %d successful and %d failed experiments",
        sum(item["status"] == "success" for item in results),
        sum(item["status"] != "success" for item in results),
    )
    return {
        "output_dir": str(destination.resolve()),
        "dataset_path": str(csv_path.resolve()),
        "experiments": results,
        "graphs": graph_names,
        "test_summary": test_summary,
        "load_error": load_error,
    }
