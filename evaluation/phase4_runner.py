"""Reproducible Phase 4 evaluation experiments for the MNIST FL framework."""

import csv
import json
import platform
import random
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from client.client import FederatedClient
from config import FLConfig
from data.dataset import (
    create_client_dataloaders,
    create_test_dataloader,
    get_client_data_sizes,
    load_mnist,
    partition_iid,
    partition_non_iid,
)
from evaluation.evaluator import GlobalEvaluator
from model.mlp import MLP
from server.server import FederatedServer
from utils.logger import FLLogger


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, CUDA and deterministic backend options."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_federated_experiment(
    train_dataset,
    test_dataset,
    client_indices: List[List[int]],
    initial_state: Dict[str, torch.Tensor],
    model_kwargs: dict,
    config: FLConfig,
    rounds: int,
    seed: int,
    device: torch.device,
    logger: FLLogger,
) -> Dict:
    """Run full-participation FedAvg using existing client and server modules."""
    seed_everything(seed, config.deterministic_experiments)
    client_loaders = create_client_dataloaders(
        train_dataset,
        client_indices,
        batch_size=config.batch_size,
        seed=seed + 1,
    )
    test_loader = create_test_dataloader(test_dataset, batch_size=256)
    global_model = MLP(**model_kwargs)
    global_model.load_state_dict(initial_state)
    clients = [
        FederatedClient(
            client_id=client_id,
            dataloader=loader,
            device=device,
            local_epochs=config.local_epochs,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            optimizer_name=config.optimizer_name,
        )
        for client_id, loader in enumerate(client_loaders, start=1)
    ]
    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        test_loader=test_loader,
        device=device,
        num_rounds=rounds,
        logger=logger,
        num_classes=config.num_classes,
    )
    metrics = server.run(MLP, model_kwargs)
    return {
        "metrics": metrics,
        "client_sizes": get_client_data_sizes(client_indices),
        "rounds": rounds,
        "total_duration_seconds": metrics["total_training_duration_seconds"],
    }


def run_centralized_experiment(
    train_dataset,
    test_dataset,
    initial_state: Dict[str, torch.Tensor],
    model_kwargs: dict,
    config: FLConfig,
    epochs: int,
    seed: int,
    device: torch.device,
) -> Dict:
    """Train one model on all training examples and evaluate it once."""
    seed_everything(seed, config.deterministic_experiments)
    generator = torch.Generator().manual_seed(seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    test_loader = create_test_dataloader(test_dataset, batch_size=256)
    model = MLP(**model_kwargs).to(device)
    model.load_state_dict(initial_state)
    if config.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=1e-4,
        )
    criterion = nn.CrossEntropyLoss()

    _sync_device(device)
    started = time.perf_counter()
    model.train()
    for _ in range(epochs):
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    _sync_device(device)

    test_loss, test_accuracy = GlobalEvaluator(
        test_loader, device, config.num_classes
    ).evaluate(model)
    _sync_device(device)
    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "duration_seconds": time.perf_counter() - started,
        "epochs": epochs,
    }


def _write_evaluation_csv(path: Path, iid_result: Dict) -> None:
    metrics = iid_result["metrics"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "experiment", "round", "global_test_loss", "global_test_accuracy",
            "round_duration_seconds", "total_federated_duration_seconds",
            "client_id", "client_training_loss", "client_training_accuracy",
        ])
        for position, round_num in enumerate(metrics["rounds"]):
            duration = "" if round_num == 0 else metrics["round_duration_seconds"][round_num - 1]
            client_ids = [None] if round_num == 0 else list(metrics["client_losses"])
            for client_id in client_ids:
                client_position = -1 if round_num == 0 else round_num - 1
                writer.writerow([
                    "iid", round_num,
                    metrics["global_test_loss"][position],
                    metrics["global_test_accuracy"][position],
                    duration,
                    iid_result["total_duration_seconds"],
                    "" if client_id is None else client_id,
                    "" if client_id is None else metrics["client_losses"][client_id][client_position],
                    "" if client_id is None else metrics["client_accuracies"][client_id][client_position],
                ])


def _write_comparison_csv(path: Path, iid_result: Dict, centralized: Dict) -> float:
    iid_metrics = iid_result["metrics"]
    federated_accuracy = iid_metrics["global_test_accuracy"][-1]
    federated_loss = iid_metrics["global_test_loss"][-1]
    accuracy_difference = centralized["test_accuracy"] - federated_accuracy
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "method", "test_loss", "test_accuracy", "accuracy_difference_centralized_minus_federated_pp",
            "elapsed_seconds", "training_epochs", "rounds", "training_procedure",
        ])
        writer.writerow([
            "federated_iid", federated_loss, federated_accuracy, accuracy_difference,
            iid_result["total_duration_seconds"], "", iid_result["rounds"],
            "10 rounds; one local epoch per client per round with FedAvg",
        ])
        writer.writerow([
            "centralized", centralized["test_loss"], centralized["test_accuracy"], accuracy_difference,
            centralized["duration_seconds"], centralized["epochs"], 0,
            f"One model; {centralized['epochs']} full-dataset epochs with {config.optimizer_name}",
        ])
    return accuracy_difference


def _write_distribution_csv(path: Path, iid_result: Dict, noniid_result: Dict, alpha: float) -> None:
    iid_metrics = iid_result["metrics"]
    noniid_metrics = noniid_result["metrics"]
    compared_rounds = noniid_result["rounds"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "partition", "round", "dirichlet_alpha", "global_test_loss",
            "global_test_accuracy", "round_duration_seconds", "client_sample_counts",
        ])
        for label, result, metrics in (
            ("iid", iid_result, iid_metrics),
            ("non_iid", noniid_result, noniid_metrics),
        ):
            sizes = ";".join(
                f"{client_id}:{size}"
                for client_id, size in enumerate(result["client_sizes"], start=1)
            )
            for position, round_num in enumerate(metrics["rounds"][:compared_rounds + 1]):
                duration = "" if round_num == 0 else metrics["round_duration_seconds"][round_num - 1]
                writer.writerow([
                    label,
                    round_num,
                    "" if label == "iid" else alpha,
                    metrics["global_test_loss"][position],
                    metrics["global_test_accuracy"][position],
                    duration,
                    sizes,
                ])


def _generate_phase4_plots(
    output_dir: Path,
    iid_result: Dict,
    noniid_result: Dict,
    centralized: Dict,
) -> None:
    plt.rcParams.update({"figure.figsize": (9, 5), "font.size": 10})
    iid_metrics = iid_result["metrics"]
    rounds = iid_metrics["rounds"]

    fig, ax = plt.subplots()
    ax.plot(rounds, iid_metrics["global_test_accuracy"], "o-", color="#2a788e", label="IID FedAvg")
    ax.set(title="Global Test Accuracy by Communication Round", xlabel="Communication round", ylabel="Accuracy (%)")
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_round.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(rounds, iid_metrics["global_test_loss"], "s-", color="#d95f02", label="IID FedAvg")
    ax.set(title="Global Test Loss by Communication Round", xlabel="Communication round", ylabel="Cross-entropy loss")
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_vs_round.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    names = ["Federated IID", "Centralized"]
    scores = [iid_metrics["global_test_accuracy"][-1], centralized["test_accuracy"]]
    ax.bar(names, scores, color=["#2a788e", "#e07a5f"])
    ax.set(title="Centralized and Federated Test Accuracy", ylabel="Accuracy (%)", ylim=(0, 100))
    for index, score in enumerate(scores):
        ax.text(index, score + 0.5, f"{score:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "centralized_vs_federated.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, result, color in (
        ("IID", iid_result, "#2a788e"),
        ("Non-IID", noniid_result, "#d95f02"),
    ):
        limit = noniid_result["rounds"] + 1
        ax.plot(
            result["metrics"]["rounds"][:limit],
            result["metrics"]["global_test_accuracy"][:limit],
            "o-", color=color, label=label,
        )
    ax.set(title="IID and Non-IID Accuracy (Matched Two-Round Budget)", xlabel="Communication round", ylabel="Accuracy (%)")
    ax.set_xticks(range(noniid_result["rounds"] + 1))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "iid_vs_noniid.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    client_rounds = rounds[1:]
    for client_id, losses in iid_metrics["client_losses"].items():
        ax.plot(client_rounds, losses, marker="o", label=f"Client {client_id}")
    ax.set(title="Per-Client Local Training Loss", xlabel="Communication round", ylabel="Average local training loss")
    ax.set_xticks(client_rounds)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "client_loss_comparison.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    for label, result, color in (
        ("IID", iid_result, "#2a788e"),
        ("Non-IID", noniid_result, "#d95f02"),
    ):
        duration_rounds = range(1, result["rounds"] + 1)
        ax.plot(duration_rounds, result["metrics"]["round_duration_seconds"], "o-", color=color, label=label)
    ax.set(title="Federated Round Duration", xlabel="Communication round", ylabel="Wall-clock duration (seconds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "round_duration.png", dpi=150)
    plt.close(fig)


def _write_report(
    output_path: Path,
    config: FLConfig,
    iid_result: Dict,
    noniid_result: Dict,
    centralized: Dict,
    accuracy_difference: float,
    device: torch.device,
) -> None:
    iid_metrics = iid_result["metrics"]
    noniid_metrics = noniid_result["metrics"]
    client_summary = []
    for client_id in iid_metrics["client_losses"]:
        client_summary.append(
            f"| {client_id} | {iid_metrics['client_losses'][client_id][0]:.4f} | "
            f"{iid_metrics['client_losses'][client_id][-1]:.4f} | "
            f"{iid_metrics['client_accuracies'][client_id][0]:.2f}% | "
            f"{iid_metrics['client_accuracies'][client_id][-1]:.2f}% |"
        )
    iid_noniid_final_round = noniid_result["rounds"]
    iid_accuracy_at_comparison = iid_metrics["global_test_accuracy"][iid_noniid_final_round]
    noniid_accuracy = noniid_metrics["global_test_accuracy"][-1]
    round_table_rows = []
    for position, (round_num, loss, accuracy) in enumerate(zip(
        iid_metrics["rounds"],
        iid_metrics["global_test_loss"],
        iid_metrics["global_test_accuracy"],
    )):
        duration = "-" if round_num == 0 else f"{iid_metrics['round_duration_seconds'][round_num - 1]:.2f}"
        round_table_rows.append(
            f"| {round_num} | {loss:.4f} | {accuracy:.2f}% | {duration} |"
        )
    round_table = "\n".join(round_table_rows)
    report = f"""# Phase 4: Federated Learning Evaluation

## Setup and Repeatability

- Dataset/model: MNIST / existing MLP; device: `{device}`.
- Primary run: {config.num_clients} IID clients, {config.num_rounds} rounds, {config.local_epochs} local epoch, batch size {config.batch_size}, {config.optimizer_name}, learning rate {config.learning_rate}.
- Partitioning: IID uses a seeded NumPy permutation and equal shards. Non-IID uses the existing per-class Dirichlet partitioner with alpha {config.phase4_noniid_alpha}, seed {config.seed}.
- The non-IID comparison uses {config.phase4_noniid_rounds} rounds and is compared to the IID run at the same round budget to limit runtime.
- Centralized training uses the same initialized MLP state, {config.optimizer_name} optimizer family and learning rate, complete MNIST training set, and {config.phase4_centralized_epochs} full-data epochs. This matches sample-epoch exposure approximately; it is not the same optimization procedure as periodic client training and FedAvg.
- Seeds cover Python, NumPy, PyTorch, CUDA, and per-loader shuffling. Deterministic PyTorch operations are requested with warnings enabled. Reproducibility across hardware/library versions is not guaranteed.
- Test data is reserved for evaluation and is not used for training or hyperparameter selection.

## Federated IID Results

Initial test loss/accuracy: {iid_metrics['global_test_loss'][0]:.4f} / {iid_metrics['global_test_accuracy'][0]:.2f}%.

Final test loss/accuracy: {iid_metrics['global_test_loss'][-1]:.4f} / {iid_metrics['global_test_accuracy'][-1]:.2f}% (accuracy change {iid_metrics['global_test_accuracy'][-1] - iid_metrics['global_test_accuracy'][0]:+.2f} percentage points).

Federated elapsed time across rounds: {iid_result['total_duration_seconds']:.2f} seconds. Round time includes local training, aggregation, and global evaluation; initial round-0 evaluation is excluded.

| Round | Test loss | Test accuracy | Round duration (s) |
|---:|---:|---:|---:|
{round_table}

### Client training metrics

| Client | Round 1 loss | Final-round loss | Round 1 training accuracy | Final-round training accuracy |
|---:|---:|---:|---:|---:|
{chr(10).join(client_summary)}

## Centralized Comparison

| Method | Test loss | Test accuracy | Elapsed seconds |
|---|---:|---:|---:|
| Federated IID | {iid_metrics['global_test_loss'][-1]:.4f} | {iid_metrics['global_test_accuracy'][-1]:.2f}% | {iid_result['total_duration_seconds']:.2f} |
| Centralized ({centralized['epochs']} epochs) | {centralized['test_loss']:.4f} | {centralized['test_accuracy']:.2f}% | {centralized['duration_seconds']:.2f} |

Centralized minus federated accuracy: {accuracy_difference:+.2f} percentage points. These measurements compare these specific runs only; neither training method is inherently more accurate in all settings. Federated elapsed time includes evaluation every round, while centralized elapsed time includes its training and final evaluation.

## IID Versus Non-IID

| Partition | Round | Test loss | Test accuracy |
|---|---:|---:|---:|
| IID | {iid_noniid_final_round} | {iid_metrics['global_test_loss'][iid_noniid_final_round]:.4f} | {iid_accuracy_at_comparison:.2f}% |
| Non-IID (alpha={config.phase4_noniid_alpha}) | {iid_noniid_final_round} | {noniid_metrics['global_test_loss'][-1]:.4f} | {noniid_accuracy:.2f}% |

This is a two-round diagnostic, not a same-budget comparison to the full ten-round headline run. Dirichlet class allocation makes client label distributions heterogeneous; the limited run should not be treated as a convergence estimate.

## Correctness and Privacy

- FedAvg tests cover equal and sample-count-weighted averages, non-floating buffers, empty updates, invalid counts, and incompatible shapes/keys.
- Server smoke tests verify same round-start weights, changed client weights, exact aggregation, next-round distribution, test-only evaluation, and round timing.
- Client data loaders remain attached to client objects; the server receives state dictionaries, sample counts, aggregate training loss/accuracy scalars, and never reads raw batches directly. This is a logical boundary in a single-process simulation, not a process or network isolation boundary.
- Model updates may leak information. FedAvg does not provide differential privacy or secure aggregation, and this implementation does not provide encrypted inter-organization networking.

## Limitations and Readiness

The framework demonstrates the FedAvg workflow on MNIST but is not production-ready for sensitive financial data. Before that use case, it needs a threat model, actual isolated clients and authenticated encrypted transport, privacy/security controls, data governance, robust failure handling, and domain-specific validation. No financial model or processing is included in this phase.
"""
    output_path.write_text(report, encoding="utf-8")


def run_phase4(config: FLConfig = None) -> Dict:
    """Execute the primary IID, centralized, and reduced-budget non-IID studies."""
    config = config or FLConfig()
    output_dir = Path(config.phase4_results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    logger = FLLogger(results_dir=str(output_dir))
    seed_everything(config.seed, config.deterministic_experiments)

    train_dataset, test_dataset = load_mnist(data_dir=config.data_dir)
    model_kwargs = {
        "input_dim": config.input_dim,
        "hidden_dim1": config.hidden_dim1,
        "hidden_dim2": config.hidden_dim2,
        "num_classes": config.num_classes,
        "dropout_p": 0.2,
    }
    initial_model = MLP(**model_kwargs)
    initial_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in initial_model.state_dict().items()
    }

    logger.section("PHASE 4: FULL IID FEDERATED RUN")
    iid_indices = partition_iid(train_dataset, config.num_clients, seed=config.seed)
    iid_result = run_federated_experiment(
        train_dataset, test_dataset, iid_indices, initial_state, model_kwargs,
        config, config.num_rounds, config.seed, device, logger,
    )

    logger.section("PHASE 4: CENTRALIZED COMPARISON")
    centralized = run_centralized_experiment(
        train_dataset, test_dataset, initial_state, model_kwargs, config,
        config.phase4_centralized_epochs, config.seed, device,
    )

    logger.section("PHASE 4: NON-IID DIAGNOSTIC RUN")
    noniid_indices = partition_non_iid(
        train_dataset,
        config.num_clients,
        alpha=config.phase4_noniid_alpha,
        seed=config.seed,
    )
    noniid_result = run_federated_experiment(
        train_dataset, test_dataset, noniid_indices, initial_state, model_kwargs,
        config, config.phase4_noniid_rounds, config.seed, device, logger,
    )

    _write_evaluation_csv(output_dir / "evaluation_metrics.csv", iid_result)
    accuracy_difference = _write_comparison_csv(
        output_dir / "centralized_comparison.csv", iid_result, centralized
    )
    _write_distribution_csv(
        output_dir / "iid_vs_noniid.csv", iid_result, noniid_result,
        config.phase4_noniid_alpha,
    )

    config_record = {
        "dataset": config.dataset_name,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "model": "MLP",
        "model_kwargs": model_kwargs,
        "num_clients": config.num_clients,
        "iid_rounds": config.num_rounds,
        "noniid_rounds": config.phase4_noniid_rounds,
        "noniid_alpha": config.phase4_noniid_alpha,
        "local_epochs": config.local_epochs,
        "centralized_epochs": config.phase4_centralized_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "optimizer": config.optimizer_name,
        "iid_partition_seed": config.seed,
        "noniid_partition_seed": config.seed,
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "deterministic_algorithms_requested": config.deterministic_experiments,
        "random_seeds": {
            "python": config.seed,
            "numpy": config.seed,
            "pytorch": config.seed,
            "dataloader_base": config.seed + 1,
        },
        "notes": [
            "IID is the full ten-round primary experiment.",
            "Non-IID is a two-round compute-limited diagnostic compared at a matched two-round IID budget.",
            f"Centralized training uses the same initialized MLP and {config.phase4_centralized_epochs} full-data epochs; its optimizer updates are not periodically averaged.",
            "No test data is used for training or hyperparameter selection.",
        ],
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )
    _generate_phase4_plots(output_dir, iid_result, noniid_result, centralized)
    _write_report(
        output_dir / "phase4_report.md", config, iid_result, noniid_result,
        centralized, accuracy_difference, device,
    )

    logger.section("PHASE 4 COMPLETE")
    logger.info(
        f"IID final test: loss={iid_result['metrics']['global_test_loss'][-1]:.4f}, "
        f"accuracy={iid_result['metrics']['global_test_accuracy'][-1]:.2f}%"
    )
    logger.info(
        f"Centralized: loss={centralized['test_loss']:.4f}, "
        f"accuracy={centralized['test_accuracy']:.2f}%"
    )
    logger.info(
        f"Non-IID at round {noniid_result['rounds']}: "
        f"accuracy={noniid_result['metrics']['global_test_accuracy'][-1]:.2f}%"
    )
    logger.info(f"Phase 4 artifacts saved under {output_dir}")
    return {
        "iid": iid_result,
        "centralized": centralized,
        "non_iid": noniid_result,
        "centralized_accuracy_difference_pp": accuracy_difference,
        "output_dir": str(output_dir),
    }
