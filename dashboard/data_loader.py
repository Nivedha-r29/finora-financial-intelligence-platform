"""Validated readers and inference helpers for the Phase 9 dashboard."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from model.financial_fraud_mlp import FinancialFraudMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHASE8_ROOT = Path("results") / "phase8"
FINAL_METRIC_COLUMNS = {
    "experiment_id",
    "method",
    "partition",
    "status",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc_average_precision",
    "false_positive_rate",
    "false_negative_rate",
    "training_duration_seconds",
    "communication_rounds",
    "confusion_matrix",
}
ROUND_METRIC_COLUMNS = {
    "experiment_id",
    "method",
    "partition",
    "round",
    "test_loss",
    "accuracy",
}
PREDICTION_COLUMNS = {
    "experiment_id",
    "test_row_index",
    "label",
    "fraud_probability",
    "prediction_at_0_5",
}


class DashboardDataError(RuntimeError):
    """Raised when a dashboard artifact is absent or invalid."""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise DashboardDataError(f"Required dashboard data is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DashboardDataError(f"Could not read JSON data at {path}: {exc}") from exc


def _read_csv(path: Path, required_columns: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise DashboardDataError(f"Required dashboard data is missing: {path}")
    try:
        frame = pd.read_csv(path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise DashboardDataError(f"Could not read CSV data at {path}: {exc}") from exc
    missing = required_columns - set(frame.columns)
    if missing:
        raise DashboardDataError(
            f"CSV data at {path} is missing required columns: {sorted(missing)}"
        )
    return frame


def load_phase8_artifacts(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load available historical results without opening the raw transaction CSV."""
    project_root = Path(project_root)
    phase8 = project_root / PHASE8_ROOT
    artifacts: dict[str, Any] = {
        "root": project_root,
        "phase8_root": phase8,
        "errors": {},
        "graphs": [],
    }

    json_files = {
        "dataset_statistics": phase8 / "metrics" / "dataset_statistics.json",
        "experiment_config": phase8 / "metrics" / "experiment_config.json",
        "experiment_results": phase8 / "metrics" / "experiment_results.json",
        "privacy_accounting": phase8 / "metrics" / "privacy_accounting.json",
        "client_statistics": phase8 / "client_statistics" / "client_fraud_distribution.json",
    }
    for key, path in json_files.items():
        try:
            value = _read_json(path)
            if key == "experiment_results" and not isinstance(value, list):
                raise DashboardDataError(f"Experiment results must be a JSON list: {path}")
            if key != "experiment_results" and not isinstance(value, dict):
                raise DashboardDataError(f"{key} must be a JSON object: {path}")
            artifacts[key] = value
        except DashboardDataError as exc:
            artifacts[key] = None
            artifacts["errors"][key] = str(exc)

    csv_files = {
        "final_metrics": (
            phase8 / "metrics" / "final_metrics.csv",
            FINAL_METRIC_COLUMNS,
        ),
        "per_round_metrics": (
            phase8 / "metrics" / "per_round_metrics.csv",
            ROUND_METRIC_COLUMNS,
        ),
        "communication_overhead": (
            phase8 / "metrics" / "communication_overhead.csv",
            {"experiment_id", "estimated_total_communication_bytes"},
        ),
        "client_distribution": (
            phase8 / "client_statistics" / "client_fraud_distribution.csv",
            {"partition", "client_id", "sample_count", "fraud_count", "fraud_rate"},
        ),
    }
    for key, (path, columns) in csv_files.items():
        try:
            frame = _read_csv(path, columns)
            if frame.empty:
                raise DashboardDataError(f"CSV data is empty: {path}")
            numeric_columns = {
                "final_metrics": [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "roc_auc",
                    "pr_auc_average_precision",
                    "false_positive_rate",
                    "false_negative_rate",
                    "training_duration_seconds",
                    "communication_rounds",
                ],
                "per_round_metrics": ["round", "test_loss", "accuracy"],
                "communication_overhead": ["estimated_total_communication_bytes"],
                "client_distribution": ["client_id", "sample_count", "fraud_count", "fraud_rate"],
            }[key]
            for column in numeric_columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            valid_rows = (
                frame["status"].eq("success")
                if key == "final_metrics"
                else pd.Series(True, index=frame.index)
            )
            if frame.loc[valid_rows, numeric_columns].isna().any().any():
                raise DashboardDataError(f"Required numeric values are missing or invalid in {path}")
            if not np.isfinite(frame.loc[valid_rows, numeric_columns].to_numpy(dtype=np.float64)).all():
                raise DashboardDataError(f"Required numeric values are non-finite in {path}")
            artifacts[key] = frame
        except DashboardDataError as exc:
            artifacts[key] = None
            artifacts["errors"][key] = str(exc)

    report_path = phase8 / "reports" / "phase8_final_report.md"
    try:
        if not report_path.is_file():
            raise DashboardDataError(f"Historical Phase 8 report is missing: {report_path}")
        artifacts["report_path"] = report_path
        artifacts["report"] = report_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, DashboardDataError) as exc:
        artifacts["report"] = None
        artifacts["report_path"] = report_path
        artifacts["errors"]["report"] = str(exc)

    graphs_dir = phase8 / "graphs"
    if graphs_dir.is_dir():
        artifacts["graphs"] = sorted(graphs_dir.glob("*.png"))
    else:
        artifacts["errors"]["graphs"] = f"Historical graph directory is missing: {graphs_dir}"

    return artifacts


def load_predictions(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    path = Path(project_root) / PHASE8_ROOT / "metrics" / "test_predictions.csv"
    frame = _read_csv(path, PREDICTION_COLUMNS)
    for column in ("label", "fraud_probability", "prediction_at_0_5"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["label", "fraud_probability", "prediction_at_0_5"]].isna().any().any():
        raise DashboardDataError(f"Prediction data contains invalid numeric values: {path}")
    if not frame["fraud_probability"].between(0, 1).all():
        raise DashboardDataError(f"Prediction probabilities must be in [0, 1]: {path}")
    if not frame["label"].isin([0, 1]).all() or not frame["prediction_at_0_5"].isin([0, 1]).all():
        raise DashboardDataError(f"Prediction labels must be binary: {path}")
    return frame


def load_preprocessing(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Load and cross-check feature ordering and train-fitted StandardScaler values."""
    project_root = Path(project_root)
    phase8 = project_root / PHASE8_ROOT
    metadata_path = phase8 / "metrics" / "preprocessing_metadata.json"
    scaler_path = phase8 / "metrics" / "preprocessing_scaler.npz"
    metadata = _read_json(metadata_path)
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise DashboardDataError(f"Preprocessing metadata has no valid feature order: {metadata_path}")
    if metadata.get("scaler_fit_source") != "stratified training split only":
        raise DashboardDataError(
            f"Scaler training provenance is missing or unexpected: {metadata_path}"
        )
    if not scaler_path.is_file():
        raise DashboardDataError(f"Required preprocessing scaler is missing: {scaler_path}")
    try:
        with np.load(scaler_path, allow_pickle=False) as archive:
            mean = np.asarray(archive["mean"], dtype=np.float64)
            scale = np.asarray(archive["scale"], dtype=np.float64)
            stored_features = archive["feature_names"].astype(str).tolist()
    except (OSError, ValueError, KeyError) as exc:
        raise DashboardDataError(f"Could not load preprocessing scaler at {scaler_path}: {exc}") from exc
    if stored_features != feature_columns:
        raise DashboardDataError(
            "Scaler feature ordering does not match preprocessing metadata."
        )
    if mean.shape != (len(feature_columns),) or scale.shape != mean.shape:
        raise DashboardDataError("Scaler vectors do not match the recorded feature count.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise DashboardDataError("Scaler contains invalid means or scales.")
    config = _read_json(phase8 / "metrics" / "experiment_config.json")
    threshold = config.get("decision_threshold")
    if not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise DashboardDataError("Experiment configuration has no valid decision threshold.")
    return {
        "feature_columns": feature_columns,
        "mean": mean,
        "scale": scale,
        "metadata": metadata,
        "threshold": float(threshold),
    }


def load_model_checkpoint(
    checkpoint_path: Path,
    feature_count: int,
) -> tuple[FinancialFraudMLP, dict[str, Any]]:
    """Load a verified state-dict checkpoint; never fall back to random weights."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise DashboardDataError(f"Model checkpoint is missing: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as exc:
        raise DashboardDataError(
            f"Could not safely load model checkpoint at {checkpoint_path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or payload.get("model_class") != "FinancialFraudMLP":
        raise DashboardDataError(f"Unsupported or malformed model checkpoint: {checkpoint_path}")
    model_kwargs = payload.get("model_kwargs")
    state_dict = payload.get("state_dict")
    if not isinstance(model_kwargs, dict) or not isinstance(state_dict, dict):
        raise DashboardDataError(f"Checkpoint is missing model metadata or weights: {checkpoint_path}")
    if int(model_kwargs.get("input_dim", -1)) != feature_count:
        raise DashboardDataError(
            f"Checkpoint input size does not match the recorded preprocessing features: {checkpoint_path}"
        )
    try:
        model = FinancialFraudMLP(**model_kwargs)
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise DashboardDataError(
            f"Checkpoint architecture or weights are incompatible: {checkpoint_path}: {exc}"
        ) from exc
    model.eval()
    return model, payload


def prepare_transaction(
    values: dict[str, float],
    preprocessing: dict[str, Any],
) -> torch.Tensor:
    """Order raw feature values and apply the saved train-only scaler."""
    columns = preprocessing["feature_columns"]
    missing = [column for column in columns if column not in values]
    if missing:
        raise DashboardDataError(f"Transaction is missing model features: {missing}")
    try:
        raw = np.asarray([float(values[column]) for column in columns], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DashboardDataError("Transaction feature values must all be numeric.") from exc
    if not np.isfinite(raw).all():
        raise DashboardDataError("Transaction feature values must be finite.")
    normalized = (raw - preprocessing["mean"]) / preprocessing["scale"]
    if not np.isfinite(normalized).all():
        raise DashboardDataError("Scaled transaction features are not finite.")
    return torch.as_tensor(normalized, dtype=torch.float32).reshape(1, -1)


def predict_transaction(
    values: dict[str, float],
    model: FinancialFraudMLP,
    preprocessing: dict[str, Any],
) -> float:
    """Return a real checkpoint inference probability for one transaction."""
    features = prepare_transaction(values, preprocessing)
    model.eval()
    with torch.inference_mode():
        probability = float(torch.sigmoid(model(features).reshape(-1)[0]).item())
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise DashboardDataError("Loaded checkpoint produced an invalid probability.")
    return probability


def load_demo_transaction(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, float]:
    """Read one reproducible held-out record on demand without retaining the dataset."""
    project_root = Path(project_root)
    preprocessing = load_preprocessing(project_root)
    config = _read_json(project_root / PHASE8_ROOT / "metrics" / "experiment_config.json")
    dataset_path = Path(config.get("dataset_path", ""))
    if not dataset_path.is_absolute():
        dataset_path = project_root / dataset_path
    if not dataset_path.is_file():
        raise DashboardDataError(
            f"Source dataset for the real demonstration sample is missing: {dataset_path}"
        )
    target_column = preprocessing["metadata"].get("label_column")
    if not isinstance(target_column, str) or not target_column:
        raise DashboardDataError("Preprocessing metadata has no target column.")
    try:
        labels = pd.read_csv(dataset_path, usecols=[target_column])[target_column]
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DashboardDataError(f"Could not read labels for the demo sample: {exc}") from exc
    if labels.isna().any():
        raise DashboardDataError("Source dataset labels contain missing values.")
    if not labels.isin([0, 1]).all():
        raise DashboardDataError("Source dataset labels are not binary.")
    indices = np.arange(len(labels))
    try:
        _, test_indices = train_test_split(
            indices,
            test_size=float(config["test_size"]),
            stratify=labels.to_numpy(),
            random_state=int(config["seed"]),
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise DashboardDataError(f"Could not reproduce the held-out demo split: {exc}") from exc
    selected_test_position = int(np.random.default_rng(int(config["seed"])).integers(len(test_indices)))
    source_row_index = int(test_indices[selected_test_position])

    try:
        with dataset_path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.reader(source)
            header = next(reader)
            missing_features = set(preprocessing["feature_columns"]) - set(header)
            if missing_features:
                raise DashboardDataError(
                    f"Source dataset is missing model features: {sorted(missing_features)}"
                )
            feature_positions = {
                name: header.index(name) for name in preprocessing["feature_columns"]
            }
            row: list[str] | None = None
            for index, candidate in enumerate(reader):
                if index == source_row_index:
                    row = candidate
                    break
    except (OSError, StopIteration, csv.Error) as exc:
        raise DashboardDataError(f"Could not read the selected real demo record: {exc}") from exc
    if row is None:
        raise DashboardDataError("Selected held-out demo record is not present in the source dataset.")
    try:
        values = {name: float(row[position]) for name, position in feature_positions.items()}
    except (IndexError, ValueError) as exc:
        raise DashboardDataError("Selected held-out demo record has malformed feature values.") from exc
    if not np.isfinite(list(values.values())).all():
        raise DashboardDataError("Selected held-out demo record contains non-finite features.")
    return values
