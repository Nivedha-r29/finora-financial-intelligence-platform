import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATASET_SOURCE = {
    "name": "Credit Card Fraud Detection",
    "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    "license": "Kaggle terms of use / dataset-specific restrictions apply; verify before redistribution.",
    "description": "European cardholder transactions with anonymized numerical features and a binary Class label.",
}


def ensure_financial_dataset(data_dir: str, csv_name: str = "creditcard.csv") -> Path:
    """Download or locate the public credit-card fraud dataset.

    The project intentionally does not silently fabricate data. If the dataset is absent,
    the function raises a clear FileNotFoundError with guidance to add the CSV file.
    """
    data_path = Path(data_dir) / csv_name
    if data_path.exists():
        return data_path

    raise FileNotFoundError(
        "Credit Card Fraud Detection dataset not found. Download the CSV from Kaggle "
        f"({DATASET_SOURCE['source']}) and save it as '{data_path}'."
    )


def inspect_financial_dataset(
    dataframe: pd.DataFrame,
    label_column: str = "Class",
) -> Dict[str, object]:
    """Return dataset metadata and a reproducible summary without leaking raw data."""
    if label_column not in dataframe.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset columns: {list(dataframe.columns)}")
    labels = dataframe[label_column]
    summary = {
        "rows": int(len(dataframe)),
        "columns": list(dataframe.columns),
        "dtypes": {str(col): str(dtype) for col, dtype in dataframe.dtypes.items()},
        "missing_values": {str(col): int(value) for col, value in dataframe.isna().sum().items() if value > 0},
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "class_distribution": {
            str(k): int(v) for k, v in labels.value_counts().sort_index().to_dict().items()
        },
        "fraud_count": int((labels == 1).sum()),
        "legitimate_count": int((labels == 0).sum()),
    }
    return summary


def preprocess_financial_data(
    dataframe: pd.DataFrame,
    label_column: str = "Class",
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, Dict[str, object]]:
    """Return scaled training/test features, labels, and preprocessing metadata."""
    if label_column not in dataframe.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset columns: {list(dataframe.columns)}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if feature_columns is None:
        feature_columns = [col for col in dataframe.columns if col != label_column]
    if not feature_columns or label_column in feature_columns:
        raise ValueError("feature_columns must be non-empty and must not include the target column.")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature_columns must not contain duplicates.")
    missing_columns = set(feature_columns) - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Feature columns not found in dataset: {sorted(missing_columns)}")
    non_numeric = [column for column in feature_columns if not pd.api.types.is_numeric_dtype(dataframe[column])]
    if non_numeric:
        raise ValueError(f"Selected features must be numeric: {non_numeric}")

    missing = dataframe[feature_columns].isna().sum()
    if missing.any():
        raise ValueError(f"Missing values detected in selected features: {missing[missing > 0].to_dict()}")

    data = dataframe[feature_columns].copy()
    labels = dataframe[label_column].copy()
    if labels.isna().any() or not pd.api.types.is_numeric_dtype(labels):
        raise ValueError(f"Label column '{label_column}' must contain non-missing numeric binary labels.")
    if not set(labels.unique()).issubset({0, 1}) or labels.nunique() != 2:
        raise ValueError(f"Label column '{label_column}' must contain both binary labels 0 and 1.")
    labels = labels.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    metadata = {
        "label_column": label_column,
        "feature_columns": feature_columns,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "y_train_distribution": {str(k): int(v) for k, v in y_train.value_counts().sort_index().to_dict().items()},
        "y_test_distribution": {str(k): int(v) for k, v in y_test.value_counts().sort_index().to_dict().items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    return (
        pd.DataFrame(X_train_scaled, columns=feature_columns, index=X_train.index),
        pd.DataFrame(X_test_scaled, columns=feature_columns, index=X_test.index),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        scaler,
        metadata,
    )


def partition_financial_clients(
    labels: pd.Series,
    num_clients: int = 5,
    strategy: str = "iid",
    alpha: float = 0.5,
    seed: int = 42,
    minimum_fraction_per_class: float = 0.5,
) -> List[List[int]]:
    """Partition training indices; non-IID uses constrained per-class Dirichlet shares."""
    if num_clients < 1:
        raise ValueError("num_clients must be positive.")
    if not 0 < minimum_fraction_per_class <= 1:
        raise ValueError("minimum_fraction_per_class must be in (0, 1].")
    if strategy not in {"iid", "non_iid"}:
        raise ValueError(f"Unsupported financial partition strategy: {strategy!r}")

    rng = np.random.default_rng(seed)
    values = np.asarray(labels)
    if values.ndim != 1 or len(values) == 0 or pd.isna(values).any():
        raise ValueError("labels must be a non-empty one-dimensional sequence without missing values.")
    class_values = np.unique(values)
    if len(class_values) < 2:
        raise ValueError("Financial client partitioning requires at least two label classes.")

    if strategy == "iid":
        client_indices = [[] for _ in range(num_clients)]
        for class_value in class_values:
            class_indices = np.flatnonzero(values == class_value)
            rng.shuffle(class_indices)
            for client_id, shard in enumerate(np.array_split(class_indices, num_clients)):
                client_indices[client_id].extend(shard.tolist())
        for indices in client_indices:
            rng.shuffle(indices)
        return client_indices

    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Dirichlet alpha must be finite and positive.")
    client_indices = [[] for _ in range(num_clients)]
    for class_value in class_values:
        class_indices = np.flatnonzero(values == class_value)
        rng.shuffle(class_indices)
        if len(class_indices) < num_clients:
            raise ValueError(
                f"Class {class_value!r} has fewer examples than clients; "
                "non-IID partitioning requires at least one per client."
            )
        minimum_count = max(
            1,
            int(len(class_indices) * minimum_fraction_per_class / num_clients),
        )
        remaining_count = len(class_indices) - minimum_count * num_clients
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        exact_counts = proportions * remaining_count
        counts = np.floor(exact_counts).astype(int)
        remainder = remaining_count - int(counts.sum())
        if remainder:
            order = np.argsort(exact_counts - counts)[::-1]
            counts[order[:remainder]] += 1
        counts += minimum_count

        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(class_indices[start:start + count].tolist())
            start += count
    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def summarize_client_distributions(
    train_df: pd.DataFrame,
    client_indices: List[List[int]],
    label_column: str,
) -> List[Dict[str, object]]:
    """Describe each client by sample count and fraud prevalence."""
    summaries = []
    for client_id, indices in enumerate(client_indices, start=1):
        local_labels = train_df.iloc[indices][label_column]
        summaries.append({
            "client_id": client_id,
            "sample_count": int(len(indices)),
            "fraud_count": int((local_labels == 1).sum()),
            "legitimate_count": int((local_labels == 0).sum()),
            "fraud_rate": float((local_labels == 1).mean()) if len(local_labels) else 0.0,
        })
    return summaries


def save_preprocessing_metadata(path: str, metadata: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
