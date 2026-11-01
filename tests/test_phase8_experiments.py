import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from client.client import FederatedClient
from data.financial_dataset import (
    ensure_financial_dataset,
    inspect_financial_dataset,
    partition_financial_clients,
    preprocess_financial_data,
)
from evaluation.phase8_experiments import calculate_fraud_metrics
from model.financial_fraud_mlp import FinancialFraudMLP
from privacy.config import DPConfig
from server.secure_aggregation import SecureAggregationConfig
from server.server import FederatedServer


class Phase8ExperimentsTests(unittest.TestCase):
    def test_missing_dataset_reports_exact_placement_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = Path(temp_dir) / "creditcard.csv"
            with self.assertRaisesRegex(FileNotFoundError, str(expected).replace("\\", "\\\\")):
                ensure_financial_dataset(temp_dir)

    def test_dataset_inspection_uses_named_target_column(self):
        frame = pd.DataFrame({
            "Class": [0, 1, 0],
            "feature": [1.0, 2.0, 3.0],
        })
        summary = inspect_financial_dataset(frame, label_column="Class")
        self.assertEqual(summary["class_distribution"], {"0": 2, "1": 1})
        self.assertEqual(summary["fraud_count"], 1)

    def test_preprocessing_fits_scaler_on_training_rows_only(self):
        frame = pd.DataFrame({
            "feature": [0.0, 1.0, 2.0, 3.0, 4.0, 1000.0, 6.0, 7.0, 8.0, 9.0],
            "other": np.arange(10, dtype=float),
            "Class": [0, 1] * 5,
        })
        train_indices, _ = train_test_split(
            np.arange(len(frame)),
            test_size=0.3,
            stratify=frame["Class"],
            random_state=17,
        )
        x_train, x_test, _, _, scaler, metadata = preprocess_financial_data(
            frame,
            test_size=0.3,
            seed=17,
        )
        expected_train_mean = frame.iloc[train_indices]["feature"].mean()
        self.assertAlmostEqual(scaler.mean_[0], expected_train_mean)
        self.assertNotAlmostEqual(scaler.mean_[0], frame["feature"].mean())
        self.assertEqual(len(x_train), metadata["train_rows"])
        self.assertEqual(len(x_test), metadata["test_rows"])
        self.assertAlmostEqual(float(x_train["feature"].mean()), 0.0, places=10)

    def test_partitioning_is_complete_reproducible_and_stratified(self):
        labels = pd.Series([0] * 900 + [1] * 100)
        iid_first = partition_financial_clients(
            labels, num_clients=5, strategy="iid", seed=29
        )
        iid_second = partition_financial_clients(
            labels, num_clients=5, strategy="iid", seed=29
        )
        self.assertEqual(iid_first, iid_second)
        self.assertEqual(sorted(index for client in iid_first for index in client), list(range(len(labels))))
        self.assertLessEqual(max(map(len, iid_first)) - min(map(len, iid_first)), 1)
        iid_rates = [labels.iloc[indices].mean() for indices in iid_first]
        self.assertLess(max(iid_rates) - min(iid_rates), 0.01)

        non_iid_first = partition_financial_clients(
            labels, num_clients=5, strategy="non_iid", alpha=0.2, seed=29
        )
        non_iid_second = partition_financial_clients(
            labels, num_clients=5, strategy="non_iid", alpha=0.2, seed=29
        )
        self.assertEqual(non_iid_first, non_iid_second)
        self.assertEqual(sorted(index for client in non_iid_first for index in client), list(range(len(labels))))
        self.assertTrue(all(labels.iloc[indices].sum() > 0 for indices in non_iid_first))
        non_iid_rates = [labels.iloc[indices].mean() for indices in non_iid_first]
        self.assertGreater(max(non_iid_rates) - min(non_iid_rates), 0.02)

    def test_partitioning_rejects_undersized_class_for_noniid(self):
        with self.assertRaisesRegex(ValueError, "fewer examples than clients"):
            partition_financial_clients(
                pd.Series([0, 0, 1]),
                num_clients=5,
                strategy="non_iid",
                alpha=0.3,
            )

    def test_fraud_metrics_include_imbalance_metrics_and_rates(self):
        metrics = calculate_fraud_metrics(
            [0, 0, 1, 1],
            [0.1, 0.8, 0.6, 0.2],
            threshold=0.5,
        )
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["f1_score"], 0.5)
        self.assertEqual(metrics["false_positive_rate"], 0.5)
        self.assertEqual(metrics["false_negative_rate"], 0.5)
        self.assertEqual(metrics["threshold"], 0.5)

    def test_fraud_metrics_reject_invalid_scores_and_single_class_labels(self):
        with self.assertRaisesRegex(ValueError, "probabilities"):
            calculate_fraud_metrics([0, 1], [0.1, 1.2])
        with self.assertRaisesRegex(ValueError, "Both binary labels"):
            calculate_fraud_metrics([0, 0], [0.1, 0.2])

    def test_client_local_training_and_all_federated_modes(self):
        torch.manual_seed(101)
        device = torch.device("cpu")
        model_kwargs = {
            "input_dim": 4,
            "hidden_dim1": 8,
            "hidden_dim2": 4,
            "binary_classification": True,
        }
        initial_model = FinancialFraudMLP(**model_kwargs)
        initial_state = {
            key: value.detach().clone() for key, value in initial_model.state_dict().items()
        }
        inputs = torch.randn(24, 4)
        labels = torch.tensor([0, 1] * 12)
        test_loader = DataLoader(
            TensorDataset(inputs[:8], labels[:8]),
            batch_size=4,
        )
        for mode in ("fedavg", "dp", "secure", "secure_dp"):
            with self.subTest(mode=mode):
                clients = []
                for client_id, indices in enumerate((range(0, 8), range(8, 16), range(16, 24)), start=1):
                    loader = DataLoader(
                        TensorDataset(inputs[list(indices)], labels[list(indices)]),
                        batch_size=4,
                        shuffle=False,
                    )
                    clients.append(FederatedClient(
                        client_id=client_id,
                        dataloader=loader,
                        device=device,
                        local_epochs=1,
                        learning_rate=0.001,
                        optimizer_name="Adam",
                        positive_class_weight=2.0,
                        dp_config=DPConfig(
                            enable_dp=mode in {"dp", "secure_dp"},
                            noise_multiplier=1.0,
                            max_grad_norm=1.0,
                            target_delta=1e-5,
                            local_epochs=1,
                            privacy_rounds=1,
                        ),
                    ))
                model = FinancialFraudMLP(**model_kwargs)
                model.load_state_dict(initial_state)
                server = FederatedServer(
                    global_model=model,
                    clients=clients,
                    test_loader=test_loader,
                    device=device,
                    num_rounds=1,
                    num_classes=1,
                    secure_config=SecureAggregationConfig(
                        enabled=mode in {"secure", "secure_dp"},
                        min_clients=2,
                        max_dropout_fraction=0.0,
                        recovery_threshold=2,
                    ),
                )
                history = server.run(FinancialFraudMLP, model_kwargs)
                self.assertEqual(history["rounds"], [0, 1])
                self.assertTrue(torch.isfinite(torch.tensor(history["global_test_loss"])).all())
                if mode in {"secure", "secure_dp"}:
                    self.assertIsNone(clients[0]._secure_participant._private_key)


if __name__ == "__main__":
    unittest.main()
