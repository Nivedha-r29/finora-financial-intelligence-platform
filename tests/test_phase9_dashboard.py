import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dashboard.data_loader import (
    DashboardDataError,
    PROJECT_ROOT,
    load_demo_transaction,
    load_model_checkpoint,
    load_phase8_artifacts,
    load_preprocessing,
    predict_transaction,
    prepare_transaction,
)


class Phase9DashboardTests(unittest.TestCase):
    def test_phase8_artifacts_load_with_actual_results(self):
        artifacts = load_phase8_artifacts(PROJECT_ROOT)
        metrics = artifacts["final_metrics"]
        self.assertIsNotNone(metrics)
        self.assertEqual(int(metrics["status"].eq("success").sum()), 9)
        self.assertEqual(len(artifacts["graphs"]), 15)
        self.assertIsInstance(artifacts["report"], str)
        self.assertIn("284,807 rows", artifacts["report"])

    def test_preprocessing_matches_checkpoint_feature_order(self):
        preprocessing = load_preprocessing(PROJECT_ROOT)
        self.assertEqual(len(preprocessing["feature_columns"]), 30)
        self.assertEqual(preprocessing["metadata"]["scaler_fit_source"], "stratified training split only")
        raw_values = {name: float(index) for index, name in enumerate(preprocessing["feature_columns"])}
        input_tensor = prepare_transaction(raw_values, preprocessing)
        self.assertEqual(tuple(input_tensor.shape), (1, 30))
        self.assertTrue(torch.isfinite(input_tensor).all())
        expected_first = (
            raw_values[preprocessing["feature_columns"][0]] - preprocessing["mean"][0]
        ) / preprocessing["scale"][0]
        self.assertAlmostEqual(float(input_tensor[0, 0]), float(expected_first), places=5)

    def test_missing_phase8_artifacts_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifacts = load_phase8_artifacts(Path(temporary_directory))
            self.assertIn("final_metrics", artifacts["errors"])
            self.assertIsNone(artifacts["final_metrics"])

    def test_missing_preprocessing_artifacts_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(DashboardDataError, "preprocessing_metadata.json"):
                load_preprocessing(Path(temporary_directory))

    def test_real_checkpoint_predicts_without_random_weight_fallback(self):
        preprocessing = load_preprocessing(PROJECT_ROOT)
        checkpoint = PROJECT_ROOT / "results" / "phase8" / "models" / "centralized.pt"
        model, payload = load_model_checkpoint(checkpoint, len(preprocessing["feature_columns"]))
        probability = predict_transaction(
            {name: 0.0 for name in preprocessing["feature_columns"]},
            model,
            preprocessing,
        )
        self.assertEqual(payload["model_class"], "FinancialFraudMLP")
        self.assertTrue(np.isfinite(probability))
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)

    def test_missing_checkpoint_is_reported_instead_of_using_random_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "missing.pt"
            with self.assertRaisesRegex(DashboardDataError, "checkpoint is missing"):
                load_model_checkpoint(missing, feature_count=30)

    def test_real_demo_sample_uses_recorded_feature_order_without_returning_label(self):
        values = load_demo_transaction(PROJECT_ROOT)
        preprocessing = load_preprocessing(PROJECT_ROOT)
        self.assertEqual(list(values), preprocessing["feature_columns"])
        self.assertTrue(np.isfinite(list(values.values())).all())
        self.assertNotIn("Class", values)

    def test_non_finite_manual_input_is_rejected(self):
        preprocessing = {
            "feature_columns": ["Time", "V1"],
            "mean": np.zeros(2),
            "scale": np.ones(2),
        }
        with self.assertRaisesRegex(DashboardDataError, "finite"):
            prepare_transaction({"Time": 0.0, "V1": float("nan")}, preprocessing)


if __name__ == "__main__":
    unittest.main()
