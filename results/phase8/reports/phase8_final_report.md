# Phase 8: End-to-End Financial Fraud Detection Experiments

## Objective

Compare centralized training with FedAvg, DP-FedAvg, Secure FedAvg, and Secure DP-FedAvg on the real Kaggle Credit Card Fraud Detection data, under IID and non-IID client partitions.

## Dataset description and observed statistics

- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Shape: 284,807 rows × 31 columns.
- Target: `Class`; counts: {'0': 284315, '1': 492}.
- Class prevalence: 0.172749% fraud.
- Missing values: 0; duplicate rows: 1,081 (retained; no deduplication).
- Dtypes: {'Time': 'float64', 'V1': 'float64', 'V2': 'float64', 'V3': 'float64', 'V4': 'float64', 'V5': 'float64', 'V6': 'float64', 'V7': 'float64', 'V8': 'float64', 'V9': 'float64', 'V10': 'float64', 'V11': 'float64', 'V12': 'float64', 'V13': 'float64', 'V14': 'float64', 'V15': 'float64', 'V16': 'float64', 'V17': 'float64', 'V18': 'float64', 'V19': 'float64', 'V20': 'float64', 'V21': 'float64', 'V22': 'float64', 'V23': 'float64', 'V24': 'float64', 'V25': 'float64', 'V26': 'float64', 'V27': 'float64', 'V28': 'float64', 'Amount': 'float64', 'Class': 'int64'}.
- Split: stratified 227,845 train / 56,962 test; train classes {'0': 227451, '1': 394}; test classes {'0': 56864, '1': 98}.
- StandardScaler was fitted only on the training features and applied unchanged to test features.
- The test split was used only for evaluation; no threshold tuning was performed.

## System architecture and experimental setup

A compact tabular MLP (30→32→16→1, ReLU, BCE-with-logits) is reused across methods. It has no BatchNorm, so per-example gradients are not coupled across examples. Training uses the existing `FederatedClient`, `FederatedServer`, weighted FedAvg, DP-SGD implementation, and Phase 7 pairwise-mask secure aggregation. The centralized run uses the same model, initial weights, loss weighting, optimizer family, and number of passes over the training data.

- Configuration: {"batch_size": 2048, "dataset_path": "S:\\Flora\\creditcard.csv", "dataset_source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", "decision_threshold": 0.5, "device": "cpu", "dp_delta": 1e-05, "dp_max_grad_norm": 1.0, "dp_noise_multiplier": 1.0, "initial_model_state_bytes": 6148, "local_epochs": 1, "minimum_fraction_per_class": 0.5, "model": {"activation": "ReLU", "architecture": "30\u219232\u219216\u21921", "class": "FinancialFraudMLP", "learning_rate": 0.001, "loss": "BCEWithLogitsLoss", "normalization": "none (per-example DP gradient independence)", "optimizer": "Adam", "positive_class_weight": 577.2868020304569}, "noniid_alpha": 0.3, "num_clients": 5, "partition_strategies": {"iid": "stratified shuffled equal-size split", "non_iid": "constrained per-class Dirichlet"}, "positive_class_weight": 577.2868020304569, "random_seeds_run": [42], "rounds": 2, "seed": 42, "test_data_used_for_training_or_threshold_tuning": false, "test_size": 0.2}
- Positive-class loss weight calculated from training labels only: 577.2868020304569.
- Client data is partitioned from training rows only. Federated server APIs receive model states or masked states, counts, and per-client loss/accuracy scalars—not transaction feature/label tensors. Clients are nevertheless in one process and hold private loaders reachable through Python objects; this is not process/network isolation.

## Client distributions

IID uses reproducibly shuffled, class-stratified splits. Non-IID uses separate per-class Dirichlet allocations (configured alpha) with a 50% equal-share floor per class, so each institution retains at least half the class's equal-share allocation while prevalence remains skewed.

| Partition | Client | Rows | Fraud | Fraud rate |
|---|---:|---:|---:|---:|
| iid | 1 | 45,570 | 79 | 0.1734% |
| iid | 2 | 45,569 | 79 | 0.1734% |
| iid | 3 | 45,569 | 79 | 0.1734% |
| iid | 4 | 45,569 | 79 | 0.1734% |
| iid | 5 | 45,568 | 78 | 0.1712% |
| non_iid | 1 | 110,542 | 90 | 0.0814% |
| non_iid | 2 | 23,733 | 187 | 0.7879% |
| non_iid | 3 | 26,305 | 39 | 0.1483% |
| non_iid | 4 | 44,474 | 39 | 0.0877% |
| non_iid | 5 | 22,791 | 39 | 0.1711% |

## Final test-set results

| Method | Distribution | Status | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC (AP) | FPR | FNR | Seconds | Time Δ vs FedAvg | Rounds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| centralized | not_applicable | success | 0.9784 | 0.0672 | 0.8980 | 0.1251 | 0.9728 | 0.6826 | 0.0215 | 0.1020 | 9.43 | — | 0 |
| fedavg | iid | success | 0.8388 | 0.0096 | 0.9082 | 0.0190 | 0.9542 | 0.6830 | 0.1613 | 0.0918 | 20.19 | +0.0% | 2 |
| dp_fedavg | iid | success | 0.1173 | 0.0019 | 1.0000 | 0.0039 | 0.8461 | 0.0230 | 0.8842 | 0.0000 | 25.57 | +26.6% | 2 |
| secure_fedavg | iid | success | 0.8272 | 0.0090 | 0.9082 | 0.0178 | 0.9545 | 0.6815 | 0.1730 | 0.0918 | 19.42 | -3.8% | 2 |
| secure_dp_fedavg | iid | success | 0.0806 | 0.0019 | 1.0000 | 0.0037 | 0.8648 | 0.0286 | 0.9210 | 0.0000 | 24.55 | +21.6% | 2 |
| fedavg | non_iid | success | 0.9899 | 0.1316 | 0.8673 | 0.2285 | 0.9465 | 0.6942 | 0.0099 | 0.1327 | 18.71 | +0.0% | 2 |
| dp_fedavg | non_iid | success | 0.0989 | 0.0019 | 1.0000 | 0.0038 | 0.8559 | 0.0260 | 0.9026 | 0.0000 | 25.89 | +38.4% | 2 |
| secure_fedavg | non_iid | success | 0.9888 | 0.1204 | 0.8776 | 0.2118 | 0.9447 | 0.6956 | 0.0110 | 0.1224 | 19.26 | +2.9% | 2 |
| secure_dp_fedavg | non_iid | success | 0.1883 | 0.0020 | 0.9388 | 0.0040 | 0.7109 | 0.0102 | 0.8130 | 0.0612 | 26.98 | +44.1% | 2 |

Centralized is partition-independent and appears once. PR-AUC is reported as average precision. Confusion matrices use the fixed probability threshold 0.5.

## Per-round convergence

The machine-readable `metrics/per_round_metrics.csv` contains the initial round and the actual server-evaluated test loss and accuracy after each federated round. Per-round evaluations are diagnostic only and were not used for training or threshold selection.

## Privacy accounting and communication

DP-FedAvg and Secure DP-FedAvg use per-example clipping and Gaussian noise from the existing Phase 6 DP-SGD path. Accountant values are the project's existing lightweight RDP-style estimates, not formally validated privacy guarantees. The accountant's sampling model is approximate for shuffled mini-batches; no formal epsilon claim is made. Secure aggregation hides individual updates from the simulated aggregator under Phase 7's assumptions but is not differential privacy.

| Method | Distribution | Approx. max client epsilon | Delta | Estimated total bytes | Bytes Δ vs FedAvg |
|---|---|---:|---:|---:|---:|
| centralized | not_applicable | not applicable | — | 0 | — |
| fedavg | iid | not applicable | — | 123,040 | +0.0% |
| dp_fedavg | iid | 1.956659199019409 | 1e-05 | 123,040 | +0.0% |
| secure_fedavg | iid | not applicable | — | 187,560 | +52.4% |
| secure_dp_fedavg | iid | 1.956659199019409 | 1e-05 | 187,560 | +52.4% |
| fedavg | non_iid | not applicable | — | 123,040 | +0.0% |
| dp_fedavg | non_iid | 2.267397339871046 | 1e-05 | 123,040 | +0.0% |
| secure_fedavg | non_iid | not applicable | — | 187,560 | +52.4% |
| secure_dp_fedavg | non_iid | 2.267397339871046 | 1e-05 | 187,560 | +52.4% |

The companion `metrics/communication_overhead.csv` also reports estimated byte overhead versus same-partition FedAvg.

Communication is estimated from tensor sizes and one fixed public-key/share setup estimate per secure round. The estimate excludes TLS, framing, dropout recovery messages, and actual network behavior; clients run sequentially in one process. Centralized has zero inter-client communication.

## Interpretation of the observed run

The centralized baseline reached PR-AUC 0.6826, recall 0.8980, and accuracy 0.9784; IID FedAvg reached PR-AUC 0.6830, recall 0.9082, and accuracy 0.8388.
On IID data, FedAvg reached PR-AUC 0.6830 and recall 0.9082; Secure FedAvg reached PR-AUC 0.6815 and recall 0.9082. Their close scores are consistent with the fixed-point masked sum approximating the same weighted aggregate.
On non-IID data, FedAvg reached PR-AUC 0.6942 and recall 0.8673; Secure FedAvg reached PR-AUC 0.6956 and recall 0.8776.
The highest-prevalence non-IID client had 187 training frauds (0.79% prevalence); one partition and seed cannot establish that non-IID generally improves utility.
IID DP-FedAvg had precision 0.0019, recall 1.0000, and PR-AUC 0.0230; Secure DP-FedAvg had precision 0.0019, recall 1.0000, and PR-AUC 0.0286 at threshold 0.5. The high training-derived class weight plus clipping/noise is a plausible contributor to the measured utility change, not evidence of an optimal privacy/utility setting.
non-IID DP-FedAvg had precision 0.0019, recall 1.0000, and PR-AUC 0.0260; Secure DP-FedAvg had precision 0.0020, recall 0.9388, and PR-AUC 0.0102 at threshold 0.5. The high training-derived class weight plus clipping/noise is a plausible contributor to the measured utility change, not evidence of an optimal privacy/utility setting.
Federated run times ranged from 18.71 to 26.98 seconds. Runtime differences at this scale include training, protocol work, and evaluation and are sensitive to timing noise.
For IID runs, estimated total traffic was 123,040 bytes for FedAvg and 187,560 bytes for Secure FedAvg (+52.4%); the masked tensor upload is twice the plaintext tensor payload before setup traffic.

## Figures

- `../graphs/01_centralized_vs_fedavg.png`
- `../graphs/02_fedavg_vs_dp-fedavg.png`
- `../graphs/03_fedavg_vs_secure_fedavg.png`
- `../graphs/04_fedavg_vs_secure_dp-fedavg.png`
- `../graphs/05_iid_vs_noniid_performance.png`
- `../graphs/06_accuracy_across_rounds.png`
- `../graphs/07_loss_across_rounds.png`
- `../graphs/08_precision_recall_f1.png`
- `../graphs/09_roc_curves.png`
- `../graphs/10_precision_recall_curves.png`
- `../graphs/11_confusion_matrices.png`
- `../graphs/12_client_fraud_distribution.png`
- `../graphs/13_training_time.png`
- `../graphs/14_communication_overhead.png`
- `../graphs/15_privacy_budget_comparison.png`

## Analysis and limitations

Fraud-class recall, precision, F1, and PR-AUC should be prioritized over accuracy because legitimate transactions dominate the test set. DP noise can reduce utility by perturbing clipped local gradients; secure aggregation adds key exchange, masking, modular encoding, and protocol handling. Non-IID fraud prevalence changes local gradient distributions and can make global convergence less stable. These are mechanisms, not claims about the direction or size of differences in these results; use the measured tables and plots.

The server still observes client participation, sample counts, round timing, per-client training loss/accuracy, and the final aggregate model. Secure aggregation does not prevent leakage from repeated/differenced aggregates, authenticated transport is not implemented, clients are not isolated processes, and malicious-client robustness is absent. Local DP and secure aggregation provide distinct protections. The DP accountant is approximate and no formal privacy guarantee is claimed.

Only one seed was run because this CPU-only environment and the per-example DP implementation make repeated full-data runs expensive. Each requested method/distribution was attempted independently; every failure is recorded in the results table and logs rather than omitted.

## Automated tests

Full test suite: 42 run, 0 failures, 0 errors.

## Conclusion

This experiment evaluates the five training approaches using the actual provided Kaggle CSV and a single stratified held-out test split. Interpret all measured scores together with the severe class imbalance, the fixed threshold, single-seed scope, approximate privacy accounting, and in-memory simulation limitations.
