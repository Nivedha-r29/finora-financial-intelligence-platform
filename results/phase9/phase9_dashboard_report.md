# Phase 9: Finora Dashboard Report

## Objective and status

Phase 9 delivers a local, interactive research dashboard for the Privacy-Preserving
Federated Learning Framework for Financial Fraud Detection. It visualizes actual
Phase 8 outputs and provides a separate, local single-record inference
demonstration. The dashboard does not initiate training and does not change
previous phase outputs.

## Architecture and technology

- **UI:** Streamlit with a custom responsive dark navy / teal theme.
- **Interactive charts:** Plotly.
- **Historical data:** pandas and JSON/CSV/PNG artifacts under `results/phase8/`.
- **Inference:** PyTorch `FinancialFraudMLP` checkpoints.
- **Application modules:** `dashboard/app.py` owns navigation and theme;
  `dashboard/pages.py` implements the eight dashboard pages;
  `dashboard/data_loader.py` validates result files, preprocessing data,
  checkpoints, and inference inputs.
- **Launcher:** `run_dashboard.py` invokes Streamlit only; it does not call any
  training entry point.

## Pages and implemented features

1. **Executive overview:** actual dataset totals, class balance, institutions,
   rounds, successful federated/privacy modes, best observed PR-AUC, an
   experiment summary, and a federated workflow diagram.
2. **Fraud detection demo:** manual raw-feature entry or a reproducibly selected
   held-out record. It loads a saved checkpoint and the recorded scaler in the
   original feature order, displays the real model probability and uses the
   recorded threshold. Sample features and labels are not displayed.
3. **Federated learning:** IID/non-IID client sizes and fraud prevalence,
   historical run status, saved global accuracy/loss by round, and explicit
   disclosure that per-client training metrics were not saved.
4. **Privacy & security:** DP-SGD settings and per-client approximate epsilon
   estimates, plus an explanation of pairwise masks, key derivation, quantized
   modular sums, dropout recovery, and simulation limitations.
5. **Model comparison:** interactive method/partition filters, recorded
   classification metrics, ROC and Precision–Recall curves computed from saved
   test predictions, confusion matrices, training duration, and estimated
   communication.
6. **IID vs Non-IID:** actual client counts and fraud rates, model PR-AUC, and
   convergence comparisons.
7. **System architecture:** an interactive data-flow diagram that distinguishes
   transaction records from client model updates.
8. **Experiment results:** saved configuration, dataset/client/privacy
   artifacts, CSV/JSON downloads, the Phase 8 report, and saved graph gallery.

The app validates required CSV columns and numeric values. Missing results are
identified as unavailable. Missing or incompatible model/scaler artifacts
produce explicit errors; there is no random-weight fallback. Model loading is
lazy and cached. The full source dataset is not loaded on dashboard startup; a
single demo row is read only after the user requests it.

## Actual Phase 8 results displayed

The dashboard reads the final Phase 8 metrics, not newly generated values. At
the recorded fixed threshold of 0.5:

| Method | Partition | Precision | Recall | PR-AUC (average precision) |
|---|---|---:|---:|---:|
| Centralized | N/A | 0.0672 | 0.8980 | 0.6826 |
| FedAvg | IID | 0.0096 | 0.9082 | 0.6830 |
| DP-FedAvg | IID | 0.0019 | 1.0000 | 0.0230 |
| Secure FedAvg | IID | 0.0090 | 0.9082 | 0.6815 |
| Secure DP-FedAvg | IID | 0.0019 | 1.0000 | 0.0286 |
| FedAvg | Non-IID | 0.1316 | 0.8673 | 0.6942 |
| DP-FedAvg | Non-IID | 0.0019 | 1.0000 | 0.0260 |
| Secure FedAvg | Non-IID | 0.1204 | 0.8776 | 0.6956 |
| Secure DP-FedAvg | Non-IID | 0.0020 | 0.9388 | 0.0102 |

These are one-seed CPU results on a highly imbalanced dataset, not estimates of
production performance. DP epsilon remains an approximate accountant output,
not a validated formal privacy guarantee. Secure aggregation and differential
privacy provide distinct protections; secure aggregation does not itself
provide differential privacy.

## Run instructions

Install project requirements in the active environment, then launch using:

```powershell
streamlit run dashboard/app.py
```

or:

```powershell
python run_dashboard.py
```

Both start the dashboard independently of model training.

## Validation performed

- Full existing and Phase 9 test suite: **50 tests passed, 0 failures**.
- Dashboard browser smoke test: application started on localhost; all eight
  navigation pages loaded without a rendered exception.
- Manual-entry prediction was exercised using the selected saved checkpoint.
- Held-out sample prediction was exercised; the dashboard displayed the
  checkpoint's probability without revealing the sample feature values or label.
- Model-comparison charts, filters, saved result tables, and historical graph
  rendering loaded from Phase 8 artifacts.
- Static Problems checks reported no errors in the new dashboard, launcher, or
  Phase 9 tests.

## Screenshot

`dashboard_overview.png` is a screenshot of the running dashboard overview.

## Limitations

- This is a local research visualization, not a production banking interface.
- Federated clients and server remain an in-memory simulation; process
  isolation, authenticated transport, and production key management are absent.
- Sample prediction depends on the original Kaggle CSV being available at the
  path recorded in Phase 8 configuration. Manual prediction uses only saved
  checkpoint and preprocessing artifacts.
- The displayed DP accountant is approximate and no formal guarantee is
  claimed.
- Per-client local training loss/accuracy were not included in Phase 8 saved
  outputs and are intentionally reported as unavailable.
- A single prediction is not a calibrated or operational fraud decision.
