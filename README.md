# Federated Learning Framework — MNIST

> **Final-Year Project** | Privacy-Preserving Federated Learning with PyTorch  
> **Phase 3**: Simulated five-client FedAvg with MNIST handwritten digits  
> **Future work**: Financial dataset integration

---

## Table of Contents

1. [What is Federated Learning?](#1-what-is-federated-learning)
2. [Why is Federated Learning Useful?](#2-why-is-federated-learning-useful)
3. [How the Client-Server Process Works](#3-how-the-client-server-process-works)
4. [What is MNIST?](#4-what-is-mnist)
5. [Why an MLP?](#5-why-an-mlp)
6. [What is FedAvg?](#6-what-is-fedavg)
7. [Data Partitioning](#7-data-partitioning)
8. [What is Shared — and What is NOT](#8-what-is-shared--and-what-is-not)
9. [How the Global Model is Created](#9-how-the-global-model-is-created)
10. [Project Structure](#10-project-structure)
11. [Setup & Installation](#11-setup--installation)
12. [How to Run](#12-how-to-run)
13. [Configuration](#13-configuration)
14. [Interpreting the Results](#14-interpreting-the-results)
15. [Limitations of Phase 1](#15-limitations-of-phase-1)
16. [Adapting to a Financial Dataset (Phase 2)](#16-adapting-to-a-financial-dataset-phase-2)
17. [Phase 7: Secure Aggregation](#phase-7-secure-aggregation)
18. [Phase 8: Financial Fraud Experiments](#phase-8-financial-fraud-experiments)

---

## 1. What is Federated Learning?

**Federated Learning (FL)** is a distributed machine learning paradigm where multiple participants (*clients*) collaboratively train a shared model **without ever sharing their raw data**.

Instead of uploading private data to a central server, each client:
1. Receives the current global model weights from the server.
2. Trains the model locally on its own private dataset.
3. Sends only the **updated model weights** (not data) back to the server.

The server aggregates these weight updates into a new, improved global model — and the cycle repeats.

```
┌─────────────────────────────────────────────────────────────────┐
│                         FEDERATED LEARNING                       │
│                                                                   │
│  Server                                                           │
│  ┌──────────────┐                                                 │
│  │ Global Model │ ──── sends weights ──► Client 1 (local data)   │
│  │              │ ◄─── weights only ─── Client 2 (local data)    │
│  │   FedAvg     │ ──── sends weights ──► Client 3 (local data)   │
│  │  Aggregation │ ◄─── weights only ─── Client 4 (local data)    │
│  └──────────────┘                        Client 5 (local data)   │
│                                                                   │
│  Raw data: NEVER leaves the client   ❌                           │
│  Model weights: shared with server  ✅                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Why is Federated Learning Useful?

| Benefit | Description |
|---|---|
| **Privacy preservation** | Raw data stays on client devices — never uploaded |
| **Regulatory compliance** | Aligns with GDPR, HIPAA, and financial data regulations |
| **Data heterogeneity** | Works even when clients have different data distributions |
| **Edge training** | Training can happen on mobile devices, hospital systems, banks |
| **Reduced bandwidth** | Only model weights transmitted, not entire datasets |

**Concrete example**: Five banks want to build a fraud-detection model together without revealing their customers' transaction records to each other or a central authority. Federated Learning enables exactly this.

---

## 3. How the Client-Server Process Works

```
Round 1
  [Server]  Initialise global model with random weights
  [Server]  Send global weights to Client 1, 2, 3, 4, 5
  [Client 1] Train on local MNIST subset → produce updated weights
  [Client 2] Train on local MNIST subset → produce updated weights
  ...
  [Server]  Collect updated weights from all 5 clients
  [Server]  Run FedAvg: compute weighted average of all client weights
  [Server]  Update global model with aggregated weights
  [Server]  Evaluate global model on test set → record accuracy/loss

Round 2 ... Round 10
  (repeat the above process)
```

---

## 4. What is MNIST?

MNIST (*Modified National Institute of Standards and Technology*) is a classic benchmark dataset containing:

- **60,000 training images** of handwritten digits (0–9)
- **10,000 test images**
- Each image: **28 × 28 pixels**, grayscale

MNIST is used here as a well-understood, publicly available dataset to **validate that the FL framework itself works correctly** before moving to a financial dataset in Phase 2.

---

## 5. Why an MLP?

An **MLP (Multi-Layer Perceptron)** is chosen because:

- It is simple and interpretable — suitable for a demonstration framework.
- It has a fixed architecture that is easy to replicate across clients.
- Its **state_dict** (weights + biases) is compact and straightforward to average.
- In Phase 2, the MLP can be swapped for a DNN, LSTM, or tree-boosted model without changing the FL infrastructure.

### Architecture

```
Input (28×28 image)
  │
  ▼ Flatten
784 neurons
  │
  ▼ Linear → BatchNorm → ReLU → Dropout(0.2)
128 neurons
  │
  ▼ Linear → BatchNorm → ReLU → Dropout(0.2)
64 neurons
  │
  ▼ Linear
10 neurons (logits for digits 0–9)
  │
  ▼ CrossEntropyLoss (during training)
```

---

## 6. What is FedAvg?

**Federated Averaging (FedAvg)** (McMahan et al., 2017) is the most widely used FL aggregation algorithm.

### Formula

```
θ_global = Σ_k  (n_k / N) × θ_k

where:
  θ_k = model parameters from client k after local training
  n_k = number of training samples at client k
  N   = total samples across all clients = Σ n_k
```

### Why weighted?

Clients with **more training samples** contribute proportionally more to the global model. This is important when clients have different dataset sizes (common in practice).

---

## 7. Data Partitioning

The 60,000 MNIST training samples are split into 5 client datasets.

### IID Partitioning (default)

All samples are randomly shuffled and divided equally. Each client receives ~12,000 samples with a representative mix of all 10 digit classes. This is the idealistic, well-studied setting.

### Non-IID Partitioning (Dirichlet)

Uses a **Dirichlet(α)** distribution to simulate real-world data heterogeneity, where different clients have skewed class distributions.

- **α = 0.1** → Highly non-IID (each client specialises in 1–2 digit classes)
- **α = 1.0** → Moderately non-IID
- **α = 100** → Nearly IID

Configure via `partition_strategy = "non_iid"` and `dirichlet_alpha` in `config.py`.

---

## 8. What is Shared — and What is NOT

| Data | Client → Server | Server → Client |
|---|---|---|
| **Raw MNIST images** | ❌ NEVER | ❌ NEVER |
| **Sample labels** | ❌ NEVER | ❌ NEVER |
| **Model weights/biases** | ✅ YES | ✅ YES |
| **Dataset size** (n_k) | ✅ YES (for FedAvg weighting) | — |
| **Average training loss** | ✅ YES (for monitoring only) | — |
| **Local accuracy** | ✅ YES (for monitoring only) | — |

The server's access to loss/accuracy values is a monitoring feature. In a real privacy-preserving system these would be protected by differential privacy or secure aggregation.

---

## 9. How the Global Model is Created

1. The server initialises a fresh MLP with random weights.
2. After each round, it receives one `state_dict` per client.
3. The `FedAvgAggregator` computes the **weighted average** across all `state_dicts`.
4. The resulting averaged `state_dict` is loaded back into the global model.
5. The global model is evaluated on the shared test set.
6. Steps 2–5 repeat for `num_rounds` communication rounds.

---

## 10. Project Structure

```
federated_learning/          (project root: s:\Flora)
│
├── config.py                ← All hyperparameters and paths in one place
├── main.py                  ← Entry point — run this file
├── requirements.txt
├── README.md
│
├── model/
│   ├── base_model.py        ← Abstract BaseModel (get/set_parameters)
│   └── mlp.py               ← MLP for MNIST (784→128→64→10)
│
├── data/
│   └── dataset.py           ← MNIST loading + IID / Non-IID partitioning
│
├── client/
│   ├── client.py            ← FederatedClient (thin wrapper, private DataLoader)
│   └── trainer.py           ← LocalTrainer (actual training loop)
│
├── server/
│   ├── server.py            ← FederatedServer (orchestrates all FL rounds)
│   └── aggregator.py        ← FedAvgAggregator (weighted + unweighted)
│
├── evaluation/
│   └── evaluator.py         ← GlobalEvaluator (loss, accuracy, per-class)
│
├── utils/
│   ├── logger.py            ← FLLogger (console + file logging)
│   └── visualizer.py        ← generate_all_plots() — 5 publication-quality plots
│
├── data_storage/            ← MNIST dataset cached here after first download
└── results/                 ← Saved model, metrics JSON, PNG plots, log file
```

---

## 11. Setup & Installation

### Prerequisites

- Python 3.10 or newer
- pip

### Install dependencies

```powershell
pip install torch torchvision numpy matplotlib
```

Or using the requirements file:

```powershell
pip install -r requirements.txt
```

No GPU is required — the framework auto-detects CUDA and falls back to CPU.

---

## 12. How to Run

```powershell
cd s:\Flora
python main.py
```

On the first run, MNIST (~10 MB) is downloaded automatically to `./data_storage/`.

Expected console output:

```
══════════════════════════════════════════════════════════════
  FEDERATED LEARNING FRAMEWORK — PHASE 1
══════════════════════════════════════════════════════════════
  Dataset           : MNIST
  Model             : MLP  (784→128→64→10)
  Clients           : 5
  ...

──────────────────────────────────────────────────────────────
  ROUND 1/10
──────────────────────────────────────────────────────────────
  [Client 0] Received global model parameters ...
  [Client 0] Training locally...
  [Client 0] Done  | samples=12,000  loss=0.3412  acc=90.21%
  ...
  [Server]   Aggregating client updates via FedAvg...
  [Server]   Global Evaluation  →  Loss: 0.2891  |  Accuracy: 91.84%
...

══════════════════════════════════════════════════════════════
  FINAL SUMMARY
══════════════════════════════════════════════════════════════
  Baseline Accuracy : 9.83%
  Final Accuracy    : 97.XX%
  Final Loss        : 0.0XXX
```

---

## 13. Configuration

Edit [`config.py`](file:///s:/Flora/config.py) to customise training:

| Parameter | Default | Description |
|---|---|---|
| `num_clients` | 5 | Number of federated clients |
| `num_rounds` | 10 | Communication rounds |
| `local_epochs` | 1 | Epochs per client per round |
| `batch_size` | 64 | Mini-batch size for local training |
| `learning_rate` | 0.001 | SGD/Adam learning rate |
| `optimizer_name` | `"Adam"` | `"SGD"` or `"Adam"` |
| `partition_strategy` | `"iid"` | `"iid"` or `"non_iid"` |
| `dirichlet_alpha` | 0.5 | Non-IID skew (lower = more skewed) |
| `hidden_dim1` | 128 | First hidden layer size |
| `hidden_dim2` | 64 | Second hidden layer size |

---

## 14. Interpreting the Results

After training, the `results/` directory contains:

| File | Contents |
|---|---|
| `federated_global_model.pth` | Saved PyTorch state_dict of the final global model |
| `federated_metrics.csv` | Global test loss and accuracy for rounds 0–10 |
| `client_training_metrics.csv` | Per-client training loss and accuracy by round |
| `global_accuracy_vs_round.png` | Accuracy vs. round curve |
| `global_loss_vs_round.png` | Loss vs. round curve |
| `client_losses.png` | All clients' training loss curves |
| `initial_vs_final.png` | Bar chart: baseline vs. final model |
| `dashboard.png` | 2×2 summary panel of all metrics |
| `fl_run_<timestamp>.log` | Full structured training log |

### What good results look like

- **Baseline accuracy** ≈ 9–11% (random MLP, 10 classes → ~10% by chance)
- **After 10 rounds**: ≈ 96–98% accuracy on MNIST with IID partitioning
- **Loss** should decrease monotonically round-by-round
- **Per-client losses** should converge and cluster together

If accuracy plateaus early, try increasing `local_epochs`, `num_rounds`, or adjusting `learning_rate`.

---

## 15. Limitations of Phase 1

Ordinary FedAvg is not a complete privacy guarantee. Individual model updates
can leak information; this implementation does not provide differential
privacy, secure aggregation, or encrypted client-server networking. The five
clients are simulated sequentially in one Python process, so this validates the
learning algorithm rather than production-grade distributed security.

| Limitation | Details |
|---|---|
| **No differential privacy** | Model weights can theoretically leak information about training data through model inversion attacks |
| **No secure aggregation** | The server sees individual client weight updates, not just the aggregate |
| **Synchronous training** | All clients train and report every round (no stragglers, no dropouts) |
| **Simulated clients** | All clients run on one machine in sequence; real FL uses network communication |
| **IID assumption** | Default partitioning is IID; real-world data is almost always non-IID |
| **MNIST only** | Phase 1 is a framework validation, not a deployment-ready system |

---

## 16. Adapting to a Financial Dataset (Phase 2)

## Phase 4 Evaluation

Run the reproducible evaluation suite from the project root with:

```powershell
python run_phase4.py
```

The primary run uses the configured full IID federated setup. The centralized
comparison uses the same initialization, optimizer family, learning rate, and
approximately matched number of training examples. The non-IID run uses the
per-class Dirichlet partitioner with a configurable alpha and a shorter round
budget (`phase4_noniid_rounds`) to keep this diagnostic bounded on CPU. Its
results must be compared at that same round count, not against the final
ten-round IID score. Phase 4 outputs are isolated under `results/phase4/` and
do not overwrite earlier model artifacts.

Randomness is seeded for Python, NumPy, PyTorch, CUDA, and client/centralized
DataLoader shuffling. PyTorch deterministic operations are requested when
configured; results may still vary across device types and library versions.
The evaluation/test split is never used for training or hyperparameter tuning.

FedAvg and the current single-process simulation are not complete privacy
controls. Model updates can leak information; there is no differential privacy,
secure aggregation, or encrypted networking between isolated organizations.

## 17. Adapting to a Financial Dataset (Future Phase)

The FL infrastructure is designed to be **dataset- and model-agnostic**. Moving to a financial use case requires only targeted changes:

### What changes in Phase 2

| Component | Phase 1 | Phase 2 Change |
|---|---|---|
| `data/dataset.py` | MNIST torchvision loader | Replace with CSV/Parquet financial data loader |
| `model/mlp.py` | 784→128→64→10 MLP | New DNN/Tabular model inheriting `BaseModel` |
| `config.py` | `input_dim=784, num_classes=10` | Update dimensions for financial features |
| Loss function | CrossEntropyLoss (classification) | BCE (fraud detection) or MSE (regression) |

### What does NOT change

- `server/server.py` — FL orchestration loop
- `server/aggregator.py` — FedAvg aggregation
- `client/client.py` — Client abstraction
- `client/trainer.py` — Local training loop (only loss function changes)
- `evaluation/evaluator.py` — Minor metric updates
- `utils/logger.py` — No changes needed
- `utils/visualizer.py` — No changes needed

Because the FedAvg aggregator operates on raw `state_dict` tensors, it is completely model-agnostic and requires **zero modification** for Phase 2.

---

## References

- McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS 2017. https://arxiv.org/abs/1602.05629
- LeCun, Y., Cortes, C., & Burges, C. J. C. (1998). *The MNIST Database of Handwritten Digits*. http://yann.lecun.com/exdb/mnist/
- Kairouz, P. et al. (2021). *Advances and Open Problems in Federated Learning*. Foundations and Trends in Machine Learning. https://arxiv.org/abs/1912.04977

## Phase 7: Secure Aggregation

The root project supports four optional aggregation modes in `FLConfig`:
`fedavg` (the unchanged standard path), `secure`, `dp`, and `secure_dp`.
Selecting `secure_dp` applies the existing client-side DP-SGD before secure
masking; secure aggregation itself is not a differential-privacy mechanism.

Run the dataset-free synthetic protocol checks and overhead measurement with:

```powershell
python run_phase7.py
```

Outputs are written to `results/phase7/`. Secure updates use fresh X25519
ephemeral keys, HKDF-SHA-256 key derivation, HMAC-SHA-256 pairwise masks, and
Shamir recovery shares for clients that fail before uploading. Updates are
quantized by rounding `update * secure_quantization_scale`, sample-weighted,
and summed modulo `secure_modulus`; configured magnitude and aggregate bounds
prevent signed modular wraparound. Clients that fail after uploading remain
included in that round. A missing pre-upload client is recoverable only when
the minimum uploader count, dropout budget, and recovery threshold all hold;
otherwise aggregation aborts.

The implementation provides protocol arithmetic and control flow in the
single-process simulator, not production isolation. Public-key authentication,
authenticated private channels, network replay protection, malicious-client
proofs, and secure hardware are not implemented. The server sees sample counts
and existing per-client training metrics. Integer buffers are decoded as
rounded weighted averages in secure mode; standard FedAvg's separate `max`
rule for arbitrary differing integer buffers is unchanged.

Bonawitz, K. et al. (2017). *Practical Secure Aggregation for Privacy-Preserving
Machine Learning*. ACM CCS 2017.
https://doi.org/10.1145/3133956.3133982

## Phase 8: Financial Fraud Experiments

The end-to-end Phase 8 experiments use the Kaggle Credit Card Fraud Detection
CSV. Place `creditcard.csv` at the project root (or pass a different file with
`--dataset`). The checked-in configuration points to `./creditcard.csv`; the
runner reports a clear error if that file is missing and does not download or
fabricate data.

Run the five-method comparison for IID and non-IID partitions with:

```powershell
python run_phase8.py
```

By default this uses five clients, two federated communication rounds, seed 42,
and Dirichlet alpha 0.3. To override settings, use `--dataset`, `--rounds`,
`--batch-size`, `--clients`, `--seed`, and `--noniid-alpha`. The complete test
suite runs after the experiments; `--skip-tests` is available for iteration.

The experiment creates a stratified 80/20 split, fits StandardScaler on
training rows only, keeps duplicate source records, and calculates positive
class weighting from the training split only. IID distribution is stratified
across clients; non-IID distribution uses per-class Dirichlet allocations with
a half-of-equal-share floor per class to keep every institution large enough
while retaining fraud-prevalence skew.
A small tabular MLP without BatchNorm is used so individual-example gradients
are not coupled by batch statistics. All methods use the same initial model,
model shape, optimizer family, and pass count. The fraud probability threshold
is fixed at 0.5; it is not tuned on the test set.

Outputs are organized under `results/phase8/`: `reports/`, `graphs/`,
`metrics/`, `models/`, `logs/`, and `client_statistics/`. The report lists each
attempted experiment, including failures. Metrics include fraud-class
precision/recall/F1, ROC-AUC, average-precision PR-AUC, confusion matrices,
false-positive/negative rates, timing, round histories, communication
estimates, and approximate existing-accountant outputs.

DP-FedAvg uses the existing per-example clipping and Gaussian-noise training
path. For models without BatchNorm, the per-example gradient calculation uses
`torch.func.vmap` to avoid a Python-level backward pass per sample; the
clipping/noise mechanism is unchanged. The included accountant is a
lightweight approximation, not a validated formal privacy guarantee. Secure
aggregation masks model states in the existing in-memory simulation; it is not
differential privacy and is not production-grade isolation. The server still
observes sample counts, participation, timing, and per-client loss/accuracy.
Communication figures are tensor-size estimates, not measured network traffic.

## Phase 9: Finora Dashboard

The Streamlit dashboard presents the actual Phase 8 dataset statistics,
experiment metrics, client partitions, privacy-accounting estimates, saved
plots, and report. It also provides a local inference demonstration using the
saved `FinancialFraudMLP` checkpoints and the exact feature order and scaler
recorded during Phase 8. Historical results are clearly separated from this
single-transaction prediction demo.

Install the requirements in the active environment and launch either with:

```powershell
streamlit run dashboard/app.py
```

or:

```powershell
python run_dashboard.py
```

The dashboard does not start training. Checkpoints and preprocessing artifacts
are loaded only for the inference page; missing or incompatible files produce
an explicit error instead of a random-weight fallback. Manual entry accepts raw
feature values in the recorded order. The sample option reads one reproducibly
selected held-out record on demand from the original Kaggle CSV, does not show
the record's source features or label, and does not retain the dataset in
memory. If the original CSV is unavailable, manual prediction remains
available, but the sample action reports that it cannot load its source.

Phase 9 artifacts and validation notes are saved in
`results/phase9/phase9_dashboard_report.md`. The dashboard is a research
visualization, not a banking decision system. Reported epsilon is an
approximation, secure aggregation is an in-memory simulation, and neither the
dashboard nor the model is production-ready.
