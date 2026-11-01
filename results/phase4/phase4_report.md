# Phase 4: Federated Learning Evaluation

## Setup and Repeatability

- Dataset/model: MNIST / existing MLP; device: `cpu`.
- Primary run: 5 IID clients, 10 rounds, 1 local epoch, batch size 64, Adam, learning rate 0.001.
- Partitioning: IID uses a seeded NumPy permutation and equal shards. Non-IID uses the existing per-class Dirichlet partitioner with alpha 0.3, seed 42.
- The non-IID comparison uses 2 rounds and is compared to the IID run at the same round budget to limit runtime.
- Centralized training uses the same initialized MLP state, Adam learning rate, complete MNIST training set, and 10 full-data epochs. This matches sample-epoch exposure approximately; it is not the same optimization procedure as periodic client training and FedAvg.
- Seeds cover Python, NumPy, PyTorch, CUDA, and per-loader shuffling. Deterministic PyTorch operations are requested with warnings enabled. Reproducibility across hardware/library versions is not guaranteed.
- Test data is reserved for evaluation and is not used for training or hyperparameter selection.

## Federated IID Results

Initial test loss/accuracy: 3.1374 / 8.49%.

Final test loss/accuracy: 0.0839 / 97.34% (accuracy change +88.85 percentage points).

Federated elapsed time across rounds: 444.62 seconds. Round time includes local training, aggregation, and global evaluation; initial round-0 evaluation is excluded.

| Round | Test loss | Test accuracy | Round duration (s) |
|---:|---:|---:|---:|
| 0 | 3.1374 | 8.49% | - |
| 1 | 0.2940 | 92.37% | 43.62 |
| 2 | 0.1856 | 94.64% | 45.79 |
| 3 | 0.1500 | 95.51% | 48.56 |
| 4 | 0.1292 | 96.09% | 47.06 |
| 5 | 0.1138 | 96.61% | 43.99 |
| 6 | 0.1036 | 96.85% | 43.48 |
| 7 | 0.0952 | 97.06% | 42.74 |
| 8 | 0.0909 | 97.28% | 42.90 |
| 9 | 0.0866 | 97.34% | 42.80 |
| 10 | 0.0839 | 97.34% | 43.69 |

### Client training metrics

| Client | Round 1 loss | Final-round loss | Round 1 training accuracy | Final-round training accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.7713 | 0.1527 | 92.58% | 98.41% |
| 2 | 0.7848 | 0.1432 | 92.58% | 98.45% |
| 3 | 0.7707 | 0.1522 | 93.11% | 98.48% |
| 4 | 0.7817 | 0.1473 | 92.78% | 98.37% |
| 5 | 0.7644 | 0.1548 | 92.46% | 98.42% |

## Centralized Comparison

| Method | Test loss | Test accuracy | Elapsed seconds |
|---|---:|---:|---:|
| Federated IID | 0.0839 | 97.34% | 444.62 |
| Centralized (10 epochs) | 0.0678 | 97.84% | 224.76 |

Centralized minus federated accuracy: +0.50 percentage points. These measurements compare these specific runs only; neither training method is inherently more accurate in all settings. Federated elapsed time includes evaluation every round, while centralized elapsed time includes its training and final evaluation.

## IID Versus Non-IID

| Partition | Round | Test loss | Test accuracy |
|---|---:|---:|---:|
| IID | 2 | 0.1856 | 94.64% |
| Non-IID (alpha=0.3) | 2 | 0.4007 | 90.95% |

This is a two-round diagnostic, not a same-budget comparison to the full ten-round headline run. Dirichlet class allocation makes client label distributions heterogeneous; the limited run should not be treated as a convergence estimate.

## Correctness and Privacy

- FedAvg tests cover equal and sample-count-weighted averages, non-floating buffers, empty updates, invalid counts, and incompatible shapes/keys.
- Server smoke tests verify same round-start weights, changed client weights, exact aggregation, next-round distribution, test-only evaluation, and round timing.
- Client data loaders remain attached to client objects; the server receives state dictionaries, sample counts, aggregate training loss/accuracy scalars, and never reads raw batches directly. This is a logical boundary in a single-process simulation, not a process or network isolation boundary.
- Model updates may leak information. FedAvg does not provide differential privacy or secure aggregation, and this implementation does not provide encrypted inter-organization networking.

## Limitations and Readiness

The framework demonstrates the FedAvg workflow on MNIST but is not production-ready for sensitive financial data. Before that use case, it needs a threat model, actual isolated clients and authenticated encrypted transport, privacy/security controls, data governance, robust failure handling, and domain-specific validation. No financial model or processing is included in this phase.
