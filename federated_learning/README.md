# MNIST MLP — Single Client Baseline (Step 1)

> **Final-Year Project | Federated Learning Framework**
>
> ⚠️ **This is Step 1 only.**
> Federated Learning has NOT been implemented yet.
> This script establishes the single-client baseline:
> `MNIST → MLP → Local Training → Evaluation`

---

## What is MNIST?

MNIST (*Modified National Institute of Standards and Technology*) is a
classic benchmark dataset of handwritten digits.

| Property | Value |
|---|---|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28 × 28 pixels (grayscale) |
| Classes | 10 (digits 0 – 9) |

---

## What is an MLP?

An **MLP (Multi-Layer Perceptron)** is the simplest form of a neural network.
It consists of fully-connected (linear) layers with non-linear activation functions.

---

## Why MLP?

- Simple and interpretable — ideal for a baseline demonstration.
- Easy to replicate across multiple clients in Federated Learning.
- Its `state_dict` (weights + biases) is straightforward to average using FedAvg.
- Can later be replaced with a more complex model (DNN, LSTM, etc.) for financial data.

---

## Model Architecture

```
Input: 28 × 28 MNIST image
         ↓
     Flatten
         ↓
      784 neurons
         ↓
   Linear(784, 128)
         ↓
       ReLU
         ↓
    Linear(128, 64)
         ↓
       ReLU
         ↓
    Linear(64, 10)
         ↓
   10 class logits
   (digits 0 – 9)
```

- **Loss function**: `CrossEntropyLoss` (handles softmax internally)
- **Optimizer**: `Adam` (lr = 0.001)
- **Total parameters**: ~109,386

---

## Training Process

1. Load MNIST training set (60,000 images)
2. Pass images through the MLP in mini-batches of 64
3. Compute CrossEntropyLoss
4. Backpropagate gradients
5. Update weights with Adam optimizer
6. Repeat for 3 epochs

---

## Evaluation Process

After training:
1. Switch model to `eval()` mode
2. Pass the 10,000 test images through the MLP
3. Compute test loss and accuracy
4. The test set is **never used during training**

---

## Project Structure

```
federated_learning/
├── client/
│   ├── __init__.py
│   └── trainer.py       ← training loop (single client for now)
├── server/
│   └── __init__.py      ← reserved for Federated Server (Step 2+)
├── model/
│   ├── __init__.py
│   └── mlp.py           ← MLP architecture
├── data/
│   ├── __init__.py
│   └── dataset.py       ← MNIST loading and DataLoaders
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py     ← test set evaluation
├── utils/
│   ├── __init__.py
│   ├── logger.py        ← print helpers
│   └── visualizer.py    ← matplotlib plots
├── results/             ← saved model and graphs
├── main.py              ← entry point
└── requirements.txt
```

---

## Setup & Installation

```powershell
pip install -r requirements.txt
```

---

## How to Run

```powershell
cd s:\Flora\federated_learning
python main.py
```

MNIST will be downloaded automatically (~10 MB) on the first run.

---

## Expected Output

```
========================================
  MNIST MLP TRAINING
========================================

Device : CPU   (or CUDA + GPU name)

Loading MNIST...
  Training samples : 60,000
  Testing samples  : 10,000
  Batch size       : 64

Creating MLP model...
  Architecture     : 784 → 128 → ReLU → 64 → ReLU → 10
  Total parameters : 109,386

  Loss function    : CrossEntropyLoss
  Optimizer        : Adam  (lr=0.001)
  Epochs           : 3

--- Training ---
  Epoch 1/3  |  Loss: ...  |  Accuracy: ...%
  Epoch 2/3  |  Loss: ...  |  Accuracy: ...%
  Epoch 3/3  |  Loss: ...  |  Accuracy: ...%

--- Evaluating on Test Set ---
  Test Loss     : ...
  Test Accuracy : ...%

--- Saving Graphs ---
  Plot saved: ./results/training_loss.png
  Plot saved: ./results/training_accuracy.png

  Model saved: ./results/mnist_mlp_single_client.pth

--- Training Summary ---
  Final Training Loss     : ...
  Final Training Accuracy : ...%
  Final Test Loss         : ...
  Final Test Accuracy     : ...%

========================================
  STEP 1 COMPLETED
========================================
```

---

## What is NOT implemented yet

The following will be added in later steps:

- ❌ Federated Server
- ❌ Multiple clients with private data partitions
- ❌ Client-server communication
- ❌ FedAvg model aggregation
- ❌ Multiple communication rounds
- ❌ Financial dataset
