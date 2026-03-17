# Torch-Threat-Inference

> A deep learning binary classifier built with PyTorch for real-time cyber threat detection on system-level event logs from the BETH dataset.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Training Methodology](#training-methodology)
- [Validation Logic](#validation-logic)
- [Dataset Handling & DataLoader](#dataset-handling--dataloader)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## Overview

Torch-Threat-Inference is a supervised deep learning system designed to classify system-level process events as either **benign (0)** or **suspicious/malicious (1)**. It is trained on the **BETH dataset**, which simulates real-world host-based intrusion logs, making it directly applicable to enterprise cybersecurity pipelines.

The model ingests 7 numerical process-level features (e.g., `processId`, `userId`, `mountNamespace`) and outputs a binary threat classification. The full pipeline covers data preprocessing, tensor conversion, model training with batch-level loss accumulation, and threshold-based validation with best-model checkpointing.

---

## Dataset

The **BETH (Behaviour-based Exploit Threat Hunting)** dataset provides pre-labeled system event logs representing realistic host activity. Each record corresponds to a single process event captured from kernel-level audit logs.

| Feature | Type | Description |
|---|---|---|
| `processId` | int64 | Unique identifier for the process generating the event |
| `threadId` | int64 | ID of the thread spawning the log |
| `parentProcessId` | int64 | ID of the parent process that spawned this log |
| `userId` | int64 | ID of the user who owns the spawning process |
| `mountNamespace` | int64 | Mount namespace the process operates within |
| `argsNum` | int64 | Number of arguments passed to the event |
| `returnValue` | int64 | Return value from the event (typically 0 on success) |
| `sus_label` | int64 | **Target** — `1` = suspicious/malicious, `0` = benign |

> The dataset is split into three pre-labelled CSV files: `labelled_train.csv`, `labelled_validation.csv`, and `labelled_test.csv`. Feature scaling is applied using `StandardScaler` before tensor conversion.

---

## Architecture

The model is a fully-connected feedforward neural network (Multi-Layer Perceptron) implemented in PyTorch. The architecture is deliberately compact to balance expressiveness against the risk of overfitting on structured tabular data.

```
Input (7 features)
      │
      ▼
Linear(7 → 128)   +   ReLU
      │
      ▼
Linear(128 → 64)  +   ReLU
      │
      ▼
Linear(64 → 1)    +   Sigmoid
      │
      ▼
Output (binary probability ∈ [0, 1])
```

### Layer-by-Layer Breakdown

| Layer | Type | Input Dim | Output Dim | Activation |
|---|---|---|---|---|
| Layer 1 | `nn.Linear` | 7 | 128 | ReLU |
| Layer 2 | `nn.Linear` | 128 | 64 | ReLU |
| Layer 3 (Output) | `nn.Linear` | 64 | 1 | Sigmoid |

**Design Decisions:**

- **ReLU Activations** — Rectified Linear Units are used in hidden layers to introduce non-linearity and avoid the vanishing gradient problem common with sigmoid/tanh in deep networks.
- **Sigmoid Output** — The final layer applies a sigmoid function to squash the raw logit into a probability in `[0, 1]`, making it directly interpretable as the probability of a malicious event.
- **Progressive Dimensionality Reduction** — The network compresses the representation from 128 → 64 → 1, forcing the model to learn increasingly abstract features before making a binary decision.
- **L2 Regularization (Weight Decay)** — Applied via the optimizer (`weight_decay=0.0001`) to penalize large weights and reduce overfitting on the high-volume training set.

---

## Training Methodology

### Loss Function — `BCELoss`

Binary Cross-Entropy Loss (`nn.BCELoss`) is used as the training objective. Since the final layer applies a `Sigmoid` activation, the raw probability is fed directly into `BCELoss`. The loss for a single sample is defined as:

```
L = -[y · log(ŷ) + (1 - y) · log(1 - ŷ)]
```

where `y` is the true label and `ŷ` is the predicted probability.

### Optimizer — Adam

The Adam optimizer is configured with the following hyperparameters:

| Hyperparameter | Value |
|---|---|
| Learning Rate | `0.001` |
| Weight Decay (L2) | `0.0001` |
| β₁ (momentum) | `0.9` (default) |
| β₂ (RMSProp term) | `0.999` (default) |
| Epochs | `10` |

Adam combines momentum and adaptive learning rates per parameter, making it well-suited for large, imbalanced datasets typical in cybersecurity contexts.

### Mathematically Accurate Epoch Loss via `running_loss`

A key implementation detail is the use of **sample-weighted loss accumulation** to compute a statistically correct epoch-level loss. A naive average of per-batch losses is biased when the final batch has fewer samples than the rest. This implementation corrects for that:

```python
running_loss += loss.item() * X_batch.size(0)   # weight by actual batch size
epoch_loss = running_loss / len(train_loader.dataset)  # divide by total samples
```

This yields the true mean loss across all training samples, regardless of whether the dataset divides evenly into the batch size.

### Training Loop

```python
for epoch in range(num_ep):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
```

`optimizer.zero_grad()` is called at the start of every batch to prevent gradient accumulation across iterations, which is a common source of training bugs.

---

## Validation Logic

After each training epoch, the model is evaluated on the held-out validation set using `torch.no_grad()` to disable gradient computation (reducing memory overhead and improving inference speed).

### Threshold-Based Classification

Raw sigmoid outputs (continuous probabilities) are converted to hard binary predictions using a **decision threshold of 0.5**:

```python
val_predictions = (val_outputs >= 0.5).float()
```

This is a standard operating point for balanced datasets. Any event with a predicted probability ≥ 0.5 is classified as malicious.

### Accuracy Metric

Batch-level accuracy is computed using `torchmetrics.Accuracy` configured for binary classification:

```python
accuracy = Accuracy(task='binary', threshold=0.5)
```

Batch accuracies are averaged across all validation batches to yield the epoch-level validation accuracy.

### Best Model Tracking

A `best_val_accuracy` variable is maintained across epochs. At the end of each epoch, if the current validation accuracy exceeds the historical best, it is updated:

```python
if current_val_acc > best_val_accuracy:
    best_val_accuracy = current_val_acc
    # torch.save(model.state_dict(), 'best_model.pth')  # checkpoint hook
```

This pattern ensures that the reported final accuracy corresponds to the **optimal epoch**, not simply the last one — which is critical in scenarios where validation performance fluctuates due to dataset noise or learning rate dynamics.

---

## Dataset Handling & DataLoader

### Preprocessing Pipeline

Raw features are standardized using `sklearn.preprocessing.StandardScaler` to zero-mean, unit-variance form before being converted to PyTorch `float32` tensors. Standardization is essential here because features like `mountNamespace` contain values in the billions while `argsNum` ranges from 0–10, which would otherwise cause gradient instability.

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(train_df.drop('sus_label', axis=1))
# fit_transform on train; transform only on val/test to prevent data leakage
```

### DataLoader Configuration

| Parameter | Training | Validation |
|---|---|---|
| Batch Size | `64` | `64` |
| Shuffle | `True` | `False` |
| Dataset Size | 763,144 samples | — |

**Batch Size of 64** — A batch size of 64 provides a good trade-off between gradient noise (which helps escape local minima) and computational efficiency on the 763K-sample training set.

**Shuffling for Cybersecurity Data** — Shuffling the training loader is particularly important in cybersecurity datasets because real-world logs are temporally ordered. Without shuffling, the model may learn spurious temporal patterns (e.g., bursts of suspicious activity from a single session) rather than generalizable features. Shuffling breaks this sequential correlation and forces the model to learn from a representative mini-batch distribution at every step.

**Validation is not shuffled** — Evaluation order does not affect accuracy metrics, so `shuffle=False` is used on the validation loader to keep evaluation deterministic and reproducible.

---

## Results

| Epoch | Training Loss | Validation Accuracy |
|---|---|---|
| 1 | 0.0040 | 100.00% |
| 2 | 0.0023 | 100.00% |
| 3 | 0.0023 | 100.00% |
| 4 | 0.0023 | 100.00% |
| 5 | 0.0024 | 100.00% |
| 6 | 0.0024 | 99.99% |
| 7 | 0.0024 | 100.00% |
| 8 | 0.0023 | 100.00% |
| 9 | 0.0023 | 99.98% |
| 10 | 0.0023 | 99.99% |

**Best Validation Accuracy: 99%** (integer-truncated from floating-point best)

> The rapid convergence to near-perfect validation accuracy from Epoch 1 suggests that the 7 engineered process-level features are highly discriminative for this threat detection task, and the MLP architecture is well-matched to the classification boundary in this feature space.

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### `requirements.txt`

```
pandas
scikit-learn
torch
torchmetrics
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Torch-Threat-Inference.git
cd Torch-Threat-Inference

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Running the Notebook

```bash
jupyter notebook notebook.ipynb
```

### Running as a Script

Convert the notebook to a Python script and execute:

```bash
jupyter nbconvert --to script notebook.ipynb
python notebook.py
```

### Expected Output

```
Epoch 1/10, Loss: 0.0040, Val Acc: 1.0000
Epoch 2/10, Loss: 0.0023, Val Acc: 1.0000
...
Epoch 10/10, Loss: 0.0023, Val Acc: 0.9999
Best Validation Accuracy achieved: 99%
```

### Saving the Best Model

To persist the best-performing model weights, uncomment the checkpoint line in the training loop:

```python
if current_val_acc > best_val_accuracy:
    best_val_accuracy = current_val_acc
    torch.save(model.state_dict(), 'best_model.pth')  # ← uncomment this
```

To reload the saved model for inference:

```python
model = nn.Sequential(
    nn.Linear(7, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 1), nn.Sigmoid()
)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

---

## Project Structure

```
Torch-Threat-Inference/
├── notebook.ipynb           # Main training and evaluation notebook
├── labelled_train.csv       # Preprocessed training split
├── labelled_validation.csv  # Preprocessed validation split
├── labelled_test.csv        # Preprocessed test split
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## Future Work

- **Model Checkpointing** — Fully integrate `torch.save` to persist best weights across runs.
- **Class Imbalance Handling** — Explore `pos_weight` in `BCEWithLogitsLoss` or oversampling techniques (SMOTE) for more adversarial distributions.
- **Threshold Tuning** — Sweep decision thresholds and plot a ROC/PR curve to optimize for precision-recall trade-offs relevant to security contexts (minimizing false negatives).
- **Feature Importance** — Apply SHAP values to interpret which process-level features most strongly drive malicious classifications.
- **Deployment** — Wrap the inference pipeline in a FastAPI endpoint for integration with SIEM (Security Information and Event Management) systems.

---

## Acknowledgements

Dataset sourced from the **BETH (Behaviour-based Exploit Threat Hunting)** dataset. See [`accreditation.md`](accreditation.md) for full citation details.
