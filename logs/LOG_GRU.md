# Experiment Log - GRU

This log records the GRU experiments in `ML_CAPSTONE.ipynb`.

> Cell numbers below are the **one-based notebook cell positions** in the uploaded notebook.

## Model structure used in the final combined-input run

The final combined GRU model contains:
- Embedding layer
- GRU with 64 hidden units
- Dropout `0.3`
- Dense sigmoid output

The final combined model uses:
- Maximum sequence length: `50`
- Learning rate: `0.001`
- Batch size: `128`
- Up to `10` epochs
- Early stopping with `patience=2`
- `restore_best_weights=True`

## Headline-only experiment

### Cell 29 — Headline tokenization
A Keras tokenizer is fitted on training text only.

Observed vocabulary size:
- `33338`

### Cell 30 — Headline padding
Sequences are padded/truncated to 50 tokens.

Observed shapes:
- Training: `(42914, 50)`
- Validation: `(14305, 50)`
- Test: `(14305, 50)`

### Cells 31–35 — Headline model construction / diagnostics
The notebook creates and then rebuilds the GRU model, performs diagnostic checks, and prepares the final headline model.

Cell 35 is the final headline GRU architecture before training.

### Cell 36 — Train headline GRU
The final headline model is trained for up to 10 epochs with early stopping.

### Cell 38 — Headline-only test evaluation
Observed test results:
- Accuracy: `0.9058371199`
- Precision: `0.8929231582`
- Recall: `0.9265652829`
- F1: `0.9094332011`
- ROC-AUC: `0.9683810839`

## Content-only experiment

### Cell 39 — Content GRU model
Creates the content-only GRU architecture.

### Cell 41 — Content tokenization
Tokenizer is fitted on the training content.

### Cell 42 — Content padding
Observed shapes:
- Training: `(42423, 50)`
- Validation: `(14141, 50)`
- Test: `(14141, 50)`

### Cell 43 — Train content GRU
The content-only model is trained with early stopping.

### Cell 44 — Content-only test evaluation
Observed test results:
- Accuracy: `0.9352238173`
- Precision: `0.9206113065`
- Recall: `0.9538957399`
- F1: `0.9369580179`
- ROC-AUC: `0.9814316648`

## Headline + Content experiment

### Cell 45 — Combined tokenization
The tokenizer is fitted only on combined training text.

### Cell 46 — Combined padding
Observed shapes:
- Training: `(42921, 50)`
- Validation: `(14308, 50)`
- Test: `(14308, 50)`

### Cell 47 — Build final combined GRU
The final combined GRU is:
- Embedding: 128 dimensions
- GRU: 64 units
- Dropout: 0.3
- Dense sigmoid output
- Adam learning rate: 0.001

### Cell 48 — Train combined GRU
Training is run for up to 10 epochs with early stopping.

Observed training log:
- Epoch 1 validation accuracy: `0.9547`
- Epoch 2 validation accuracy: `0.9567`
- Epoch 3 validation accuracy: `0.9560`
- Epoch 4 validation accuracy: `0.9577`

### Cell 49 — Combined test evaluation
Observed test results printed by the notebook:
- Accuracy: `0.9526`
- Precision: `0.9522`
- Recall: `0.9551`
- F1: `0.9536`
- ROC-AUC: `0.9882`

### Cell 50 — Macro-averaged combined metrics
The notebook additionally computes macro precision, macro recall, and macro F1.

Observed macro results:
- Precision: `0.9526`
- Recall: `0.9526`
- F1: `0.9526`

The cell reprints the already-computed accuracy and ROC-AUC:
- Accuracy: `0.9526`
- ROC-AUC: `0.9882`

## Important reproducibility note

The notebook contains several GRU rebuild/debug cells. The values above are taken from the final executed evaluation cells, not from the earlier diagnostic model.
