# Experiment Log — ALBERT

This log records the dedicated ALBERT experiment in `ALBERT (3).ipynb`.

> Cell numbers below are the **one-based notebook cell positions** in the uploaded notebook.

## Environment and dataset setup

### Cell 1 — Install libraries
Installs:
- Transformers
- SentencePiece
- Accelerate
- NLTK
- tqdm

### Cell 2 — Imports
Imports NumPy, pandas, PyTorch, NLTK, scikit-learn splitting utilities, Transformers, and PyTorch dataset/DataLoader utilities.

### Cell 3 — GPU check
The executed run reports:
- PyTorch: `2.11.0+cu130`
- CUDA: available
- GPU: `Tesla T4`
- GPU memory reported: `14.56 GB`
- GPU matrix test later succeeds

### Cell 4 — Upload dataset
Uploads `WELFake_Clean.csv`.

### Cell 5 — Load dataset
The uploaded dataset contains:
- `71537` initial rows
- Columns: `title`, `text`, `label`

### Cell 6 — Cleaning and preprocessing
The notebook:
- removes missing rows
- removes duplicate rows
- lowercases text
- removes URLs
- removes HTML tags
- removes mentions
- removes hashtags
- removes punctuation
- removes numbers
- normalizes whitespace
- removes English stopwords
- applies WordNet lemmatization

Observed rows after missing/duplicate removal:
- `63121`

### Cell 7 — Verify labels
Label values are confirmed as:
- `0`
- `1`

The notebook does not flip the labels in this pipeline.

### Cell 8 — Create input scenarios
Creates:
- headline-only
- content-only
- headline + content

### Cell 9 — Save cleaned scenario datasets
Writes:
- `Headline_Only.csv`
- `Content_Only.csv`
- `Headline_Content.csv`

### Cell 10 — Stratified 60/20/20 split
Uses `random_state=42`.

The resulting combined dataset split is:
- Training: `37872`
- Validation: `12624`
- Test: `12625`

### Cell 11 — Verify stratification
Checks label proportions across the three splits.

### Cell 12 — Data validation
Checks missing and empty text values.

### Cell 13 — Reproducibility
Sets Python, NumPy, and PyTorch seeds to `42`.

### Cell 14 — Final GPU check
Sets `device` to CUDA when available.

## ALBERT input pipeline

### Cell 15 — Prepare ALBERT headline + content text
Reconstructs combined headline + article content for the split indices.

Observed:
- Training: `37872`
- Validation: `12624`
- Test: `12625`
- Empty training texts reported: `5`
- Empty validation texts: `0`
- Empty test texts: `0`

### Cell 16 — Load tokenizer
Loads:
- `albert-base-v2`

Observed tokenizer vocabulary:
- `30000`

Model maximum input length reported by the tokenizer:
- `512`

### Cell 17 — Tokenization
Uses:
- maximum sequence length: `200`

Observed tokenized shapes:
- Training: `(37872, 200)`
- Validation: `(12624, 200)`
- Test: `(12625, 200)`

### Cell 18 — PyTorch DataLoaders
Creates DataLoaders with:
- batch size: `16`

Observed:
- Training batches: `2367`
- Validation batches: `789`
- Test batches: `790`

### Cell 19 — Load ALBERT model
Loads `albert-base-v2` for 2-class classification.

Observed:
- Total parameters: `11,685,122`
- Trainable parameters: `11,685,122`

### Cell 20 — Optimizer and scheduler
Configuration:
- Learning rate: `2e-5`
- Epochs: `5`
- Weight decay: `0.01`
- Warmup ratio: `0.10`
- Optimizer: AdamW
- Scheduler: linear warmup/decay

### Cells 21–24 — Runtime/debug checks
The notebook verifies CUDA operation, reloads the model/tokenizer when needed, and checks that the ALBERT pipeline variables exist.

### Cell 27 — Recover ALBERT input pipeline
Rebuilds tokenized datasets/DataLoaders in memory using the existing split variables.

### Cell 28 — ALBERT forward-pass test
Runs a forward pass on one training batch before the full training run.

### Cell 29 — Final optimizer/scheduler setup
Recreates the 5-epoch AdamW + linear-warmup training schedule.

## ALBERT training

### Cell 30 — Train ALBERT on Headline + Content
The notebook uses:
- 5 epochs
- mixed precision when CUDA is available
- gradient clipping
- validation weighted F1 for model selection
- checkpoint saved whenever validation weighted F1 improves

Observed validation F1 by epoch:
- Epoch 1: `0.9781`
- Epoch 2: `0.9750`
- Epoch 3: `0.9844`
- Epoch 4: `0.9864`
- Epoch 5: `0.9861`

Best epoch:
- `4`

Best validation weighted F1:
- `0.9864`

## Final ALBERT test evaluation

### Cell 31 — Final test evaluation
Loads the best validation checkpoint and evaluates on the held-out test set.

Observed final results:
- Accuracy: `0.9863`
- Precision: `0.9863`
- Recall: `0.9860`
- F1: `0.9861`
- ROC-AUC: `0.9980`

Best epoch:
- `4`

Test inference time:
- `58.74 sec`

Confusion matrix:
```text
[[6884   75]
 [  98 5568]]
```

The printed precision/recall/F1 values here use the macro-averaged metric code present in the evaluation cell.

## Explainability — Integrated Gradients

### Cell 32 — Install/import Captum
Loads `LayerIntegratedGradients` and prepares the final trained ALBERT model.

### Cell 33 — Choose one test example
Selects a correctly classified fake-news test example when available.

### Cell 34 — Token-level Integrated Gradients
Targets the ALBERT fake-class logit and attributes importance through the ALBERT word embedding layer.

The explanation completes successfully for one article.

### Cell 35 — Plot and save the single-example explanation
Plots the 20 most strongly attributed tokens and saves:
- `/content/ALBERT_explainability.csv`

### Cells 36–38 — 15-article explainability check
Runs Integrated Gradients over 15 test samples and saves:
- `/content/ALBERT_explainability_15_summary.csv`
- `/content/ALBERT_explainability_15_tokens.csv`

The 15-sample check reports:
- `15 / 15` correct predictions

## Explainability runtime benchmark

### Cell 39 — 50-article benchmark
Runs Integrated Gradients with:
- `N_BENCHMARK = 50`
- `N_STEPS = 16`

Observed:
- Total time: `36.01 sec`
- Average: `0.72 sec/article`

### Cells 40–41 — Runtime extrapolation
Uses the measured per-article runtime to estimate larger runs.

### Cell 42 — 500-article explainability run
Runs Integrated Gradients on 500 test articles with 16 steps and a live progress bar.

Observed:
- Total time: `282.99 sec`
- Average: `0.57 sec/article`
- Estimated 500-article runtime: `4.72 minutes`

Saved:
- `/content/ALBERT_500_explainability_summary.csv`
- `/content/ALBERT_500_explainability_tokens.csv`

## Important interpretation note

The Integrated Gradients experiment is an **additional explainability analysis** carried out in this project. It should not be described as a replacement that exactly reproduces the reference paper's SHAP implementation.

The notebook contains the actual measured runtimes and outputs above; no additional runtime or accuracy claims are added here.
