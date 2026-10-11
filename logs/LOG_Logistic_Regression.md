# Experiment Log — Logistic Regression

This log records the Logistic Regression work in `Copy_of_ML_CAPSTONE.ipynb`.

> Cell numbers below are the **one-based notebook cell positions** in the uploaded notebook, not Python execution-count values. Each entry names the actual cell purpose so the log remains traceable.

## Model

Logistic Regression is used as the linear TF-IDF baseline.

Configuration visible in the notebook:
- `max_iter=1000`
- `random_state=42`

## Headline-only experiment

### Cell 13 — Save input datasets
The notebook saves:
- `Headline_Only.csv`
- `Content_Only.csv`
- `Headline_Content.csv`

### Cell 14 — Headline-only train/validation/test split
A stratified split is created using `random_state=42`:
- 60% training
- 20% validation
- 20% test

### Cell 16 — Headline-only TF-IDF preparation
The headline data is cleaned, missing rows are removed, and TF-IDF is fitted on the training data.

Observed training TF-IDF shape:
- `(42914, 1000)`

### Cell 18 — Train and evaluate Logistic Regression
The model is fitted on the headline-only TF-IDF training matrix, then validation and test predictions/probabilities are generated.

Observed test results:
- Accuracy: `0.8684376092`
- Precision: `0.8687542546`
- Recall: `0.8742293465`
- F1: `0.8714832013`
- ROC-AUC: `0.9444384703`

## Content-only experiment

### Cell 21 — Content-only TF-IDF preparation
The content-only dataset is loaded, missing rows are removed, and TF-IDF is prepared.

Observed training size:
- `42423`
- TF-IDF features: `1000`

### Cell 22 — Train and evaluate Logistic Regression
The content-only Logistic Regression model is trained and evaluated.

Observed test results:
- Accuracy: `0.9331023266`
- Precision: `0.9322625698`
- Recall: `0.9353979821`
- F1: `0.9338276441`
- ROC-AUC: `0.9815137850`

## Headline + Content experiment

### Cell 25 — Combined TF-IDF preparation
The combined headline + article-content dataset is loaded and converted to a 1000-feature TF-IDF representation.

Observed training shape:
- `(42921, 1000)`

### Cell 26 — Train and evaluate Logistic Regression
The combined-input Logistic Regression model is trained and evaluated.

Observed test results:
- Accuracy: `0.9389851831`
- Precision: `0.9355100935`
- Recall: `0.9456313339`
- F1: `0.9405434857`
- ROC-AUC: `0.9834584990`

## Important metric note

The Logistic Regression cells use scikit-learn's default `f1_score(...)` call for these printed F1 values. Therefore these values are the binary/positive-class F1 printed by the notebook, not the paper's macro-F1 metric.

## Final observation recorded from the notebook

Adding article content improved the Logistic Regression test F1 from:
- Headline only: `0.8715`
- Content only: `0.9338`
- Headline + Content: `0.9405`

No additional performance claims are made here beyond the values printed by the notebook.
