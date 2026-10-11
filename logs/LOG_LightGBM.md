# Experiment Log - LightGBM

This log records the LightGBM work in `ML_CAPSTONE.ipynb`.

> Cell numbers below are the **one-based notebook cell positions** in the uploaded notebook.

## Model

LightGBM configuration visible in the notebook:
- `n_estimators=100`
- `learning_rate=0.1`
- `random_state=42`
- `n_jobs=-1`
- `verbosity=-1`

## Headline-only experiment

### Cell 13 — Save input datasets
The notebook saves headline-only, content-only, and headline + content datasets.

### Cell 14 — Headline-only stratified split
The headline dataset is split into training, validation, and test portions using stratification.

### Cell 16 — Headline-only TF-IDF preparation
The headline-only representation uses 1000 TF-IDF features.

### Cell 20 — Train and evaluate LightGBM
Observed test results:
- Accuracy: `0.8632645928`
- Precision: `0.8474444011`
- Recall: `0.8927250308`
- F1: `0.8694955965`
- ROC-AUC: `0.9418497667`

## Content-only experiment

### Cell 21 — Content-only TF-IDF preparation
Content-only data is transformed into a 1000-feature TF-IDF representation.

### Cell 24 — Train and evaluate LightGBM
Observed test results:
- Accuracy: `0.9531150555`
- Precision: `0.9429314356`
- Recall: `0.9655269058`
- F1: `0.9540954095`
- ROC-AUC: `0.9899614465`

## Headline + Content experiment

### Cell 25 — Combined TF-IDF preparation
Headline + content is converted into a 1000-feature TF-IDF representation.

### Cell 28 — Train and evaluate LightGBM
Observed test results:
- Accuracy: `0.9596729103`
- Precision: `0.9497124515`
- Recall: `0.9724732950`
- F1: `0.9609581162`
- ROC-AUC: `0.9926837176`

## Important metric note

The printed F1 values in these cells come from the notebook's default scikit-learn binary `f1_score(...)` calculation. They are not automatically macro-F1.

## Final observation recorded from the notebook

Observed LightGBM test F1:
- Headline only: `0.8695`
- Content only: `0.9541`
- Headline + Content: `0.9610`

The notebook therefore shows its strongest LightGBM result for the combined representation.
