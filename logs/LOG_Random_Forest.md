# Experiment Log - Random Forest

This log records the Random Forest work in `ML_CAPSTONE.ipynb`.

> Cell numbers below are the **one-based notebook cell positions** in the uploaded notebook.

## Model

Random Forest configuration visible in the notebook:
- `n_estimators=100`
- `random_state=42`
- `n_jobs=-1`

## Headline-only experiment

### Cell 13 — Save input datasets
The notebook saves the headline-only, content-only, and headline + content datasets.

### Cell 14 — Headline-only split
The headline data is stratified into:
- 60% training
- 20% validation
- 20% test

### Cell 16 — Headline-only TF-IDF preparation
TF-IDF is fitted using 1000 features.

### Cell 19 — Train and evaluate Random Forest
Random Forest is trained on the headline-only TF-IDF matrix.

Observed test results:
- Accuracy: `0.8793428871`
- Precision: `0.8765031752`
- Recall: `0.8887518838`
- F1: `0.8825850340`
- ROC-AUC: `0.9482994964`

## Content-only experiment

### Cell 21 — Content-only TF-IDF preparation
The content-only data is prepared using the same general TF-IDF workflow.

### Cell 23 — Train and evaluate Random Forest
Observed test results:
- Accuracy: `0.9459727035`
- Precision: `0.9363188168`
- Recall: `0.9580997758`
- F1: `0.9470840837`
- ROC-AUC: `0.9884707292`

## Headline + Content experiment

### Cell 25 — Combined TF-IDF preparation
The combined representation is created with 1000 TF-IDF features.

### Cell 27 — Train and evaluate Random Forest
Observed test results:
- Accuracy: `0.9510064300`
- Precision: `0.9407130458`
- Recall: `0.9648041632`
- F1: `0.9526063147`
- ROC-AUC: `0.9911955871`

## Important metric note

The notebook prints scikit-learn's default binary F1 for these model cells. These are not automatically the macro-F1 values reported in the reference paper.

## Final observation recorded from the notebook

The Random Forest test F1 increased as richer text was used:
- Headline only: `0.8826`
- Content only: `0.9471`
- Headline + Content: `0.9526`

No claims beyond the notebook's recorded results are made.
