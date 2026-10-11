# Automated Fake News Detection using Machine Learning and Deep Learning

A comparative fake-news detection project using the **WELFake** dataset. The project evaluates classical machine-learning baselines, a GRU-based neural model, and **ALBERT (`albert-base-v2`)**, followed by token-level explainability using **Integrated Gradients**.

The implementation is based on the methodology of:

> Tian, Y.; Xu, S.; Cao, Y.; Wang, Z.; Wei, Z. *An Empirical Comparison of Machine Learning and Deep Learning Models for Automated Fake News Detection.* Mathematics, 2025, 13, 2086.

Paper: https://doi.org/10.3390/math13132086

---

## Project Pipeline

```text
WELFake
   │
   ├── Data cleaning
   │   ├── missing-value removal
   │   ├── duplicate removal
   │   ├── lowercase
   │   ├── URL / HTML / mention / hashtag removal
   │   ├── punctuation / number removal
   │   ├── whitespace normalization
   │   └── stopword removal + lemmatization
   │
   ├── Input representations
   │   ├── Headline only
   │   ├── Content only
   │   └── Headline + Content
   │
   ├── Classical ML
   │   ├── Logistic Regression
   │   ├── Random Forest
   │   └── LightGBM
   │
   ├── Deep Learning
   │   ├── GRU
   │   └── ALBERT
   │
   └── Explainability
       └── Integrated Gradients
```

---

## Dataset

The experiments use `WELFake_Clean.csv` with the columns:

- `title`
- `text`
- `label`

The uploaded clean dataset initially contains **71,537 rows**. In the ALBERT pipeline, missing and duplicate records are removed before modeling, leaving **63,121 records** for that run.

The label convention used in the project is:

- `0` = Real
- `1` = Fake

The dataset is split using stratified sampling into:

- **60% training**
- **20% validation**
- **20% test**

The ALBERT run uses a maximum sequence length of **200 tokens**.

> Note: the classical-model notebook and the ALBERT notebook were run as separate experiments. Their intermediate row counts are therefore not identical.

---

## Preprocessing

The preprocessing pipeline includes:

1. Lowercasing
2. URL removal
3. HTML-tag removal
4. `@mention` removal
5. hashtag removal
6. punctuation removal
7. number removal
8. whitespace normalization
9. English stopword removal
10. WordNet lemmatization

For classical ML, the text is represented using **TF-IDF with up to 1,000 features and unigram + bigram features**.

---

## Models

### 1. Logistic Regression

A linear TF-IDF baseline with:

```text
max_iter = 1000
random_state = 42
```

### 2. Random Forest

```text
n_estimators = 100
random_state = 42
n_jobs = -1
```

### 3. LightGBM

```text
n_estimators = 100
learning_rate = 0.1
random_state = 42
n_jobs = -1
```

### 4. GRU

The combined headline + content GRU uses:

```text
Embedding dimension = 128
GRU hidden units     = 64
Dropout              = 0.3
Learning rate        = 0.001
Batch size           = 128
Maximum sequence     = 50 tokens
Epochs               = up to 10
Early stopping       = validation loss
```

The combined-input GRU achieved approximately:

```text
Accuracy       0.9550
Macro Precision 0.9553
Macro Recall    0.9548
Macro F1        0.9549
ROC-AUC         0.9884
```

### 5. ALBERT

The transformer model is:

```text
Model: albert-base-v2
Input: Headline + Content
Maximum sequence length: 200
Batch size: 16
Learning rate: 2e-5
Epochs: 5
Weight decay: 0.01
Warmup ratio: 0.10
Optimizer: AdamW
Mixed precision: FP16 on GPU
```

The best checkpoint was selected using **validation weighted F1**.

The best validation checkpoint occurred at **epoch 4**:

```text
Validation weighted F1 = 0.9864
```

Final test performance:

```text
Accuracy  = 0.9863
Precision = 0.9863
Recall    = 0.9860
F1        = 0.9861
ROC-AUC   = 0.9980
```

The final ALBERT confusion matrix was:

```text
[[6884,   75],
 [  98, 5568]]
```

---

## Explainability

The project adds explainability for the deep model using **Integrated Gradients** through Captum.

### Why Integrated Gradients?

The reference paper attempted SHAP-based explanations for its deep models but reported that the computation was prohibitively expensive and did not finish after more than 100 hours.

Instead of reproducing an impractical SHAP run, this project uses a gradient-based attribution method that can provide token-level explanations on a practical timescale.

For ALBERT:

```text
Positive attribution → pushes the prediction toward FAKE
Negative attribution → pushes the prediction toward REAL
```

The explainability workflow includes:

- single-article local explanation
- 15-article sanity check
- 50-article runtime benchmark
- 500-article explainability run
- token-level attribution CSV output
- summary CSV output

### Runtime benchmark

Using:

```text
Integrated Gradients steps = 16
```

the benchmark produced:

```text
50 articles  → 36.01 seconds
```

The actual 500-article run completed in:

```text
500 articles → 282.99 seconds
               ≈ 4.72 minutes
               ≈ 0.57 seconds/article
```

The explainability outputs generated by the notebook are:

```text
ALBERT_500_explainability_summary.csv
ALBERT_500_explainability_tokens.csv
```

---

## Repository Contents

| File | Description |
|---|---|
| `Copy_of_ML_CAPSTONE.ipynb` | Main notebook containing preprocessing, classical ML, GRU and the original ALBERT workflow |
| `ALBERT (3).ipynb` | Dedicated ALBERT notebook with the cleaned data pipeline, 5-epoch training, final evaluation and Integrated Gradients experiments |
| `vertopal.com_Copy_of_ML_CAPSTONE.pdf` | PDF export of the main ML/GRU notebook |
| `vertopal.com_ALBERT (3).pdf` | PDF export of the ALBERT notebook including the explainability experiments |
| `README.md` | Project documentation |

---

## Running the Project

The notebooks were developed for **Google Colab**.

### Requirements

- Google Colab
- NVIDIA GPU recommended for GRU/ALBERT
- WELFake dataset in CSV format
- Python 3
- TensorFlow
- PyTorch
- Transformers
- scikit-learn
- LightGBM
- NLTK
- Captum

### Basic workflow

1. Open the appropriate notebook in Google Colab.
2. Upload `WELFake_Clean.csv`.
3. Run the preprocessing and train/validation/test split cells.
4. Run the classical/GRU notebook for the baseline experiments.
5. Run the ALBERT notebook for transformer fine-tuning.
6. Run the final explainability cells to generate Integrated Gradients attributions.

---

## Evaluation Metrics

The project uses:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

For the ALBERT final evaluation, precision, recall and F1 are reported using **macro averaging**, matching the project's final evaluation code.

Validation model selection uses **weighted F1**.

---

## Explainability Interpretation

Integrated Gradients produces an attribution value for each input token.

Conceptually:

```text
                 ALBERT
                   │
             Prediction
                   │
          Integrated Gradients
                   │
        ┌──────────┴──────────┐
        │                     │
 Positive attribution    Negative attribution
        │                     │
   Supports FAKE          Supports REAL
```

This makes it possible to inspect which words or subword tokens most influenced individual predictions.

---

## Important Reproducibility Notes

This repository contains the actual experimental notebooks used for the project rather than claiming a perfectly identical reproduction of every hidden training detail of the reference paper.

In particular:

- The ALBERT notebook uses a fixed 5-epoch training run with the best validation checkpoint retained.
- Several implementation details are necessarily engineering choices because the reference paper does not specify every low-level training parameter.
- Explainability is an additional project analysis using Integrated Gradients rather than a reproduction of the paper's unfinished SHAP experiment.
- Neural-network results can vary slightly across Colab GPU sessions because of hardware and training nondeterminism.

---

## Reference

Tian, Y.; Xu, S.; Cao, Y.; Wang, Z.; Wei, Z.  
**An Empirical Comparison of Machine Learning and Deep Learning Models for Automated Fake News Detection.**  
*Mathematics* 2025, 13, 2086.  
https://doi.org/10.3390/math13132086

---

## License

This repository is intended for academic/research use. Check the licensing terms of the underlying WELFake dataset and third-party pretrained models before redistribution.
