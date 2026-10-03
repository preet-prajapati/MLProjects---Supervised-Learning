# Breast Cancer Classification

Classify benign and malignant records from numeric cell measurements.

| Item | Specification |
| --- | --- |
| Task | Classification |
| Estimator | Support vector classifier |
| Target | `diagnosis` |
| Validation | 80/20 holdout, seed 42; 3-fold shuffled training-only cross-validation |
| Baseline | Most-frequent class |

## Run

From the repository root after the [environment setup](../../README.md#quick-start):

```bash
python -m ml_portfolio list
python -m ml_portfolio run breast-cancer
```

Explore [the notebook](notebooks/analysis.ipynb) for training-only EDA, validation, diagnostics, and example predictions. The notebook and CLI use the same tested implementation in `src/ml_portfolio`.

## Evaluation and interpretation

Select hyperparameters using macro F1 on training folds; report accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix on the held-out test set. Binary projects also report positive-class precision, recall, F1, and ROC AUC.

Preprocessing is fitted inside each cross-validation fold. Exact repeated predictor rows are deduplicated before splitting; conflicting labels for identical predictors are excluded and counted. IDs are excluded. All metrics, cleaning counts, dataset fingerprints, versions, and selected parameters are saved to `reports/breast-cancer/metrics.json`.

See [measured results](../../reports/RESULTS.md). A score on one holdout is evidence about this dataset, not a guarantee on new populations.

## Reproduced result

On 114 held-out records, **macro_f1 = 0.9615**, compared with **0.3871** for the dummy baseline.

Malignant-class recall is about 0.905 on 42 malignant test records. Four malignant records are missed despite high overall accuracy; the confusion matrix is essential to interpretation.

![Held-out diagnostics](../../reports/breast-cancer/diagnostics.png)

## Limitations

Small benchmark dataset without prospective validation. Malignant is the positive class. This is an educational model, not a medical device.

## Interview discussion

- Explain why this model suits the problem and compare it with the dummy baseline.
- Walk through preprocessing and show how the training pipeline prevents leakage.
- Use the diagnostic plot to discuss errors and their practical consequences.
- Propose an independent validation set and justify the next experiment before deployment.

[Back to portfolio](../../README.md)
