# SMS Spam Detection

Identify spam messages while measuring missed spam and false alarms.

| Item | Specification |
| --- | --- |
| Task | Classification |
| Estimator | Multinomial Naive Bayes |
| Target | `label` |
| Validation | 80/20 holdout, seed 42; 3-fold shuffled training-only cross-validation |
| Baseline | Most-frequent class |

## Run

From the repository root after the [environment setup](../../README.md#quick-start):

```bash
python -m ml_portfolio list
python -m ml_portfolio run sms-spam
```

Explore [the notebook](notebooks/analysis.ipynb) for training-only EDA, validation, diagnostics, and example predictions. The notebook and CLI use the same tested implementation in `src/ml_portfolio`.

## Evaluation and interpretation

Select hyperparameters using macro F1 on training folds; report accuracy, macro F1, per-class precision/recall/F1, and a confusion matrix on the held-out test set. Binary projects also report positive-class precision, recall, F1, and ROC AUC.

Preprocessing is fitted inside each cross-validation fold. Exact repeated predictor rows are deduplicated before splitting; conflicting labels for identical predictors are excluded and counted. IDs are excluded. All metrics, cleaning counts, dataset fingerprints, versions, and selected parameters are saved to `reports/sms-spam/metrics.json`.

See [measured results](../../reports/RESULTS.md). A score on one holdout is evidence about this dataset, not a guarantee on new populations.

## Reproduced result

On 1,032 held-out records, **macro_f1 = 0.9711**, compared with **0.4669** for the dummy baseline.

Positive-class precision is about 0.946 and recall about 0.953 on this split. Similar message templates may still occur across partitions; a campaign-grouped or newer corpus would test generalization more rigorously.

![Held-out diagnostics](../../reports/sms-spam/diagnostics.png)

## Limitations

Exact repeated messages are removed before splitting. Related templates can still cross partitions; language and campaign drift limit deployment.

## Interview discussion

- Explain why this model suits the problem and compare it with the dummy baseline.
- Walk through preprocessing and show how the training pipeline prevents leakage.
- Use the diagnostic plot to discuss errors and their practical consequences.
- Propose an independent validation set and justify the next experiment before deployment.

[Back to portfolio](../../README.md)
