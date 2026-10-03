# Reproduced results

Fixed 80/20 holdouts, seed 42. Model selection uses 3-fold training-only CV. These are educational benchmark results, not production claims. Different datasets and metrics must not be ranked against each other.

| Project | Test rows | Metric | Model | Dummy baseline |
| --- | ---: | --- | ---: | ---: |
| [Breast Cancer Classification](breast-cancer/metrics.json) | 114 | macro_f1 | 0.9615 | 0.3871 |
| [California Housing Value Prediction](california-housing/metrics.json) | 4128 | rmse | 0.5668 | 1.1449 |
| [Car Price Prediction](car-price/metrics.json) | 38 | rmse | 2339.1652 | 7530.5271 |
| [Heart Disease Classification](heart-disease/metrics.json) | 61 | macro_f1 | 0.8195 | 0.3511 |
| [Medical Insurance Cost Prediction](insurance-cost/metrics.json) | 267 | rmse | 3529.7627 | 11853.2428 |
| [SMS Spam Detection](sms-spam/metrics.json) | 1032 | macro_f1 | 0.9711 | 0.4669 |
| [Student Grade Classification](student-grades/metrics.json) | 479 | macro_f1 | 0.5425 | 0.1346 |

Higher macro F1 is better; lower RMSE is better. Housing RMSE is in $100,000 units. Car and insurance targets retain the raw dataset units, whose provenance is unverified. Each project folder includes a diagnostic plot, CV results, data fingerprint, cleaning counts, selected parameters, and package versions.
