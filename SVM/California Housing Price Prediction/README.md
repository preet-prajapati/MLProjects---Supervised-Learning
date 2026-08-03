# California Housing Price Prediction

Predicting median house values for California districts (1990 census, 20,640
block groups) using Support Vector Regression, benchmarked against Linear
Regression and Random Forest baselines.

## Results

| Model               | R²     | RMSE   | MAE    |
|---------------------|--------|--------|--------|
| Linear Regression   | 0.645  | 0.682  | 0.500  |
| SVR (default)       | 0.762  | 0.558  | 0.374  |
| SVR (tuned)         | 0.768  | 0.552  | 0.368  |
| **Random Forest**   | **0.805** | **0.506** | **0.328** |

*(RMSE/MAE in units of $100,000)*

## What's inside

- Full EDA: target/feature distributions, correlation heatmap, a
  latitude/longitude scatter colored by price (clearly showing the Bay Area
  and LA/San Diego price clusters), and outlier diagnostics.
- Missing-value imputation and 99th-percentile outlier capping.
- Baselines (Linear Regression, Random Forest) to contextualize SVR's result.
- SVR hyperparameter tuning via `RandomizedSearchCV` on a subsample (to keep
  runtime tractable) followed by a refit on the full training set.
- Model comparison table + bar charts.
- Residual analysis and permutation feature importance.
- Discussion of the dataset's $500k top-coding artifact and its effect on
  achievable accuracy.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook CaliforniaHousingPricePrediction.ipynb
```

The notebook tries `sklearn.datasets.fetch_california_housing()` first and
falls back to a public CSV mirror of the same source data if that download
is blocked (e.g. in restricted network environments).

## Dataset

California Housing dataset, derived from the 1990 U.S. Census (StatLib /
Pace & Barry, 1997). 8 features, 20,640 rows, target is median house value
in units of $100,000, top-coded at $500,000.
