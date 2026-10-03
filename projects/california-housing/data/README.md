# California Housing Value Prediction: data

The dataset is fetched explicitly from the scikit-learn California housing loader. Run `python -m ml_portfolio download-housing` from the repository root. The download is cached locally and is not committed.

Expected file: `raw/california_housing.csv`. Target: `MedHouseVal`.

Keep raw data immutable. Cleaning happens in memory inside the shared loader.
