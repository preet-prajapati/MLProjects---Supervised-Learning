# Evaluation protocol

## What is being measured

Each project evaluates one model family against a dummy baseline on an 80/20 holdout.
The seed is 42. Classification splits are stratified. Three shuffled training folds select
hyperparameters using macro F1 (classification) or negative RMSE (regression). Linear regression
has an empty parameter grid and is still evaluated by cross-validation. CV standard deviation
describes variation across these folds; it is not a confidence interval.

The holdout does not select models, transformations, or hyperparameters. Training-only EDA appears
in each notebook. Numeric imputation and scaling, categorical imputation and one-hot encoding,
and text vocabulary learning occur inside the pipeline fitted independently in each fold.
The reported model is refitted on all training rows after parameter selection.

## Data cleaning decisions

Raw CSVs are preserved. The loader strips column names, excludes identifiers, applies fixed
brand spelling corrections, rejects unsupported labels, removes missing targets, and removes
exact duplicates after feature exclusions. SMS also trims outer whitespace and removes empty
messages. Non-finite numeric values become missing values and are imputed during training.

After exact deduplication, all remaining rows with identical predictors and conflicting targets
are excluded. Their count is reported. This conservative policy avoids placing the same inputs
in train and test, but changes the evaluated population; it is not a universal data-cleaning rule.
The original rows remain in the raw CSVs for alternative analyses.

StudentID and GPA are excluded from grade classification: the former is an identifier and the
latter is a direct outcome proxy. The project predicts GradeClass 0–4, not a binary pass/fail flag.
Integer-coded nominal categories in the student and heart datasets are one-hot encoded.
The code does not infer undocumented clinical meanings from those codes.

Medical label mapping is explicit: malignant `M` is 1, benign `B` is 0. SMS spam is 1 and ham is 0.
Heart target 1 is evaluated as positive, subject to verification of the original data dictionary.
No automatic transformation invents a new outcome.

## Metrics and diagnostics

Classification reports include accuracy, macro F1, per-class precision/recall/F1, and a confusion
matrix. Binary tasks additionally include positive-class precision, recall, F1, and ROC AUC using
decision scores or probabilities. Default model thresholds are retained; thresholds are not
optimized against the test set. A most-frequent-class dummy model establishes the baseline.

Regression reports include MAE, RMSE, R², actual-versus-predicted points, and residual plots.
The baseline predicts the training-target mean. Housing values and errors use $100,000 units.
Car and insurance values retain their original units; currency and collection metadata are unverified.
Higher R² is better; lower MAE and RMSE are better.

Reports record training metrics to reveal overfitting, candidate search results, dataset checksum,
cleaning counts, exact selected parameters, and numerical-library versions. Small floating-point
differences may remain across operating systems and numerical libraries.

## Known limitations

- A single holdout is not repeated validation. Small test sets produce uncertain estimates.
- Deduplication does not resolve related individuals, SMS templates, households, or geographic groups.
- Housing's random split can overstate spatial generalization. A geographic holdout is a next experiment.
- Unknown categories use an all-zero encoding; with a dropped reference category this conflates them
  with that reference. Warnings are retained. New categories should be monitored before deployment.
- Fairness, feature availability at prediction time, calibration, drift, and independent-population
  performance have not been established. Demographic features require careful review for real use.
- Dataset sources and rights for inherited CSVs are unverified. No client outcomes or business impact
  figures are claimed.

The next iteration should verify provenance, define the decision cost, collect an independent
validation set, and predeclare acceptance criteria. Tuning against these published test results
would require a new final holdout.
