# Supervised Learning Portfolio

**Seven case studies. Six model families. One reproducible evaluation workflow.**

A Python and scikit-learn portfolio covering tabular regression, multiclass classification,
binary classification, and text classification. Each project connects a prediction problem
to an executable notebook, a tested training pipeline, measured results, and explicit limitations.

[Explore the projects](#project-gallery) · [Measured results](reports/RESULTS.md) ·
[Interview walkthrough](docs/PORTFOLIO_GUIDE.md) · [Evaluation protocol](docs/METHODOLOGY.md)

## Project gallery

| Project | Method | What it demonstrates |
| --- | --- | --- |
| [Car prices](projects/car-price/) | Linear regression | Mixed feature types, brand normalization, residual analysis |
| [Insurance costs](projects/insurance-cost/) | Decision tree regression | Nonlinear relationships, tree regularization, error in target units |
| [Heart disease](projects/heart-disease/) | k-nearest neighbors | Duplicate control, scaling, stratified validation |
| [Student grades](projects/student-grades/) | Logistic regression | Five-class prediction, outcome-proxy exclusion, macro F1 |
| [SMS spam](projects/sms-spam/) | Multinomial Naive Bayes | Training-only text vocabulary, n-grams, precision/recall tradeoffs |
| [Breast cancer](projects/breast-cancer/) | Support vector classification | Kernel selection, malignant-class recall, ROC AUC |
| [California housing](projects/california-housing/) | Support vector regression | Full-dataset regression, standardized inputs, geographic limitations |

## Start here

For a quick review, open the [results table](reports/RESULTS.md), choose a project above,
and read its executed notebook. The notebooks include visible results and plots for browsing on GitHub.
For a deeper review, inspect [the shared implementation](src/ml_portfolio/) and [tests](tests/).

This is an educational portfolio. Medical examples are not diagnostic systems; benchmark
performance is not evidence of readiness for client deployment. Dataset provenance and
redistribution terms still need verification for the six inherited CSVs.

## Quick start

Use **Python 3.11 or 3.12**. The project is designed for an editable installation from this clone.

```bash
git clone https://github.com/preet-prajapati/MLProjects---Supervised-Learning.git
cd MLProjects---Supervised-Learning
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

```bash
python -m pip install -r requirements-dev.txt
python -m ml_portfolio list
python -m ml_portfolio run sms-spam
python -m jupyterlab
```

In JupyterLab, open `projects/<project>/notebooks/analysis.ipynb` and select the environment's
Python kernel. **Restart Kernel and Run All** reproduces the walkthrough. The editable install
lets notebooks run from their own folders without fragile relative dataset paths.

## Reproduce the portfolio

```bash
# Six projects with bundled CSVs; no dataset downloads
python -m ml_portfolio run bundled

# Explicit one-time download, then all seven projects
python -m ml_portfolio download-housing
python -m ml_portfolio run all

# Quality checks and clean-kernel notebook execution
pytest -q
ruff check src tests scripts
ruff format --check src tests scripts
python scripts/check_notebooks.py
python scripts/check_notebooks.py --execute bundled
```

The California housing run uses all available records and can take several minutes.
The download command requires internet access; subsequent runs use the local CSV.
There is no silent fallback to a dataset with different columns or target units.

Runs save `metrics.json`, `cross_validation.csv`, and `diagnostics.png` under `reports/<project>/`.
The results table is rebuilt from available reports. Use `--output artifacts/my-run` for a separate run.
Notebook execution writes copies to `reports/executed/`; add `--in-place` to deliberately refresh
the committed notebooks. The source builder in `scripts/` resets notebook outputs when run.

## Repository structure

```text
.
├── projects/
│   └── <project>/
│       ├── README.md              # Problem, model, validation, limitations
│       ├── project.json           # Project metadata and dataset contract
│       ├── data/
│       │   ├── README.md          # Acquisition and provenance notes
│       │   └── raw/               # Immutable source CSV
│       └── notebooks/
│           └── analysis.ipynb     # Executed walkthrough
├── src/ml_portfolio/              # Data cleaning, pipelines, evaluation, CLI
├── tests/                        # Leakage, schema, and integration checks
├── scripts/                      # Notebook generation and validation
├── reports/                      # Measured metrics, diagnostics, CV results
├── docs/                         # Methodology, data register, presentation guide
├── .github/                      # Automated checks and contribution templates
├── pyproject.toml                # Package and pinned numerical dependencies
├── requirements.txt              # Notebook environment
└── requirements-dev.txt          # Testing and formatting tools
```

## Engineering and evaluation choices

- Reserve a fixed holdout before EDA; fit imputation, scaling, encoding, and vocabulary inside CV folds.
- Remove repeated predictor rows before splitting and report conflicting records separately.
- Select hyperparameters on training folds and compare held-out performance with a dummy model.
- Report per-class errors and regression residuals alongside headline scores.
- Record dataset SHA-256 fingerprints, cleaning counts, package versions, and selected parameters.
- Check Python 3.11 and 3.12 in CI; execute the six bundled notebooks on Python 3.12.
  The network-dependent housing job runs through **Actions → Portfolio quality → Run workflow**.

## Further reading

- [Methodology and limitations](docs/METHODOLOGY.md)
- [Data provenance register](docs/DATASETS.md)
- [Client and interview presentation guide](docs/PORTFOLIO_GUIDE.md)
- [Migration notes and original project paths](docs/MIGRATION.md)
- [Contributing](CONTRIBUTING.md)
- [Reuse and licensing status](NOTICE.md)

Maintained by [Preet Prajapati](https://github.com/preet-prajapati).
