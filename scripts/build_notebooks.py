"""Generate consistent, editable notebook walkthroughs from the project registry."""

import nbformat as nbf

from ml_portfolio.data import PROJECTS, ROOT, config


def build(project):
    spec = config(project)
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    setup = """from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from ml_portfolio.data import ROOT, config, load_project, split_data
from ml_portfolio.models import build_model
from ml_portfolio.evaluate import fit_project, save_result
"""
    cells = [
        md(
            f"# {spec['title']}\n\n{spec['objective']}\n\n"
            f"**Model:** {spec['model']} · **Target:** `{spec['target']}`\n\n"
            "Run the environment setup in the root README first, then use **Restart Kernel "
            "and Run All**. This notebook uses the same tested functions as the command line. "
            "Its output is reproducible from the committed dataset and pinned numerical dependencies."
        ),
        md(
            "## 1. Setup and data contract\n\nThe loader preserves raw files, excludes IDs and "
            "outcome proxies, removes identical predictor rows, and reports each cleaning step. "
            + (
                "First run `python -m ml_portfolio download-housing` in a terminal. "
                "Housing uses the full dataset; training can take several minutes."
                if project == "california-housing"
                else "No network is needed for this dataset."
            )
        ),
        code(
            setup + f'\nPROJECT = "{project}"\nspec = config(PROJECT)\n'
            'X, y, audit = load_project(PROJECT)\ndisplay(pd.Series(audit, name="Data audit"))'
        ),
        md(
            "## 2. Reserve the holdout before exploration\n\nThe fixed 80/20 split uses seed 42. "
            "Classification splits are stratified. EDA below uses only training records. "
            "The holdout is used only for the final evaluation; it does not select parameters."
        ),
        code(
            "X_train, X_test, y_train, y_test = split_data(PROJECT, X, y)\n"
            'print(f"Training: {len(y_train):,}; held out: {len(y_test):,}")\n'
            "display(X_train.head())\ndisplay(X_train.describe())"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            'if spec["task"] == "classification":\n'
            '    y_train.value_counts().sort_index().plot.bar(ax=ax, color="#247a94")\n'
            '    ax.set_ylabel("Training records")\n'
            "else:\n"
            '    ax.hist(y_train, bins=30, color="#247a94", edgecolor="white")\n'
            '    ax.set_ylabel("Training records")\n'
            'ax.set_xlabel(spec["target"])\nax.set_title("Training target distribution")\n'
            "display(fig)\nplt.close(fig)"
        ),
        md(
            "## 3. Pipeline and model selection\n\nNumeric features are median-imputed and "
            "scaled. Categorical features are imputed and one-hot encoded; unseen categories "
            "map to the all-zero encoding. SMS uses a vocabulary learned only from training "
            "messages. All transformations are fitted inside each CV fold.\n\n"
            "Three shuffled folds select macro F1 for classification or negative RMSE for "
            "regression. Linear regression has no tuning grid but receives the same CV evaluation."
        ),
        code(
            "pipeline, parameter_grid = build_model(PROJECT, X_train)\n"
            'display(pipeline)\nprint("Search space:", parameter_grid)'
        ),
        code(
            'result = fit_project(PROJECT)\nreport = result["report"]\n'
            'print("Selected parameters:", report["best_parameters"])\n'
            'print("CV scoring:", report["cv_scoring"])\n'
            'print("CV mean and standard deviation:", report["cv_best_score"], report["cv_score_std"])\n'
            'display(pd.DataFrame(result["search"].cv_results_)[\n'
            '    ["params", "mean_test_score", "std_test_score", "rank_test_score"]\n'
            '].sort_values("rank_test_score"))'
        ),
        md(
            "## 4. Held-out results and baseline\n\nThe dummy classifier predicts the most "
            "frequent training class; the dummy regressor predicts the training-target mean. "
            "Compare train and test performance to discuss overfitting. Macro F1 gives each "
            "class equal weight. Binary recall treats spam, malignant, or heart target 1 as "
            "positive. Regression errors retain the target units."
        ),
        code(
            'display(pd.DataFrame({"model_test": report["test_metrics"],\n'
            '                      "dummy_test": report["baseline_metrics"],\n'
            '                      "model_train": report["training_metrics"]}))\n'
            'if "classification_report" in report:\n'
            '    display(pd.DataFrame(report["classification_report"]).T)\n'
            'output = ROOT / "reports" / PROJECT\nsave_result(result, output)\n'
            'display(Image(filename=str(output / "diagnostics.png")))'
        ),
        md(
            "## 5. Example predictions\n\nThese are held-out examples, not a production API. "
            "Inference accepts the cleaned feature schema produced by the loader. Inspect "
            "the source to understand encoding and feature exclusions."
        ),
        code(
            'examples = result["X_test"].iloc[:5]\n'
            "display(examples)\ndisplay(pd.DataFrame({\n"
            '    "actual": result["y_test"].iloc[:5].to_numpy(),\n'
            '    "predicted": result["model"].predict(examples),\n}))'
        ),
        md(
            f"## 6. Interpretation and next experiment\n\n{spec['limitations']}\n\n"
            "Use the confusion matrix or residuals to identify the most costly errors. "
            "Before client deployment, verify data rights and feature availability, evaluate "
            "an independent population, assess subgroup performance where relevant, and "
            "define monitoring and acceptance criteria.\n\n"
            "[Project guide](../README.md) · [Portfolio](../../../README.md) · "
            "[Evaluation protocol](../../../docs/METHODOLOGY.md)"
        ),
    ]
    feature = {
        "car-price": "horsepower",
        "insurance-cost": "bmi",
        "heart-disease": "age",
        "student-grades": "Absences",
        "breast-cancer": "radius_mean",
        "california-housing": "MedInc",
        "sms-spam": "message_length",
    }[project]
    eda = (
        "training = X_train.to_frame() if X_train.ndim == 1 else X_train.copy()\n"
        'training["target"] = y_train\n'
        + (
            'training["message_length"] = training["message"].str.len()\n'
            if project == "sms-spam"
            else ""
        )
        + f'FEATURE = "{feature}"\n'
        + "fig, ax = plt.subplots(figsize=(8, 4))\n"
        + 'if spec["task"] == "classification":\n'
        + '    for label, group in training.groupby("target"):\n'
        + "        ax.hist(group[FEATURE], bins=20, alpha=0.45, density=True, label=str(label))\n"
        + '    ax.legend(title="Target class")\n'
        + '    ax.set_ylabel("Density")\n'
        + "else:\n"
        + '    ax.scatter(training[FEATURE], training["target"], s=10, alpha=0.35)\n'
        + '    ax.set_ylabel(spec["target"])\n'
        + 'ax.set_xlabel(FEATURE)\nax.set_title("Training-only feature exploration")\n'
        + "display(fig)\nplt.close(fig)\n"
        + 'display(training.isna().sum().rename("Training missing values"))'
    )
    cells[6:6] = [
        md(
            "### Explore a relevant predictor\n\nCompare feature distributions on "
            "the training partition. Association does not establish causation. "
            "The model's feature set is specified in the source, not selected using "
            "test-set plots."
        ),
        code(eda),
    ]
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    # Stable cell IDs avoid noisy diffs when rebuilding notebooks.
    for i, cell in enumerate(nb.cells):
        cell.id = f"{project}-{i:02d}"
    path = ROOT / "projects" / project / "notebooks" / "analysis.ipynb"
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


if __name__ == "__main__":
    for project in PROJECTS:
        build(project)
    print(f"Generated {len(PROJECTS)} notebooks")
