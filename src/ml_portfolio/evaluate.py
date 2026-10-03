"""Training-only model selection and honest held-out evaluation."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from .data import ROOT, SEED, config, load_project, split_data
from .models import build_model


def score_predictions(task, truth, predicted):
    if task == "regression":
        return {
            "mae": mean_absolute_error(truth, predicted),
            "rmse": root_mean_squared_error(truth, predicted),
            "r2": r2_score(truth, predicted),
        }
    scores = {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, average="macro", zero_division=0),
    }
    if set(truth.unique()) == {0, 1}:
        scores.update(
            precision=precision_score(truth, predicted, zero_division=0),
            recall=recall_score(truth, predicted, zero_division=0),
            f1=f1_score(truth, predicted, zero_division=0),
        )
    return scores


def fit_project(project, root=ROOT):
    spec = config(project, root)
    X, y, audit = load_project(project, root)
    X_train, X_test, y_train, y_test = split_data(project, X, y)
    estimator, grid = build_model(project, X_train)
    classification = spec["task"] == "classification"
    cv_type = StratifiedKFold if classification else KFold
    scoring = "f1_macro" if classification else "neg_root_mean_squared_error"
    search = GridSearchCV(
        estimator,
        grid,
        scoring=scoring,
        cv=cv_type(n_splits=3, shuffle=True, random_state=SEED),
        n_jobs=1,
        error_score="raise",
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    predicted = model.predict(X_test)
    baseline = DummyClassifier(strategy="most_frequent") if classification else DummyRegressor()
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    base_pred = baseline.predict(np.zeros((len(y_test), 1)))
    metrics = score_predictions(spec["task"], y_test, predicted)
    if classification and y.nunique() == 2:
        scores = (
            model.decision_function(X_test)
            if hasattr(model, "decision_function")
            else model.predict_proba(X_test)[:, 1]
        )
        metrics["roc_auc"] = roc_auc_score(y_test, scores)
    report = {
        "project": project,
        "title": spec["title"],
        "task": spec["task"],
        "model": spec["model"],
        "seed": SEED,
        "test_fraction": 0.2,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "data": audit,
        "cv_folds": 3,
        "cv_scoring": scoring,
        "cv_best_score": float(search.best_score_),
        "cv_score_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "best_parameters": search.best_params_,
        "test_metrics": metrics,
        "baseline_metrics": score_predictions(spec["task"], y_test, base_pred),
        "training_metrics": score_predictions(spec["task"], y_train, model.predict(X_train)),
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy"]},
        "python": platform.python_version(),
        "limitations": spec["limitations"],
    }
    if classification:
        report["classification_report"] = classification_report(
            y_test, predicted, output_dict=True, zero_division=0
        )
    return {
        "report": report,
        "model": model,
        "search": search,
        "X_test": X_test,
        "y_test": y_test,
        "predicted": predicted,
    }


def save_result(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = result["report"]
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    cv = pd.DataFrame(result["search"].cv_results_)
    cv[["params", "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(
        output / "cross_validation.csv", index=False
    )
    truth, predicted = result["y_test"], result["predicted"]
    if report["task"] == "classification":
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            truth, predicted, ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(report["title"])
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(truth, predicted, s=12, alpha=0.5)
        bounds = [min(truth.min(), predicted.min()), max(truth.max(), predicted.max())]
        axes[0].plot(bounds, bounds, "--", color="#c43c39")
        axes[0].set(xlabel="Actual target", ylabel="Predicted target", title="Held-out predictions")
        axes[1].scatter(predicted, truth - predicted, s=12, alpha=0.5)
        axes[1].axhline(0, color="#c43c39", linestyle="--")
        axes[1].set(xlabel="Predicted target", ylabel="Actual minus predicted", title="Residuals")
        fig.suptitle(report["title"])
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_summary(output):
    output = Path(output)
    rows = [
        "# Reproduced results",
        "",
        "Fixed 80/20 holdouts, seed 42. Model selection uses 3-fold training-only CV. "
        "These are educational benchmark results, not production claims. "
        "Different datasets and metrics must not be ranked against each other.",
        "",
        "| Project | Test rows | Metric | Model | Dummy baseline |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for path in sorted(output.glob("*/metrics.json")):
        report = json.loads(path.read_text())
        metric = "macro_f1" if report["task"] == "classification" else "rmse"
        rows.append(
            f"| [{report['title']}]({report['project']}/metrics.json) | "
            f"{report['test_rows']} | {metric} | "
            f"{report['test_metrics'][metric]:.4f} | {report['baseline_metrics'][metric]:.4f} |"
        )
    rows += [
        "",
        "Higher macro F1 is better; lower RMSE is better. Housing RMSE is in $100,000 "
        "units. Car and insurance targets retain the raw dataset units, whose provenance "
        "is unverified. Each project folder includes a diagnostic plot, CV results, data "
        "fingerprint, cleaning counts, selected parameters, and package versions.",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(rows), encoding="utf-8")
