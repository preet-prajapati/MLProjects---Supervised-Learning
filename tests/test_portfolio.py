import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ml_portfolio.data import (
    PROJECTS,
    ROOT,
    config,
    dataset_path,
    load_project,
    prepare_frame,
    split_data,
)
from ml_portfolio.evaluate import fit_project, save_result, write_summary
from ml_portfolio.models import build_model


@pytest.mark.parametrize("project", PROJECTS[:-1])
def test_clean_data_has_no_predictor_overlap(project):
    X, y, audit = load_project(project)
    train, test, y_train, y_test = split_data(project, X, y)
    assert len(X) == audit["usable_rows"]
    assert len(train) + len(test) == len(X)
    assert set(train.index).isdisjoint(test.index)
    train_hash = pd.util.hash_pandas_object(train, index=False)
    test_hash = pd.util.hash_pandas_object(test, index=False)
    assert set(train_hash).isdisjoint(test_hash)
    if config(project)["task"] == "classification":
        assert set(y_train) == set(y_test) == set(y)


def test_student_excludes_outcome_proxy_and_identifier():
    X, _, _ = load_project("student-grades")
    assert "GPA" not in X and "StudentID" not in X


def test_heart_removes_repeated_records():
    _, _, audit = load_project("heart-disease")
    assert audit["exact_duplicates_removed"] > 0
    assert audit["usable_rows"] < audit["raw_rows"]


def test_housing_rejects_incompatible_fallback_schema():
    with pytest.raises(ValueError, match="schema"):
        prepare_frame("california-housing", pd.DataFrame({"median_house_value": [1] * 30}))


def test_missing_housing_gives_explicit_download_instruction(tmp_path):
    spec_dir = tmp_path / "projects" / "california-housing"
    spec_dir.mkdir(parents=True)
    (spec_dir / "project.json").write_text(json.dumps(config("california-housing")))
    with pytest.raises(FileNotFoundError, match="download-housing"):
        load_project("california-housing", tmp_path)


def test_vocabulary_does_not_learn_heldout_text():
    pipeline, _ = build_model("sms-spam", None)
    pipeline.fit(["normal message", "win prize", "ordinary text", "free winner"], [0, 1, 0, 1])
    pipeline.predict(["unseenwordonly"])
    assert "unseenwordonly" not in pipeline.named_steps["vectorizer"].vocabulary_


def test_preprocessing_uses_training_statistics_and_handles_unknown_categories():
    X = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0], "category": ["a", "b", "a", "b"]})
    pipeline, _ = build_model("insurance-cost", X)
    pipeline.fit(X, [1.0, 2.0, 3.0, 4.0])
    imputer = (
        pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"]
    )
    assert imputer.statistics_[0] == 2.0
    assert np.isfinite(
        pipeline.predict(pd.DataFrame({"value": [999.0], "category": ["new"]}))
    ).all()
    assert imputer.statistics_[0] == 2.0
    clone(pipeline)


def test_duplicate_and_conflicting_sms_are_excluded():
    raw = pd.DataFrame({"v1": ["ham", "spam"] * 15, "v2": [f"message {i}" for i in range(30)]})
    raw = pd.concat([raw, raw.iloc[[0]], pd.DataFrame({"v1": ["spam"], "v2": ["message 0"]})])
    X, _, audit = prepare_frame("sms-spam", raw)
    assert audit["exact_duplicates_removed"] == 1
    assert audit["conflicting_rows_removed"] == 2
    assert "message 0" not in X.values


def test_end_to_end_report(tmp_path):
    result = fit_project("insurance-cost")
    save_result(result, tmp_path / "insurance-cost")
    write_summary(tmp_path)
    report = json.loads((tmp_path / "insurance-cost" / "metrics.json").read_text())
    assert report["test_metrics"]["rmse"] < report["baseline_metrics"]["rmse"]
    assert (tmp_path / "insurance-cost" / "diagnostics.png").stat().st_size > 1000
    assert "Medical Insurance" in (tmp_path / "RESULTS.md").read_text()


def test_all_project_notebooks_and_data_guides_exist():
    for project in PROJECTS:
        folder = ROOT / "projects" / project
        assert (folder / "README.md").exists()
        assert (folder / "notebooks" / "analysis.ipynb").exists()
        assert (folder / "data" / "README.md").exists()
        if project != "california-housing":
            assert dataset_path(project).exists()
