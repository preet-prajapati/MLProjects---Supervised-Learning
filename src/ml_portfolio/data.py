"""Dataset contracts, deterministic cleaning, and leakage-aware splitting."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = (
    "car-price",
    "insurance-cost",
    "heart-disease",
    "student-grades",
    "sms-spam",
    "breast-cancer",
    "california-housing",
)
SEED = 42
HOUSING_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]


def config(project, root=ROOT):
    if project not in PROJECTS:
        raise ValueError(f"Unknown project {project!r}; choose from {PROJECTS}")
    return json.loads((Path(root) / "projects" / project / "project.json").read_text())


def dataset_path(project, root=ROOT):
    return Path(root) / "projects" / project / "data" / "raw" / config(project, root)["dataset"]


def download_housing(root=ROOT):
    """Network access is explicit; never substitute a different dataset schema."""
    from sklearn.datasets import fetch_california_housing

    path = dataset_path("california-housing", root)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = fetch_california_housing(
        data_home=str(path.parent / "scikit_learn_data"), as_frame=True
    ).frame
    frame.to_csv(path, index=False)
    return path


def prepare_frame(project, raw):
    """Return model features, labels, and an auditable cleaning summary."""
    frame = raw.copy()
    frame.columns = frame.columns.str.strip()
    rows_raw = len(frame)
    target = "label" if project == "sms-spam" else config(project)["target"]
    if project == "sms-spam":
        frame = frame[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
        if not frame["label"].dropna().isin(["ham", "spam"]).all():
            raise ValueError("SMS labels must be ham or spam")
        frame["label"] = frame["label"].map({"ham": 0, "spam": 1})
        frame["message"] = frame["message"].astype("string").str.strip()
        frame.loc[frame["message"].eq(""), "message"] = pd.NA
        frame = frame.dropna(subset=["message"])
    elif project == "car-price":
        frame["brand"] = (
            frame["CarName"]
            .str.lower()
            .str.split()
            .str[0]
            .replace(
                {
                    "maxda": "mazda",
                    "porcshce": "porsche",
                    "toyouta": "toyota",
                    "vokswagen": "volkswagen",
                    "vw": "volkswagen",
                }
            )
        )
        frame = frame.drop(columns=["car_ID", "CarName"])
    elif project == "student-grades":
        # GPA is an outcome proxy; ID must not identify individual students.
        frame = frame.drop(columns=["StudentID", "GPA"])
        if not frame[target].dropna().isin(range(5)).all():
            raise ValueError("GradeClass must be an integer from 0 to 4")
        for col in ["Gender", "Ethnicity", "ParentalEducation", "ParentalSupport"]:
            frame[col] = frame[col].astype(str)
    elif project == "breast-cancer":
        frame = frame.drop(columns=["id", "Unnamed: 32"], errors="ignore")
        if not frame[target].dropna().isin(["B", "M"]).all():
            raise ValueError("Diagnosis must be B or M")
        frame[target] = frame[target].map({"B": 0, "M": 1})
    elif project == "heart-disease":
        if not frame[target].dropna().isin([0, 1]).all():
            raise ValueError("Heart target must be 0 or 1")
        for col in ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
            frame[col] = frame[col].astype(str)
    elif project == "california-housing":
        if set(frame.columns) != set(HOUSING_COLUMNS):
            raise ValueError("Expected the scikit-learn California housing schema")
        frame = frame[HOUSING_COLUMNS]
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=[target])
    rows_invalid = rows_raw - len(frame)
    before = len(frame)
    frame = frame.drop_duplicates()
    duplicates = before - len(frame)
    feature_columns = frame.columns.drop(target).tolist()
    # Identical inputs cannot be split across train and test, even with conflicting labels.
    conflicts = frame.duplicated(subset=feature_columns, keep=False)
    conflicting_rows = int(conflicts.sum())
    frame = frame.loc[~conflicts].reset_index(drop=True)
    if len(frame) < 20:
        raise ValueError("Fewer than 20 usable records remain")
    X = frame["message"] if project == "sms-spam" else frame.drop(columns=target)
    y = frame[target]
    if config(project)["task"] == "classification":
        y = y.astype(int)
        if y.nunique() < 2 or y.value_counts().min() < 5:
            raise ValueError("Each class needs at least five usable records")
    return (
        X,
        y,
        {
            "raw_rows": rows_raw,
            "invalid_rows_removed": rows_invalid,
            "exact_duplicates_removed": duplicates,
            "conflicting_rows_removed": conflicting_rows,
            "usable_rows": len(frame),
            "feature_count": 1 if X.ndim == 1 else X.shape[1],
        },
    )


def load_project(project, root=ROOT):
    path = dataset_path(project, root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {path}. "
            + (
                "Run: python -m ml_portfolio download-housing"
                if project == "california-housing"
                else "Restore the project's data/raw CSV from the repository."
            )
        )
    frame = pd.read_csv(path, encoding="latin-1" if project == "sms-spam" else "utf-8")
    X, y, audit = prepare_frame(project, frame)
    audit["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return X, y, audit


def split_data(project, X, y):
    stratify = y if config(project)["task"] == "classification" else None
    return train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=stratify)
