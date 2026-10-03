"""All learned preprocessing lives inside the cross-validated pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeRegressor

from .data import SEED


def build_model(project, X):
    if project == "sms-spam":
        return Pipeline([("vectorizer", CountVectorizer()), ("model", MultinomialNB())]), {
            "vectorizer__ngram_range": [(1, 1), (1, 2)],
            "model__alpha": [0.1, 1.0],
        }
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.columns.difference(numeric).tolist()
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False, drop="first"
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    choices = {
        "car-price": (LinearRegression(), {}),
        "insurance-cost": (
            DecisionTreeRegressor(random_state=SEED),
            {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 15]},
        ),
        "heart-disease": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 7, 15], "model__weights": ["uniform", "distance"]},
        ),
        "student-grades": (
            LogisticRegression(max_iter=3000, random_state=SEED),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "breast-cancer": (
            SVC(random_state=SEED),
            {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"]},
        ),
        "california-housing": (
            SVR(cache_size=512),
            {"model__C": [1.0, 10.0], "model__epsilon": [0.1, 0.2]},
        ),
    }
    model, grid = choices[project]
    return Pipeline([("preprocess", preprocessor), ("model", model)]), grid
