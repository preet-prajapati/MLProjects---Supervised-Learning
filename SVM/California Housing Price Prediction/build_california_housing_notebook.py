import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent
notebook_path = project_dir / "CaliforniaHousingPricePrediction.ipynb"

cells = []

def add_markdown(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    })


def add_code(code: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")],
    })

add_markdown("# California Housing Price Prediction\n\nThis notebook demonstrates a professional regression workflow for the California Housing dataset. It includes EDA, clean train/test splitting, feature engineering, baseline comparison, hyperparameter search, model evaluation, and interpretability.")
add_markdown("## Project goals\n\n- Predict median California house values using a robust regression workflow.\n- Avoid data leakage using stratified sampling by income category.\n- Compare SVR against strong baselines and tune the model with cross-validation.\n- Present clean visualizations, reusable functions, and clear evaluation metrics.")
add_code("""import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedShuffleSplit,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
""")
add_markdown("## Imports and logging\n\nWe configure a reproducible data science environment and a simple logger for runtime feedback.")
add_code("""def load_housing_data() -> pd.DataFrame:
    dataset_path = Path('data') / 'california_housing.csv'
    if dataset_path.exists():
        logging.info('Loading dataset from %s', dataset_path)
        return pd.read_csv(dataset_path)

    try:
        from sklearn.datasets import fetch_california_housing

        logging.info('Downloading California Housing dataset using scikit-learn')
        frame = fetch_california_housing(as_frame=True).frame
        frame.columns = frame.columns.str.replace(' ', '_')
        return frame
    except Exception as exc:
        logging.warning('scikit-learn dataset fetch failed: %s', exc)
        fallback_url = (
            'https://raw.githubusercontent.com/ageron/handson-ml3/'
            'main/datasets/housing/housing.csv'
        )
        logging.info('Loading California Housing dataset from fallback URL: %s', fallback_url)
        try:
            return pd.read_csv(fallback_url)
        except Exception as fallback_exc:
            raise RuntimeError(
                'Unable to load the California Housing dataset. '
                'Provide data/california_housing.csv or enable network access for the fallback URL.'
            ) from fallback_exc


def add_income_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['income_cat'] = pd.cut(
        df['MedInc'],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    return df


def stratified_train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(df, df['income_cat']):
        train_set = df.loc[train_index].copy()
        test_set = df.loc[test_index].copy()
    for dataset in (train_set, test_set):
        dataset.drop(columns=['income_cat'], inplace=True)
    return train_set, test_set


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
    df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
    df['population_per_household'] = df['Population'] / df['AveOccup']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.fillna(0)


def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline([
        ('feature_engineering', FunctionTransformer(add_features, validate=False)),
        ('scaler', StandardScaler())
    ])


def evaluate_regression_model(name: str, model, X: pd.DataFrame, y: pd.Series):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    logging.info('%s test results: R2=%.4f RMSE=%.4f MAE=%.4f', name, r2, rmse, mae)
    return {
        'model': name,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'predictions': y_pred
    }


def cross_validate_model(name: str, model, X: pd.DataFrame, y: pd.Series, cv: int = 5):
    scoring = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']
    scores = {
        'R2': cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1),
        'RMSE': -cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1),
        'MAE': -cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1),
    }
    print(f'{name} cross-validation results:')
    for metric, values in scores.items():
        print(f'  {metric}: {values.mean():.4f} ± {values.std():.4f}')
    return scores
""")
add_markdown("## Helper functions\n\nThese functions load the dataset, perform stratified splitting, generate feature ratios, and standardize the pipeline. This keeps the notebook reusable and avoids duplicated preprocessing logic.")
add_code("""housing = load_housing_data()
housing.head()
""")
add_markdown("## Dataset overview\n\nLoad the California Housing dataset and inspect the first rows.")
add_code("""housing.info()
""")
add_markdown("### Missing values and duplicates\n\nWe verify that the dataset contains no missing or duplicate records before modeling.")
add_code(r"""print('Missing values per column:')
print(housing.isnull().sum())
print('\nDuplicate rows:', housing.duplicated().sum())
""")
add_code("""housing.describe().T
""")
add_markdown("### Feature distributions\n\nWe visualize the distribution of key numeric features and the target variable.")
add_code("""fig, axes = plt.subplots(3, 3, figsize=(18, 14))
columns = housing.columns.tolist()
for ax, col in zip(axes.flatten(), columns):
    sns.histplot(housing[col], ax=ax, kde=True, color='steelblue')
    ax.set_title(col)
plt.tight_layout()
""")
add_markdown("### Correlation matrix\n\nA correlation heatmap identifies the strongest linear relationships and guides model interpretation.")
add_code("""corr_matrix = housing.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0)
plt.title('Correlation Matrix')
plt.show()
""")
add_markdown("### Geographic distribution\n\nScatter plots show how median house values vary with location.")
add_code("""plt.figure(figsize=(10, 7))
plt.scatter(housing['Longitude'], housing['Latitude'], c=housing['MedHouseVal'], cmap='viridis', s=15, alpha=0.8)
plt.colorbar(label='Median House Value ($100k)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('House Value by Location')
plt.show()
""")
add_code("""plt.figure(figsize=(10, 7))
prices_by_location = housing.groupby(['Latitude', 'Longitude'])['MedHouseVal'].median().reset_index()
plt.scatter(prices_by_location['Longitude'], prices_by_location['Latitude'],
            c=prices_by_location['MedHouseVal'], cmap='viridis', s=20)
plt.colorbar(label='Median House Value ($100k)')
plt.title('Median House Value by Location')
plt.show()
""")
add_markdown("## Train-test split\n\nWe apply stratified sampling on income category to preserve the distribution of household income in both the train and test sets.")
add_code(r"""housing_with_cat = add_income_category(housing)
train_set, test_set = stratified_train_test_split(housing_with_cat)
print('Train shape:', train_set.shape)
print('Test shape :', test_set.shape)
print('\nIncome category distribution in train set:')
print(housing_with_cat.loc[train_set.index, 'income_cat'].value_counts(normalize=True))
""")
add_markdown("## Prepare data for modeling\n\nDrop the target from the training set and build a reusable preprocessing pipeline.")
add_code("""X_train = train_set.drop('MedHouseVal', axis=1)
y_train = train_set['MedHouseVal']
X_test = test_set.drop('MedHouseVal', axis=1)
y_test = test_set['MedHouseVal']

preprocessor = build_preprocessing_pipeline()
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)
print('Prepared feature shape:', X_train_prepared.shape)
""")
add_markdown("## Baseline models\n\nWe compare Linear Regression, Random Forest, and Support Vector Regression with the same preprocessing pipeline.")
add_code("""lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])
svr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])

baseline_models = {
    'Linear Regression': lr_pipeline,
    'Random Forest': rf_pipeline,
    'SVR (default)': svr_pipeline,
}

baseline_scores = {}
for name, model in baseline_models.items():
    baseline_scores[name] = cross_validate_model(name, model, X_train, y_train)
""")
add_markdown("## Fit baseline models on full training data\n\nThe baseline pipelines are fit on the complete training set so we can compare hold-out test performance.")
add_code("""fitted_models = {}
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model

results = []
for name, model in fitted_models.items():
    evaluation = evaluate_regression_model(name, model, X_test, y_test)
    results.append(evaluation)

results_df = pd.DataFrame(results).drop(columns=['predictions'])
results_df
""")
add_markdown("## Hyperparameter tuning\n\nUse randomized search to find an improved SVR configuration without overfitting the training data.")
add_code("""param_distributions = {
    'regressor__C': [0.5, 1.0, 2.5, 5.0, 10.0, 20.0],
    'regressor__epsilon': [0.01, 0.05, 0.1, 0.2, 0.3],
    'regressor__kernel': ['rbf', 'linear'],
    'regressor__gamma': ['scale', 'auto'],
}
search_pipeline = Pipeline([
    ('preprocessor', build_preprocessing_pipeline()),
    ('regressor', SVR())
])

random_search = RandomizedSearchCV(
    search_pipeline,
    param_distributions,
    n_iter=25,
    scoring='neg_root_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)
print('Best parameters:', random_search.best_params_)
print('Best CV RMSE:', -random_search.best_score_)
""")
add_markdown("## Evaluate the tuned SVR model\n\nCompare the tuned SVR against the baseline models on the hold-out test set.")
add_code("""best_svr = random_search.best_estimator_
best_svr_results = evaluate_regression_model('SVR (tuned)', best_svr, X_test, y_test)
results_df = pd.concat([
    results_df,
    pd.DataFrame([ {k: v for k, v in best_svr_results.items() if k != 'predictions'} ])
], ignore_index=True)
results_df
""")
add_markdown("## Residual analysis\n\nResidual plots help detect heteroscedasticity and systematic prediction errors.")
add_code("""y_pred_best = best_svr.predict(X_test)
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_best, y=residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Median House Value ($100k)')
plt.ylabel('Residual')
plt.title('Residuals vs Predicted Values')
plt.show()
""")
add_code("""plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='purple', bins=30)
plt.title('Residual Distribution')
plt.xlabel('Residual')
plt.show()
""")
add_markdown("## Permutation feature importance\n\nThis model-agnostic check highlights the features that most affect SVR predictions.")
add_code("""perm_importance = permutation_importance(
    best_svr, X_test, y_test, n_repeats=15, random_state=42, n_jobs=-1
)
feature_names = X_test.columns.tolist()
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std,
}).sort_values(by='importance_mean', ascending=False)
importance_df.head(10)
""")
add_code("""plt.figure(figsize=(10, 6))
 sns.barplot(
    data=importance_df.head(10),
    x='importance_mean',
    y='feature',
    palette='mako'
)
plt.title('Top 10 Permutation Feature Importances')
plt.xlabel('Mean importance')
plt.ylabel('Feature')
plt.show()
""")
add_markdown("## Conclusion\n\nThe notebook compares a tuned SVR model with baseline regressors, uses stratified train/test splitting to reduce sampling bias, and evaluates performance with R², RMSE, and MAE. The model pipeline is reproducible and avoids data leakage by applying all feature transformations inside sklearn pipelines.")

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.14'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

notebook_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Notebook written to {notebook_path}')
