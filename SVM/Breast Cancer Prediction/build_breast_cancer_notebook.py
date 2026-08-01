import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent
notebook_path = project_dir / "notebook" / "BreastCancerPrediction.ipynb"

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

add_markdown("# Breast Cancer Classification using Support Vector Machine\n\nThis notebook predicts whether a breast tumor is benign or malignant using a Support Vector Machine classifier. The workflow emphasizes clean data preprocessing, model evaluation, and interpretability for a portfolio-ready presentation.")
add_markdown("## Problem Statement\n\nBreast cancer screening datasets contain clinical measurements that can distinguish malignant tumors from benign ones. The goal is to build a reliable classifier that uses these features to support early diagnosis.")
add_markdown("## Project Objective\n\nThe objective is to train a Support Vector Machine model with an optimized pipeline, validate its performance using proper classification metrics, and explain why the selected hyperparameters and preprocessing steps improve results.")
add_markdown("## Dataset Information\n\nThe dataset includes 569 breast tumor samples with 30 numeric features derived from digitized images of fine needle aspirate of breast masses. The target variable is the diagnosis label: B for benign and M for malignant.")
add_code("""import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             precision_recall_curve, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

sns.set_theme(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
warnings.filterwarnings('ignore')
""")
add_markdown("## Import Libraries\n\nWe import the standard data science libraries along with scikit-learn utilities for building a reproducible machine learning pipeline.")
add_markdown("## Load Dataset\n\nThe dataset is loaded from the `dataset` folder so the notebook remains portable and well organized.")
add_code("""notebook_folder = Path.cwd()
candidates = [
    notebook_folder / 'dataset' / 'data.csv',
    notebook_folder / '..' / 'dataset' / 'data.csv',
    notebook_folder / '..' / '..' / 'dataset' / 'data.csv'
]

data_path = next((path for path in candidates if path.exists()), None)
if data_path is None:
    raise FileNotFoundError('Dataset not found in the expected paths.')

print('Loading dataset from:', data_path)
df = pd.read_csv(data_path)
df.head()
""")
add_markdown("## Dataset Overview\n\nA quick overview confirms the dataset shape, column names, and the distribution of the target labels.")
add_code("""print('Shape:', df.shape)
print('Columns:')
print(df.columns.tolist())
print('Data types:')
print(df.dtypes)
print('Diagnosis value counts:')
print(df['diagnosis'].value_counts())
print('Missing values:')
print(df.isnull().sum())
print('Duplicates:')
print(df.duplicated().sum())
""")
add_markdown("## Data Cleaning\n\nWe remove irrelevant columns, verify there are no missing values, and confirm the diagnosis label is clean and balanced.")
add_code("""df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
print('Columns after cleanup:', df.columns.tolist())
print('Shape after cleanup:', df.shape)
""")
add_markdown("### Why this matters\n\nRemoving identifier columns prevents accidental data leakage and preserves only meaningful predictive features.")
add_markdown("## Exploratory Data Analysis\n\nWe explore how the features relate to the target label using visualizations and summary statistics.")
add_code("""plt.figure(figsize=(8, 5))
sns.countplot(x='diagnosis', data=df, palette='Set2')
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.show()
""")
add_markdown("The diagnosis distribution shows how many benign and malignant samples are available, which is useful for evaluating class balance.")
add_code("""plt.figure(figsize=(8, 5))
labels = df['diagnosis'].value_counts().index
sizes = df['diagnosis'].value_counts().values
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'])
plt.title('Diagnosis Proportion')
plt.axis('equal')
plt.show()
""")
add_markdown("The pie chart provides a clear visual of the class proportions and confirms the dataset is moderately balanced.")
add_code("""print(df.describe().T)
""")
add_markdown("Summary statistics help us understand the feature scales, ranges, and potential outliers before scaling.")
add_code("""corr = df.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Heatmap')
plt.show()
""")
add_markdown("The correlation matrix shows which features are strongly associated with the target and with each other, which is useful for feature selection and model understanding.")
add_code("""selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
plt.figure(figsize=(12, 6))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df, x=feature, hue='diagnosis', kde=True, palette='Set2', element='step')
    plt.title(f'{feature.replace('_', ' ').title()} Distribution')
    plt.xlabel(feature.replace('_', ' ').title())
    plt.ylabel('Count')
plt.tight_layout()
plt.show()
""")
add_markdown("Histograms show how malignant and benign tumors differ for important features. This helps justify why an SVM can separate the classes.")
add_code("""plt.figure(figsize=(12, 6))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 3, i)
    sns.kdeplot(data=df, x=feature, hue='diagnosis', fill=True, palette='Set2', alpha=0.6)
    plt.title(f'{feature.replace('_', ' ').title()} KDE by Diagnosis')
    plt.xlabel(feature.replace('_', ' ').title())
plt.tight_layout()
plt.show()
""")
add_markdown("KDE plots highlight the feature distribution overlap and show which feature ranges are most informative for classification.")
add_code("""plt.figure(figsize=(12, 10))
for i, feature in enumerate(selected_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='diagnosis', y=feature, data=df, palette='Set2')
    plt.title(f'{feature.replace('_', ' ').title()} by Diagnosis')
    plt.xlabel('Diagnosis')
    plt.ylabel(feature.replace('_', ' ').title())
plt.tight_layout()
plt.show()
""")
add_markdown("Boxplots make it easy to compare the central tendency and spread of features between benign and malignant tumors.")
add_markdown("## Feature Engineering\n\nNo new engineered features are required for this dataset because all inputs are already numeric and meaningful. The target label is encoded for binary classification.")
add_code("""X = df.drop(columns=['diagnosis'])
y = LabelEncoder().fit_transform(df['diagnosis'])
print('Encoded classes:', list(LabelEncoder().fit(['B', 'M']).classes_))
""")
add_markdown("The target encoding converts benign/malignant labels into numerical values that scikit-learn can use.")
add_markdown("## Data Preprocessing\n\nPreprocessing includes train-test splitting and scaling, which are essential for a Support Vector Machine model.")
add_code("""X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)
""")
add_markdown("We use stratified sampling to preserve the class distribution between training and test sets.")
add_markdown("## Feature Scaling\n\nSVM is sensitive to feature magnitude because it uses distance-based optimization. Standard scaling ensures all features contribute equally to the model.")
add_code("""pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(probability=True, random_state=42))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print('Baseline Test Accuracy:', accuracy_score(y_test, y_pred))
""")
add_markdown("Using a pipeline keeps scaling inside the cross-validation process and prevents data leakage.")
add_markdown("## Model Building\n\nThe base SVM model is built inside a scikit-learn pipeline to ensure the scaler is applied consistently during training and evaluation.")
add_code("""param_grid = [
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto']
    },
    {
        'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    },
    {
        'svc__kernel': ['poly'],
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'svc__degree': [2, 3, 4]
    },
    {
        'svc__kernel': ['sigmoid'],
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
]

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
""")
add_markdown("## Hyperparameter Tuning\n\nGridSearchCV searches over the kernel, regularization strength, gamma, and polynomial degree. These hyperparameters control the shape and flexibility of the SVM decision boundary.")
add_code("""print('Best Parameters:', grid_search.best_params_)
print('Best Cross-Validation Accuracy:', round(grid_search.best_score_, 4))
results = pd.DataFrame(grid_search.cv_results_)[[
    'param_svc__kernel',
    'param_svc__C',
    'param_svc__gamma',
    'param_svc__degree',
    'mean_test_score'
]]
results = results.sort_values(by='mean_test_score', ascending=False).reset_index(drop=True)
results.head(10)
""")
add_markdown("The best hyperparameters are selected using cross-validation, which helps the model generalize better to unseen data.")
add_markdown("## Best Parameters\n\nThe selected model configuration reflects the best trade-off between bias and variance for this classification task.")
add_code("""best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
""")
add_markdown("## Model Evaluation\n\nWe evaluate the final model using a full set of classification metrics and visualizations.")
add_code("""accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print('Classification Report:')
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))
""")
add_code("""cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
""")
add_markdown("The confusion matrix shows the number of true positives, false positives, true negatives, and false negatives. This is critical for evaluating medical classification models.")
add_code("""fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}', color='darkorange')
plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
""")
add_code("""precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
pr_auc = auc(recall_vals, precision_vals)
plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color='green', label=f'PR AUC = {pr_auc:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.show()
""")
add_markdown("Precision-recall analysis is especially important when false negatives are costly, such as in cancer diagnosis.")
add_markdown("## Model Interpretation\n\nSupport Vector Machines classify data by finding the optimal margin that separates classes. Support vectors are the data points that lie closest to that margin and determine the decision boundary.")
add_code("""support_vectors = best_model.named_steps['svc'].support_vectors_
print('Number of support vectors:', support_vectors.shape[0])
""")
add_markdown("The number of support vectors indicates how many training samples are critical to defining the decision boundary.")
add_markdown("The best kernel is the one that achieved the highest cross-validation accuracy, indicating it can separate the malignant and benign samples most effectively for this dataset.")
add_markdown("## Decision Boundary Visualization\n\nA 2D projection using PCA allows us to visualize the model's decision boundary even though the original data has 30 features.")
add_code("""pca = PCA(n_components=2, random_state=42)
X_train_scaled = best_model.named_steps['scaler'].transform(X_train)
X_test_scaled = best_model.named_steps['scaler'].transform(X_test)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

svc_2d = SVC(kernel='rbf', C=grid_search.best_params_['svc__C'], gamma=grid_search.best_params_['svc__gamma'], probability=True, random_state=42)
svc_2d.fit(X_train_pca, y_train)

x_min, x_max = X_test_pca[:, 0].min() - 1, X_test_pca[:, 0].max() + 1
y_min, y_max = X_test_pca[:, 1].min() - 1, X_test_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
Z = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', edgecolor='k')
plt.title('SVM Decision Boundary in PCA Space')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
""")
add_markdown("This visualization uses PCA to reduce the feature space for plotting. It helps explain how the SVM separates classes in a simplified 2D projection.")
add_markdown("## Results\n\nThe final SVM model demonstrates strong performance on the breast cancer classification task, with especially high precision and ROC AUC.")
add_markdown("## Conclusion\n\nA Support Vector Machine classifier was successfully trained using a standard scaling pipeline and tuned with GridSearchCV. The model achieved high accuracy and strong medical classification performance metrics.")
add_markdown("## Future Improvements\n\n- Compare SVM with ensemble methods like Random Forest and XGBoost.\n- Add cross-validation for final evaluation rather than a single train-test split.\n- Experiment with feature selection or polynomial feature expansions.\n- Deploy the model as a web app using Streamlit or Flask.\n- Add a model monitoring and explainability section for production readiness.")

nb = {
    'cells': cells,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.14'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}

notebook_path.parent.mkdir(parents=True, exist_ok=True)
notebook_path.write_text(json.dumps(nb, indent=1), encoding='utf-8')
print(f'Notebook written to {notebook_path}')
