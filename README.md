# Machine Learning Project Portfolio

This repository contains five machine learning prediction projects implemented as Jupyter notebooks. Each project is organized into a dedicated folder with the dataset and a polished notebook covering the problem statement, data exploration, preprocessing, model training, and evaluation.

## Project Overview

- **Decision Tree / Loan Approval Prediction**: Predicts whether loan applications are approved or rejected using a Decision Tree classifier.
- **kNN / Heart Disease Prediction**: Classifies heart disease risk using k-Nearest Neighbors with hyperparameter tuning.
- **Linear Regression / Car Price Prediction**: Forecasts car prices using Linear Regression and feature engineering.
- **Logistic Regression / Student Grade Classification**: Classifies student grade levels using Logistic Regression.
- **Naive Bayes / SMS Spam Detection**: Detects spam SMS messages using Multinomial Naive Bayes and text vectorization.

## Problem Statement

Each notebook solves a supervised learning problem using structured data or text data. The focus is on model selection, preprocessing, train-test split, evaluation metrics, and professional notebook presentation.

## Dataset

Each project folder contains a dataset in CSV format:

- `Decision Tree/Loan Approval Prediction/loan_approval_dataset.csv`
- `kNN/Heart Disease Prediction/heart.csv`
- `Linear Regression/Car Price Prediction/CarPrice_Assignment.csv`
- `Logistic Regression/Student Pass-Fail Prediction/Student_performance_data _.csv`
- `Naive Bayes/SMS Spam Detection/spam.csv`

## Technologies Used

- Python
- Jupyter Notebook
- pandas
- NumPy
- seaborn
- matplotlib
- scikit-learn

## Project Workflow

Each notebook follows a standard machine learning workflow:

1. Problem statement and dataset overview
2. Data loading and quality checks
3. Exploratory data analysis (EDA)
4. Data cleaning and preprocessing
5. Train-test split with fixed random state
6. Model training and hyperparameter tuning (where applicable)
7. Evaluation using classification/regression metrics
8. Visualization of results and model interpretation

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd MLProjects
```

2. Install required Python packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn
```

## Usage

Open the notebook for the chosen project in Jupyter Notebook or JupyterLab, then run cells sequentially to reproduce the analysis and results.

## Model Used

- Decision Tree classifier
- k-Nearest Neighbors classifier
- Linear Regression
- Logistic Regression
- Multinomial Naive Bayes

## Results

Each notebook includes model evaluation metrics, confusion matrices, and visualizations to validate performance. The projects preserve existing objectives while improving readability, structure, and ML best practices.

## Folder Structure

```
MLProjects/
├── Decision Tree/
│   └── Loan Approval Prediction/
│       ├── loan_approval_dataset.csv
│       └── loan_approval_prediction.ipynb
├── kNN/
│   └── Heart Disease Prediction/
│       ├── heart.csv
│       └── heart_disease_prediction.ipynb
├── Linear Regression/
│   └── Car Price Prediction/
│       ├── CarPrice_Assignment.csv
│       └── CarPricePrediction.ipynb
├── Logistic Regression/
│   └── Student Pass-Fail Prediction/
│       ├── Student_performance_data _.csv
│       └── StudentPassFailPrediction.ipynb
├── Naive Bayes/
│   └── SMS Spam Detection/
│       ├── spam.csv
│       └── sms_spam_detection.ipynb
└── README.md
```

## Future Improvements

- Add a `requirements.txt` or `environment.yml` for reproducible installs.
- Add unit tests and a CI pipeline for notebook validation.
- Expand feature engineering and model comparison for each project.
- Add a summary notebook or dashboard to compare all model results.
