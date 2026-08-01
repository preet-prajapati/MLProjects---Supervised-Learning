# Breast Cancer Classification using Support Vector Machine

## Project Overview
This project uses Support Vector Machine (SVM) classification to predict whether a breast tumor is benign or malignant. It is designed as a portfolio-ready machine learning project with a clean workflow, strong evaluation, and interpretable results.

## Problem Statement
Breast cancer diagnosis is a critical healthcare task. Early and accurate classification of tumors as benign or malignant can support clinicians in decision making and improve patient outcomes.

## Objective
Train and evaluate an SVM model to accurately classify breast tumors using digitized image-derived features, while following best practices for preprocessing, model tuning, and evaluation.

## Dataset Information
- Number of samples: 569
- Number of features: 30 numeric measurements
- Target variable: diagnosis (B = benign, M = malignant)
- Source: Breast Cancer Wisconsin dataset

## Technologies Used
- Python
- pandas
- NumPy
- scikit-learn
- seaborn
- matplotlib

## Project Structure
- `dataset/data.csv`: Breast cancer dataset
- `notebook/BreastCancerPrediction.ipynb`: Notebook with full analysis and model building
- `README.md`: Project documentation

## Installation
Install the required Python packages:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## Usage
Open `notebook/BreastCancerPrediction.ipynb` in Jupyter Notebook or JupyterLab and run the cells sequentially.

## Results
The optimized SVM model delivers strong classification performance with high accuracy, precision, recall, F1 score, and ROC AUC.

## Future Improvements
- Add model comparison with Random Forest and XGBoost
- Implement cross-validation for final performance reporting
- Deploy the model using Streamlit or Flask
- Add feature selection and explainability techniques

## License
This project is available under the MIT License.
