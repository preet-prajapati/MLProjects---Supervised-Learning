# Medical Insurance Cost Prediction

## Project Overview
This project predicts medical insurance charges using a Decision Tree Regressor. It is designed as a beginner-friendly, portfolio-ready machine learning notebook that demonstrates the full workflow from data loading and cleaning to model evaluation and interpretation.

## Dataset Information
The dataset used in this project is the well-known medical insurance dataset. It contains features such as:
- Age
- Sex
- BMI
- Number of children/dependents
- Smoking status
- Region
- Insurance charges (target variable)

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure
- MedicalInsuranceCostPrediction_fixed.ipynb: Main notebook with the complete workflow
- insurance.csv: Dataset used for training and evaluation
- README.md: Project documentation

## Installation
Install the required libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Workflow
1. Load and inspect the dataset
2. Clean and preprocess the data
3. Explore relationships with visualizations
4. Encode categorical variables
5. Split data into train and test sets
6. Train a Decision Tree Regressor
7. Evaluate the model using MAE, MSE, RMSE, and R²
8. Tune hyperparameters using GridSearchCV
9. Interpret the model and summarize findings

## Results
The tuned Decision Tree Regressor achieved strong predictive performance on the test set with an R² score of approximately 0.894. Smoking status, age, and BMI were the most influential features in the model.

## Screenshots
Placeholder for screenshots or visual outputs from the notebook.

## Future Improvements
- Compare with Random Forest, XGBoost, or CatBoost
- Add cross-validation for more robust evaluation
- Deploy the model with Streamlit or Flask
- Add feature selection and engineering improvements
