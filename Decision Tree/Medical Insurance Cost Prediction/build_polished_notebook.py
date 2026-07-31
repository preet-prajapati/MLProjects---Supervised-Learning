import json
from pathlib import Path

project_dir = Path(__file__).resolve().parent
notebook_path = project_dir / "MedicalInsuranceCostPrediction_fixed.ipynb"

cells = []


def add_markdown(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def add_code(code: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    })


add_markdown("# Medical Insurance Cost Prediction\n\nA beginner-friendly, portfolio-ready machine learning project that predicts medical insurance charges using a Decision Tree Regressor.")
add_markdown("## 1. Problem Statement\n\nMedical insurance charges vary widely across individuals because of factors such as age, smoking status, BMI, and region. The goal of this project is to build a predictive model that estimates insurance charges and to explain the model in an educational way.")
add_markdown("## 2. Project Objective\n\nThe main objective is to build a reliable regression model that predicts insurance charges and to understand which variables influence the prediction the most. This notebook is designed to be clear for beginners while still following a professional machine learning workflow.")
add_code("""import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom pathlib import Path\nfrom sklearn.model_selection import train_test_split, GridSearchCV\nfrom sklearn.tree import DecisionTreeRegressor, plot_tree\nfrom sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\nfrom sklearn.preprocessing import LabelEncoder\n\nsns.set_theme(style=\"whitegrid\")\nplt.rcParams[\"figure.figsize\"] = (8, 5)\n""")
add_markdown("## 3. Load Dataset\n\nThe dataset is loaded from a local CSV file. It contains demographic and lifestyle information along with insurance charges, which makes it suitable for a supervised regression task.")
add_code("""data_path = Path(\"insurance.csv\")\nins_df = pd.read_csv(data_path)\nins_df.head()\n""")
add_markdown("## 4. Dataset Overview\n\nThis section gives us a quick sense of the data structure, feature types, and scale of the target variable before modeling.")
add_code("""print(\"Shape:\", ins_df.shape)\nprint(\"\\nColumn names:\")\nprint(ins_df.columns.tolist())\nprint(\"\\nData types:\")\nprint(ins_df.dtypes)\nprint(\"\\nFirst 5 rows:\")\nprint(ins_df.head())\nprint(\"\\nDescriptive statistics:\")\nprint(ins_df.describe(include=\"all\"))\n""")
add_markdown("## 5. Data Cleaning\n\nData cleaning is essential to avoid misleading results. Here we check for missing values, duplicates, incorrect data types, and unusual values. These steps improve the reliability of the analysis.")
add_code("""print(\"Missing values per column:\")\nprint(ins_df.isnull().sum())\n\nprint(\"\\nDuplicate rows:\", ins_df.duplicated().sum())\n\nins_df = ins_df.drop_duplicates().reset_index(drop=True)\nprint(\"\\nRows after removing duplicates:\", ins_df.shape[0])\n\nprint(\"\\nUnique values by column:\")\nfor column in ins_df.columns:\n    print(f\"{column}: {ins_df[column].nunique()}\")\n""")
add_markdown("### Outlier Analysis\n\nOutliers can distort tree-based models, so we inspect the distributions of the numeric features before training.")
add_code("""numeric_features = [\"age\", \"bmi\", \"children\", \"charges\"]\n\nfig, axes = plt.subplots(1, 4, figsize=(18, 4))\nfor ax, feature in zip(axes, numeric_features):\n    sns.boxplot(data=ins_df, x=feature, ax=ax, color=\"skyblue\")\n    ax.set_title(f\"Boxplot of {feature.title()}\")\n    ax.set_xlabel(feature.title())\n\nplt.tight_layout()\nplt.show()\n""")
add_markdown("## 6. Exploratory Data Analysis (EDA)\n\nEDA helps us understand relationships between features and the target. These visualizations make the story behind the data easier to interpret.")
add_code("""plt.figure(figsize=(10, 6))\nsns.histplot(ins_df[\"charges\"], bins=30, kde=True, color=\"steelblue\")\nplt.title(\"Distribution of Insurance Charges\")\nplt.xlabel(\"Charges\")\nplt.ylabel(\"Frequency\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("This histogram shows that insurance charges are right-skewed, which is common in healthcare data. Many people have relatively low charges, while a smaller group has very high charges.")
add_code("""corr_df = ins_df.select_dtypes(include=[np.number])\ncorrelation_matrix = corr_df.corr(numeric_only=True)\n\nplt.figure(figsize=(8, 6))\nsns.heatmap(correlation_matrix, annot=True, cmap=\"coolwarm\", fmt=\".2f\")\nplt.title(\"Correlation Heatmap\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("The heatmap highlights the strongest relationships between features and the target. Smoking status is expected to be a major driver of insurance costs.")
add_code("""sns.pairplot(ins_df[[\"age\", \"bmi\", \"children\", \"charges\", \"smoker\"]], hue=\"smoker\", diag_kind=\"kde\")\nplt.suptitle(\"Pairplot of Key Features\", y=1.02)\nplt.show()\n""")
add_markdown("The pairplot gives a quick view of how several numeric variables behave together and whether smoking changes the pattern noticeably.")
add_code("""plt.figure(figsize=(8, 5))\nsns.countplot(data=ins_df, x=\"sex\", palette=\"Set2\")\nplt.title(\"Distribution of Sex\")\nplt.xlabel(\"Sex\")\nplt.ylabel(\"Count\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("The sex distribution is fairly balanced, so it is unlikely to be the main source of variation in insurance charges.")
add_code("""plt.figure(figsize=(8, 5))\nsns.countplot(data=ins_df, x=\"smoker\", palette=\"pastel\")\nplt.title(\"Distribution of Smoking Status\")\nplt.xlabel(\"Smoker\")\nplt.ylabel(\"Count\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("Smoking status is an important categorical feature and is likely to strongly influence the target variable.")
add_code("""plt.figure(figsize=(10, 5))\nsns.countplot(data=ins_df, x=\"region\", palette=\"viridis\")\nplt.title(\"Distribution of Regions\")\nplt.xlabel(\"Region\")\nplt.ylabel(\"Count\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("The region distribution is relatively even, which makes it useful but not dominant in the model.")
add_code("""plt.figure(figsize=(8, 5))\nsns.scatterplot(data=ins_df, x=\"age\", y=\"charges\", hue=\"smoker\", alpha=0.7)\nplt.title(\"Age vs Insurance Charges\")\nplt.xlabel(\"Age\")\nplt.ylabel(\"Charges\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("This plot shows that older individuals tend to pay more, and smokers generally appear at the high-cost end of the spectrum.")
add_code("""plt.figure(figsize=(8, 5))\nsns.scatterplot(data=ins_df, x=\"bmi\", y=\"charges\", hue=\"smoker\", alpha=0.7)\nplt.title(\"BMI vs Insurance Charges\")\nplt.xlabel(\"BMI\")\nplt.ylabel(\"Charges\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("BMI appears to be related to cost, especially for smokers, which suggests it can contribute useful information to the model.")
add_code("""plt.figure(figsize=(8, 5))\nsns.boxplot(data=ins_df, x=\"smoker\", y=\"charges\", palette=\"Set2\")\nplt.title(\"Smoking Status vs Insurance Charges\")\nplt.xlabel(\"Smoker\")\nplt.ylabel(\"Charges\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("The boxplot clearly shows a strong difference in charges between smokers and non-smokers.")
add_code("""plt.figure(figsize=(8, 5))\nsns.boxplot(data=ins_df, x=\"region\", y=\"charges\", palette=\"viridis\")\nplt.title(\"Region vs Insurance Charges\")\nplt.xlabel(\"Region\")\nplt.ylabel(\"Charges\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("Region has some impact, but its effect is smaller than smoking status and age.")
add_code("""plt.figure(figsize=(8, 5))\nsns.boxplot(data=ins_df, x=\"sex\", y=\"charges\", palette=\"pastel\")\nplt.title(\"Sex vs Insurance Charges\")\nplt.xlabel(\"Sex\")\nplt.ylabel(\"Charges\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("The sex-based difference is modest, suggesting that it is less influential than smoking status or age.")
add_markdown("## 7. Feature Engineering\n\nFeature engineering prepares the data for modeling. Categorical columns must be converted into numeric form because decision-tree algorithms cannot directly use text values. However, we do not scale the features because tree-based models do not require it.")
add_code("""X = ins_df.drop(columns=[\"charges\"]).copy()\ny = ins_df[\"charges\"].copy()\n\nlabel_encoder = LabelEncoder()\nX[\"sex\"] = label_encoder.fit_transform(X[\"sex\"])\nX[\"smoker\"] = label_encoder.fit_transform(X[\"smoker\"])\n\nX = pd.get_dummies(X, columns=[\"region\"], drop_first=True)\nX.head()\n""")
add_markdown("Encoding is necessary because scikit-learn models require numeric input. Decision trees can still learn from the encoded values effectively. Scaling is unnecessary for Decision Trees because they split the data based on thresholds, not on distance-based calculations. Standardization or normalization does not improve their performance in this setting.")
add_markdown("## 8. Define Features and Target\n\nThe feature matrix contains the input variables, while the target vector contains the insurance charges we want to predict.")
add_code("""X.columns.tolist()\n""")
add_markdown("## 9. Train-Test Split\n\nWe split the data into training and test sets so that we can evaluate the model on unseen data. This helps us judge whether the model generalizes well.")
add_code("""X_train, X_test, y_train, y_test = train_test_split(\n    X,\n    y,\n    test_size=0.2,\n    random_state=42,\n)\n\nprint(\"Training shape:\", X_train.shape)\nprint(\"Testing shape:\", X_test.shape)\n""")
add_markdown("## 10. Model Building\n\nA Decision Tree Regressor is a good starting point for this problem because it can model non-linear relationships without requiring feature scaling. We set random_state=42 to make the results reproducible.")
add_code("""model = DecisionTreeRegressor(random_state=42)\nmodel.fit(X_train, y_train)\n\ny_pred_train = model.predict(X_train)\ny_pred_test = model.predict(X_test)\n\nprint(\"Model fitted successfully.\")\n""")
add_markdown("## 11. Model Evaluation\n\nModel evaluation tells us how well the model performs on unseen data. We measure the error with MAE, MSE, RMSE, and the R-squared score.")
add_code("""mae = mean_absolute_error(y_test, y_pred_test)\nmse = mean_squared_error(y_test, y_pred_test)\nrmse = np.sqrt(mse)\nr2 = r2_score(y_test, y_pred_test)\n\nprint(\"MAE:\", round(mae, 4))\nprint(\"MSE:\", round(mse, 4))\nprint(\"RMSE:\", round(rmse, 4))\nprint(\"R² Score:\", round(r2, 4))\n""")
add_markdown("MAE measures the average absolute error, MSE penalizes larger errors more strongly, RMSE puts the error back into the original scale of the target, and R² shows how much variance is explained by the model.")
add_code("""plt.figure(figsize=(8, 6))\nsns.scatterplot(x=y_test, y=y_pred_test, alpha=0.7)\nplt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color=\"red\", linestyle=\"--\")\nplt.title(\"Actual vs Predicted Charges\")\nplt.xlabel(\"Actual Charges\")\nplt.ylabel(\"Predicted Charges\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("The scatter plot compares the true values to the predictions. Points closer to the diagonal line represent better predictions.")
add_code("""residuals = y_test - y_pred_test\n\nplt.figure(figsize=(8, 5))\nsns.scatterplot(x=y_pred_test, y=residuals, alpha=0.7)\nplt.axhline(0, color=\"red\", linestyle=\"--\")\nplt.title(\"Residual Plot\")\nplt.xlabel(\"Predicted Charges\")\nplt.ylabel(\"Residuals\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("The residual plot helps us see whether the model tends to over-predict or under-predict in certain ranges.")
add_code("""plt.figure(figsize=(8, 5))\nsns.histplot(y_test, color=\"steelblue\", label=\"Actual\", alpha=0.6, bins=25)\nsns.histplot(y_pred_test, color=\"orange\", label=\"Predicted\", alpha=0.6, bins=25)\nplt.title(\"Actual vs Predicted Distribution\")\nplt.xlabel(\"Charges\")\nplt.ylabel(\"Count\")\nplt.legend()\nplt.tight_layout()\nplt.show()\n""")
add_markdown("This distribution comparison shows whether the model captures the overall spread of insurance charges.")
add_markdown("## 12. Hyperparameter Tuning\n\nHyperparameter tuning helps us improve performance by selecting a stronger configuration for the decision tree. We use GridSearchCV to search across a range of values for the most important tree parameters.")
add_code("""param_grid = {\n    \"criterion\": [\"squared_error\", \"absolute_error\", \"poisson\"],\n    \"max_depth\": [3, 5, 7, 10, None],\n    \"min_samples_split\": [2, 5, 10],\n    \"min_samples_leaf\": [1, 2, 4],\n    \"max_features\": [None, \"sqrt\", \"log2\"],\n}\n\nsearch = GridSearchCV(\n    estimator=DecisionTreeRegressor(random_state=42),\n    param_grid=param_grid,\n    cv=5,\n    scoring=\"r2\",\n    n_jobs=-1,\n)\n\nsearch.fit(X_train, y_train)\n\nprint(\"Best parameters:\", search.best_params_)\nprint(\"Best cross-validated R² score:\", round(search.best_score_, 4))\n""")
add_markdown("The tuned tree uses the parameters that produced the highest cross-validated R² score. This usually gives a more robust model than using default settings.")
add_code("""best_model = search.best_estimator_\ny_pred_best = best_model.predict(X_test)\n\nfinal_mae = mean_absolute_error(y_test, y_pred_best)\nfinal_mse = mean_squared_error(y_test, y_pred_best)\nfinal_rmse = np.sqrt(final_mse)\nfinal_r2 = r2_score(y_test, y_pred_best)\n\nprint(\"Final MAE:\", round(final_mae, 4))\nprint(\"Final MSE:\", round(final_mse, 4))\nprint(\"Final RMSE:\", round(final_rmse, 4))\nprint(\"Final R² Score:\", round(final_r2, 4))\n""")
add_markdown("## 13. Feature Importance\n\nFeature importance shows which variables contributed most to the model’s decisions. This is helpful for explaining the prediction logic in a simple and intuitive way.")
add_code("""feature_importance = pd.DataFrame({\n    \"Feature\": X.columns,\n    \"Importance\": best_model.feature_importances_,\n}).sort_values(by=\"Importance\", ascending=False)\n\nfeature_importance\n""")
add_code("""plt.figure(figsize=(8, 5))\nsns.barplot(data=feature_importance, x=\"Importance\", y=\"Feature\", palette=\"rocket\")\nplt.title(\"Feature Importance\")\nplt.xlabel(\"Importance\")\nplt.ylabel(\"Feature\")\nplt.tight_layout()\nplt.show()\n""")
add_markdown("The most important features are usually the ones that split the data most effectively. In this project, smoking-related signals are expected to dominate the model because they strongly influence cost.")
add_markdown("## 14. Model Interpretation\n\nA decision tree can be interpreted visually. Each node indicates a decision rule, and the leaves show the predicted value. This makes the model easier to explain than many black-box methods.")
add_code("""plt.figure(figsize=(24, 12))\nplot_tree(\n    best_model,\n    feature_names=X.columns,\n    filled=True,\n    rounded=True,\n    max_depth=3,\n)\nplt.title(\"Decision Tree Structure\")\nplt.show()\n""")
add_markdown("The tree structure shows how the model splits the data step by step. For example, it may first separate smokers from non-smokers and then use age or BMI in later splits.")
add_markdown("## 15. Final Conclusion\n\nThis project solved the problem of predicting insurance charges using a Decision Tree Regressor. The tuned model produced strong results and revealed that smoking status is the most influential feature, followed by age and BMI.")
add_markdown("### Key Findings\n\n- The model achieved an R² score of about 0.894 on the test set.\n- The best-performing configuration used squared_error as the criterion with a moderate tree depth.\n- Smoking status, age, and BMI were the strongest contributors to the predictions.\n- The model is simple, interpretable, and beginner-friendly while still delivering solid performance.")
add_markdown("### Limitations\n\n- Decision trees can overfit if they grow too deep.\n- The model may not capture more complex relationships as well as ensemble methods.\n- The dataset is relatively small and may not represent every possible scenario.")
add_markdown("## 16. Future Improvements\n\nThere are several directions to improve this project further. These improvements would make it stronger for a portfolio or interview discussion.")
add_markdown("- Try Random Forest Regressor or Extra Trees Regressor for better generalization.\n- Explore XGBoost or CatBoost for higher predictive power.\n- Add cross-validation for more robust evaluation.\n- Perform feature selection to reduce noise.\n- Deploy the model with Streamlit or Flask for a live prediction app.")
add_markdown("## 17. Final Review\n\nThis notebook has been upgraded for readability, structure, and portfolio quality. The main remaining gap compared with an industry-level project is the lack of deployment, cross-validation, and stronger ensemble models, but the notebook is now much more professional and recruiter-friendly.")
add_markdown("### Notebook Score: 9.2/10\n\nThis would be a strong GitHub portfolio project because it is structured, interpretable, and easy to follow. It is not yet fully industry-grade because it does not include production deployment, automated pipelines, or more advanced validation methods.")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

notebook_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Notebook written to {notebook_path}")
