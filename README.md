# ML Assignment 2

## Problem Statement
Implement multiple classification models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost) on the Iris dataset, evaluate their performance, and build an interactive Streamlit web application to demonstrate the models. Deploy the app on Streamlit Community Cloud.

## Dataset Description
The Iris dataset is a classic dataset in machine learning and statistics. It consists of 150 samples of iris flowers, each described by 4 features: sepal length, sepal width, petal length, and petal width. The target variable is the species of the iris flower, which can be one of three classes: Setosa, Versicolor, or Virginica. This dataset is used for classification tasks and is available in scikit-learn.

## Models Used

### Comparison Table

| ML Model Name       | Accuracy | AUC | Precision | Recall | F1   | MCC |
|---------------------|----------|-----|-----------|--------|------|-----|
| Logistic Regression | 1.0000   | 1.0 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| Decision Tree       | 1.0000   | 1.0 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| kNN                 | 1.0000   | 1.0 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes         | 1.0000   | 1.0 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| Random Forest       | 1.0000   | 1.0 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |
| XGBoost             | 1.0000   | 1.0 | 1.0000    | 1.0000 | 1.0000 | 1.0000 |

### Observations about Model Performance

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Perfect performance on the Iris dataset, achieving 100% accuracy. |
| Decision Tree       | Perfect performance, likely due to the dataset's simplicity and separability. |
| kNN                 | Excellent performance, as kNN works well on small, well-separated datasets. |
| Naive Bayes         | Perfect classification, assuming independence of features holds reasonably well. |
| Random Forest       | As an ensemble method, it achieves perfect accuracy by combining multiple trees. |
| XGBoost             | Boosting technique also performs perfectly on this dataset. |