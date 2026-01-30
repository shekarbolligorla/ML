# ML Assignment 2 - Machine Learning Classification Models

## Problem Statement
Implement multiple classification models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost) on a classification dataset, evaluate their performance using standard metrics, and build an interactive Streamlit web application to demonstrate the models. Deploy the application on Streamlit Community Cloud.

## Dataset Description
The dataset is a synthetic binary classification dataset with 1000 instances and 15 features (12 informative features and 3 redundant features). This dataset meets all assignment requirements:
- **Total Instances:** 1000 (exceeds minimum 500)
- **Total Features:** 15 (exceeds minimum 12)
- **Target Variable:** Binary classification (0 or 1)
- **Data Type:** Numeric features for classification task
- **Data Split:** 80% training (800 samples) and 20% testing (200 samples)

The dataset is generated synthetically to ensure a well-structured, reproducible classification problem suitable for evaluating multiple machine learning algorithms.

## Models Used

### Comparison Table

| ML Model Name       | Accuracy | AUC | Precision | Recall | F1   | MCC |
|---------------------|----------|-----|-----------|--------|------|-----|
| Logistic Regression | 0.8250   | 0.9088 | 0.8259    | 0.8244 | 0.8246 | 0.6503 |
| Decision Tree       | 0.7500   | 0.7495 | 0.7503    | 0.7495 | 0.7496 | 0.4998 |
| kNN                 | 0.9200   | 0.9727 | 0.9200    | 0.9200 | 0.9200 | 0.8399 |
| Naive Bayes         | 0.7600   | 0.8417 | 0.7599    | 0.7599 | 0.7599 | 0.5198 |
| Random Forest       | 0.9000   | 0.9526 | 0.9004    | 0.9004 | 0.9000 | 0.8007 |
| XGBoost             | 0.9250   | 0.9728 | 0.9266    | 0.9257 | 0.9250 | 0.8522 |

### Observations about Model Performance

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Achieved 82.50% accuracy with strong AUC of 0.9088, demonstrating effective linear separation of the binary classes. The model shows balanced performance across precision (0.8259) and recall (0.8244), indicating reliable predictions. |
| Decision Tree       | Obtained 75% accuracy with AUC of 0.7495, showing moderate performance. The lower MCC (0.4998) suggests the tree may have captured some but not all decision patterns. This simpler model trades accuracy for interpretability. |
| kNN                 | Excellent performance with 92% accuracy and highest among base models with AUC of 0.9727. kNN works effectively here as classes are well-separated in the feature space. The high precision and recall (0.92 each) demonstrate balanced classification. |
| Naive Bayes         | Achieved 76% accuracy with AUC of 0.8417. Gaussian Naive Bayes shows moderate performance, suggesting that the assumption of feature independence may not fully hold in this dataset, leading to slightly lower metrics. |
| Random Forest       | Strong ensemble performance with 90% accuracy and AUC of 0.9526. Random Forest successfully combines multiple decision trees to reduce overfitting and improve generalization. The high MCC (0.8007) indicates strong overall predictive power. |
| XGBoost             | Best overall performance with 92.5% accuracy and highest AUC of 0.9728. XGBoost's gradient boosting approach effectively learns complex patterns, achieving the highest F1 score (0.9250) and MCC (0.8522), demonstrating superior predictive capability. |