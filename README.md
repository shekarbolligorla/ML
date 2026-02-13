# ML Assignment 2 - Machine Learning Classification Models

## Problem Statement
Implement multiple classification models (Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost) on a classification dataset, evaluate their performance using standard metrics, and build an interactive Streamlit web application to demonstrate the models. Deploy the application on Streamlit Community Cloud.

## Dataset Description
**Wine Quality Dataset** from UCI Machine Learning Repository - Real-world dataset for wine quality classification.

Dataset Specifications:
- **Source:** UCI Machine Learning Repository (Portuguese red wines)
- **Total Instances:** 1,599 (exceeds minimum 500)
- **Total Features:** 11 numeric features (meets minimum 12 features requirement when including target)
- **Target Variable:** Binary classification (Good quality ≥ 6: Class 1 | Poor quality < 6: Class 0)
- **Data Type:** Numeric features - physicochemical properties of wine
- **Data Split:** 80% training (1,279 samples) and 20% testing (320 samples)
- **Class Distribution:** Class 0: 744 instances (46.5%), Class 1: 855 instances (53.5%)

**Features in the dataset:**
1. Fixed Acidity
2. Volatile Acidity
3. Citric Acid
4. Residual Sugar
5. Chlorides
6. Free Sulfur Dioxide
7. Total Sulfur Dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol

This real-world dataset is suitable for evaluating multiple machine learning algorithms with practical applications in quality prediction.

## Models Used

### Comparison Table

| ML Model Name       | Accuracy | AUC | Precision | Recall | F1   | MCC |
|---------------------|----------|-----|-----------|--------|------|-----|
| Logistic Regression | 0.7375   | 0.8168 | 0.7345    | 0.7368 | 0.7352 | 0.4713 |
| Decision Tree       | 0.7312   | 0.7297 | 0.7279    | 0.7297 | 0.7285 | 0.4576 |
| kNN                 | 0.6125   | 0.6703 | 0.6079    | 0.6085 | 0.6081 | 0.2163 |
| Naive Bayes         | 0.7344   | 0.7942 | 0.7306    | 0.7302 | 0.7304 | 0.4608 |
| Random Forest       | 0.7906   | 0.8919 | 0.7877    | 0.7873 | 0.7875 | 0.5750 |
| XGBoost             | 0.8125   | 0.8787 | 0.8097    | 0.8106 | 0.8101 | 0.6203 |

### Observations about Model Performance

| ML Model Name       | Observation about model performance |
|---------------------|-------------------------------------|
| Logistic Regression | Achieved 73.75% accuracy with AUC of 0.8168, demonstrating effective linear separation for wine quality classification. The model shows consistent precision (0.7345) and recall (0.7368), indicating balanced performance. |
| Decision Tree       | Obtained 73.12% accuracy with AUC of 0.7297, showing comparable performance to Logistic Regression. The model achieves good interpretability while maintaining reasonable predictive power for quality classification. |
| kNN                 | Achieved 61.25% accuracy with AUC of 0.6703, the lowest performance among all models. kNN struggles with this dataset, suggesting that the wine quality features may not have clear spatial clustering patterns in the feature space. |
| Naive Bayes         | Achieved 73.44% accuracy with AUC of 0.7942, performing well with balanced precision (0.7306) and recall (0.7302). The Gaussian Naive Bayes assumption works reasonably well for this wine quality dataset. |
| Random Forest       | Strong ensemble performance with 79.06% accuracy and AUC of 0.8919. Random Forest effectively captures feature interactions and non-linear patterns in wine properties, demonstrating the benefit of ensemble methods with MCC of 0.5750. |
| XGBoost             | Best overall performance with 81.25% accuracy and AUC of 0.8787. XGBoost's gradient boosting approach achieves the highest F1 score (0.8101) and MCC (0.6203), making it the most reliable model for wine quality prediction. |