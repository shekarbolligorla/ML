import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load models
models = {}
model_names = ['Logistic_Regression', 'Decision_Tree', 'kNN', 'Naive_Bayes', 'Random_Forest', 'XGBoost']
for name in model_names:
    models[name] = joblib.load(f'model/{name}.pkl')

st.title("ML Classification Models Demo")

# Upload CSV
uploaded_file = st.file_uploader("Upload your test CSV file", type="csv")
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.dataframe(df.head())
        
        # Assume last column is target
        if df.shape[1] < 2:
            st.error("CSV must have at least features and target column.")
        else:
            X = df.iloc[:, :-1].values
            y_true = df.iloc[:, -1].values
            
            # Model selection
            selected_model = st.selectbox("Select Model", list(models.keys()))
            model = models[selected_model]
            
            # Predict
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_true, y_pred)
            if y_proba is not None:
                # For binary classification, use probability of positive class
                if y_proba.shape[1] == 2:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr')
            else:
                auc = 'N/A'
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            mcc = matthews_corrcoef(y_true, y_pred)
            
            st.subheader("Evaluation Metrics")
            st.write(f"Accuracy: {accuracy:.4f}")
            st.write(f"AUC: {auc}")
            st.write(f"Precision: {precision:.4f}")
            st.write(f"Recall: {recall:.4f}")
            st.write(f"F1 Score: {f1:.4f}")
            st.write(f"MCC: {mcc:.4f}")
            
            st.subheader("Classification Report")
            st.text(classification_report(y_true, y_pred))
            
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")