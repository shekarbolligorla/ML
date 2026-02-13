"""
Naive Bayes Model for Wine Quality Classification
This module implements Gaussian Naive Bayes for binary wine quality classification
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import joblib

def train_naive_bayes(X_train, y_train):
    """
    Train Gaussian Naive Bayes model
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training features
    y_train : array-like of shape (n_samples,)
        Training target
    
    Returns:
    --------
    model : GaussianNB
        Trained Naive Bayes model
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def evaluate_naive_bayes(model, X_test, y_test):
    """
    Evaluate Naive Bayes model
    
    Parameters:
    -----------
    model : GaussianNB
        Trained model
    X_test : array-like of shape (n_samples, n_features)
        Test features
    y_test : array-like of shape (n_samples,)
        Test target
    
    Returns:
    --------
    metrics : dict
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_proba[:, 1]),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1': f1_score(y_test, y_pred, average='macro'),
        'MCC': matthews_corrcoef(y_test, y_pred)
    }
    return metrics, y_pred

if __name__ == "__main__":
    # Example usage
    from sklearn.model_selection import train_test_split
    
    # Load dataset
    df = pd.read_csv('../dataset.csv')
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = train_naive_bayes(X_train, y_train)
    
    # Evaluate
    metrics, y_pred = evaluate_naive_bayes(model, X_test, y_test)
    
    print("Naive Bayes Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save model
    joblib.dump(model, 'Naive_Bayes.pkl')
