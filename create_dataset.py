import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

# Create a synthetic dataset that meets requirements
# 500+ instances, 12+ features, classification problem
X, y = make_classification(
    n_samples=1000,
    n_features=15,
    n_informative=12,
    n_redundant=3,
    n_classes=2,
    random_state=42
)

# Create DataFrame
feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Save to CSV
df.to_csv('dataset.csv', index=False)
print(f"Dataset created successfully!")
print(f"Shape: {df.shape}")
print(f"Instances: {df.shape[0]}")
print(f"Features: {df.shape[1]-1}")
print(f"\nFirst few rows:")
print(df.head())
