import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Download Wine Quality dataset from UCI Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

print("Downloading Wine Quality dataset from UCI Repository...")
df = pd.read_csv(url, sep=';')

print(f"Dataset downloaded successfully!")
print(f"Original shape: {df.shape}")
print(f"Instances: {df.shape[0]}")
print(f"Features: {df.shape[1]-1}")

# Display dataset info
print("\nDataset Info:")
print(df.head())
print(f"\nTarget distribution:")
print(df['quality'].value_counts().sort_index())

# Convert quality scores to binary classification (Good: >= 6, Bad: < 6)
# This makes it more suitable for binary classification models
df['quality_binary'] = (df['quality'] >= 6).astype(int)

# Remove original quality column and rename target
df = df.drop('quality', axis=1)
df = df.rename(columns={'quality_binary': 'target'})

# Reorder columns to put target at the end
cols = list(df.columns)
cols.remove('target')
df = df[cols + ['target']]

print(f"\nAfter conversion to binary classification:")
print(f"Shape: {df.shape}")
print(f"Target distribution:")
print(df['target'].value_counts())

# Save to CSV
df.to_csv('dataset.csv', index=False)
print(f"\nDataset saved to dataset.csv")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset Statistics:")
print(df.describe())
