from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

def min_max_normalize(X):
    return (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

normalizer = FunctionTransformer(min_max_normalize)
X_normalized = normalizer.fit_transform(X)
X_normalized_df = pd.DataFrame(X_normalized, columns=data.feature_names)

print("Primeiras linhas após normalização:")
print(X_normalized_df.head())
