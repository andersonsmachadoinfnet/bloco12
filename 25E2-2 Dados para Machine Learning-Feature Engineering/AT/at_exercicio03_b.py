from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import PowerTransformer
import pandas as pd

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)

power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)

X_power = power_transformer.fit_transform(X)
X_power_df = pd.DataFrame(X_power, columns=data.feature_names)

print("Primeiras linhas após PowerTransformer:")
print(X_power_df.head())
