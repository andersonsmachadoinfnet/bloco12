from sklearn.datasets import load_breast_cancer
import pandas as pd

# Carrega o dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Seleciona duas features contínuas
selected_features = ['mean radius', 'mean texture']
df_selected = df[selected_features].copy()

# Discretização por bins fixos (4 intervalos iguais)
bins = 4
discretized = pd.DataFrame()

for feature in selected_features:
    discretized[feature + '_binned'] = pd.cut(df_selected[feature], bins=bins, labels=False)

# Junta os dados originais com os discretizados
result = pd.concat([df_selected, discretized], axis=1)

# Exibe os primeiros resultados
print(result.head())