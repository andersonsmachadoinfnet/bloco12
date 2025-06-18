from sklearn.datasets import load_breast_cancer
import pandas as pd

# Carrega o dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Seleciona duas features contínuas
selected_features = ['mean radius', 'mean texture']
df_selected = df[selected_features].copy()

# Discretização por quantis (bins variáveis)
bins = 4 # Número de quantis
discretized = pd.DataFrame()

for feature in selected_features:
    discretized[feature + '_qcut'] = pd.qcut(df_selected[feature], q=bins, labels=False, duplicates='drop')

# Junta os dados originais com os discretizados
result = pd.concat([df_selected, discretized], axis=1)

# Exibe os primeiros resultados
print(result.head())