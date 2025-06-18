from sklearn.datasets import load_breast_cancer
import pandas as pd

# Carregando o dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)

# Verificando tipos de dados
print(df.dtypes)