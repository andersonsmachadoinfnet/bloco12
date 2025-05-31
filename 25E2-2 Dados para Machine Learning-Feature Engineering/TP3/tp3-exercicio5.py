import pandas as pd

# Carregar o dataset
df = pd.read_csv('tempo.csv', encoding='latin1')

# Visualizar as primeiras linhas (opcional)
print(df.head())

# Aplicar One-Hot Encoding nas variáveis categóricas
df_encoded = pd.get_dummies(df, columns=['Aspecto', 'Temp', 'Humidade', 'Vento'], drop_first=False)

# Visualizar o DataFrame resultante
print(df_encoded.head())
