import pandas as pd

# Carregar o dataset
df = pd.read_csv('tempo.csv', encoding='latin1')

# Visualizar as primeiras linhas (opcional)
print(df.head())

# Aplicar Dummy Encoding: remove a primeira categoria de cada variável
# Supondo que as variáveis categóricas sejam: 'Aspecto', 'Temp', 'Humidade', 'Vento'
df_dummy = pd.get_dummies(df, columns=['Aspecto', 'Temp', 'Humidade', 'Vento'], drop_first=True)

# Visualizar o DataFrame resultante
print(df_dummy.head())
