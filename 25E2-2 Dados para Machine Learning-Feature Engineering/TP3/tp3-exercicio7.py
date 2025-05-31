import pandas as pd

# Carregar o dataset
df = pd.read_csv('tempo.csv', encoding='latin1')

# Visualizar as primeiras linhas (opcional)
print(df.head())

# Lista de variáveis categóricas
categorical_vars = ['Aspecto', 'Temp', 'Humidade', 'Vento']

# Aplicar Effect Encoding
df_effect = df.copy()

for var in categorical_vars:
    dummies = pd.get_dummies(df[var])
    # Remover a última categoria para evitar multicolinearidade
    dummies = dummies.iloc[:, :-1]
    # Substituir os 0s por -1
    dummies = dummies.replace(0, -1)
    # Renomear as colunas para identificar a variável
    dummies.columns = [f"{var}_{col}" for col in dummies.columns]
    
    # Concatenar ao dataframe
    df_effect = pd.concat([df_effect, dummies], axis=1)
    
# Remover as variáveis originais categóricas
df_effect = df_effect.drop(columns=categorical_vars)

# Visualizar o DataFrame com Effect Encoding
print(df_effect.head())

