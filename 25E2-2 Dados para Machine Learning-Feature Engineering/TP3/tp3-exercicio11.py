import pandas as pd
from sklearn.feature_extraction import FeatureHasher

# Carregar o dataset
df = pd.read_csv('tempo.csv', encoding='latin1')

# Selecionar as variáveis categóricas
categorical_vars = ['Aspecto', 'Temp', 'Humidade', 'Vento']

# Criar um dicionário por linha para aplicar FeatureHasher
data_dicts = df[categorical_vars].to_dict(orient='records')

# Definir o número de características desejadas
# Por exemplo, 8 saídas para demonstrar
n_features = 8

# Inicializar o FeatureHasher
hasher = FeatureHasher(n_features=n_features, input_type='string')

# Transformar os dados
hashed_features = hasher.transform([{k: str(v) for k, v in record.items()} for record in data_dicts])

# Converter o resultado para DataFrame
hashed_df = pd.DataFrame(hashed_features.toarray())

# Concatenar com as variáveis restantes, se quiser
df_hashed = pd.concat([df.drop(columns=categorical_vars), hashed_df], axis=1)

# Visualizar o resultado
print(df_hashed.head())

