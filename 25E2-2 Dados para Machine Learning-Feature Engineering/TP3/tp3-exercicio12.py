import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Carregar o dataset
df = pd.read_csv('tempo.csv', encoding='latin1')

# Selecionar as variáveis categóricas
categorical_vars = ['Aspecto', 'Temp', 'Humidade', 'Vento']

# Inicializar um dicionário para armazenar os codificadores
label_encoders = {}

# Aplicar Label Encoding para cada variável categórica
for var in categorical_vars:
    le = LabelEncoder()
    df[var + '_encoded'] = le.fit_transform(df[var])
    label_encoders[var] = le

# Agora, criar uma matriz de contagem binária
# Vamos assumir que cada amostra tem apenas uma categoria por variável
# O tamanho total do vetor será a soma das categorias únicas de cada variável

n_total_bins = sum(len(le.classes_) for le in label_encoders.values())

# Inicializar a matriz de contagem
bin_count_matrix = np.zeros((len(df), n_total_bins), dtype=int)

current_idx = 0

for var in categorical_vars:
    le = label_encoders[var]
    num_classes = len(le.classes_)
    encoded_col = df[var + '_encoded']
    
    for i, val in enumerate(encoded_col):
        bin_count_matrix[i, current_idx + val] += 1
    
    current_idx += num_classes

# Converter para DataFrame
bin_count_df = pd.DataFrame(bin_count_matrix, columns=[f'bin_{i}' for i in range(n_total_bins)])

# Concatenar com outras colunas (por exemplo, 'Dia', 'Jogar ténis')
df_bin_count = pd.concat([df[['Dia', 'Jogar ténis']], bin_count_df], axis=1)

# Visualizar o DataFrame
print(df_bin_count.head())
