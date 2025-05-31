import os
import tarfile
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Baixar e extrair o dataset
def download_and_extract_imdb(data_dir='aclImdb'):
    url = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    filename = 'aclImdb_v1.tar.gz'

    if not os.path.exists(data_dir):
        print("Baixando o dataset...")
        urllib.request.urlretrieve(url, filename)

        print("Extraindo...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        print("Extração concluída.")

# 2. Carregar textos e rótulos
def load_imdb_data(data_dir='aclImdb', subset='train'):
    data = []
    labels = []

    for sentiment in ['pos', 'neg']:
        dir_path = os.path.join(data_dir, subset, sentiment)
        for fname in os.listdir(dir_path):
            if fname.endswith(".txt"):
                with open(os.path.join(dir_path, fname), encoding='utf-8') as f:
                    data.append(f.read())
                labels.append(1 if sentiment == 'pos' else 0)

    return data, labels

# 3. Aplicar TF-IDF
def apply_tfidf(texts, max_features=10000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_tfidf = vectorizer.fit_transform(texts)
    return X_tfidf, vectorizer

# Executando tudo
if __name__ == "__main__":
    download_and_extract_imdb()
    texts, labels = load_imdb_data(subset='train')  # ou 'test' para os dados de teste
    X_tfidf, vectorizer = apply_tfidf(texts)

    print(f"TF-IDF shape: {X_tfidf.shape}")  # Ex: (25000, 10000)
