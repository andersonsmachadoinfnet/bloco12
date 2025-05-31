import os
import tarfile
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

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
def apply_tfidf(train_texts, test_texts, max_features=10000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer

# Execução principal
if __name__ == "__main__":
    download_and_extract_imdb()

    # Carrega dados de treino e teste
    train_texts, train_labels = load_imdb_data(subset='train')
    test_texts, test_labels = load_imdb_data(subset='test')

    # Aplica TF-IDF
    X_train_tfidf, X_test_tfidf, vectorizer = apply_tfidf(train_texts, test_texts)

    # Modelo de Regressão Logística
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, train_labels)

    # Predição e avaliação
    predictions = clf.predict(X_test_tfidf)
    accuracy = accuracy_score(test_labels, predictions)

    print(f"\n Acurácia no conjunto de teste: {accuracy * 100:.2f}%")
