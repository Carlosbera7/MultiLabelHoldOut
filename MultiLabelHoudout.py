import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from skmultilearn.model_selection import iterative_train_test_split
from nltk.corpus import stopwords
import logging
import numpy as np

nltk.download('stopwords')

# Configurações de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para limpar o texto
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', str(text).lower())
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in text.split() if word not in stop_words]
    return ' '.join(words)

# Função para carregar e preparar os dados
def load_and_prepare_data(file_path):
    try:
        logging.info("Carregando os dados...")
        data = pd.read_csv(file_path)
        data['text'] = data['text'].apply(clean_text)
        X = data['text']
        y = data.drop(columns=['text'])
        return X, y
    except FileNotFoundError:
        logging.error(f"Arquivo {file_path} não encontrado.")
        return None, None

# Filtrar rótulos com frequência mínima
def filter_labels(y, min_count=10):
    label_counts = y.sum(axis=0)
    valid_labels = label_counts[label_counts >= min_count].index
    return y[valid_labels]

# Treinar e avaliar os modelos
def train_and_evaluate(X_train, y_train, X_test, y_test, seed):
    params = {
        'max_depth': 6,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    models = {}
    predictions = {}

    for label_idx in range(y_train.shape[1]):
        logging.info(f"Treinando modelo para o rótulo: {label_idx} (Seed {seed})")
        dtrain_label = xgb.DMatrix(data=X_train, label=y_train[:, label_idx])
        model = xgb.train(params, dtrain_label, num_boost_round=100)
        models[label_idx] = model

        logging.info(f"Fazendo previsões para o rótulo: {label_idx} (Seed {seed})")
        dtest_label = xgb.DMatrix(data=X_test)
        predictions[label_idx] = model.predict(dtest_label)

    predictions_df = pd.DataFrame(predictions)
    return predictions_df

# Função principal
def gerar():
    X, y = load_and_prepare_data('2019-05-28_portuguese_hate_speech_hierarchical_classification.csv')
    if X is None or y is None:
        return

    logging.info(f"Rótulos Originais: {list(y.columns)}  {list(y.sum(axis=0))}")
    y = filter_labels(y)
    logging.info(f"Rótulos mantidos: {list(y.columns)}  {list(y.sum(axis=0))}")

    portuguese_stopwords = stopwords.words('portuguese')
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=portuguese_stopwords)
    X_tfidf = vectorizer.fit_transform(X)

    results = []  # Para armazenar métricas para cada seed
    for seed in range(10):
        logging.info(f"Dividindo os dados (Seed {seed})...")
        np.random.seed(seed)  # Definir seed
        X_train, y_train, X_test, y_test = iterative_train_test_split(X_tfidf, y.values, test_size=0.3)

        logging.info(f"Formatos: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
        predictions_df = train_and_evaluate(X_train, y_train, X_test, y_test, seed)

        # Calcular métricas
        seed_metrics = []
        for label_idx in range(y_test.shape[1]):
            report = classification_report(
                y_test[:, label_idx],
                (predictions_df[label_idx] >= 0.5).astype(int),
                zero_division=0,
                output_dict=True
            )
            seed_metrics.append(report['accuracy'])  # Acurácia para o rótulo
        results.append(np.mean(seed_metrics))  # Média de acurácia para a seed

    # Exibir médias
    logging.info("\nResultados Médios:")
    for seed, acc in enumerate(results):
        logging.info(f"Seed {seed}: Acurácia média = {acc:.4f}")

    logging.info(f"Acurácia Geral (Média de todas as seeds): {np.mean(results):.4f}")

if __name__ == "__main__":
    gerar()
