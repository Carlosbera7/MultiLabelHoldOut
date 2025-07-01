import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report
import os
import re
import logging
import numpy as np


class ModelTrainer:
    def __init__(self, partition_dir):
        self.partition_dir = partition_dir
        self.models = {}

    def load_partitions(self):
        logging.info("Carregando partições salvas...")
        X_train = pd.read_csv(os.path.join(self.partition_dir, 'X_train.csv')).values
        X_test = pd.read_csv(os.path.join(self.partition_dir, 'X_test.csv')).values
        y_train = pd.read_csv(os.path.join(self.partition_dir, 'y_train.csv')).values
        y_test = pd.read_csv(os.path.join(self.partition_dir, 'y_test.csv')).values
        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        params = {
            'max_depth': 6,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        predictions = {}

        for label_idx in range(y_train.shape[1]):
            logging.info(f"Treinando modelo para o rótulo: {label_idx}")
            dtrain_label = xgb.DMatrix(data=X_train, label=y_train[:, label_idx])
            model = xgb.train(params, dtrain_label, num_boost_round=100)
            self.models[label_idx] = model

            # Salvar modelo treinado
            model_path = os.path.join(self.partition_dir, f"xgb_label_{label_idx}.json")
            model.save_model(model_path)
            logging.info(f"Modelo salvo em {model_path}")

            logging.info(f"Fazendo previsões para o rótulo: {label_idx}")
            dtest_label = xgb.DMatrix(data=X_test)
            predictions[label_idx] = model.predict(dtest_label)

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(os.path.join(self.partition_dir, "predictions.csv"), index=False)
        logging.info("Previsões salvas em predictions.csv")

        for label_idx in range(y_test.shape[1]):
            logging.info(f"Avaliando o rótulo: {label_idx}")
            print(classification_report(
                y_test[:, label_idx],
                (predictions[label_idx] >= 0.5).astype(int),
                zero_division=0
            ))
        return predictions_df  # <-- Adicione este retorno

    def run(self):
        results = []  # Para armazenar métricas para cada seed
        for seed in range(10):
            logging.info(f"Dividindo os dados (Seed {seed})...")
            np.random.seed(seed)  # Definir seed
            X_train, X_test, y_train, y_test = self.load_partitions()
            
            logging.info(f"Formatos: X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
            predictions_df = self.train_and_evaluate(X_train, y_train, X_test, y_test)

            # Calcular métricas
            seed_metrics = []
            for label_idx in range(y_test.shape[1]):
                report = classification_report(
                    y_test[:, label_idx],
                    (predictions_df[label_idx] >= 0.5).astype(int),
                    zero_division=0,
                    output_dict=True
                )
                seed_metrics.append(report['macro avg']['f1-score'])  # Acurácia para o rótulo
            results.append(np.mean(seed_metrics))  # Média de acurácia para a seed

        # Exibir médias
        logging.info("\nResultados Médios por Seed:")
        for seed, f1 in enumerate(results):
            print(f"Seed {seed}: F1-score médio = {f1:.4f}")

        mean_f1 = np.mean(results)
        std_f1 = np.std(results)
        print(f"\nF1-score médio geral: {mean_f1:.4f}")
        print(f"Desvio padrão do F1-score: {std_f1:.4f}")
       

if __name__ == "__main__":
      
    model_trainer = ModelTrainer(partition_dir='./partitions')
    model_trainer.run()
