# Experimento: Classificação de Discurso de Ódio em Português XGBoost Multi-Label com Holdout 10x

Este repositório contém a implementação do experimento utilizando Xgboost para Multi-Label adapatado de https://gabrielziegler3.medium.com/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d, aplicando a técnica de holdout 10x. 

## Descrição do Experimento
O experimento segue as etapas descritas no artigo:

1. **Carregamento dos Dados**:
   - O arquivo CSV 2019-05-28_portuguese_hate_speech_hierarchical_classification_reduzido.csv é carregado.   

2. **Limpeza de Texto:**:
     - Remove caracteres especiais, converte para minúsculas e exclui stopwords em português.
     - Mantém apenas os rótulos com pelo menos 10 exemplos positivos.
      
3. **Divisão dos Dados**:
   - Utiliza iterative_train_test_split para garantir uma divisão balanceada entre os conjuntos de treino e teste.
  
4. **Treinamento do Modelo**:
   - Modelos individuais são treinados para cada rótulo usando a estratégia One-vs-Rest.
   
     
## Implementação
O experimento foi implementado em Python 3.6 utilizando as bibliotecas:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `skmultilearn`
- `nltk`

## Estrutura do Repositório
- [`Script/MultiLabelHD.py`](https://github.com/Carlosbera7/MultiLabelHoldOut/blob/main/Script/MultiLabelHD.py): Script principal para executar o experimento.
- [`Data/`](https://github.com/Carlosbera7/MultiLabelHoldOut/tree/main/Data): Pasta contendo o conjunto de dados.

## Divisão
![distribuição](https://github.com/user-attachments/assets/aaf08dd0-6d50-442d-97f9-9f40698210f8)


## Resultados
Resultados Médios por Seed
Os resultados médios para cada seed de inicialização são apresentados na tabela abaixo:

## 📊 Resultados por Seed

| Seed | F1-Score Médio |
|------|----------------|
| 0    | 0.6697         |
| 1    | 0.6792         |
| 2    | 0.6621         |
| 3    | 0.6840         |
| 4    | 0.6650         |
| 5    | 0.6639         |
| 6    | 0.6651         |
| 7    | 0.6617         |
| 8    | 0.6727         |
| 9    | 0.6621         |

- **F1-score médio geral:** 0.6686  


![media](https://github.com/user-attachments/assets/d5e35965-e59a-44be-87ae-890cb30501f6)







