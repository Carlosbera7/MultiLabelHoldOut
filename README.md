# Experimento: Classificação de Discurso de Ódio em Português XGBoost Multi-Label com Holdout

Este repositório contém a implementação do experimento utilizando Xgboost para Multi-Label adapatado de https://gabrielziegler3.medium.com/multiclass-multilabel-classification-with-xgboost-66195e4d9f2d. 

## Descrição do Experimento
O experimento segue as etapas descritas no artigo:

1. **Carregamento dos Dados**:
   - O arquivo CSV 2019-05-28_portuguese_hate_speech_hierarchical_classification_reduzido.csv é carregado.
   - A coluna text é separada como as features (X), e as demais colunas são tratadas como rótulos (y).

2. **Pré-processamento dos Rótulos**:
     - Os rótulos (y) são convertidos para valores numéricos
     - Valores inválidos ou fora do intervalo [0, 1] são substituídos por 0.
     - Valores NaN são preenchidos com 0.
     - Os rótulos são convertidos para inteiros e transformados em uma matriz NumPy.   

3. **Vetorização do Texto**:
   - O texto (X) é vetorizado usando TF-IDF com um limite de 5000 features.
   - Stopwords em português são removidas utilizando a biblioteca NLTK.
      
4. **Divisão dos Dados**:
   - Os dados são divididos em conjuntos de treino e teste utilizando stratificação hierárquica com a função iterative_train_test_split da biblioteca scikit-multilearn.
   - A distribuição das classes nos conjuntos de treino e teste é verificada.
  
5. **Treinamento do Modelo**:
   - Um modelo XGBoost é treinado para cada rótulo (coluna de y).
   - O modelo utiliza a função de objetivo binary:logistic para classificação binária.
     
## Implementação
O experimento foi implementado em Python 3.6 utilizando as bibliotecas:
- pandas
- NLTK
- Scikit-learn
- XGBoost

## Divisão
![Divisao](https://github.com/user-attachments/assets/7da2dc03-7fc2-4680-8d21-094c31f174a9)

O script principal executa as seguintes etapas:
1. Carregamento das partições salvas.
2. Tokenização e padding das sequências de texto.
3. Carregamento dos embeddings GloVe.
4. Construção e treinamento do modelo LSTM.
5. Extração das representações intermediárias.
6. Treinamento e avaliação do XGBoost.
7. Busca de hiperparâmetros com validação cruzada.

## Estrutura do Repositório
- [`Scripts/ClassificadorHierarquicoValido.py`](https://github.com/Carlosbera7/ClassificadorMultiLabel/blob/main/Script/ClassificadorHierarquicoValido.py): Script principal para executar o experimento.
- [`Data/`](https://github.com/Carlosbera7/ClassificadorMultiLabel/tree/main/Data): Pasta contendo o conjunto de dados e o Embeddings GloVe pré-treinados (necessário para execução).
- [`Execução`](https://musical-space-yodel-9rpvjvw9qr39vw4.github.dev/): O código pode ser executado diretamente no ambiente virtual.

## Resultados
Resultados Médios por Seed
Os resultados médios para cada seed de inicialização são apresentados na tabela abaixo:

```
Seed	F1-Score Médio
0	0.6697
1	0.6792
2	0.6621
3	0.6840
4	0.6650
5	0.6639
6	0.6651
7	0.6617
8	0.6727
9	0.6621
```

![media](https://github.com/user-attachments/assets/d5e35965-e59a-44be-87ae-890cb30501f6)







