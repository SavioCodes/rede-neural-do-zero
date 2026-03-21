# Datasets Reais, Matriz de Confusao e Regressao

> Pagina gerada automaticamente a partir do notebook oficial do projeto.

- Arquivo fonte: `notebooks/03_datasets_reais_e_matriz_confusao.ipynb`

Este notebook usa os datasets reais empacotados com o projeto: Iris, Wine e Diabetes.

## Celula 3

```python
from rede_neural_do_zero import DataUtils, MetricUtils, RedeNeural

X_iris, y_iris, meta_iris = DataUtils.carregar_dataset_iris(normalizar="padrao")
print(meta_iris)
rede_iris = RedeNeural([X_iris.shape[1], 16, 12, 3], ativacao="relu", inicializacao="he", seed=42, funcao_custo="categorical_crossentropy")
rede_iris.treinar(X_iris, y_iris, epochs=120, taxa_aprendizado=0.01, batch_size=16, otimizador="adam", verbose=False)
resultado_iris = rede_iris.avaliar(X_iris, y_iris)
metricas_iris = MetricUtils.metricas_classificacao(y_iris, resultado_iris["predicoes"])
metricas_iris["matriz_confusao"]
```

## Celula 4

```python
X_dia, y_dia, meta_dia = DataUtils.carregar_dataset_diabetes(normalizar="padrao")
rede_reg = RedeNeural([X_dia.shape[1], 32, 16, 1], ativacao="relu", inicializacao="he", seed=42, funcao_custo="mse", ativacao_saida="linear")
rede_reg.treinar(X_dia, y_dia, epochs=160, taxa_aprendizado=0.01, batch_size=32, otimizador="adam", verbose=False)
resultado_reg = rede_reg.avaliar(X_dia, y_dia)
MetricUtils.metricas_regressao(y_dia, resultado_reg["predicoes"])
```

A ideia aqui e comparar o que muda entre classificacao e regressao, tanto na saida quanto nas metricas.
