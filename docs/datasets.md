# Datasets

O repositório agora cobre dois grupos de datasets:

## Sinteticos

- `xor`
- `binario`
- `multiclasse`
- `regressao`

Esses datasets sao bons para visualizar rapidamente o comportamento do algoritmo.

## Reais empacotados

- `iris`: classificacao multiclasse com 150 amostras
- `wine`: classificacao multiclasse com 178 amostras
- `diabetes`: regressao com 442 amostras

Eles ficam dentro do pacote em `src/datasets/` e podem ser usados mesmo depois de instalar o projeto via `pip`.

## Exemplo de uso

```python
from rede_neural_do_zero import DataUtils

X, y, meta = DataUtils.carregar_dataset_wine(normalizar="padrao")
print(X.shape, y.shape)
print(meta["feature_names"])
print(meta["tipo_tarefa"])
```

## Quando usar cada um

- `xor`: entender nao linearidade
- `binario`: estudar classificacao binaria com fronteira 2D
- `multiclasse`: estudar `softmax` e matriz de confusao
- `regressao`: estudar saida linear e `mse`
- `iris` e `wine`: mostrar que o projeto funciona em datasets reais pequenos
- `diabetes`: demonstrar regressao em um dataset real
