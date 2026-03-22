# Datasets

O repositorio cobre dois grupos de datasets: sinteticos e reais pequenos empacotados.

## Sinteticos

- `xor`
- `binario`
- `multiclasse`
- `regressao`

Esses datasets sao bons para estudar o comportamento do algoritmo de forma controlada.

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

## Onde cada coisa fica

- `src/datasets/`: CSVs reais distribuidos junto do pacote
- `src/data/`: geracao de datasets sinteticos e utilitarios
- `data/`: notas de apoio do repositorio sobre dados e exemplos
