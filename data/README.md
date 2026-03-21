# Dados

O projeto agora trabalha com datasets sinteticos e com datasets reais pequenos empacotados no proprio pacote.

## Datasets sinteticos

- `XOR`
- classificacao binaria
- classificacao multiclasse
- regressao

Exemplo:

```python
from rede_neural_do_zero import DataUtils

X, y = DataUtils.gerar_dataset_regressao(n_samples=240, random_state=42)
```

## Datasets reais empacotados

Os CSVs reais ficam em `src/datasets/` e sao distribuidos junto com o pacote:

- `iris.csv`
- `wine.csv`
- `diabetes.csv`

Eles podem ser carregados assim:

```python
from rede_neural_do_zero import DataUtils

X, y, meta = DataUtils.carregar_dataset_iris(normalizar="padrao")
print(meta["feature_names"])
print(meta["tipo_tarefa"])
```

## Arquivos auxiliares do repositório

### `examples/xor_dataset.csv`

Dataset classico do problema XOR, mantido como artefato didatico simples.

Campos:

- `x1`
- `x2`
- `y`

## Como ler um CSV customizado

```python
from rede_neural_do_zero import FileUtils

dados = FileUtils.carregar_csv("meu_dataset.csv")
```

## Recomendacoes

- normalize as features antes do treino
- mantenha treino, validacao e teste separados
- alinhe a arquitetura com o numero de features e saidas
- use `softmax` + `categorical_crossentropy` para multiclasse
- use saida `linear` + `mse` para regressao
- prefira seeds explicitas para comparacoes e benchmarks
