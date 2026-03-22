# Dados

Esta pasta nao guarda o codigo dos datasets do pacote. Ela existe como apoio de documentacao para explicar de onde os dados do projeto vem e como eles sao usados.

## Onde cada tipo de dado fica

- datasets sinteticos: gerados em `src/data/utils.py`
- datasets reais empacotados: arquivos CSV em `src/datasets/`
- notas e contexto do repositorio: esta pasta `data/`

## Datasets sinteticos disponiveis

- `xor`
- `binario`
- `multiclasse`
- `regressao`

Exemplo:

```python
from rede_neural_do_zero import DataUtils

X, y = DataUtils.gerar_dataset_regressao(n_samples=240, random_state=42)
```

## Datasets reais empacotados

Os CSVs reais distribuidos com o pacote sao:

- `iris.csv`
- `wine.csv`
- `diabetes.csv`

Exemplo:

```python
from rede_neural_do_zero import DataUtils

X, y, meta = DataUtils.carregar_dataset_iris(normalizar="padrao")
print(meta["feature_names"])
print(meta["tipo_tarefa"])
```

## Como ler um CSV customizado

```python
from rede_neural_do_zero import FileUtils

dados = FileUtils.carregar_csv("meu_dataset.csv")
```

## Recomendacoes

- normalize as features antes do treino
- mantenha treino, validacao e teste separados
- alinhe a arquitetura com o numero de entradas e saidas
- use `softmax` + `categorical_crossentropy` para multiclasse
- use saida `linear` + `mse` para regressao
