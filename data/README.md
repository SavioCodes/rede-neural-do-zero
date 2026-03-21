# Dados

Esta pasta guarda referencias e pequenos arquivos de dados usados pelos exemplos do projeto.

## Arquivos atuais

### `examples/xor_dataset.csv`

Dataset classico do problema XOR.

Campos:

- `x1`
- `x2`
- `y`

## Como gerar dados em memoria

### XOR

```python
from src import DataUtils

X, y = DataUtils.gerar_xor_dataset()
```

### Classificacao binaria

```python
from src import DataUtils

X, y = DataUtils.gerar_dataset_classificacao(n_samples=1000, random_state=42)
```

### Classificacao multiclasse

```python
from src import DataUtils

X, y = DataUtils.gerar_dataset_multiclasse(
    n_samples=600,
    n_classes=3,
    random_state=42,
)
```

## Como ler um CSV customizado

```python
from src import FileUtils

dados = FileUtils.carregar_csv("meu_dataset.csv")
```

## Recomendacoes

- normalize as features antes do treino
- mantenha treino e teste separados
- garanta consistencia entre numero de colunas e arquitetura da rede
- prefira seeds explicitas para comparacoes
- para multiclasse, escolha `categorical_crossentropy` e saida `softmax`

## Proximas extensoes naturais

- datasets pequenos de benchmark para classificacao binaria e multiclasse
- conjuntos de dados externos com script de preparacao
- exemplos de importacao e avaliacao com CSVs customizados
