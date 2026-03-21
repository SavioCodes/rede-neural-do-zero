# Dados

Esta pasta guarda referencias e pequenos arquivos de dados usados pelos exemplos do projeto.

## Arquivos atuais

### `examples/xor_dataset.csv`

Dataset classico do problema XOR.

Campos:

- `x1`
- `x2`
- `y`

## Como carregar

### Gerar XOR em memoria

```python
from src.utils import DataUtils

X, y = DataUtils.gerar_xor_dataset()
```

### Gerar dataset sintetico de classificacao

```python
from src.utils import DataUtils

X, y = DataUtils.gerar_dataset_classificacao(n_samples=1000, random_state=42)
```

### Ler um CSV customizado

```python
from src.utils import FileUtils

dados = FileUtils.carregar_csv("meu_dataset.csv")
```

## Recomendações

- normalize as features antes do treino
- mantenha treino e teste separados
- garanta consistencia entre numero de colunas e arquitetura da rede
- prefira seeds explicitas para comparacoes

## Proximas extensoes naturais

- datasets pequenos de benchmark para classificacao binaria
- conjuntos de dados externos com script de preparacao
- exemplos de importacao e avaliacao com CSVs customizados
