# rede-neural-do-zero

[![CI](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-MkDocs%20Material-00897B)](https://saviocodes.github.io/rede-neural-do-zero/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

Implementacao educacional de rede neural do zero com NumPy, agora organizada como pacote instalavel, CLI oficial, notebooks, datasets reais pequenos e checkpoint completo de treino.

## O que o projeto cobre

- forward e backward propagation
- classificacao binaria, multiclasse e regressao
- `sigmoid`, `relu`, `tanh`, `leaky_relu`, `linear` e `softmax`
- `binary_crossentropy`, `categorical_crossentropy` e `mse`
- mini-batch, `SGD` e `Adam`
- `L2`, `dropout` e `gradient clipping`
- `EarlyStopping`, `History`, `CSVLogger` e `ModelCheckpoint`
- salvar e retomar treino completo
- benchmark com multiplas `seeds`, media, desvio e ranking
- datasets reais empacotados: Iris, Wine e Diabetes
- CLI com `train`, `evaluate`, `benchmark` e `example`
- docs web com MkDocs Material
- notebooks didaticos para estudo

## Instalacao

### Desenvolvimento

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### Como pacote

Instalacao local:

```bash
python -m pip install .
```

Instalacao editavel:

```bash
python -m pip install -e ".[dev]"
```

Quando publicado no PyPI:

```bash
python -m pip install rede-neural-do-zero
```

## Uso rapido

### Importando como biblioteca

```python
from rede_neural_do_zero import DataUtils, RedeNeural

X, y, meta = DataUtils.carregar_dataset_iris(normalizar="padrao")

rede = RedeNeural(
    [X.shape[1], 16, 12, 3],
    ativacao="relu",
    inicializacao="he",
    seed=42,
    funcao_custo="categorical_crossentropy",
)

resumo = rede.treinar(
    X,
    y,
    epochs=160,
    taxa_aprendizado=0.01,
    batch_size=16,
    otimizador="adam",
    verbose=False,
)

print(resumo["acuracia_final"])
```

### Regressao

```python
from rede_neural_do_zero import DataUtils, RedeNeural

X, y, _ = DataUtils.carregar_dataset_diabetes(normalizar="padrao")

rede = RedeNeural(
    [X.shape[1], 32, 16, 1],
    ativacao="relu",
    inicializacao="he",
    seed=42,
    funcao_custo="mse",
    ativacao_saida="linear",
)
```

### Checkpoint completo e resume

```python
rede.salvar_checkpoint("results/model-checkpoint.npz")

nova_rede = RedeNeural([X.shape[1], 1], ativacao="relu", funcao_custo="mse", ativacao_saida="linear")
nova_rede.carregar_checkpoint("results/model-checkpoint.npz")
nova_rede.retomar_treinamento(X, y, epochs_adicionais=40, verbose=False)
```

## CLI oficial

Sem instalar script global:

```bash
python -m src --help
```

Comandos principais:

```bash
python -m src train --dataset iris --epochs 160 --save-dir results/iris
python -m src evaluate --dataset diabetes --epochs 180 --min-score 0.20
python -m src benchmark --mode multiclasse --dataset wine --seeds 42,52,62
python -m src example --dataset xor --save-dir results/xor
```

Depois de instalar o pacote, tambem funciona:

```bash
rede-neural-do-zero --help
rnz --help
```

## Datasets disponiveis

### Sinteticos

- `xor`
- `binario`
- `multiclasse`
- `regressao`

### Reais empacotados

- `iris`
- `wine`
- `diabetes`

## Notebooks

Os notebooks ficam em `notebooks/`:

- `01_forward_backward.ipynb`
- `02_softmax_dropout_adam.ipynb`
- `03_datasets_reais_e_matriz_confusao.ipynb`

## Documentacao

- [Landing page e docs web](https://saviocodes.github.io/rede-neural-do-zero/)
- [Teoria](./docs/teoria.md)
- [Algoritmos](./docs/algoritmos.md)
- [Tutorial](./docs/tutorial.md)
- [CLI](./docs/cli.md)
- [Publicacao PyPI](./docs/publishing.md)

## Build e publicacao

Build local:

```bash
python -m build
python -m twine check dist/*
```

O repositório inclui:

- workflow de CI
- workflow de docs com GitHub Pages
- workflow de publicacao no PyPI via Trusted Publishing

## Qualidade

```bash
python -m ruff check .
python -m mypy src rede_neural_do_zero
python -m pytest -q
python -m mkdocs build --strict
```

## Licenca

MIT. Veja [LICENSE](./LICENSE).
