# rede-neural-do-zero

[![CI](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

Implementacao educacional de uma rede neural do zero com NumPy, focada em estudo, reproducibilidade e evolucao incremental de recursos importantes de ML.

## Visao Geral

Este projeto existe para ensinar os fundamentos de redes neurais sem esconder a matematica atras de frameworks.
Hoje o repositório cobre:

- inicializacao de pesos
- forward propagation e backpropagation
- classificacao binaria e multiclasse
- funcoes de ativacao e funcoes de custo
- mini-batch training, `SGD` e `Adam`
- regularizacao com `L2`, `dropout` e `gradient clipping`
- callbacks como `EarlyStopping`, `CSVLogger`, `History` e `ModelCheckpoint`
- configs organizadas com `ModelConfig` e `TrainingConfig`
- metricas, visualizacoes e benchmark simples

## Destaques

- `RedeNeural(..., seed=...)` para experimentos reproduziveis
- `funcao_custo="binary_crossentropy"`, `"categorical_crossentropy"` ou `"mse"`
- `ativacao_saida` automatica: `sigmoid` para binario e `softmax` para multiclasse
- `treinar(..., batch_size=..., otimizador="sgd"|"adam")`
- regularizacao por `l2_lambda`, `dropout` e `gradient_clip`
- callbacks reutilizaveis em `src/callbacks.py`
- `ModelConfig` e `TrainingConfig` para deixar a API mais organizada
- `scripts/evaluate.py` para smoke deterministico
- `scripts/benchmark.py` para comparar configuracoes
- `pytest`, `coverage`, `ruff`, `mypy` e `pre-commit`

## Estrutura

```text
.github/workflows/   # CI
docs/                # teoria, algoritmos e tutorial
examples/            # scripts de demonstracao
logs/                # artefatos de avaliacao e benchmark
scripts/             # avaliacao e benchmark
src/                 # implementacao principal
tests/               # testes unitarios e de integracao
```

## Setup Rapido

### Runtime basico

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
python -m pip install -r requirements.txt
python -m pytest -q
```

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

### Ambiente de desenvolvimento

```bash
python -m pip install -r requirements-dev.txt
pre-commit install
python -m ruff check .
python -m mypy src
python -m pytest -q
```

## Uso Basico

### API direta

```python
from src import DataUtils, RedeNeural

X, y = DataUtils.gerar_xor_dataset()

rede = RedeNeural(
    [2, 4, 1],
    ativacao="sigmoid",
    inicializacao="xavier",
    seed=42,
    funcao_custo="binary_crossentropy",
)
resumo = rede.treinar(
    X,
    y,
    epochs=1200,
    taxa_aprendizado=0.05,
    batch_size=2,
    otimizador="adam",
    embaralhar=False,
    verbose=False,
)

print(resumo["acuracia_final"])
print(rede.prever_classes(X))
```

### API organizada com configs

```python
from src import ModelConfig, RedeNeural, TrainingConfig

modelo = RedeNeural.from_config(
    ModelConfig(
        arquitetura=[2, 16, 12, 3],
        ativacao="relu",
        inicializacao="he",
        seed=42,
        funcao_custo="categorical_crossentropy",
    )
)

config_treino = TrainingConfig(
    epochs=160,
    taxa_aprendizado=0.01,
    batch_size=16,
    otimizador="adam",
    l2_lambda=1e-3,
    dropout=0.1,
    gradient_clip=1.0,
    verbose=False,
)
```

## Exemplos

```bash
python examples/xor_exemplo.py
python examples/classificacao.py
python examples/multiclasse.py --save-dir results/multiclasse --no-plots
python examples/exemplo.py --save-dir results/demo --no-plots
```

## Avaliacao e Benchmark

### Avaliacao deterministica

```bash
python scripts/evaluate.py --seed 42 --epochs 500 --samples 300
```

Arquivos gerados:

- `logs/eval-summary.json`
- `logs/eval-history.jsonl`

### Benchmark simples

```bash
python scripts/benchmark.py --mode binario --samples 240 --epochs 120
python scripts/benchmark.py --mode multiclasse --samples 240 --epochs 120
```

Arquivos gerados:

- `logs/benchmark.json`
- `logs/benchmark.csv`

## Automacao

Comandos principais:

```bash
make install
make install-dev
make lint
make typecheck
make test
make test-cov
make eval
make benchmark
make verify
```

## Documentacao

- [Fundamentos teoricos](./docs/teoria.md)
- [Detalhes dos algoritmos](./docs/algoritmos.md)
- [Tutorial guiado](./docs/tutorial.md)
- [Notas sobre datasets](./data/README.md)

## Decisoes de Projeto

- O foco e clareza didatica, nao performance de producao.
- A API publica tenta equilibrar simplicidade e extensibilidade.
- Classificacao binaria usa `sigmoid`; multiclasse usa `softmax`.
- `batch_size=None` mantem o treino em batch completo para estudo.
- `Adam` e a opcao recomendada para exemplos maiores.
- O script de avaliacao continua deterministico para reduzir flakiness na CI.
- O projeto evita depender do estado global do NumPy quando usa seeds.

## Qualidade Atual

- `ruff` para lint
- `mypy` para type check
- `pytest` com `coverage`
- `pre-commit` para checagens locais
- workflow de CI com lint, type-check, testes, avaliacao e benchmark smoke

## Licenca

MIT. Veja [LICENSE](./LICENSE).
