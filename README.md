# rede-neural-do-zero

[![CI](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

Implementacao educacional de uma rede neural do zero com NumPy, exemplos reproduziveis, documentacao organizada e pipeline simples de qualidade.

## Visao Geral

Este projeto existe para estudar os fundamentos de redes neurais sem esconder a matematica atras de frameworks.
Ele cobre:

- inicializacao de pesos
- forward propagation
- backpropagation
- funcoes de ativacao
- metricas de classificacao binaria
- geracao e normalizacao de datasets
- avaliacao deterministica para CI

## Destaques

- `RedeNeural(..., seed=...)` para experimentos reproduziveis
- validacoes de entrada no treino, previsao, metricas e split de dados
- `prever_classes()` para converter probabilidades em classes binarias
- historico de treino e validacao salvo no modelo
- `scripts/evaluate.py` gerando artefatos JSON e JSONL em `logs/`
- configuracao centralizada em `pyproject.toml`

## Estrutura

```text
.github/workflows/   # CI
docs/                # Notas teoricas e explicacao dos algoritmos
examples/            # Scripts de demonstracao
logs/                # Artefatos da avaliacao deterministica
src/                 # Implementacao principal
tests/               # Testes unitarios e de integracao
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
python -m pip install -r requirements.txt
python -m pytest -q
```

### Ambiente de desenvolvimento

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
```

## Uso Basico

```python
from src.rede_neural import RedeNeural
from src.utils import DataUtils

X, y = DataUtils.gerar_xor_dataset()

rede = RedeNeural([2, 4, 1], ativacao="sigmoid", inicializacao="xavier", seed=42)
resumo = rede.treinar(X, y, epochs=2000, taxa_aprendizado=0.5, verbose=False)

probabilidades = rede.prever(X)
classes = rede.prever_classes(X)
metricas = rede.avaliar(X, y)

print(resumo)
print(probabilidades)
print(classes)
print(metricas["acuracia"])
```

## Exemplos

```bash
python examples/xor_exemplo.py
python examples/classificacao.py
python examples/exemplo.py --save-dir results/demo --no-plots
```

## Avaliacao Reproduzivel

```bash
python scripts/evaluate.py --seed 42 --epochs 500 --samples 300
```

Arquivos gerados:

- `logs/eval-summary.json`
- `logs/eval-history.jsonl`

O sumario inclui:

- configuracao do modelo
- seed usada
- MSE, acuracia, precisao, recall e F1-score
- matriz de confusao

## Automacao

Comandos principais:

```bash
make install
make install-dev
make lint
make test
make eval
make verify
```

## Documentacao

- [Fundamentos teoricos](./docs/teoria.md)
- [Detalhes dos algoritmos](./docs/algoritmos.md)
- [Notas sobre datasets](./data/README.md)

## Decisoes de Projeto

- O foco e clareza didatica, nao performance de producao.
- A camada de saida usa sigmoid para classificacao binaria.
- O script de avaliacao e deterministico para reduzir flakiness na CI.
- O projeto evita depender do estado global do NumPy quando usa seeds.

## Proximos Passos

- adicionar outras funcoes de custo para comparacao didatica
- exportar artefatos visuais automaticamente em pipelines
- incluir mais exemplos com datasets externos pequenos

## Licenca

MIT. Veja [LICENSE](./LICENSE).
