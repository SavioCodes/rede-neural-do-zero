# rede-neural-do-zero

[![CI](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

Implementacao educacional de uma rede neural do zero com NumPy, testes automatizados e pipeline de avaliacao reproduzivel.

## Visao Geral

Este repositorio existe para estudar o funcionamento interno de uma rede neural sem depender de frameworks de alto nivel.
O projeto cobre:

- inicializacao de pesos
- forward propagation
- backpropagation
- funcoes de ativacao
- utilitarios de dados e metricas
- exemplos executaveis
- avaliacao deterministica para CI

## Destaques

- `RedeNeural(..., seed=...)` para experimentos reproduziveis sem depender do estado global do NumPy
- validacoes de entrada para treino, previsao, split e normalizacao
- `prever_classes()` para converter probabilidades em classes binarias
- historico de treino e de validacao salvo no objeto
- script de avaliacao que gera artefatos JSON e JSONL em `logs/`

## Estrutura

```text
src/         # Implementacao principal da rede neural e utilitarios
tests/       # Testes unitarios e de integracao
examples/    # Exemplos de uso
scripts/     # Scripts auxiliares, incluindo avaliacao deterministica
docs/        # Notas teoricas e algoritmos
logs/        # Saida da avaliacao (mantido com .gitkeep)
data/        # Dados e notas de datasets
```

## Setup Rapido

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Uso Basico

```python
import numpy as np

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

## Avaliacao Reproduzivel

```bash
python scripts/evaluate.py --seed 42 --epochs 500 --samples 300
```

Arquivos gerados:

- `logs/eval-summary.json`
- `logs/eval-history.jsonl`

O sumario salvo inclui configuracao do modelo, seed, metrica de erro, acuracia, precisao, recall e F1-score.

## Qualidade

Comandos principais:

```bash
python -m pytest -q
python scripts/evaluate.py --seed 42 --epochs 400 --samples 240
```

CI atual:

- executa testes automatizados
- roda um smoke test deterministico da avaliacao
- usa Python 3.11

## Decisoes de Projeto

- O foco e aprendizado e clareza, nao performance de producao.
- A camada de saida usa sigmoid para classificacao binaria.
- A avaliacao e deterministicamente configurada para reduzir flakiness na CI.
- Os utilitarios evitam alterar o estado global do gerador aleatorio quando recebem `random_state`.

## Proximos Passos

- adicionar comparacoes de hiperparametros com exportacao de resultados
- gerar artefatos de visualizacao para analise de fronteira de decisao
- expandir exemplos com persistencia de modelos e avaliacao em datasets customizados

## Licenca

MIT. Veja [LICENSE](./LICENSE).
