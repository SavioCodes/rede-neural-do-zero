# Experimentos Versionados

Esta pasta organiza os experimentos oficiais do projeto.

## Estrutura

- `manifests/`: descreve comandos, configs e saidas esperadas
- `runs/`: destino padrao para artefatos locais gerados pela CLI

## Fluxo recomendado

1. Escolha um manifesto em `experiments/manifests/`
2. Rode o comando correspondente com `python -m src`
3. Confira os artefatos gerados em `experiments/runs/`
4. Versione apenas manifests e configuracoes, nao os resultados pesados
