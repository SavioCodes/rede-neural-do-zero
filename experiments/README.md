# Experimentos Versionados

Esta pasta organiza os experimentos oficiais do projeto.

## Estrutura

- `manifests/`: descreve comandos, configs e saidas esperadas
- `runs/`: destino padrao para artefatos locais gerados pela CLI

## Fluxo recomendado

1. escolha um manifesto em `experiments/manifests/`
2. rode o comando correspondente com `python -m rede_neural_do_zero`
3. confira os artefatos gerados em `experiments/runs/`
4. versione apenas manifests e configuracoes, nao os resultados pesados
