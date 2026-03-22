# Modelo de Branches

Esta pagina documenta o fluxo oficial de branches do projeto.

## Branches permanentes

- `main`: branch estavel e publica do repositorio
- `develop`: branch de integracao para a proxima rodada de melhorias

## Branches curtas recomendadas

- `feat/*`: novas funcionalidades
- `fix/*`: correcoes
- `docs/*`: documentacao, wiki e textos oficiais
- `chore/*`: manutencao, limpeza e tooling
- `release/*`: preparacao de release

## Fluxo sugerido

1. criar uma branch curta a partir de `develop`
2. implementar a mudanca
3. validar com `python -m rede_neural_do_zero verify --build-package`
4. integrar em `develop`
5. promover `develop` para `main` quando a rodada estiver pronta para release

## Regras praticas

- `main` deve representar o estado mais confiavel do projeto
- tags e releases saem de `main`
- `develop` pode receber trabalho em andamento da proxima versao
- mudancas urgentes em `main` devem ser sincronizadas de volta para `develop`

## Objetivo desse modelo

O foco aqui e manter o projeto:

- profissional
- organizado
- facil de manter
- simples o bastante para um repositorio educacional e de portfolio
