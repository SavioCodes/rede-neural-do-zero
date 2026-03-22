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
- `hotfix/*`: correcoes urgentes que nasceram de `main`
- `release/*`: preparacao de release

## Padrao exato

Nomes aceitos:

- `main`
- `develop`
- `feat/<slug-kebab-case>`
- `fix/<slug-kebab-case>`
- `docs/<slug-kebab-case>`
- `chore/<slug-kebab-case>`
- `hotfix/<slug-kebab-case>`
- `release/vX.Y.Z`

## Exemplos validos

- `feat/add-branch-policy`
- `fix/checkpoint-restore-bug`
- `docs/update-wiki-links`
- `chore/reorganize-ci-cache`
- `hotfix/fix-release-link`
- `release/v2.2.5`

## Exemplos invalidos

- `feature/nova-coisa`
- `docs/wiki links`
- `Feat/maiuscula`
- `release/2.2.5`
- `minha-branch`

## Fluxo sugerido

1. criar uma branch curta a partir de `develop`
2. implementar a mudanca
3. validar com `python -m rede_neural_do_zero verify --build-package`
4. integrar em `develop`
5. promover `develop` para `main` quando a rodada estiver pronta para release

## Verificacao automatica

O repositorio agora valida esse padrao de tres formas:

- comando local: `python -m rede_neural_do_zero check-branch --name feat/add-branch-policy`
- script leve para automacao: `python src/interfaces/branch_policy.py --name feat/add-branch-policy --json`
- workflow `Branch Policy` no GitHub Actions para pushes e pull requests

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
