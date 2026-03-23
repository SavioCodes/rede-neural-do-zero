# Contribuindo

Obrigado por querer melhorar o `rede-neural-do-zero`.

## Objetivo do projeto

O projeto busca equilibrar tres coisas ao mesmo tempo:

- didatica para quem esta aprendendo redes neurais
- qualidade de engenharia para funcionar como pacote e portfolio
- reproducibilidade para exemplos, benchmarks e CI

## Setup de desenvolvimento

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
pre-commit install
```

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
pre-commit install
```

## Fluxo recomendado

1. Crie ou atualize uma branch para sua mudanca.
2. Rode o fluxo de verificacao local com `python -m rede_neural_do_zero verify --build-package`.
3. Atualize documentacao, configs e exemplos quando a API mudar.
4. Se a mudanca impactar usuarios, registre no `CHANGELOG.md`.
5. Rode `python -m rede_neural_do_zero release-check` se a mudanca tocar versao, changelog, release notes ou governanca de release.
6. Abra um pull request descrevendo o problema, a solucao e como foi validado.

## Modelo de branches

O projeto adota um fluxo simples para continuar profissional sem virar burocratico:

- `main`: branch estavel e oficial, usada para tags, releases e estado publico do projeto
- `develop`: branch de integracao para a proxima rodada de melhorias
- `feat/*`: novas funcionalidades
- `fix/*`: correcoes de bugs
- `docs/*`: mudancas de documentacao, wiki e textos oficiais
- `chore/*`: manutencao, tooling e organizacao interna
- `hotfix/*`: correcoes urgentes a partir de `main`
- `release/*`: preparacao de release quando fizer sentido

Fluxo sugerido:

1. abra uma branch curta a partir de `develop`
2. valide com a CLI oficial, inclusive nome da branch e destino do PR
3. integre em `develop`
4. promova `develop` para `main` quando a rodada estiver pronta para release

Se houver hotfix urgente em producao, a correcao pode sair de `main`, mas depois `develop` deve ser sincronizada.

Comando util:

```bash
python -m rede_neural_do_zero check-branch --name feat/add-branch-policy --target develop
```

Destino esperado de PR:

- `feat/*`, `fix/*`, `docs/*` e `chore/*` -> `develop`
- `hotfix/*` e `release/*` -> `main`
- promocao de versao -> `develop` abrindo PR para `main`
- sincronizacao depois de hotfix -> `main` abrindo PR para `develop`
- PR de release devem usar o template `.github/PULL_REQUEST_TEMPLATE/release.md`

## Padroes do repositorio

- Use ASCII por padrao nos arquivos.
- Mantenha a CLI como interface oficial do projeto sempre que fizer sentido.
- Prefira configuracoes versionadas em `configs/` para exemplos reproduziveis.
- Coloque artefatos locais gerados em `results/`, `logs/` ou `experiments/runs/`.
- Nao versione arquivos gerados pesados.

## Checklist de qualidade

```bash
python -m ruff check .
python -m mypy src rede_neural_do_zero
python -m pytest -q
python scripts/validate_notebooks.py
python scripts/export_notebooks_to_docs.py
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

Ou, de forma centralizada:

```bash
python -m rede_neural_do_zero verify --build-package
```

## Mudancas em documentacao

Se voce alterar:

- comandos da CLI
- formato de configs
- exemplos oficiais
- benchmark
- release process

entao atualize tambem README, docs e, quando aplicavel, os links de `tags` e `releases`.

## Governanca automatica

O repositorio tambem possui algumas automacoes oficiais:

- `CODEOWNERS` para docs, nucleo, CLI e workflows
- `SECURITY.md`, `SUPPORT.md` e onboarding oficial para quem esta chegando agora
- reviewers padrao resolvidos a partir do `CODEOWNERS`
- labels automaticas para o tipo de branch do PR
- sincronizacao automatica de `main` para `develop` depois de hotfix
- draft de release notes a partir do `CHANGELOG.md`
- workflow `Release Readiness` para validar versao, changelog e release notes antes da release
- protecao oficial em `main` e `develop` com checks obrigatorios

## Onboarding e suporte

- comece por [docs/onboarding.md](./docs/onboarding.md)
- use [SUPPORT.md](./SUPPORT.md) para saber onde pedir ajuda
- use [SECURITY.md](./SECURITY.md) para relatos sensiveis
- acompanhe o ciclo atual em [roadmaps/README.md](./roadmaps/README.md)
