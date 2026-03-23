# Governanca do Repositorio

Esta pagina concentra as regras oficiais que deixam o projeto organizado, publico e previsivel.

## Protecao de branches

As branches `main` e `develop` sao tratadas como branches protegidas no GitHub.

Regras aplicadas:

- CI obrigatoria antes de merge
- workflow `Branch Policy` obrigatorio antes de merge
- review de code owner obrigatoria quando o PR toca arquivos cobertos pelo `CODEOWNERS`
- conversa do PR precisa estar resolvida
- historico linear obrigatorio
- force push bloqueado
- exclusao da branch protegida bloqueada
- PR e review fazem parte do fluxo oficial

Checks obrigatorios hoje:

- `CI / quality`
- `Branch Policy / branch_name`

## CODEOWNERS

O repositorio agora tem um arquivo oficial em `.github/CODEOWNERS`.

Areas cobertas explicitamente:

- documentacao e site
- datasets, configs e manifests
- nucleo da rede neural
- camada de dados e metricas
- treinamento, callbacks e configuracoes
- workflows experimentais
- CLI publica e interfaces
- workflows e arquivos de governanca do GitHub

O repositorio tambem passa a usar um workflow de reviewers para solicitar revisao automaticamente com base nessas regras quando fizer sentido.

## Labels automaticas

Os pull requests recebem labels automaticamente de acordo com a branch de origem:

- `feat/*` -> `feat`
- `fix/*` -> `fix`
- `docs/*` -> `docs`
- `chore/*` -> `chore`
- `hotfix/*` -> `hotfix`
- `release/*` -> `release`
- `develop -> main` -> `release`
- `main -> develop` -> `governance`

Isso ajuda a bater o olho e entender o tipo de mudanca rapidamente.

## Politica de merge

O repositorio agora padroniza merge por `squash` no GitHub:

- merge commit desativado
- rebase merge desativado
- squash merge mantido como padrao oficial
- ruleset adicional reforcando historico linear em `main` e `develop`

## Sincronizacao depois de hotfix

Quando um PR de `hotfix/*` entra em `main`, o workflow `Hotfix Sync` cria automaticamente um PR de `main` para `develop` se ainda existir diferenca entre as duas branches.

Objetivo:

- evitar que a correcao urgente fique so em `main`
- manter `develop` alinhada com a linha estavel
- reduzir esquecimento manual depois de hotfix

## Template oficial de release PR

Agora existe um template separado para release PR em:

- `.github/PULL_REQUEST_TEMPLATE/release.md`

Use esse template quando a mudanca for:

- `develop -> main`
- `release/* -> main`

## Draft de release notes

O workflow `Release Draft` extrai a secao certa do `CHANGELOG.md` e cria ou atualiza um draft release no GitHub.

Fluxo esperado:

- atualizar `pyproject.toml`, `src/__init__.py` e `CHANGELOG.md`
- validar com `python -m rede_neural_do_zero verify --build-package`
- deixar o workflow montar o draft das release notes
- revisar o draft antes de publicar a release final

## GitHub Pages oficial

O deploy da documentacao agora faz parte do fluxo oficial do repositorio:

- PRs validam build das docs e checagem de links
- `main` publica automaticamente no GitHub Pages
- a URL final esperada e `https://saviocodes.github.io/rede-neural-do-zero/`
- a homepage do repositorio deve apontar para essa mesma URL

## CLI de governanca

A CLI tambem virou interface oficial para inspecionar a governanca:

- `python -m rede_neural_do_zero governance-report`
- `python -m rede_neural_do_zero rules-check`
- `python -m rede_neural_do_zero release-status`
- `python -m rede_neural_do_zero pr-summary`

## Links importantes

- [Docs publicadas](https://saviocodes.github.io/rede-neural-do-zero/)
- [Modelo de branches](./branching.md)
- [Projeto oficial](./project.md)
- [Contribuindo](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/CONTRIBUTING.md)
- [Tags](https://github.com/SavioCodes/rede-neural-do-zero/tags)
- [Releases](https://github.com/SavioCodes/rede-neural-do-zero/releases)
