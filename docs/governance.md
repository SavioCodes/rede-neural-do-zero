# Governanca do Repositorio

Esta pagina concentra as regras oficiais que deixam o projeto organizado, publico e previsivel.

## Protecao de branches

As branches `main` e `develop` sao tratadas como branches protegidas no GitHub.

Regras aplicadas:

- CI obrigatoria antes de merge
- workflow `Branch Policy` obrigatorio antes de merge
- conversa do PR precisa estar resolvida
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
- nucleo da rede neural
- CLI publica e interfaces
- workflows e arquivos de governanca do GitHub

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

## Links importantes

- [Modelo de branches](./branching.md)
- [Projeto oficial](./project.md)
- [Contribuindo](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/CONTRIBUTING.md)
- [Tags](https://github.com/SavioCodes/rede-neural-do-zero/tags)
- [Releases](https://github.com/SavioCodes/rede-neural-do-zero/releases)
