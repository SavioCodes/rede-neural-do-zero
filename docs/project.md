# Projeto Oficial

Esta pagina centraliza os links, regras e convencoes do repositorio como projeto publico.

## Links oficiais

- [Docs publicadas](https://saviocodes.github.io/rede-neural-do-zero/)
- [Issues](https://github.com/SavioCodes/rede-neural-do-zero/issues)
- [Releases](https://github.com/SavioCodes/rede-neural-do-zero/releases)
- [Tags](https://github.com/SavioCodes/rede-neural-do-zero/tags)
- [Wiki](https://github.com/SavioCodes/rede-neural-do-zero/wiki)
- [Repositorio](https://github.com/SavioCodes/rede-neural-do-zero)

## Arquivos de governanca

- [`CONTRIBUTING.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/CONTRIBUTING.md)
- [`SECURITY.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/SECURITY.md)
- [`SUPPORT.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/SUPPORT.md)
- [`ROADMAP.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/ROADMAP.md)
- [`roadmaps/README.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/roadmaps/README.md)
- [`CHANGELOG.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/CHANGELOG.md)
- [`docs/project-structure.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/docs/project-structure.md)
- [`docs/faq.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/docs/faq.md)

## Como navegar no projeto

Se voce esta chegando agora:

1. leia o README
2. siga por `docs/getting-started.md`
3. use `docs/project-structure.md` para se localizar
4. consulte `docs/faq.md` quando surgir duvida de estrutura

## Branches oficiais

O projeto usa um fluxo simples:

- `main`: estado oficial, estavel e publicavel
- `develop`: integracao da proxima rodada de melhorias
- branches curtas como `feat/*`, `fix/*`, `docs/*`, `chore/*`, `hotfix/*` e `release/*`

Guia rapido:

- novas melhorias devem preferir `develop`
- releases, tags e estado publico seguem saindo de `main`
- hotfixes em `main` devem ser sincronizados de volta para `develop`
- nomes de branch e branch-base de PR sao validados automaticamente no workflow `Branch Policy`
- `main` e `develop` contam com protecao oficial no GitHub
- PRs recebem labels automaticas e existe um `CODEOWNERS` versionado no repositorio

## Governanca do GitHub

Para manter o repositorio mais profissional, o projeto agora tambem tem:

- `CODEOWNERS` granular cobrindo docs, datasets, core, data, training, workflows e interfaces
- reviewers padrao automaticos baseados no `CODEOWNERS`
- labels automaticas a partir da branch do PR
- template de release PR
- sincronizacao automatica depois de hotfix
- draft de release notes e politica de squash merge no GitHub
- workflow `Release Readiness`
- CLI de governanca com `governance-report`, `rules-check`, `release-status`, `release-check` e `pr-summary`

Veja tambem: [`docs/governance.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/docs/governance.md)

## Publicacao

O repositorio esta preparado para:

- build local com wheel e sdist
- deploy de docs via GitHub Pages
- publicacao no PyPI via Trusted Publishing
- releases versionadas com tags no GitHub
- CLI oficial com `python -m rede_neural_do_zero` e os aliases instalados `rede-neural-do-zero` e `rnz`

## Artefatos locais

Arquivos e pastas como `site/`, `dist/`, `results/`, `logs/`, `.coverage` e `*.egg-info/` sao artefatos locais. Eles nao fazem parte da leitura normal do projeto e podem ser apagados sem afetar o codigo-fonte versionado.
