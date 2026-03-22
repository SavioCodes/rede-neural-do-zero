# Projeto Oficial

Esta pagina centraliza os artefatos oficiais e o funcionamento do repositorio como projeto publico.

## Links oficiais

- [Issues](https://github.com/SavioCodes/rede-neural-do-zero/issues)
- [Releases](https://github.com/SavioCodes/rede-neural-do-zero/releases)
- [Tags](https://github.com/SavioCodes/rede-neural-do-zero/tags)
- [Wiki](https://github.com/SavioCodes/rede-neural-do-zero/wiki)
- [Repositorio](https://github.com/SavioCodes/rede-neural-do-zero)

## Arquivos de governanca

- [`CONTRIBUTING.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/CONTRIBUTING.md)
- [`ROADMAP.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/ROADMAP.md)
- [`CHANGELOG.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/CHANGELOG.md)
- [`docs/project-structure.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/docs/project-structure.md)

## Estrutura oficial

O repositorio agora esta organizado por dominios claros:

- `src/core`: logica numerica principal da rede neural
- `src/training`: callbacks e configuracoes
- `src/data`: datasets sinteticos, metricas, plots e IO simples
- `src/workflows`: treino, avaliacao e benchmark
- `src/interfaces`: CLI oficial e suporte a arquivos de config

Os caminhos antigos de `src.*` permanecem disponiveis como wrappers de compatibilidade.

## Branches oficiais

O projeto agora usa um modelo simples de branches:

- `main`: estado oficial, estavel e publicavel
- `develop`: integracao da proxima rodada de melhorias
- branches curtas como `feat/*`, `fix/*`, `docs/*`, `chore/*`, `hotfix/*` e `release/*`

Guia rapido:

- novas melhorias devem preferir `develop`
- releases, tags e estado publico seguem saindo de `main`
- hotfixes em `main` devem ser sincronizados de volta para `develop`
- nomes de branch agora sao validados automaticamente no workflow `Branch Policy`

Detalhes:

- [`docs/branching.md`](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/docs/branching.md)

## Gestao de issues

O projeto agora usa formularios mais guiados para:

- bugs
- melhorias
- documentacao
- perguntas

As issues devem apontar claramente impacto em CLI, docs, configs, benchmark ou publicacao quando isso fizer parte do problema.

## Publicacao

O repositorio esta preparado para:

- build local com wheel e sdist
- deploy de docs via GitHub Pages
- publicacao no PyPI via Trusted Publishing
- releases versionadas com tags no GitHub
- CLI oficial com `python -m rede_neural_do_zero` e os aliases instalados `rede-neural-do-zero` e `rnz`

Enquanto Pages e PyPI nao estiverem ativados externamente, os workflows permanecem preparados e seguros para nao falhar por configuracao ausente.
