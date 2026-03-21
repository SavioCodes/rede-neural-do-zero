# Changelog

Todas as mudancas relevantes deste projeto passam a ser registradas aqui seguindo versionamento semantico.

## [2.1.0] - 2026-03-21

### Added

- suporte oficial a arquivos de configuracao da CLI em JSON, TOML e YAML
- comandos centrais da CLI para `resume`, `build-docs`, `build-package` e `verify`
- pasta `configs/` com exemplos reutilizaveis e versionados
- pasta `experiments/` com manifests oficiais para execucoes reproduziveis
- benchmark em suite multi-dataset com relatorio Markdown e leaderboard
- exportacao automatica de notebooks para paginas da documentacao
- validacao automatica de notebooks para CI
- referencia de API com `mkdocstrings`
- `CONTRIBUTING.md`, `ROADMAP.md`, templates de issues e template de pull request

### Changed

- documentacao reorganizada para destacar CLI, configuracoes, releases e governanca
- scripts `evaluate.py` e `benchmark.py` passaram a reutilizar a camada oficial de `src`
- automacao do projeto foi alinhada para usar a CLI como interface central

## [2.0.3] - 2026-03-21

### Fixed

- serializacao JSON da avaliacao deterministica usada no smoke de CI

## [2.0.2] - 2026-03-21

### Changed

- gating do deploy de Pages e da publicacao no PyPI ate a configuracao externa do repositorio estar pronta

## [2.0.1] - 2026-03-21

### Fixed

- estabilizacao dos workflows de CI, docs e publicacao

## [2.0.0] - 2026-03-21

### Added

- CLI oficial do projeto
- docs com MkDocs Material
- checkpoint completo de treino e `resume`
- datasets reais empacotados
- suporte a regressao

## [1.5.0] - 2026-03-21

### Added

- suporte a classificacao multiclasse
- callbacks de treinamento
- melhoria forte em tooling, exemplos e documentacao

## [0.1.0] - 2026-03-14

### Added

- baseline publico inicial do projeto com README e sinais de engenharia
