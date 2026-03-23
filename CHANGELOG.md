# Changelog

Todas as mudancas relevantes deste projeto passam a ser registradas aqui seguindo versionamento semantico.

## [2.3.0] - 2026-03-23

### Added

- `CODEOWNERS` oficial cobrindo documentacao, nucleo, CLI e workflows
- labels automaticas de PR a partir do prefixo da branch e do fluxo entre `main` e `develop`
- workflow `Hotfix Sync` para abrir PR de sincronizacao de `main` para `develop` depois de hotfix
- template oficial de release PR e documentacao dedicada de governanca

### Changed

- governanca do repositorio agora documenta branch protection, labels, hotfix sync e release PR
- `main` e `develop` passaram a ter configuracao oficial de protecao no GitHub
- README, contribuicao e docs oficiais ficaram mais claros sobre ownership e fluxo de release

## [2.2.6] - 2026-03-23

### Added

- validacao do destino correto do pull request para reduzir erro entre `main` e `develop`
- cobertura de testes para branch-base correta no fluxo oficial

### Changed

- comando `check-branch` agora aceita `--target`
- workflow `Branch Policy` passou a validar tambem a branch-base dos pull requests
- templates e docs de branches agora mostram com mais clareza quando usar `develop` ou `main`

## [2.2.5] - 2026-03-22

### Changed

- documentacao principal reorganizada para onboarding mais claro e leitura mais rapida
- explicacoes de estrutura, CLI, datasets, testes e artefatos locais ficaram mais diretas
- exemplos, scripts e pasta `data/` foram alinhados com o fluxo oficial atual
- `Makefile` agora usa a CLI oficial em vez de `python -m src`
- alvo `make clean` ficou mais completo para limpar artefatos locais comuns

### Removed

- arquivo `examples/xor_dataset.csv`, que estava parado e nao fazia parte do fluxo oficial

## [2.2.4] - 2026-03-22

### Fixed

- prioridade da deteccao de branch agora respeita `BRANCH_NAME` antes do ambiente do GitHub Actions
- cobertura de testes do `branch_policy` foi ajustada para validar o cenario real da CI

## [2.2.3] - 2026-03-22

### Added

- comando `check-branch` na CLI oficial para validar nomes de branch
- workflow `Branch Policy` para validar nomes de branch em pushes e pull requests
- testes automatizados para o padrao oficial de nomes de branch

### Changed

- CI principal agora roda em `main` e `develop`
- documentacao oficial ganhou exemplos validos e invalidos de nomes de branch
- template de pull request agora lembra o padrao oficial de branches

## [2.2.2] - 2026-03-22

### Added

- branch oficial `develop` para integracao da proxima rodada de melhorias
- documentacao oficial do modelo de branches em `docs/branching.md`

### Changed

- `README.md`, `CONTRIBUTING.md` e docs oficiais agora descrevem o fluxo de branches do projeto

## [2.2.1] - 2026-03-21

### Added

- wiki oficial do projeto no GitHub com paginas de onboarding, CLI, datasets, estrutura e publicacao

### Changed

- README e documentacao oficial agora apontam para a wiki publica do projeto

## [2.2.0] - 2026-03-21

### Added

- pagina oficial de estrutura do repositorio em `docs/project-structure.md`
- `README.md` internos para `src/`, `examples/`, `scripts/` e `tests/`
- testes cobrindo a entrada publica `python -m rede_neural_do_zero`

### Changed

- codigo-fonte reorganizado em subpacotes: `src/core`, `src/training`, `src/data`, `src/workflows` e `src/interfaces`
- modulos antigos de `src.*` passaram a funcionar como camadas de compatibilidade
- CLI, workflows, templates e documentacao passaram a destacar `python -m rede_neural_do_zero` como interface oficial
- referencia de API e documentacao agora refletem a nova organizacao interna do projeto

## [2.1.1] - 2026-03-21

### Added

- formularios de issue para documentacao e perguntas
- links oficiais de roadmap, changelog e docs dentro da configuracao de issues

### Changed

- formularios de bug e feature ficaram mais guiados por area afetada
- pagina oficial do projeto agora destaca melhor a gestao de issues

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
