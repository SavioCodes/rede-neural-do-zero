# Estrutura do Repositorio

Esta pagina documenta a organizacao oficial do projeto para facilitar manutencao, onboarding e contribuicao.

## Visao geral

```text
.
|-- configs/               # configs versionadas para a CLI
|-- data/                  # notas sobre dados e fontes
|-- docs/                  # documentacao fonte do site MkDocs
|-- examples/              # exemplos curtos e didaticos
|-- experiments/           # manifests versionados e saidas locais ignoradas
|-- notebooks/             # notebooks fonte
|-- rede_neural_do_zero/   # pacote publico do PyPI
|-- scripts/               # automacoes de avaliacao, benchmark e docs
|-- src/                   # implementacao principal organizada por dominio
|-- tests/                 # suite de testes automatizados
|-- dist/                  # artefatos de build locais (ignorado no Git)
|-- logs/                  # saidas locais de avaliacao/benchmark (ignorado no Git)
|-- results/               # artefatos locais de treino e exemplos (ignorado no Git)
`-- site/                  # build local do MkDocs (ignorado no Git)
```

## Layout interno de `src`

- `src/core/`: rede neural, forward/backward e funcoes de ativacao.
- `src/training/`: callbacks e configs de treinamento.
- `src/data/`: datasets sinteticos, metricas, plots e CSV helpers.
- `src/workflows/`: treino completo, avaliacao e benchmark.
- `src/interfaces/`: CLI oficial e suporte a `--config`.
- `src/datasets/`: datasets reais empacotados em CSV.

## Compatibilidade

Para manter estabilidade, os caminhos antigos continuam validos:

- `src.rede_neural`
- `src.utils`
- `src.cli`
- `src.benchmarking`
- `src.experiments`

Esses modulos agora sao wrappers finos para a nova estrutura interna. Isso permite reorganizar o projeto sem quebrar os imports ja usados em exemplos, testes e documentacao antiga.

## Convencoes praticas

- Codigo-fonte oficial fica em `src/` e no wrapper publico `rede_neural_do_zero/`.
- Artefatos gerados nao entram no Git: `dist/`, `site/`, `logs/`, `results/` e `experiments/runs/`.
- Configs reproduziveis vivem em `configs/`.
- Manifests de experimentos vivem em `experiments/manifests/`.
- O comando oficial do projeto e `python -m rede_neural_do_zero`.
