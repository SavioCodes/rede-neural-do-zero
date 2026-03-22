# Estrutura do Repositorio

Esta pagina documenta a organizacao oficial do projeto para facilitar onboarding, manutencao e contribuicao.

## Visao geral

```text
.
|-- configs/               # configs versionadas para a CLI
|-- data/                  # notas de apoio sobre dados e fontes
|-- docs/                  # documentacao fonte do site MkDocs
|-- examples/              # exemplos curtos e didaticos
|-- experiments/           # manifests versionados e saidas locais ignoradas
|-- notebooks/             # notebooks fonte
|-- rede_neural_do_zero/   # pacote publico do PyPI
|-- scripts/               # automacoes de avaliacao, benchmark e docs
|-- src/                   # implementacao principal organizada por dominio
|-- tests/                 # suite de testes automatizados
|-- dist/                  # artefatos locais de build (ignorado no Git)
|-- logs/                  # saidas locais de avaliacao/benchmark (ignorado no Git)
|-- results/               # artefatos locais de treino e exemplos (ignorado no Git)
`-- site/                  # build local do MkDocs (ignorado no Git)
```

## O que editar em cada tipo de mudanca

- quer mudar a logica numerica da rede: `src/core/`
- quer mexer em callbacks ou configs: `src/training/`
- quer mexer em datasets, metricas ou plots: `src/data/`
- quer mexer em treino, avaliacao ou benchmark: `src/workflows/`
- quer mexer na CLI ou em `--config`: `src/interfaces/`
- quer atualizar datasets empacotados: `src/datasets/`
- quer atualizar onboarding e explicacoes: `README.md` e `docs/`
- quer proteger comportamento: `tests/`

## O que e publico e o que e interno

- `rede_neural_do_zero/` e a interface publica recomendada para quem instala o pacote
- `src/` concentra a implementacao interna do repositorio
- `python -m rede_neural_do_zero` e o comando oficial
- `python -m src` foi mantido por compatibilidade com material antigo

## Compatibilidade

Alguns caminhos antigos de `src.*` continuam validos como wrappers:

- `src.rede_neural`
- `src.utils`
- `src.cli`
- `src.benchmarking`
- `src.experiments`

Eles existem para nao quebrar exemplos, testes e material antigo enquanto a estrutura interna fica mais organizada.

## O que pode ser ignorado ou apagado localmente

- `dist/`
- `site/`
- `logs/`
- `results/`
- `.coverage`
- `*.egg-info/`

Esses itens sao gerados localmente durante teste, build, docs ou exemplos. Eles nao fazem parte do codigo-fonte oficial.
