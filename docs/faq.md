# FAQ

## Qual e o comando oficial do projeto?

Use `python -m rede_neural_do_zero`.

`python -m src` ainda existe por compatibilidade com material antigo, mas nao e mais o caminho recomendado para uso novo.

## Por que existem `src/` e `rede_neural_do_zero/`?

- `src/` concentra a implementacao interna do repositorio
- `rede_neural_do_zero/` expoe a interface publica do pacote instalado

Isso deixa o projeto mais facil de evoluir sem quebrar quem ja usa a interface oficial.

## Onde eu comeco a ler o codigo?

Se voce quer entender a rede:

1. `src/core/`
2. `src/training/`
3. `src/data/`
4. `src/interfaces/`

Se voce quer usar o projeto antes de entender tudo, comece pela CLI.

## Quais pastas sao codigo-fonte de verdade?

- `src/`
- `rede_neural_do_zero/`
- `tests/`
- `docs/`
- `configs/`

## O que e so artefato local?

Voce pode apagar sem medo quando aparecer:

- `site/`
- `dist/`
- `results/`
- `logs/`
- `.coverage`
- `*.egg-info/`

Esses itens sao recriados por build, docs, exemplos e verificacoes locais.

Se quiser limpar tudo de uma vez, use:

```bash
make clean
```

## Onde entram novas mudancas?

- logica da rede: `src/core/`
- CLI e config: `src/interfaces/`
- treino, avaliacao e benchmark: `src/workflows/`
- datasets, metricas e plots: `src/data/`
- documentacao: `README.md` e `docs/`
- testes: `tests/`
