# Scripts de automacao

Os scripts desta pasta apoiam tarefas de manutencao, validacao e reproducao fora da CLI principal.

## Scripts incluidos

- `evaluate.py`: avaliacao deterministica usada em smoke tests e CI.
- `benchmark.py`: benchmark rapido ou suite multi-dataset.
- `export_notebooks_to_docs.py`: converte notebooks para paginas Markdown da documentacao.
- `validate_notebooks.py`: valida notebooks antes de publicar docs.

Quando possivel, prefira a CLI oficial (`python -m rede_neural_do_zero ...`) e use os scripts como apoio de automacao.
