# Notebooks

Os notebooks do projeto ficam na pasta `notebooks/`, sao versionados no repositorio e tambem sao exportados para paginas Markdown dentro da documentacao.

## Arquivos

- `01_forward_backward.ipynb`
- `02_softmax_dropout_adam.ipynb`
- `03_datasets_reais_e_matriz_confusao.ipynb`

## O que cada notebook cobre

### `01_forward_backward.ipynb`

- forward propagation
- backward propagation
- intuicao do gradiente
- leitura das camadas

### `02_softmax_dropout_adam.ipynb`

- `softmax`
- `categorical_crossentropy`
- `dropout`
- `Adam`
- checkpoint completo e resume

### `03_datasets_reais_e_matriz_confusao.ipynb`

- Iris e Wine
- Diabetes para regressao
- matriz de confusao
- comparacao de metricas

## Paginas geradas no site

- [Forward e Backward](notebooks/generated/01-forward-backward.md)
- [Softmax, Dropout e Adam](notebooks/generated/02-softmax-dropout-adam.md)
- [Datasets Reais e Matriz de Confusao](notebooks/generated/03-datasets-reais-e-matriz-confusao.md)

## Como abrir

```bash
python -m pip install -e ".[dev]"
jupyter notebook
```

Se voce preferir, tambem pode abrir os arquivos `.ipynb` direto no VS Code.

## Como regenerar

```bash
python scripts/validate_notebooks.py
python scripts/export_notebooks_to_docs.py
```
