# Notebooks

Os notebooks do projeto ficam na pasta `notebooks/` e foram pensados para apoiar estudo e demonstracao.

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

## Como abrir

```bash
python -m pip install -e ".[dev]"
jupyter notebook
```

Se voce preferir, tambem pode abrir os arquivos `.ipynb` direto no VS Code.
