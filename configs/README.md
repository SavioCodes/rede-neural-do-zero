# Configs Oficiais

Esta pasta guarda arquivos versionados para reproduzir comandos da CLI sem repetir flags manualmente.

## Exemplos incluidos

- `train/iris.yaml`: treino didatico para Iris
- `evaluate/diabetes.toml`: avaliacao deterministica para regressao
- `benchmark/suite.yaml`: suite multi-dataset para comparar configuracoes
- `example/wine.json`: execucao rapida de exemplo com dataset real

## Como usar

```bash
python -m src train --config configs/train/iris.yaml
python -m src evaluate --config configs/evaluate/diabetes.toml
python -m src benchmark --config configs/benchmark/suite.yaml
python -m src example --config configs/example/wine.json
```

Os arquivos aceitam JSON, TOML e YAML. Campos passados explicitamente na linha de comando continuam tendo prioridade sobre o arquivo.
