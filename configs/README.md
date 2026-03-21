# Configs Oficiais

Esta pasta guarda arquivos versionados para reproduzir comandos da CLI sem repetir flags manualmente.

## Exemplos incluidos

- `train/iris.yaml`: treino didatico para Iris
- `evaluate/diabetes.toml`: avaliacao deterministica para regressao
- `benchmark/suite.yaml`: suite multi-dataset para comparar configuracoes
- `example/wine.json`: execucao rapida de exemplo com dataset real

## Como usar

```bash
python -m rede_neural_do_zero train --config configs/train/iris.yaml
python -m rede_neural_do_zero evaluate --config configs/evaluate/diabetes.toml
python -m rede_neural_do_zero benchmark --config configs/benchmark/suite.yaml
python -m rede_neural_do_zero example --config configs/example/wine.json
```

Os arquivos aceitam JSON, TOML e YAML. Campos passados explicitamente na linha de comando continuam tendo prioridade sobre o arquivo.

Para compatibilidade, `python -m src ...` ainda funciona, mas a interface oficial do projeto agora e `python -m rede_neural_do_zero ...`.

