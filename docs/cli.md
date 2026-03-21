# CLI Oficial

O projeto agora tem uma CLI publica que centraliza treino, resume, avaliacao, benchmark, exemplos e automacao de qualidade.

## Usando sem instalar script global

```bash
python -m src --help
python -m src train --help
```

## Usando via pacote instalado

```bash
rede-neural-do-zero --help
rnz --help
```

## Comandos principais

### `train`

Treina um modelo, salva parametros, checkpoint completo, resumo JSON e a configuracao efetiva da execucao.

```bash
python -m src train --dataset iris --epochs 160 --save-dir results/iris
python -m src train --config configs/train/iris.yaml
```

### `resume`

Retoma um treino completo a partir de um checkpoint salvo.

```bash
python -m src resume --checkpoint results/iris/model-checkpoint.npz --dataset iris --epochs 40
```

### `evaluate`

Roda uma avaliacao deterministica com gate minimo.

```bash
python -m src evaluate --dataset diabetes --min-score 0.20
python -m src evaluate --config configs/evaluate/diabetes.toml
```

### `benchmark`

Executa benchmark com multiplas `seeds`, media, desvio, ranking e relatorio Markdown.

```bash
python -m src benchmark --mode multiclasse --dataset wine --seeds 42,52,62
python -m src benchmark --datasets iris,wine,diabetes --seeds 42,52,62
python -m src benchmark --config configs/benchmark/suite.yaml
```

### `example`

Executa um fluxo pronto com datasets sinteticos ou reais.

```bash
python -m src example --dataset xor
python -m src example --config configs/example/wine.json
```

### `build-docs`

Exporta notebooks e monta o site da documentacao.

```bash
python -m src build-docs --strict
```

### `build-package`

Gera wheel e sdist, com opcao de validar metadados.

```bash
python -m src build-package --check
```

### `verify`

Executa lint, tipos, testes, notebooks, docs e, opcionalmente, build do pacote.

```bash
python -m src verify
python -m src verify --build-package
```

## Arquivos de configuracao

Os subcomandos `train`, `resume`, `evaluate`, `benchmark` e `example` aceitam `--config`.

Formatos suportados:

- JSON
- TOML
- YAML

Estrutura recomendada:

```yaml
common:
  seed: 42
  no_plots: true

train:
  dataset: iris
  epochs: 180
  save_dir: experiments/runs/iris-baseline
```

Se o mesmo campo aparecer no arquivo e na linha de comando, a flag explicita vence.

## Resume programatico

```python
from rede_neural_do_zero import RedeNeural

rede = RedeNeural([4, 1], ativacao_saida="linear", funcao_custo="mse")
rede.carregar_checkpoint("results/diabetes/model-checkpoint.npz")
```
