# CLI Oficial

O projeto tem uma CLI publica que centraliza treino, resume, avaliacao, benchmark, exemplos e automacao de qualidade.

## Usando sem instalar script global

```bash
python -m rede_neural_do_zero --help
python -m rede_neural_do_zero train --help
```

## Usando via pacote instalado

```bash
rede-neural-do-zero --help
rnz --help
```

## Quatro comandos para comecar

```bash
python -m rede_neural_do_zero example --config configs/example/wine.json
python -m rede_neural_do_zero train --config configs/train/iris.yaml
python -m rede_neural_do_zero evaluate --config configs/evaluate/diabetes.toml
python -m rede_neural_do_zero verify --build-package
```

## Comandos principais

### `train`

Treina um modelo, salva parametros, checkpoint completo, resumo JSON e a configuracao efetiva da execucao.

```bash
python -m rede_neural_do_zero train --dataset iris --epochs 160 --save-dir results/iris
python -m rede_neural_do_zero train --config configs/train/iris.yaml
```

### `resume`

Retoma um treino completo a partir de um checkpoint salvo.

```bash
python -m rede_neural_do_zero resume --checkpoint results/iris/model-checkpoint.npz --dataset iris --epochs 40
```

### `evaluate`

Roda uma avaliacao deterministica com gate minimo.

```bash
python -m rede_neural_do_zero evaluate --dataset diabetes --min-score 0.20
python -m rede_neural_do_zero evaluate --config configs/evaluate/diabetes.toml
```

### `benchmark`

Executa benchmark com multiplas `seeds`, media, desvio, ranking e relatorio Markdown.

```bash
python -m rede_neural_do_zero benchmark --mode multiclasse --dataset wine --seeds 42,52,62
python -m rede_neural_do_zero benchmark --datasets iris,wine,diabetes --seeds 42,52,62
python -m rede_neural_do_zero benchmark --config configs/benchmark/suite.yaml
```

### `example`

Executa um fluxo pronto com datasets sinteticos ou reais.

```bash
python -m rede_neural_do_zero example --dataset xor
python -m rede_neural_do_zero example --config configs/example/wine.json
```

### `build-docs`

Exporta notebooks e monta o site da documentacao.

```bash
python -m rede_neural_do_zero build-docs --strict
```

### `build-package`

Gera wheel e sdist, com opcao de validar metadados.

```bash
python -m rede_neural_do_zero build-package --check
```

### `check-branch`

Valida nomes de branch pelo padrao oficial do projeto.

```bash
python -m rede_neural_do_zero check-branch --name feat/add-branch-policy
python -m rede_neural_do_zero check-branch --name docs/update-wiki-links
```

### `verify`

Executa lint, tipos, testes, notebooks, docs e, opcionalmente, build do pacote.

```bash
python -m rede_neural_do_zero verify
python -m rede_neural_do_zero verify --build-package
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

## Qual comando usar em cada caso

- quer ver o projeto rodando: `example`
- quer treinar e salvar artefatos: `train`
- quer continuar um treino salvo: `resume`
- quer medir um setup com menos variacao: `evaluate`
- quer comparar configuracoes: `benchmark`
- quer validar o repositorio inteiro: `verify`
