# CLI Oficial

O projeto agora tem uma CLI publica para treino, avaliacao, benchmark e exemplos.

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

Treina um modelo, salva parametros, checkpoint completo e resumo JSON.

```bash
python -m src train --dataset iris --epochs 160 --save-dir results/iris
python -m src train --dataset diabetes --epochs 200 --save-dir results/diabetes
```

### `evaluate`

Roda uma avaliacao deterministica com gate minimo.

```bash
python -m src evaluate --dataset binario --min-score 70
python -m src evaluate --dataset diabetes --min-score 0.20
```

### `benchmark`

Executa benchmark com multiplas `seeds`, media, desvio e ranking.

```bash
python -m src benchmark --mode binario --seeds 42,52,62
python -m src benchmark --mode multiclasse --dataset wine --seeds 42,52,62
python -m src benchmark --mode regressao --dataset diabetes --seeds 42,52,62
```

### `example`

Executa um fluxo pronto com datasets sinteticos ou reais.

```bash
python -m src example --dataset xor
python -m src example --dataset iris
python -m src example --dataset wine
python -m src example --dataset diabetes
```

## Resume de treino

Depois de um `train`, o arquivo `model-checkpoint.npz` pode ser carregado para continuar:

```python
from rede_neural_do_zero import RedeNeural

rede = RedeNeural([4, 1], ativacao_saida="linear", funcao_custo="mse")
rede.carregar_checkpoint("results/diabetes/model-checkpoint.npz")
```
