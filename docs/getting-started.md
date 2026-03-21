# Instalar e Rodar

## Instalacao rapida

### Desenvolvimento

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pre-commit install
```

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
pre-commit install
```

### Instalacao como pacote

```bash
python -m pip install .
```

### Instalacao editavel com extras

```bash
python -m pip install -e ".[dev]"
```

## Testes rapidos

```bash
python -m pytest -q
python scripts/evaluate.py --dataset binario --seed 42 --epochs 150 --samples 240
python scripts/benchmark.py --mode multiclasse --epochs 80 --seeds 42,52,62
```

## Primeiros comandos uteis

```bash
python -m src train --dataset iris --epochs 160 --save-dir results/iris
python -m src evaluate --dataset diabetes --epochs 180 --min-score 0.20
python -m src benchmark --mode regressao --seeds 42,52,62
python -m src example --dataset wine --save-dir results/wine
```

## Importacao via Python

```python
from rede_neural_do_zero import DataUtils, RedeNeural

X, y, meta = DataUtils.carregar_dataset_iris(normalizar="padrao")
rede = RedeNeural(
    [X.shape[1], 16, 12, 3],
    ativacao="relu",
    inicializacao="he",
    seed=42,
    funcao_custo="categorical_crossentropy",
)
```
