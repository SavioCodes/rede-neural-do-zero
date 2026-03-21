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

## Verificacao rapida

```bash
python -m rede_neural_do_zero verify
python -m rede_neural_do_zero evaluate --dataset binario --seed 42 --epochs 150 --samples 240
python -m rede_neural_do_zero benchmark --mode multiclasse --epochs 80 --seeds 42,52,62
```

## Primeiros comandos uteis

```bash
python -m rede_neural_do_zero train --config configs/train/iris.yaml
python -m rede_neural_do_zero evaluate --config configs/evaluate/diabetes.toml
python -m rede_neural_do_zero benchmark --config configs/benchmark/suite.yaml
python -m rede_neural_do_zero example --config configs/example/wine.json
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

## Pastas importantes

- `configs/`: configuracoes oficiais reutilizaveis
- `experiments/manifests/`: manifestos versionados
- `experiments/runs/`: destino sugerido para artefatos locais
- `docs/`: documentacao oficial

