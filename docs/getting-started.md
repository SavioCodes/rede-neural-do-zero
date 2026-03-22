# Instalar e Rodar

Esta pagina serve para quem quer ver o projeto funcionando sem precisar entender toda a estrutura primeiro.

## Instalar o ambiente

### Linux ou macOS

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

### Instalacao editavel

```bash
python -m pip install -e ".[dev]"
```

## Primeiros 10 minutos

### 1. Validar o projeto

```bash
python -m rede_neural_do_zero verify --build-package
```

### 2. Rodar um exemplo pronto

```bash
python -m rede_neural_do_zero example --config configs/example/wine.json
```

### 3. Treinar um modelo

```bash
python -m rede_neural_do_zero train --config configs/train/iris.yaml
```

### 4. Rodar uma avaliacao

```bash
python -m rede_neural_do_zero evaluate --config configs/evaluate/diabetes.toml
```

## Se voce quer ler o codigo

Comece nesta ordem:

1. `src/core/` para a logica numerica da rede
2. `src/training/` para callbacks e configuracoes
3. `src/data/` para datasets, metricas e visualizacao
4. `src/interfaces/` para a CLI

O pacote publico para quem usa a biblioteca e `rede_neural_do_zero`.

## Pastas importantes

- `configs/`: comandos prontos para a CLI
- `examples/`: exemplos simples e curtos
- `experiments/manifests/`: execucoes reproduziveis
- `docs/`: documentacao oficial
- `tests/`: suite automatizada
