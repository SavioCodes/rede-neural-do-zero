.PHONY: install install-dev test test-cov lint typecheck eval benchmark format docs docs-serve build verify notebooks clean

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest -q

test-cov:
	python -m pytest -q --cov=src --cov-report=term-missing

lint:
	python -m ruff check .

typecheck:
	python -m mypy src rede_neural_do_zero

format:
	python -m black src tests scripts examples rede_neural_do_zero

eval:
	python -m rede_neural_do_zero evaluate --dataset binario --seed 42 --epochs 180 --samples 240

benchmark:
	python -m rede_neural_do_zero benchmark --config configs/benchmark/suite.yaml

notebooks:
	python scripts/validate_notebooks.py
	python scripts/export_notebooks_to_docs.py

build:
	python -m rede_neural_do_zero build-package --check

docs:
	python -m rede_neural_do_zero build-docs --strict

docs-serve:
	python -m mkdocs serve

verify:
	python -m rede_neural_do_zero verify --build-package

clean:
	python -c "from pathlib import Path; import shutil; \
for path in ['.coverage', 'build', 'dist', 'site', '.mypy_cache', '.pytest_cache', '.ruff_cache', 'results', 'rede_neural_do_zero.egg-info']: \
    p = Path(path); \
    (shutil.rmtree(p) if p.exists() and p.is_dir() else p.unlink() if p.exists() else None); \
logs = Path('logs'); \
[(shutil.rmtree(p) if p.is_dir() else p.unlink()) for p in logs.iterdir() if logs.exists() and p.name != '.gitkeep']; \
runs = Path('experiments/runs'); \
[(shutil.rmtree(p) if p.is_dir() else p.unlink()) for p in runs.iterdir() if runs.exists() and p.name != '.gitkeep']"
