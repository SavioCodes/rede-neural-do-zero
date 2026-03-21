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
	python -m src evaluate --dataset binario --seed 42 --epochs 180 --samples 240

benchmark:
	python -m src benchmark --config configs/benchmark/suite.yaml

notebooks:
	python scripts/validate_notebooks.py
	python scripts/export_notebooks_to_docs.py

build:
	python -m src build-package --check

docs:
	python -m src build-docs --strict

docs-serve:
	python -m mkdocs serve

verify:
	python -m src verify --build-package

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('logs').glob('*') if p.is_file() and p.name != '.gitkeep']"
