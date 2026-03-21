.PHONY: install install-dev test test-cov lint typecheck eval benchmark format verify clean

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
	python -m mypy src

format:
	python -m black src tests scripts examples

eval:
	python scripts/evaluate.py --seed 42 --epochs 500 --samples 300

benchmark:
	python scripts/benchmark.py --mode binario --samples 240 --epochs 120

verify: lint typecheck test-cov eval

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('logs').glob('eval-*.json*')]"
