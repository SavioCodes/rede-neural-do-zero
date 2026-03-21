.PHONY: install install-dev test lint eval verify clean

install:
	python -m pip install -r requirements.txt

install-dev:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest -q

lint:
	python -m ruff check .

eval:
	python scripts/evaluate.py --seed 42 --epochs 500 --samples 300

verify: lint test eval

clean:
	python -c "from pathlib import Path; [p.unlink() for p in Path('logs').glob('eval-*.json*')]"
