.PHONY: install test eval verify

install:
	pip install -r requirements.txt

test:
	pytest -q

eval:
	python scripts/evaluate.py --seed 42 --epochs 500 --samples 300

verify: test eval
