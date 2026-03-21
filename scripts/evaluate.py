#!/usr/bin/env python3
"""Entrypoint deterministico de avaliacao reutilizando `src.evaluation`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import run_evaluation  # noqa: E402


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Tipo nao serializavel: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic NN evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument(
        "--dataset",
        choices=["binario", "multiclasse", "regressao", "iris", "wine", "diabetes"],
        default="binario",
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--json-output", type=Path, default=Path("logs/eval-summary.json"))
    parser.add_argument("--history-output", type=Path, default=Path("logs/eval-history.jsonl"))
    args = parser.parse_args()

    min_score = args.min_score
    if min_score is None:
        min_score = 0.20 if args.dataset in {"regressao", "diabetes"} else 65.0

    summary = run_evaluation(args.seed, args.epochs, args.samples, args.dataset, min_score)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.history_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(summary, indent=2, default=_json_default),
        encoding="utf-8",
    )
    with args.history_output.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(summary, default=_json_default) + "\n")

    print("Evaluation completed")
    print(f"Dataset: {summary['dataset']}")
    print(
        f"{summary['metrics']['nome_score_principal']}: "
        f"{summary['metrics']['score_principal']:.4f}"
    )
    print(f"Loss: {summary['metrics']['loss']:.6f}")
    print(f"MSE: {summary['metrics']['mse']:.6f}")
    print(f"Output: {args.json_output}")


if __name__ == "__main__":
    main()
