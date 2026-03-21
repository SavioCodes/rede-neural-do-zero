#!/usr/bin/env python3
"""Deterministic evaluation entrypoint for CI and local validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import RedeNeural  # noqa: E402
from src.experiments import (  # noqa: E402
    avaliar_modelo,
    carregar_dataset,
    criar_configs_padrao,
    dividir_treino_validacao_teste,
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Tipo nao serializavel: {type(value)!r}")


def run_evaluation(seed: int, epochs: int, samples: int, dataset: str, min_score: float) -> dict:
    bundle = carregar_dataset(dataset, seed=seed, samples=samples)
    splits = dividir_treino_validacao_teste(bundle.X, bundle.y, seed=seed)
    model_config, train_config = criar_configs_padrao(
        bundle, seed=seed, epochs=epochs, verbose=False
    )

    model = RedeNeural.from_config(model_config)
    treino = model.treinar_com_config(
        splits["X_train"],
        splits["y_train"],
        train_config,
        validacao_X=splits["X_val"],
        validacao_y=splits["y_val"],
    )

    avaliacao = avaliar_modelo(model, splits["X_test"], splits["y_test"])
    metricas = {"loss": float(avaliacao["loss"]), "mse": float(avaliacao["mse"])}

    if avaliacao["tipo_problema"].startswith("regressao"):
        metricas.update(
            {
                "mae": float(avaliacao["mae"]),
                "rmse": float(avaliacao["rmse"]),
                "r2": float(avaliacao["r2"]),
                "score_principal": float(avaliacao["r2"]),
                "nome_score_principal": "r2",
            }
        )
    elif avaliacao["tipo_problema"] == "classificacao_multiclasse":
        classificacao = avaliacao["metricas_classificacao"]
        metricas.update(
            {
                "accuracy": float(avaliacao["acuracia"]),
                "f1_macro": float(classificacao["f1_macro"]),
                "confusion_matrix": classificacao["matriz_confusao"].tolist(),
                "score_principal": float(avaliacao["acuracia"]),
                "nome_score_principal": "accuracy",
            }
        )
    else:
        classificacao = avaliacao["metricas_classificacao"]
        metricas.update(
            {
                "accuracy": float(avaliacao["acuracia"]),
                "precision": float(classificacao["precisao"]),
                "recall": float(classificacao["recall"]),
                "specificity": float(classificacao["especificidade"]),
                "balanced_accuracy": float(classificacao["balanced_accuracy"]),
                "f1_score": float(classificacao["f1_score"]),
                "confusion_matrix": classificacao["matriz_confusao"].tolist(),
                "score_principal": float(avaliacao["acuracia"]),
                "nome_score_principal": "accuracy",
            }
        )

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "epochs": epochs,
        "samples": samples,
        "dataset": bundle.nome,
        "dataset_metadata": bundle.metadata,
        "data_split": {
            "train_samples": int(splits["X_train"].shape[0]),
            "validation_samples": int(splits["X_val"].shape[0]),
            "test_samples": int(splits["X_test"].shape[0]),
        },
        "model": model.resumir_modelo(),
        "training": treino,
        "metrics": metricas,
        "thresholds": {
            "min_score": float(min_score),
            "metric_name": metricas["nome_score_principal"],
        },
    }

    if summary["metrics"]["score_principal"] < min_score:
        raise SystemExit(
            f"Score gate failed: {summary['metrics']['score_principal']:.4f} < {min_score:.4f}"
        )

    return summary


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
