#!/usr/bin/env python3
"""Deterministic evaluation entrypoint for CI and local validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rede_neural import RedeNeural  # noqa: E402
from src.utils import DataUtils, MetricUtils  # noqa: E402


def run_evaluation(seed: int, epochs: int, samples: int, min_accuracy: float) -> dict:
    X, y = DataUtils.gerar_dataset_classificacao(
        n_samples=samples,
        n_features=2,
        noise=0.12,
        random_state=seed,
    )
    X_norm, _ = DataUtils.normalizar_dados(X, metodo="padrao")
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
        X_norm,
        y,
        test_size=0.2,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = DataUtils.dividir_treino_teste(
        X_train,
        y_train,
        test_size=0.25,
        random_state=seed,
    )

    model = RedeNeural(
        [2, 8, 1],
        ativacao="relu",
        inicializacao="he",
        seed=seed,
        funcao_custo="binary_crossentropy",
    )
    treino = model.treinar(
        X_train,
        y_train,
        epochs=epochs,
        taxa_aprendizado=0.01,
        validacao_X=X_val,
        validacao_y=y_val,
        paciencia=40,
        min_delta=1e-4,
        batch_size=32,
        otimizador="adam",
        verbose=False,
    )

    eval_result = model.avaliar(X_test, y_test)
    prf = MetricUtils.precisao_recall_f1(y_test, eval_result["predicoes"])

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "epochs": epochs,
        "samples": samples,
        "data_split": {
            "train_samples": int(X_train.shape[0]),
            "validation_samples": int(X_val.shape[0]),
            "test_samples": int(X_test.shape[0]),
        },
        "model": {
            **model.resumir_modelo(),
        },
        "training": treino,
        "metrics": {
            "loss": float(eval_result["loss"]),
            "mse": float(eval_result["mse"]),
            "accuracy": float(eval_result["acuracia"]),
            "precision": float(prf["precisao"]),
            "recall": float(prf["recall"]),
            "specificity": float(prf["especificidade"]),
            "balanced_accuracy": float(prf["balanced_accuracy"]),
            "f1_score": float(prf["f1_score"]),
            "confusion_matrix": prf["matriz_confusao"].tolist(),
        },
        "thresholds": {
            "min_accuracy": float(min_accuracy),
        },
    }

    if summary["metrics"]["accuracy"] < min_accuracy:
        raise SystemExit(
            f"Accuracy gate failed: {summary['metrics']['accuracy']:.2f} < {min_accuracy:.2f}"
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic NN evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--min-accuracy", type=float, default=65.0)
    parser.add_argument("--json-output", type=Path, default=Path("logs/eval-summary.json"))
    parser.add_argument("--history-output", type=Path, default=Path("logs/eval-history.jsonl"))
    args = parser.parse_args()

    summary = run_evaluation(args.seed, args.epochs, args.samples, args.min_accuracy)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.history_output.parent.mkdir(parents=True, exist_ok=True)

    args.json_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.history_output.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(summary) + "\n")

    print("Evaluation completed")
    print(f"Accuracy: {summary['metrics']['accuracy']:.2f}%")
    print(f"Loss: {summary['metrics']['loss']:.6f}")
    print(f"MSE: {summary['metrics']['mse']:.6f}")
    print(f"Output: {args.json_output}")


if __name__ == "__main__":
    main()
