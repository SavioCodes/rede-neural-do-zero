"""Helpers de avaliacao reutilizados por scripts e CLI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .experiments import (
    avaliar_modelo,
    carregar_dataset,
    criar_configs_padrao,
    dividir_treino_validacao_teste,
)
from .rede_neural import RedeNeural


def run_evaluation(seed: int, epochs: int, samples: int, dataset: str, min_score: float) -> dict:
    """Executa um treino deterministico e resume as metricas finais."""
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
    metricas: dict[str, Any] = {"loss": float(avaliacao["loss"]), "mse": float(avaliacao["mse"])}

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

    summary: dict[str, Any] = {
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
