#!/usr/bin/env python3
"""Benchmark simples para comparar configuracoes do projeto."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    DataUtils,
    FileUtils,
    MetricUtils,
    ModelConfig,
    RedeNeural,
    TrainingConfig,
)


def _gerar_dados(modo: str, amostras: int, seed: int):
    if modo == "multiclasse":
        X, y = DataUtils.gerar_dataset_multiclasse(
            n_samples=amostras,
            random_state=seed,
        )
    else:
        X, y = DataUtils.gerar_dataset_classificacao(
            n_samples=amostras,
            random_state=seed,
        )

    X_norm, _ = DataUtils.normalizar_dados(X)
    return DataUtils.dividir_treino_teste(X_norm, y, test_size=0.25, random_state=seed)


def executar_benchmark(modo: str, amostras: int, seed: int, epochs: int) -> list[dict]:
    X_train, X_test, y_train, y_test = _gerar_dados(modo, amostras, seed)

    if modo == "multiclasse":
        configuracoes = [
            (
                "relu_adam",
                ModelConfig(
                    [2, 16, 12, 3],
                    ativacao="relu",
                    inicializacao="he",
                    seed=seed,
                    funcao_custo="categorical_crossentropy",
                ),
                TrainingConfig(
                    epochs=epochs,
                    taxa_aprendizado=0.01,
                    batch_size=16,
                    otimizador="adam",
                    verbose=False,
                ),
            ),
            (
                "tanh_adam_reg",
                ModelConfig(
                    [2, 12, 3],
                    ativacao="tanh",
                    inicializacao="xavier",
                    seed=seed + 1,
                    funcao_custo="categorical_crossentropy",
                ),
                TrainingConfig(
                    epochs=epochs,
                    taxa_aprendizado=0.01,
                    batch_size=12,
                    otimizador="adam",
                    l2_lambda=1e-3,
                    dropout=0.1,
                    verbose=False,
                ),
            ),
        ]
    else:
        configuracoes = [
            (
                "relu_adam",
                ModelConfig(
                    [2, 8, 1],
                    ativacao="relu",
                    inicializacao="he",
                    seed=seed,
                    funcao_custo="binary_crossentropy",
                ),
                TrainingConfig(
                    epochs=epochs,
                    taxa_aprendizado=0.01,
                    batch_size=16,
                    otimizador="adam",
                    verbose=False,
                ),
            ),
            (
                "sigmoid_sgd",
                ModelConfig(
                    [2, 8, 1],
                    ativacao="sigmoid",
                    inicializacao="xavier",
                    seed=seed + 1,
                    funcao_custo="binary_crossentropy",
                ),
                TrainingConfig(
                    epochs=epochs,
                    taxa_aprendizado=0.1,
                    batch_size=None,
                    otimizador="sgd",
                    verbose=False,
                ),
            ),
        ]

    resultados = []
    for nome, model_config, train_config in configuracoes:
        rede = RedeNeural.from_config(model_config)
        resumo = rede.treinar_com_config(X_train, y_train, train_config)
        avaliacao = rede.avaliar(X_test, y_test)

        linha = {
            "nome": nome,
            "modo": modo,
            "loss": avaliacao["loss"],
            "mse": avaliacao["mse"],
            "acuracia": avaliacao["acuracia"],
            "epochs": resumo["epocas_executadas"],
            "otimizador": resumo["otimizador"],
            "batch_size": resumo["batch_size"],
        }

        if modo == "multiclasse":
            metricas = MetricUtils.metricas_classificacao(y_test, avaliacao["predicoes"])
            linha["f1_macro"] = metricas["f1_macro"]
        else:
            metricas = MetricUtils.precisao_recall_f1(y_test, avaliacao["predicoes"])
            linha["f1"] = metricas["f1_score"]

        resultados.append(linha)

    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simples do projeto")
    parser.add_argument("--mode", choices=["binario", "multiclasse"], default="binario")
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path, default=Path("logs/benchmark.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("logs/benchmark.csv"))
    args = parser.parse_args()

    resultados = executar_benchmark(args.mode, args.samples, args.seed, args.epochs)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(resultados, indent=2), encoding="utf-8")
    FileUtils.salvar_csv(
        {chave: [linha[chave] for linha in resultados] for chave in resultados[0].keys()},
        str(args.csv_output),
    )

    print("Benchmark concluido")
    for linha in resultados:
        print(linha)


if __name__ == "__main__":
    main()
