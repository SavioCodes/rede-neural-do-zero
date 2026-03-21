#!/usr/bin/env python3
"""Synthetic classification experiments for the educational network.

Este exemplo continua sendo uma porta de entrada curta para comparar
arquiteturas binarias sem depender da CLI oficial.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import DataUtils, MetricUtils, ModelConfig, RedeNeural, TrainingConfig  # noqa: E402


def avaliar_modelo(
    X_train,
    X_test,
    y_train,
    y_test,
    arquitetura,
    ativacao: str,
    inicializacao: str,
    epochs: int,
    taxa_aprendizado: float,
    seed: int,
    batch_size: int,
    otimizador: str,
) -> dict:
    rede = RedeNeural.from_config(
        ModelConfig(
            arquitetura=arquitetura,
            ativacao=ativacao,
            inicializacao=inicializacao,
            seed=seed,
            funcao_custo="binary_crossentropy",
        )
    )
    treino = rede.treinar_com_config(
        X_train,
        y_train,
        TrainingConfig(
            epochs=epochs,
            taxa_aprendizado=taxa_aprendizado,
            paciencia=25,
            min_delta=1e-4,
            batch_size=batch_size,
            otimizador=otimizador,
            gradient_clip=1.0,
            verbose=False,
        ),
        validacao_X=X_test,
        validacao_y=y_test,
    )
    resultado = rede.avaliar(X_test, y_test)
    metricas = MetricUtils.precisao_recall_f1(y_test, resultado["predicoes"])
    return {
        "acuracia": resultado["acuracia"],
        "loss": resultado["loss"],
        "erro": resultado["erro"],
        "mse": resultado["mse"],
        "f1": metricas["f1_score"],
        "precisao": metricas["precisao"],
        "recall": metricas["recall"],
        "especificidade": metricas["especificidade"],
        "epocas_executadas": treino["epocas_executadas"],
        "batch_size": treino["batch_size"],
        "otimizador": treino["otimizador"],
    }


def experimento_funcoes_ativacao(samples: int, epochs: int, seed: int) -> None:
    X, y = DataUtils.gerar_dataset_classificacao(n_samples=samples, noise=0.15, random_state=seed)
    X_norm, _ = DataUtils.normalizar_dados(X)
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
        X_norm, y, test_size=0.25, random_state=seed
    )

    print("\nComparando funcoes de ativacao")
    print("------------------------------")
    print("Treino: otimizador=adam, batch_size=32")
    print(f"{'funcao':<14}{'acuracia':<12}{'f1':<10}{'loss':<12}{'epocas':<10}")

    for indice, funcao in enumerate(["sigmoid", "relu", "tanh", "leaky_relu"]):
        inicializacao = "he" if "relu" in funcao else "xavier"
        resultado = avaliar_modelo(
            X_train,
            X_test,
            y_train,
            y_test,
            arquitetura=[2, 10, 6, 1],
            ativacao=funcao,
            inicializacao=inicializacao,
            epochs=epochs,
            taxa_aprendizado=0.01,
            seed=seed + indice,
            batch_size=32,
            otimizador="adam",
        )
        print(
            f"{funcao:<14}{resultado['acuracia']:<12.2f}"
            f"{resultado['f1']:<10.4f}{resultado['loss']:<12.6f}"
            f"{resultado['epocas_executadas']:<10}"
        )


def experimento_normalizacao(samples: int, epochs: int, seed: int) -> None:
    X, y = DataUtils.gerar_dataset_classificacao(n_samples=samples, noise=0.1, random_state=seed)
    X[:, 0] *= 100
    X[:, 1] *= 0.01

    print("\nComparando metodos de normalizacao")
    print("----------------------------------")
    print("Treino: otimizador=adam, batch_size=32")
    print(f"{'metodo':<14}{'acuracia':<12}{'f1':<10}{'loss':<12}{'epocas':<10}")

    for indice, metodo in enumerate(["padrao", "minmax", "robusto"]):
        X_norm, _ = DataUtils.normalizar_dados(X, metodo=metodo)
        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
            X_norm, y, test_size=0.2, random_state=seed
        )
        resultado = avaliar_modelo(
            X_train,
            X_test,
            y_train,
            y_test,
            arquitetura=[2, 8, 1],
            ativacao="relu",
            inicializacao="he",
            epochs=epochs,
            taxa_aprendizado=0.01,
            seed=seed + indice,
            batch_size=32,
            otimizador="adam",
        )
        print(
            f"{metodo:<14}{resultado['acuracia']:<12.2f}"
            f"{resultado['f1']:<10.4f}{resultado['loss']:<12.6f}"
            f"{resultado['epocas_executadas']:<10}"
        )


def experimento_hiperparametros(samples: int, epochs: int, seed: int) -> None:
    X, y = DataUtils.gerar_dataset_classificacao(n_samples=samples, noise=0.1, random_state=seed)
    X_norm, _ = DataUtils.normalizar_dados(X)
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
        X_norm, y, test_size=0.2, random_state=seed
    )

    configuracoes = [
        (0.001, [2, 4, 1]),
        (0.01, [2, 8, 1]),
        (0.01, [2, 8, 4, 1]),
        (0.05, [2, 12, 6, 1]),
    ]

    melhor = None
    print("\nComparando configuracoes")
    print("-----------------------")
    print("Treino: otimizador=adam, batch_size=32")
    print(f"{'lr':<10}{'arquitetura':<18}{'acuracia':<12}{'f1':<10}{'epocas':<10}")

    for indice, (taxa, arquitetura) in enumerate(configuracoes):
        resultado = avaliar_modelo(
            X_train,
            X_test,
            y_train,
            y_test,
            arquitetura=arquitetura,
            ativacao="relu",
            inicializacao="he",
            epochs=epochs,
            taxa_aprendizado=taxa,
            seed=seed + indice,
            batch_size=32,
            otimizador="adam",
        )

        print(
            f"{taxa:<10.3f}{str(arquitetura):<18}"
            f"{resultado['acuracia']:<12.2f}{resultado['f1']:<10.4f}"
            f"{resultado['epocas_executadas']:<10}"
        )

        if melhor is None or resultado["f1"] > melhor[2]:
            melhor = (taxa, arquitetura, resultado["f1"], resultado["acuracia"])

    assert melhor is not None
    print(
        f"\nMelhor configuracao: lr={melhor[0]}, arquitetura={melhor[1]}, "
        f"F1={melhor[2]:.4f}, acuracia={melhor[3]:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exemplos de classificacao sintetica")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Rede Neural do Zero - Classificacao")
    print("===================================")
    print(f"Samples: {args.samples}")
    print(f"Epocas: {args.epochs}")
    print(f"Seed: {args.seed}")
    print("Treino recomendado: otimizador=adam, batch_size=32")
    print("API recomendada: ModelConfig + TrainingConfig")

    experimento_funcoes_ativacao(args.samples, args.epochs, args.seed)
    experimento_normalizacao(args.samples, args.epochs, args.seed)
    experimento_hiperparametros(args.samples, args.epochs, args.seed)


if __name__ == "__main__":
    main()
