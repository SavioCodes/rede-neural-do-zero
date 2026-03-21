#!/usr/bin/env python3
"""Detailed XOR example with deterministic comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import DataUtils, ModelConfig, RedeNeural, TrainingConfig  # noqa: E402


def imprimir_tabela_verdade() -> None:
    print("\nTabela verdade do XOR")
    print("----------------------")
    print("| A | B | A XOR B |")
    print("| 0 | 0 |    0    |")
    print("| 0 | 1 |    1    |")
    print("| 1 | 0 |    1    |")
    print("| 1 | 1 |    0    |")


def comparar_arquiteturas(epochs: int, taxa_aprendizado: float, seed: int) -> None:
    X, y = DataUtils.gerar_xor_dataset()
    arquiteturas = [
        ([2, 2, 1], "minima"),
        ([2, 3, 1], "pequena"),
        ([2, 4, 1], "padrao"),
        ([2, 4, 3, 1], "duas_camadas"),
    ]

    print("\nComparacao de arquiteturas")
    print("--------------------------")
    print(f"{'nome':<14}{'arquitetura':<16}{'acuracia':<12}{'erro':<12}")

    melhor = None
    for indice, (arquitetura, nome) in enumerate(arquiteturas):
        rede = RedeNeural.from_config(
            ModelConfig(
                arquitetura=arquitetura,
                ativacao="sigmoid",
                inicializacao="xavier",
                seed=seed + indice,
                funcao_custo="binary_crossentropy",
            )
        )
        rede.treinar_com_config(
            X,
            y,
            TrainingConfig(
                epochs=epochs,
                taxa_aprendizado=taxa_aprendizado,
                batch_size=2,
                otimizador="adam",
                embaralhar=False,
                verbose=False,
            ),
        )
        resultado = rede.avaliar(X, y)

        print(
            f"{nome:<14}{str(arquitetura):<16}"
            f"{resultado['acuracia']:<12.2f}{resultado['erro']:<12.6f}"
        )

        if melhor is None or resultado["acuracia"] > melhor[2]:
            melhor = (nome, arquitetura, resultado["acuracia"], resultado["erro"])

    assert melhor is not None
    print(
        f"\nMelhor arquitetura: {melhor[0]} {melhor[1]} "
        f"com {melhor[2]:.2f}% de acuracia e erro {melhor[3]:.6f}"
    )


def treinamento_detalhado(epochs: int, taxa_aprendizado: float, seed: int) -> None:
    X, y = DataUtils.gerar_xor_dataset()
    rede = RedeNeural.from_config(
        ModelConfig(
            arquitetura=[2, 4, 1],
            ativacao="sigmoid",
            inicializacao="xavier",
            seed=seed,
            funcao_custo="binary_crossentropy",
        )
    )

    resumo = rede.treinar_com_config(
        X,
        y,
        TrainingConfig(
            epochs=epochs,
            taxa_aprendizado=taxa_aprendizado,
            batch_size=2,
            otimizador="adam",
            embaralhar=False,
            verbose=False,
        ),
    )
    probabilidades = rede.prever(X)
    classes = rede.prever_classes(X)

    print("\nPredicoes finais")
    print("----------------")
    print(f"{'entrada':<12}{'esperado':<10}{'prob':<10}{'classe':<8}")

    for entrada, esperado, prob, classe in zip(X, y, probabilidades, classes):
        entrada_str = f"[{int(entrada[0])}, {int(entrada[1])}]"
        print(f"{entrada_str:<12}{int(esperado[0]):<10}{prob[0]:<10.4f}{int(classe[0]):<8}")

    print("\nResumo do treino")
    print("----------------")
    print(f"Epocas planejadas: {resumo['epochs_planejadas']}")
    print(f"Epocas executadas: {resumo['epocas_executadas']}")
    print(f"Taxa de aprendizado: {resumo['taxa_aprendizado']}")
    print(f"Otimizador: {resumo['otimizador']}")
    print(f"Batch size: {resumo['batch_size']}")
    print(f"Erro final: {resumo['erro_final']:.6f}")
    print(f"Acuracia final: {resumo['acuracia_final']:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Exemplo detalhado de XOR")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Rede Neural do Zero - XOR")
    print("=========================")
    print(f"Seed: {args.seed}")
    print(f"Epocas: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print("Treino recomendado: otimizador=adam, batch_size=2")
    print("API recomendada: ModelConfig + TrainingConfig")

    imprimir_tabela_verdade()
    treinamento_detalhado(args.epochs, args.learning_rate, args.seed)
    comparar_arquiteturas(args.epochs, args.learning_rate, args.seed)


if __name__ == "__main__":
    main()
