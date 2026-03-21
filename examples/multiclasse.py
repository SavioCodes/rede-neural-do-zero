#!/usr/bin/env python3
"""Exemplo de classificacao multiclasse com softmax, callbacks e plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import (  # noqa: E402
    CSVLogger,
    DataUtils,
    MetricUtils,
    ModelCheckpoint,
    ModelConfig,
    RedeNeural,
    TrainingConfig,
    VisualizationUtils,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exemplo multiclasse com softmax")
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path("results/multiclasse"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    X, y = DataUtils.gerar_dataset_multiclasse(
        n_samples=args.samples,
        random_state=args.seed,
    )
    X_norm, _ = DataUtils.normalizar_dados(X)
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
        X_norm,
        y,
        test_size=0.25,
        random_state=args.seed,
    )
    y_train_one_hot = DataUtils.one_hot_encode(y_train, n_classes=3)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    rede = RedeNeural.from_config(
        ModelConfig(
            arquitetura=[2, 16, 12, 3],
            ativacao="relu",
            inicializacao="he",
            seed=args.seed,
            funcao_custo="categorical_crossentropy",
        )
    )
    resumo = rede.treinar_com_config(
        X_train,
        y_train_one_hot,
        TrainingConfig(
            epochs=args.epochs,
            taxa_aprendizado=0.01,
            batch_size=16,
            otimizador="adam",
            l2_lambda=1e-3,
            dropout=0.1,
            gradient_clip=1.0,
            callbacks=[
                CSVLogger(str(args.save_dir / "treino_multiclasse.csv")),
                ModelCheckpoint(str(args.save_dir / "melhor_multiclasse.npz"), monitor="loss"),
            ],
            verbose=False,
        ),
    )
    resultado = rede.avaliar(X_test, y_test)
    metricas = MetricUtils.metricas_classificacao(y_test, resultado["predicoes"])

    print("Rede Neural do Zero - Multiclasse")
    print("=================================")
    print(f"Acuracia: {resultado['acuracia']:.2f}%")
    print(f"Loss: {resultado['loss']:.6f}")
    print(f"F1 macro: {metricas['f1_macro']:.4f}")
    print(f"Epocas executadas: {resumo['epocas_executadas']}")
    print(f"Callbacks: {', '.join(resumo['callbacks']) if resumo['callbacks'] else 'nenhum'}")
    print(f"Matriz de confusao:\n{metricas['matriz_confusao']}")

    if not args.no_plots:
        VisualizationUtils.plotar_historico_treinamento(
            rede.historico_erro,
            rede.historico_acuracia,
            salvar=str(args.save_dir / "historico_multiclasse.png"),
            mostrar=False,
        )
        VisualizationUtils.plotar_fronteira_decisao(
            rede,
            X_test,
            y_test,
            salvar=str(args.save_dir / "fronteira_multiclasse.png"),
            mostrar=False,
        )
        VisualizationUtils.plotar_matriz_confusao(
            metricas["matriz_confusao"],
            labels=["0", "1", "2"],
            salvar=str(args.save_dir / "matriz_multiclasse.png"),
            mostrar=False,
        )


if __name__ == "__main__":
    main()
