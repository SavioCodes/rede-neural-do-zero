#!/usr/bin/env python3
"""End-to-end example showing training, evaluation, and artifact export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rede_neural import RedeNeural  # noqa: E402
from src.utils import DataUtils, MetricUtils, VisualizationUtils  # noqa: E402


def treinar_xor(seed: int) -> tuple[RedeNeural, dict]:
    X_xor, y_xor = DataUtils.gerar_xor_dataset()
    rede = RedeNeural(
        [2, 4, 1],
        ativacao="sigmoid",
        inicializacao="xavier",
        seed=seed,
        funcao_custo="binary_crossentropy",
    )
    rede.treinar(
        X_xor,
        y_xor,
        epochs=1200,
        taxa_aprendizado=0.05,
        batch_size=2,
        otimizador="adam",
        embaralhar=False,
        verbose=False,
    )
    resultado = rede.avaliar(X_xor, y_xor)
    return rede, resultado


def treinar_classificacao(samples: int, seed: int) -> tuple[RedeNeural, dict, tuple]:
    X, y = DataUtils.gerar_dataset_classificacao(n_samples=samples, noise=0.12, random_state=seed)
    X_norm, _ = DataUtils.normalizar_dados(X, metodo="padrao")
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
        X_norm, y, test_size=0.2, random_state=seed
    )

    rede = RedeNeural(
        [2, 8, 4, 1],
        ativacao="relu",
        inicializacao="he",
        seed=seed,
        funcao_custo="binary_crossentropy",
    )
    rede.treinar(
        X_train,
        y_train,
        epochs=800,
        taxa_aprendizado=0.01,
        validacao_X=X_test,
        validacao_y=y_test,
        paciencia=40,
        min_delta=1e-4,
        batch_size=32,
        otimizador="adam",
        verbose=False,
    )

    resultado = rede.avaliar(X_test, y_test)
    return rede, resultado, (X_train, X_test, y_train, y_test)


def salvar_artefatos(
    save_dir: Path,
    rede_xor: RedeNeural,
    rede_classificacao: RedeNeural,
    X_test,
    y_test,
    plots: bool,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    rede_xor.salvar_parametros(str(save_dir / "modelo_xor.npz"))
    rede_classificacao.salvar_parametros(str(save_dir / "modelo_classificacao.npz"))

    if not plots:
        return

    VisualizationUtils.plotar_historico_treinamento(
        rede_xor.historico_erro,
        rede_xor.historico_acuracia,
        salvar=str(save_dir / "historico_xor.png"),
    )
    VisualizationUtils.plotar_historico_treinamento(
        rede_classificacao.historico_erro,
        rede_classificacao.historico_acuracia,
        salvar=str(save_dir / "historico_classificacao.png"),
    )
    VisualizationUtils.plotar_dados_classificacao(
        X_test,
        y_test,
        titulo="Dataset de teste",
        salvar=str(save_dir / "dados_teste.png"),
    )
    VisualizationUtils.plotar_fronteira_decisao(
        rede_classificacao,
        X_test,
        y_test,
        titulo="Fronteira de decisao",
        salvar=str(save_dir / "fronteira_decisao.png"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Exemplo principal do repositorio")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path("results/exemplo"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    print("Rede Neural do Zero - Exemplo principal")
    print("=======================================")
    print(f"Seed: {args.seed}")
    print(f"Samples: {args.samples}")
    print(f"Diretorio de saida: {args.save_dir}")
    print("Treino recomendado: otimizador=adam, batch_size=32")

    rede_xor, resultado_xor = treinar_xor(args.seed)
    rede_classificacao, resultado_classificacao, (_, X_test, _, y_test) = treinar_classificacao(
        args.samples, args.seed
    )

    metricas = MetricUtils.precisao_recall_f1(y_test, resultado_classificacao["predicoes"])

    print("\nResumo XOR")
    print("----------")
    print(f"Acuracia: {resultado_xor['acuracia']:.2f}%")
    print(f"Loss: {resultado_xor['loss']:.6f}")
    print(f"MSE: {resultado_xor['mse']:.6f}")

    print("\nResumo classificacao")
    print("--------------------")
    print(f"Acuracia: {resultado_classificacao['acuracia']:.2f}%")
    print(f"Loss: {resultado_classificacao['loss']:.6f}")
    print(f"MSE: {resultado_classificacao['mse']:.6f}")
    print(f"Precisao: {metricas['precisao']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"Especificidade: {metricas['especificidade']:.4f}")
    print(f"Balanced accuracy: {metricas['balanced_accuracy']:.4f}")
    print(f"F1-score: {metricas['f1_score']:.4f}")
    print(f"Matriz de confusao:\n{metricas['matriz_confusao']}")

    salvar_artefatos(
        save_dir=args.save_dir,
        rede_xor=rede_xor,
        rede_classificacao=rede_classificacao,
        X_test=X_test,
        y_test=y_test,
        plots=not args.no_plots,
    )

    print("\nArtefatos gerados")
    print("-----------------")
    print(f"- {args.save_dir / 'modelo_xor.npz'}")
    print(f"- {args.save_dir / 'modelo_classificacao.npz'}")
    if not args.no_plots:
        print(f"- {args.save_dir / 'historico_xor.png'}")
        print(f"- {args.save_dir / 'historico_classificacao.png'}")
        print(f"- {args.save_dir / 'dados_teste.png'}")
        print(f"- {args.save_dir / 'fronteira_decisao.png'}")


if __name__ == "__main__":
    main()
