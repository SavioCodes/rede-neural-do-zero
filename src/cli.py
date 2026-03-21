"""CLI oficial do projeto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import FileUtils, RedeNeural, VisualizationUtils
from .benchmarking import executar_benchmark, nome_dataset_padrao, parse_seeds
from .evaluation import run_evaluation
from .experiments import (
    avaliar_modelo,
    carregar_dataset,
    criar_configs_padrao,
    dividir_treino_validacao_teste,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    """Converte arrays NumPy para tipos serializaveis em JSON."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Tipo nao serializavel: {type(value)!r}")


def _salvar_artefatos_treinamento(
    save_dir: Path,
    bundle_nome: str,
    splits: dict[str, Any],
    model: RedeNeural,
    avaliacao: dict[str, Any],
    no_plots: bool,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.salvar_parametros(str(save_dir / "model-params.npz"))
    model.salvar_checkpoint(str(save_dir / "model-checkpoint.npz"))

    if no_plots:
        return

    if avaliacao["tipo_problema"].startswith("regressao"):
        VisualizationUtils.plotar_regressao(
            splits["y_test"],
            avaliacao["predicoes"],
            salvar=str(save_dir / "regression.png"),
            mostrar=False,
        )
        return

    VisualizationUtils.plotar_historico_treinamento(
        model.historico_erro,
        model.historico_acuracia,
        historico_validacao_erro=model.historico_validacao_erro,
        historico_validacao_acuracia=model.historico_validacao_acuracia,
        salvar=str(save_dir / "history.png"),
        mostrar=False,
    )

    metricas = avaliacao["metricas_classificacao"]
    VisualizationUtils.plotar_matriz_confusao(
        metricas["matriz_confusao"],
        labels=[str(rotulo) for rotulo in metricas["labels"]],
        salvar=str(save_dir / "confusion-matrix.png"),
        mostrar=False,
    )
    if splits["X_test"].shape[1] == 2:
        VisualizationUtils.plotar_fronteira_decisao(
            model,
            splits["X_test"],
            splits["y_test"],
            titulo=f"Fronteira de decisao - {bundle_nome}",
            salvar=str(save_dir / "decision-boundary.png"),
            mostrar=False,
        )


def _dummy_model_for_checkpoint(input_dim: int) -> RedeNeural:
    """Cria um modelo minimo so para receber um checkpoint carregado."""
    return RedeNeural([input_dim, 1], ativacao="relu", ativacao_saida="linear", funcao_custo="mse")


def cmd_train(args: argparse.Namespace) -> None:
    bundle = carregar_dataset(args.dataset, seed=args.seed, samples=args.samples)
    splits = dividir_treino_validacao_teste(bundle.X, bundle.y, seed=args.seed)
    save_dir = Path(args.save_dir)

    if args.resume_from:
        model = _dummy_model_for_checkpoint(splits["X_train"].shape[1])
        checkpoint_info = model.carregar_checkpoint(args.resume_from)
        resumo = model.retomar_treinamento(
            splits["X_train"],
            splits["y_train"],
            epochs_adicionais=args.epochs,
            validacao_X=splits["X_val"],
            validacao_y=splits["y_val"],
            verbose=not args.no_verbose,
        )
    else:
        model_config, train_config = criar_configs_padrao(
            bundle,
            seed=args.seed,
            epochs=args.epochs,
            verbose=not args.no_verbose,
        )
        model = RedeNeural.from_config(model_config)
        resumo = model.treinar_com_config(
            splits["X_train"],
            splits["y_train"],
            train_config,
            validacao_X=splits["X_val"],
            validacao_y=splits["y_val"],
        )
        checkpoint_info = None

    avaliacao = avaliar_modelo(model, splits["X_test"], splits["y_test"])
    _salvar_artefatos_treinamento(
        save_dir,
        bundle.nome,
        splits,
        model,
        avaliacao,
        no_plots=args.no_plots,
    )

    payload = {
        "dataset": bundle.nome,
        "metadata": bundle.metadata,
        "checkpoint_info": checkpoint_info,
        "training": resumo,
        "evaluation": avaliacao,
    }
    _write_json(save_dir / "train-summary.json", payload)
    print(json.dumps(payload, indent=2, default=_json_default))


def cmd_evaluate(args: argparse.Namespace) -> None:
    min_score = args.min_score
    if min_score is None:
        min_score = 0.20 if args.dataset == "diabetes" or args.dataset == "regressao" else 65.0
    summary = run_evaluation(
        seed=args.seed,
        epochs=args.epochs,
        samples=args.samples,
        dataset=args.dataset,
        min_score=min_score,
    )
    output_path = Path(args.output)
    _write_json(output_path, summary)
    print(json.dumps(summary, indent=2, default=_json_default))


def cmd_benchmark(args: argparse.Namespace) -> None:
    dataset_nome = args.dataset or nome_dataset_padrao(args.mode)
    relatorio = executar_benchmark(
        dataset_nome=dataset_nome,
        amostras=args.samples,
        seeds=parse_seeds(args.seeds),
        epochs=args.epochs,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "benchmark.json", relatorio)

    resumo = relatorio["summary"]
    bruto = relatorio["raw_results"]
    FileUtils.salvar_csv(
        {chave: [linha[chave] for linha in resumo] for chave in resumo[0].keys()},
        str(output_dir / "benchmark-summary.csv"),
    )
    FileUtils.salvar_csv(
        {chave: [linha[chave] for linha in bruto] for chave in bruto[0].keys()},
        str(output_dir / "benchmark-runs.csv"),
    )
    print(json.dumps(relatorio, indent=2, default=_json_default))


def cmd_example(args: argparse.Namespace) -> None:
    namespace = argparse.Namespace(
        dataset=args.dataset,
        seed=args.seed,
        samples=args.samples,
        epochs=args.epochs,
        save_dir=args.save_dir,
        resume_from=None,
        no_plots=args.no_plots,
        no_verbose=args.no_verbose,
    )
    cmd_train(namespace)


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser principal da CLI."""
    parser = argparse.ArgumentParser(
        prog="rede-neural-do-zero", description="CLI oficial do projeto"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Treina um modelo e salva artefatos")
    parser_train.add_argument(
        "--dataset",
        choices=["xor", "binario", "multiclasse", "regressao", "iris", "wine", "diabetes"],
        default="binario",
    )
    parser_train.add_argument("--samples", type=int, default=240)
    parser_train.add_argument("--epochs", type=int, default=120)
    parser_train.add_argument("--seed", type=int, default=42)
    parser_train.add_argument("--save-dir", type=str, default="results/cli-train")
    parser_train.add_argument("--resume-from", type=str, default=None)
    parser_train.add_argument("--no-plots", action="store_true")
    parser_train.add_argument("--no-verbose", action="store_true")
    parser_train.set_defaults(func=cmd_train)

    parser_eval = subparsers.add_parser("evaluate", help="Roda avaliacao deterministica")
    parser_eval.add_argument(
        "--dataset",
        choices=["binario", "multiclasse", "regressao", "iris", "wine", "diabetes"],
        default="binario",
    )
    parser_eval.add_argument("--samples", type=int, default=240)
    parser_eval.add_argument("--epochs", type=int, default=150)
    parser_eval.add_argument("--seed", type=int, default=42)
    parser_eval.add_argument("--min-score", type=float, default=None)
    parser_eval.add_argument("--output", type=str, default="logs/cli-evaluation.json")
    parser_eval.set_defaults(func=cmd_evaluate)

    parser_bench = subparsers.add_parser("benchmark", help="Executa benchmark multi-seed")
    parser_bench.add_argument(
        "--mode", choices=["binario", "multiclasse", "regressao"], default="binario"
    )
    parser_bench.add_argument("--dataset", type=str, default=None)
    parser_bench.add_argument("--samples", type=int, default=240)
    parser_bench.add_argument("--epochs", type=int, default=120)
    parser_bench.add_argument("--seeds", type=str, default="42,52,62")
    parser_bench.add_argument("--output-dir", type=str, default="logs/cli-benchmark")
    parser_bench.set_defaults(func=cmd_benchmark)

    parser_example = subparsers.add_parser("example", help="Executa um exemplo pronto")
    parser_example.add_argument(
        "--dataset",
        choices=["xor", "binario", "multiclasse", "regressao", "iris", "wine", "diabetes"],
        default="iris",
    )
    parser_example.add_argument("--samples", type=int, default=240)
    parser_example.add_argument("--epochs", type=int, default=120)
    parser_example.add_argument("--seed", type=int, default=42)
    parser_example.add_argument("--save-dir", type=str, default="results/cli-example")
    parser_example.add_argument("--no-plots", action="store_true")
    parser_example.add_argument("--no-verbose", action="store_true")
    parser_example.set_defaults(func=cmd_example)

    return parser


def main() -> None:
    """Entrypoint principal da CLI."""
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
