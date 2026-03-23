"""CLI oficial do projeto."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..core.rede_neural import RedeNeural
from ..data.utils import FileUtils, VisualizationUtils
from ..workflows.benchmarking import (
    executar_benchmark,
    executar_suite_benchmark,
    gerar_relatorio_markdown,
    nome_dataset_padrao,
    parse_datasets,
    parse_seeds,
)
from ..workflows.evaluation import run_evaluation
from ..workflows.experiments import (
    avaliar_modelo,
    carregar_dataset,
    criar_configs_padrao,
    dividir_treino_validacao_teste,
)
from .branch_policy import (
    detectar_branch_atual,
    exemplos_branch,
    exemplos_destino_branch,
    validar_destino_pr,
    validar_nome_branch,
)
from .cli_config import (
    aplicar_config_cli,
    argv_comando_atual,
    serializar_config_efetiva,
)
from .pypi_status import carregar_nome_projeto, obter_status_pypi
from .release_notes import construir_release_notes


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    """Converte arrays NumPy para tipos serializaveis em JSON."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo nao serializavel: {type(value)!r}")


def _adicionar_argumento_config(parser: argparse.ArgumentParser) -> None:
    """Adiciona suporte a `--config` em subcomandos da CLI."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Arquivo JSON/TOML/YAML com configuracoes do subcomando.",
    )


def _persistir_config_efetiva(save_dir: Path, args: argparse.Namespace) -> None:
    """Salva a configuracao efetiva resolvida pela CLI."""
    save_dir.mkdir(parents=True, exist_ok=True)
    _write_json(save_dir / "effective-config.json", getattr(args, "_effective_config", {}))


def _executar_comando(comando: list[str]) -> None:
    """Executa um comando do projeto preservando falhas no exit code."""
    subprocess.run(comando, check=True)


def _artefatos_dist() -> list[str]:
    """Resolve os arquivos gerados em `dist/` para validacao com Twine."""
    artefatos = sorted(Path("dist").glob("*"))
    if not artefatos:
        raise FileNotFoundError("Nenhum artefato encontrado em dist/. Rode o build antes do check.")
    return [str(artefato) for artefato in artefatos]


def _limpar_artefatos_build() -> None:
    """Remove arquivos antigos de `build/` e `dist/` antes de gerar um novo pacote."""
    for diretorio in [Path("build"), Path("dist")]:
        if not diretorio.exists():
            continue
        for item in diretorio.glob("*"):
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


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


def _executar_fluxo_treino(
    args: argparse.Namespace,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """Executa o fluxo completo de treino ou resume e devolve um payload serializavel."""
    bundle = carregar_dataset(args.dataset, seed=args.seed, samples=args.samples)
    splits = dividir_treino_validacao_teste(bundle.X, bundle.y, seed=args.seed)
    save_dir = Path(args.save_dir)
    checkpoint_info = None

    checkpoint_origem = checkpoint or getattr(args, "resume_from", None)
    if checkpoint_origem:
        model = _dummy_model_for_checkpoint(splits["X_train"].shape[1])
        checkpoint_info = model.carregar_checkpoint(checkpoint_origem)
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

    avaliacao = avaliar_modelo(model, splits["X_test"], splits["y_test"])
    _salvar_artefatos_treinamento(
        save_dir,
        bundle.nome,
        splits,
        model,
        avaliacao,
        no_plots=args.no_plots,
    )
    _persistir_config_efetiva(save_dir, args)

    return {
        "dataset": bundle.nome,
        "metadata": bundle.metadata,
        "checkpoint_info": checkpoint_info,
        "training": resumo,
        "evaluation": avaliacao,
    }


def cmd_train(args: argparse.Namespace) -> None:
    payload = _executar_fluxo_treino(args)
    save_dir = Path(args.save_dir)
    _write_json(save_dir / "train-summary.json", payload)
    print(json.dumps(payload, indent=2, default=_json_default))


def cmd_resume(args: argparse.Namespace) -> None:
    payload = _executar_fluxo_treino(args, checkpoint=args.checkpoint)
    save_dir = Path(args.save_dir)
    _write_json(save_dir / "resume-summary.json", payload)
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
    seeds = parse_seeds(args.seeds)
    dataset_nomes = (
        parse_datasets(args.datasets)
        if args.datasets
        else [args.dataset or nome_dataset_padrao(args.mode)]
    )
    if len(dataset_nomes) == 1:
        relatorio = executar_benchmark(
            dataset_nome=dataset_nomes[0],
            amostras=args.samples,
            seeds=seeds,
            epochs=args.epochs,
        )
    else:
        relatorio = executar_suite_benchmark(
            dataset_nomes=dataset_nomes,
            amostras=args.samples,
            seeds=seeds,
            epochs=args.epochs,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "benchmark.json", relatorio)
    (output_dir / "benchmark-report.md").write_text(
        gerar_relatorio_markdown(relatorio),
        encoding="utf-8",
    )
    _persistir_config_efetiva(output_dir, args)

    resumo = relatorio["summary"]
    bruto = relatorio["raw_results"]
    FileUtils.salvar_linhas_csv(resumo, str(output_dir / "benchmark-summary.csv"))
    FileUtils.salvar_linhas_csv(bruto, str(output_dir / "benchmark-runs.csv"))
    FileUtils.salvar_linhas_csv(
        relatorio["leaderboard"],
        str(output_dir / "benchmark-leaderboard.csv"),
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
        _effective_config=getattr(args, "_effective_config", {}),
    )
    cmd_train(namespace)


def cmd_build_docs(args: argparse.Namespace) -> None:
    _executar_comando([sys.executable, "scripts/export_notebooks_to_docs.py"])
    comando = [sys.executable, "-m", "mkdocs", "build"]
    if args.strict:
        comando.append("--strict")
    _executar_comando(comando)


def cmd_build_package(args: argparse.Namespace) -> None:
    _limpar_artefatos_build()
    _executar_comando([sys.executable, "-m", "build"])
    if args.check:
        _executar_comando([sys.executable, "-m", "twine", "check", *_artefatos_dist()])


def cmd_verify(args: argparse.Namespace) -> None:
    _executar_comando([sys.executable, "-m", "ruff", "check", "."])
    _executar_comando([sys.executable, "-m", "mypy", "src", "rede_neural_do_zero"])
    _executar_comando([sys.executable, "-m", "pytest", "-q"])
    _executar_comando([sys.executable, "scripts/validate_notebooks.py"])
    _executar_comando([sys.executable, "scripts/export_notebooks_to_docs.py"])
    _executar_comando([sys.executable, "-m", "mkdocs", "build", "--strict"])
    if args.build_package:
        _limpar_artefatos_build()
        _executar_comando([sys.executable, "-m", "build"])
        _executar_comando([sys.executable, "-m", "twine", "check", *_artefatos_dist()])


def cmd_check_branch(args: argparse.Namespace) -> None:
    """Valida uma branch pelo padrao oficial do projeto."""
    branch_name = args.name or detectar_branch_atual()
    if not branch_name:
        raise SystemExit("Nao foi possivel detectar a branch atual. Use --name.")

    resultado = validar_nome_branch(branch_name)
    payload = {
        **resultado.to_dict(),
        "exemplos": exemplos_branch(),
    }
    destino_resultado = None
    if args.target is not None:
        destino_resultado = validar_destino_pr(branch_name, args.target)
        payload["target_validation"] = destino_resultado.to_dict()
        payload["target_examples"] = exemplos_destino_branch()
    print(json.dumps(payload, indent=2, ensure_ascii=True))
    if not resultado.valid or (destino_resultado is not None and not destino_resultado.valid):
        raise SystemExit(1)


def cmd_release_notes(args: argparse.Namespace) -> None:
    """Extrai release notes oficiais para uso manual ou em automacoes."""
    payload = construir_release_notes(
        version=args.version,
        changelog_path=args.changelog,
        pyproject_path=args.pyproject,
    )
    if args.output:
        Path(args.output).write_text(payload.body, encoding="utf-8")

    if args.json:
        print(json.dumps(payload.to_dict(), indent=2, ensure_ascii=True))
        return

    print(payload.body)


def cmd_pypi_status(args: argparse.Namespace) -> None:
    """Inspeciona o estado do pacote no PyPI e a configuracao do Trusted Publisher."""
    payload = obter_status_pypi(
        project_name=args.project_name or carregar_nome_projeto(args.pyproject),
        owner=args.owner,
        repository=args.repository,
        workflow_filename=args.workflow_filename,
        environment=args.environment,
        pyproject_path=args.pyproject,
    )
    print(json.dumps(payload.to_dict(), indent=2, ensure_ascii=True))


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    """Monta o parser principal da CLI."""
    parser = argparse.ArgumentParser(
        prog="rede-neural-do-zero", description="CLI oficial do projeto"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    parsers_por_comando: dict[str, argparse.ArgumentParser] = {}

    parser_train = subparsers.add_parser("train", help="Treina um modelo e salva artefatos")
    _adicionar_argumento_config(parser_train)
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
    parsers_por_comando["train"] = parser_train

    parser_eval = subparsers.add_parser("evaluate", help="Roda avaliacao deterministica")
    _adicionar_argumento_config(parser_eval)
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
    parsers_por_comando["evaluate"] = parser_eval

    parser_bench = subparsers.add_parser("benchmark", help="Executa benchmark multi-seed")
    _adicionar_argumento_config(parser_bench)
    parser_bench.add_argument(
        "--mode", choices=["binario", "multiclasse", "regressao"], default="binario"
    )
    parser_bench.add_argument("--dataset", type=str, default=None)
    parser_bench.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Lista separada por virgula para rodar uma suite de benchmarks.",
    )
    parser_bench.add_argument("--samples", type=int, default=240)
    parser_bench.add_argument("--epochs", type=int, default=120)
    parser_bench.add_argument("--seeds", type=str, default="42,52,62")
    parser_bench.add_argument("--output-dir", type=str, default="logs/cli-benchmark")
    parser_bench.set_defaults(func=cmd_benchmark)
    parsers_por_comando["benchmark"] = parser_bench

    parser_example = subparsers.add_parser("example", help="Executa um exemplo pronto")
    _adicionar_argumento_config(parser_example)
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
    parsers_por_comando["example"] = parser_example

    parser_resume = subparsers.add_parser(
        "resume",
        help="Retoma um treinamento a partir de um checkpoint completo",
    )
    _adicionar_argumento_config(parser_resume)
    parser_resume.add_argument("--checkpoint", type=str, required=True)
    parser_resume.add_argument(
        "--dataset",
        choices=["xor", "binario", "multiclasse", "regressao", "iris", "wine", "diabetes"],
        default="binario",
    )
    parser_resume.add_argument("--samples", type=int, default=240)
    parser_resume.add_argument("--epochs", type=int, default=40)
    parser_resume.add_argument("--seed", type=int, default=42)
    parser_resume.add_argument("--save-dir", type=str, default="results/cli-resume")
    parser_resume.add_argument("--no-plots", action="store_true")
    parser_resume.add_argument("--no-verbose", action="store_true")
    parser_resume.set_defaults(func=cmd_resume)
    parsers_por_comando["resume"] = parser_resume

    parser_build_docs = subparsers.add_parser(
        "build-docs",
        help="Gera as paginas de docs e exporta notebooks para o site",
    )
    parser_build_docs.add_argument("--strict", action="store_true")
    parser_build_docs.set_defaults(func=cmd_build_docs)
    parsers_por_comando["build-docs"] = parser_build_docs

    parser_build_package = subparsers.add_parser(
        "build-package",
        help="Gera distribuicoes do pacote e valida metadados",
    )
    parser_build_package.add_argument("--check", action="store_true")
    parser_build_package.set_defaults(func=cmd_build_package)
    parsers_por_comando["build-package"] = parser_build_package

    parser_verify = subparsers.add_parser(
        "verify",
        help="Executa o fluxo principal de qualidade do projeto",
    )
    parser_verify.add_argument("--build-package", action="store_true")
    parser_verify.set_defaults(func=cmd_verify)
    parsers_por_comando["verify"] = parser_verify

    parser_check_branch = subparsers.add_parser(
        "check-branch",
        help="Valida o nome de uma branch pelo padrao oficial do projeto",
    )
    parser_check_branch.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nome da branch. Se omitido, tenta detectar a branch atual.",
    )
    parser_check_branch.add_argument(
        "--target",
        type=str,
        default=None,
        help="Branch-base do pull request para validar o fluxo do PR.",
    )
    parser_check_branch.set_defaults(func=cmd_check_branch)
    parsers_por_comando["check-branch"] = parser_check_branch

    parser_release_notes = subparsers.add_parser(
        "release-notes",
        help="Extrai release notes do CHANGELOG para a versao atual ou informada",
    )
    parser_release_notes.add_argument("--version", type=str, default=None)
    parser_release_notes.add_argument("--changelog", type=str, default="CHANGELOG.md")
    parser_release_notes.add_argument("--pyproject", type=str, default="pyproject.toml")
    parser_release_notes.add_argument("--output", type=str, default=None)
    parser_release_notes.add_argument("--json", action="store_true")
    parser_release_notes.set_defaults(func=cmd_release_notes)
    parsers_por_comando["release-notes"] = parser_release_notes

    parser_pypi_status = subparsers.add_parser(
        "pypi-status",
        help="Mostra o estado do pacote no PyPI e a configuracao esperada do Trusted Publisher",
    )
    parser_pypi_status.add_argument("--project-name", type=str, default=None)
    parser_pypi_status.add_argument("--owner", type=str, default="SavioCodes")
    parser_pypi_status.add_argument("--repository", type=str, default="rede-neural-do-zero")
    parser_pypi_status.add_argument("--workflow-filename", type=str, default="publish.yml")
    parser_pypi_status.add_argument("--environment", type=str, default="pypi")
    parser_pypi_status.add_argument("--pyproject", type=str, default="pyproject.toml")
    parser_pypi_status.set_defaults(func=cmd_pypi_status)
    parsers_por_comando["pypi-status"] = parser_pypi_status

    return parser, parsers_por_comando


def main(argv: list[str] | None = None) -> None:
    """Entrypoint principal da CLI."""
    parser, parsers_por_comando = build_parser()
    argv_raiz, argv_comando = argv_comando_atual(argv)
    args = parser.parse_args(argv_raiz)
    parser_comando = parsers_por_comando.get(str(args.command))
    if parser_comando is not None:
        args, _ = aplicar_config_cli(args, parser_comando, argv_comando)
    args._effective_config = serializar_config_efetiva(args)
    args.func(args)


if __name__ == "__main__":
    main()
