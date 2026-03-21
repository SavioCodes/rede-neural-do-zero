#!/usr/bin/env python3
"""Entrypoint de benchmark reutilizando a camada oficial de `src.benchmarking`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import FileUtils  # noqa: E402
from src.benchmarking import (  # noqa: E402
    executar_benchmark,
    executar_suite_benchmark,
    gerar_relatorio_markdown,
    nome_dataset_padrao,
    parse_datasets,
    parse_seeds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multi-seed do projeto")
    parser.add_argument(
        "--mode", choices=["binario", "multiclasse", "regressao"], default="binario"
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--json-output", type=Path, default=Path("logs/benchmark.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("logs/benchmark-summary.csv"))
    parser.add_argument("--raw-csv-output", type=Path, default=Path("logs/benchmark-runs.csv"))
    parser.add_argument(
        "--leaderboard-output",
        type=Path,
        default=Path("logs/benchmark-leaderboard.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("logs/benchmark-report.md"),
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    dataset_nomes = (
        parse_datasets(args.datasets)
        if args.datasets
        else [args.dataset or nome_dataset_padrao(args.mode)]
    )
    if len(dataset_nomes) == 1:
        relatorio = executar_benchmark(dataset_nomes[0], args.samples, seeds, args.epochs)
    else:
        relatorio = executar_suite_benchmark(dataset_nomes, args.samples, seeds, args.epochs)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(relatorio, indent=2), encoding="utf-8")
    args.markdown_output.write_text(gerar_relatorio_markdown(relatorio), encoding="utf-8")

    resumo = relatorio["summary"]
    brutos = relatorio["raw_results"]
    vencedores = relatorio["leaderboard"]
    FileUtils.salvar_linhas_csv(resumo, str(args.csv_output))
    FileUtils.salvar_linhas_csv(brutos, str(args.raw_csv_output))
    FileUtils.salvar_linhas_csv(vencedores, str(args.leaderboard_output))

    print("Benchmark concluido")
    print(f"Datasets: {', '.join(dataset_nomes)}")
    print(f"Seeds: {seeds}")
    print(f"Saida JSON: {args.json_output}")
    print(f"Saida Markdown: {args.markdown_output}")


if __name__ == "__main__":
    main()
