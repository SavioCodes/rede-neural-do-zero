"""Reusable high-level workflows for training, evaluation, and benchmarking."""

from .benchmarking import (
    executar_benchmark,
    executar_suite_benchmark,
    gerar_relatorio_markdown,
    nome_dataset_padrao,
    parse_datasets,
    parse_seeds,
)
from .evaluation import run_evaluation
from .experiments import DatasetBundle, avaliar_modelo, carregar_dataset, criar_configs_padrao

__all__ = [
    "DatasetBundle",
    "carregar_dataset",
    "criar_configs_padrao",
    "avaliar_modelo",
    "executar_benchmark",
    "executar_suite_benchmark",
    "gerar_relatorio_markdown",
    "nome_dataset_padrao",
    "parse_datasets",
    "parse_seeds",
    "run_evaluation",
]
