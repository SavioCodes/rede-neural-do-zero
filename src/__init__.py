"""Public package exports for rede-neural-do-zero."""

from .core import FuncoesAtivacao, RedeNeural
from .data import DataUtils, FileUtils, MetricUtils, VisualizationUtils
from .training import Callback, CSVLogger, EarlyStopping, History, ModelCheckpoint
from .training.config import ModelConfig, TrainingConfig
from .workflows import (
    executar_benchmark,
    executar_suite_benchmark,
    gerar_relatorio_markdown,
    parse_datasets,
    run_evaluation,
)
from .workflows.experiments import (
    DatasetBundle,
    avaliar_modelo,
    carregar_dataset,
    criar_configs_padrao,
)

__version__ = "2.2.0"
__author__ = "Savio"

__all__ = [
    "RedeNeural",
    "ModelConfig",
    "TrainingConfig",
    "Callback",
    "History",
    "EarlyStopping",
    "ModelCheckpoint",
    "CSVLogger",
    "FuncoesAtivacao",
    "DataUtils",
    "VisualizationUtils",
    "FileUtils",
    "MetricUtils",
    "DatasetBundle",
    "carregar_dataset",
    "criar_configs_padrao",
    "avaliar_modelo",
    "executar_benchmark",
    "executar_suite_benchmark",
    "gerar_relatorio_markdown",
    "parse_datasets",
    "run_evaluation",
]
