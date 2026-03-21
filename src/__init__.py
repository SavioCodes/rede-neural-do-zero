"""Public package exports for rede-neural-do-zero."""

from .benchmarking import executar_benchmark
from .callbacks import Callback, CSVLogger, EarlyStopping, History, ModelCheckpoint
from .config import ModelConfig, TrainingConfig
from .evaluation import run_evaluation
from .experiments import DatasetBundle, avaliar_modelo, carregar_dataset, criar_configs_padrao
from .funcoes_ativacao import FuncoesAtivacao
from .rede_neural import RedeNeural
from .utils import DataUtils, FileUtils, MetricUtils, VisualizationUtils

__version__ = "2.0.3"
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
    "run_evaluation",
]
