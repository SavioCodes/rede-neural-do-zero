"""Public package exports for rede-neural-do-zero."""

from .callbacks import Callback, CSVLogger, EarlyStopping, History, ModelCheckpoint
from .config import ModelConfig, TrainingConfig
from .funcoes_ativacao import FuncoesAtivacao
from .rede_neural import RedeNeural
from .utils import DataUtils, FileUtils, MetricUtils, VisualizationUtils

__version__ = "1.5.0"
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
]
