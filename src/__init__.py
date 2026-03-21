"""Public package exports for rede-neural-do-zero."""

from .funcoes_ativacao import FuncoesAtivacao
from .rede_neural import RedeNeural
from .utils import DataUtils, FileUtils, MetricUtils, VisualizationUtils

__version__ = "1.2.0"
__author__ = "Savio"

__all__ = [
    "RedeNeural",
    "FuncoesAtivacao",
    "DataUtils",
    "VisualizationUtils",
    "FileUtils",
    "MetricUtils",
]
