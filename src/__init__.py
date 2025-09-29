"""
Rede Neural do Zero - Implementação em Python puro
Autor: Sávio (https://github.com/SavioCodes)
"""

from .rede_neural import RedeNeural
from .funcoes_ativacao import FuncoesAtivacao
from .utils import DataUtils, VisualizationUtils, FileUtils, MetricUtils

__version__ = "1.0.0"
__author__ = "Sávio"

__all__ = [
    'RedeNeural',
    'FuncoesAtivacao', 
    'DataUtils',
    'VisualizationUtils',
    'FileUtils',
    'MetricUtils'
]
