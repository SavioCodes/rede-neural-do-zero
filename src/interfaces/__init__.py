"""Public interfaces such as the official CLI."""

from .cli import build_parser, main
from .cli_config import (
    aplicar_config_cli,
    argv_comando_atual,
    carregar_arquivo_config,
    resolver_config_comando,
    serializar_config_efetiva,
)

__all__ = [
    "build_parser",
    "main",
    "aplicar_config_cli",
    "argv_comando_atual",
    "carregar_arquivo_config",
    "resolver_config_comando",
    "serializar_config_efetiva",
]
