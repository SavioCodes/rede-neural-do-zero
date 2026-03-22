"""Public interfaces such as the official CLI."""

from .branch_policy import (
    BranchPolicyResult,
    detectar_branch_atual,
    exemplos_branch,
    validar_nome_branch,
)
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
    "BranchPolicyResult",
    "validar_nome_branch",
    "detectar_branch_atual",
    "exemplos_branch",
    "aplicar_config_cli",
    "argv_comando_atual",
    "carregar_arquivo_config",
    "resolver_config_comando",
    "serializar_config_efetiva",
]
