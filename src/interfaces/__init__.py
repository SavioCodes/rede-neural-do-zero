"""Public interfaces such as the official CLI."""

from .branch_labels import LabelDefinition, definicoes_para_labels, labels_para_pull_request
from .branch_policy import (
    BranchPolicyResult,
    BranchTargetResult,
    detectar_branch_atual,
    exemplos_branch,
    exemplos_destino_branch,
    validar_destino_pr,
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
    "BranchTargetResult",
    "LabelDefinition",
    "validar_nome_branch",
    "validar_destino_pr",
    "detectar_branch_atual",
    "exemplos_branch",
    "exemplos_destino_branch",
    "labels_para_pull_request",
    "definicoes_para_labels",
    "aplicar_config_cli",
    "argv_comando_atual",
    "carregar_arquivo_config",
    "resolver_config_comando",
    "serializar_config_efetiva",
]
