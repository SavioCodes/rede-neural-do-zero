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
from .codeowners_reviewers import CodeownersEntry, ReviewerResolution, resolve_reviewers
from .pypi_status import (
    PyPIStatusResult,
    TrustedPublisherConfig,
    obter_status_pypi,
)
from .release_notes import ReleaseNotesResult, construir_release_notes, extrair_secao_changelog

__all__ = [
    "build_parser",
    "main",
    "BranchPolicyResult",
    "BranchTargetResult",
    "LabelDefinition",
    "CodeownersEntry",
    "ReviewerResolution",
    "ReleaseNotesResult",
    "PyPIStatusResult",
    "TrustedPublisherConfig",
    "validar_nome_branch",
    "validar_destino_pr",
    "detectar_branch_atual",
    "exemplos_branch",
    "exemplos_destino_branch",
    "labels_para_pull_request",
    "definicoes_para_labels",
    "resolve_reviewers",
    "obter_status_pypi",
    "construir_release_notes",
    "extrair_secao_changelog",
    "aplicar_config_cli",
    "argv_comando_atual",
    "carregar_arquivo_config",
    "resolver_config_comando",
    "serializar_config_efetiva",
]
