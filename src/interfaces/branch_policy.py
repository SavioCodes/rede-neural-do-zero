"""Validador leve para o padrao oficial de nomes e fluxo de branches.

Este modulo foi mantido sem dependencias externas para poder rodar tanto:

- pela CLI oficial do projeto
- diretamente em workflows do GitHub Actions
- em validacoes locais rapidas
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from typing import Final

PERMANENT_BRANCHES: Final[set[str]] = {"main", "develop"}
TOPIC_BRANCH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(feat|fix|docs|chore|hotfix)/[a-z0-9]+(?:-[a-z0-9]+)*$"
)
RELEASE_BRANCH_PATTERN: Final[re.Pattern[str]] = re.compile(r"^release/v\d+\.\d+\.\d+$")
BRANCH_TARGET_RULES: Final[dict[str, tuple[str, ...]]] = {
    "feat": ("develop",),
    "fix": ("develop",),
    "docs": ("develop",),
    "chore": ("develop",),
    "hotfix": ("main",),
    "release": ("main",),
    "develop": ("main",),
    "main": ("develop",),
}


@dataclass(slots=True)
class BranchPolicyResult:
    """Representa o resultado da validacao de um nome de branch."""

    branch_name: str
    valid: bool
    category: str
    message: str

    def to_dict(self) -> dict[str, object]:
        """Converte o resultado para um dicionario serializavel."""
        return asdict(self)


@dataclass(slots=True)
class BranchTargetResult:
    """Representa a validacao do destino correto de um pull request."""

    branch_name: str
    target_branch: str
    valid: bool
    category: str
    allowed_targets: tuple[str, ...]
    message: str

    def to_dict(self) -> dict[str, object]:
        """Converte o resultado para um dicionario serializavel."""
        return asdict(self)


def exemplos_branch() -> dict[str, list[str]]:
    """Retorna exemplos validos e invalidos para ajudar o usuario."""
    return {
        "validas": [
            "main",
            "develop",
            "feat/add-multiclass-report",
            "fix/checkpoint-restore-bug",
            "docs/update-wiki-links",
            "chore/reorganize-ci-cache",
            "hotfix/fix-release-tag-link",
            "release/v2.2.6",
        ],
        "invalidas": [
            "feature/nova-coisa",
            "bugfix/erro-x",
            "Feat/maiuscula",
            "docs/wiki links",
            "release/2.2.6",
            "minha-branch",
        ],
    }


def exemplos_destino_branch() -> dict[str, list[str]]:
    """Retorna exemplos de combinacoes validas e invalidas de branch/base."""
    return {
        "validas": [
            "feat/add-metrics -> develop",
            "fix/checkpoint-bug -> develop",
            "docs/update-wiki -> develop",
            "hotfix/fix-release-link -> main",
            "release/v2.2.6 -> main",
            "develop -> main",
            "main -> develop",
        ],
        "invalidas": [
            "feat/add-metrics -> main",
            "fix/checkpoint-bug -> main",
            "hotfix/fix-release-link -> develop",
            "release/v2.2.6 -> develop",
            "develop -> develop",
            "main -> main",
        ],
    }


def validar_nome_branch(nome: str) -> BranchPolicyResult:
    """Valida o nome de branch segundo a convencao oficial do projeto."""
    branch = nome.strip()
    if not branch:
        return BranchPolicyResult(
            branch_name=branch,
            valid=False,
            category="invalida",
            message="Forneca um nome de branch nao vazio.",
        )

    if branch in PERMANENT_BRANCHES:
        return BranchPolicyResult(
            branch_name=branch,
            valid=True,
            category="permanente",
            message=f"Branch permanente '{branch}' aprovada.",
        )

    if TOPIC_BRANCH_PATTERN.fullmatch(branch):
        categoria = branch.split("/", 1)[0]
        return BranchPolicyResult(
            branch_name=branch,
            valid=True,
            category=categoria,
            message=f"Branch '{branch}' segue o padrao oficial.",
        )

    if RELEASE_BRANCH_PATTERN.fullmatch(branch):
        return BranchPolicyResult(
            branch_name=branch,
            valid=True,
            category="release",
            message=f"Branch de release '{branch}' aprovada.",
        )

    return BranchPolicyResult(
        branch_name=branch,
        valid=False,
        category="invalida",
        message=(
            "Nome fora do padrao. Use `main`, `develop`, "
            "`feat/<slug>`, `fix/<slug>`, `docs/<slug>`, `chore/<slug>`, "
            "`hotfix/<slug>` ou `release/vX.Y.Z`."
        ),
    )


def destinos_permitidos(branch_name: str) -> tuple[str, ...]:
    """Retorna as branches-base aceitas para o fluxo da branch."""
    resultado = validar_nome_branch(branch_name)
    if not resultado.valid:
        return ()

    if branch_name in PERMANENT_BRANCHES:
        return BRANCH_TARGET_RULES.get(branch_name, ())

    if resultado.category in BRANCH_TARGET_RULES:
        return BRANCH_TARGET_RULES[resultado.category]

    return ()


def validar_destino_pr(branch_name: str, target_branch: str) -> BranchTargetResult:
    """Valida se a branch esta apontando para a base correta do PR."""
    nome_resultado = validar_nome_branch(branch_name)
    destino = target_branch.strip()
    permitidos = destinos_permitidos(branch_name)

    if not nome_resultado.valid:
        return BranchTargetResult(
            branch_name=branch_name.strip(),
            target_branch=destino,
            valid=False,
            category=nome_resultado.category,
            allowed_targets=permitidos,
            message=f"Corrija primeiro o nome da branch. {nome_resultado.message}",
        )

    if not destino:
        return BranchTargetResult(
            branch_name=branch_name.strip(),
            target_branch=destino,
            valid=False,
            category=nome_resultado.category,
            allowed_targets=permitidos,
            message="Informe a branch-base do pull request com --target.",
        )

    if destino in permitidos:
        return BranchTargetResult(
            branch_name=branch_name.strip(),
            target_branch=destino,
            valid=True,
            category=nome_resultado.category,
            allowed_targets=permitidos,
            message=(
                f"Fluxo aprovado: '{branch_name.strip()}' pode abrir PR para '{destino}'."
            ),
        )

    destinos_texto = ", ".join(permitidos) if permitidos else "nenhum"
    return BranchTargetResult(
        branch_name=branch_name.strip(),
        target_branch=destino,
        valid=False,
        category=nome_resultado.category,
        allowed_targets=permitidos,
        message=(
            f"Destino incorreto. '{branch_name.strip()}' deve abrir PR para "
            f"{destinos_texto}, nao para '{destino}'."
        ),
    )


def detectar_branch_atual() -> str | None:
    """Tenta descobrir a branch atual pelo ambiente ou pelo Git local."""
    # `BRANCH_NAME` vem primeiro para permitir override explicito em testes
    # locais, scripts e chamadas controladas pela nossa CLI.
    for chave in ("BRANCH_NAME", "GITHUB_HEAD_REF", "GITHUB_REF_NAME"):
        valor = str(os.environ.get(chave, "")).strip()
        if valor:
            return valor

    try:
        resultado = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    branch = resultado.stdout.strip()
    return branch or None


def _formatar_resultado_humano(
    resultado: BranchPolicyResult,
    destino_resultado: BranchTargetResult | None = None,
) -> str:
    linhas = [
        f"Branch: {resultado.branch_name or '<vazia>'}",
        f"Status: {'valida' if resultado.valid else 'invalida'}",
        f"Categoria: {resultado.category}",
        f"Mensagem: {resultado.message}",
    ]
    exemplos = exemplos_branch()
    linhas.append("Exemplos validos: " + ", ".join(exemplos["validas"]))

    if destino_resultado is not None:
        linhas.extend(
            [
                f"Base do PR: {destino_resultado.target_branch or '<vazia>'}",
                f"Destino valido: {'sim' if destino_resultado.valid else 'nao'}",
                "Bases permitidas: "
                + (", ".join(destino_resultado.allowed_targets) or "nenhuma"),
                f"Mensagem destino: {destino_resultado.message}",
            ]
        )
    return "\n".join(linhas)


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser do validador standalone."""
    parser = argparse.ArgumentParser(description="Valida nomes de branch do projeto.")
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Nome da branch a validar. Se omitido, tenta detectar a branch atual.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Branch-base do pull request para validar o fluxo da branch.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime o resultado em JSON para automacoes.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint standalone usado localmente e nos workflows."""
    parser = build_parser()
    args = parser.parse_args(argv)

    branch_name = args.name or detectar_branch_atual()
    if not branch_name:
        resultado = BranchPolicyResult(
            branch_name="",
            valid=False,
            category="invalida",
            message="Nao foi possivel detectar a branch atual. Use --name.",
        )
    else:
        resultado = validar_nome_branch(branch_name)

    destino_resultado = None
    if args.target is not None and branch_name:
        destino_resultado = validar_destino_pr(branch_name, args.target)

    if args.json:
        payload: dict[str, object] = {
            **resultado.to_dict(),
            "exemplos": exemplos_branch(),
        }
        if destino_resultado is not None:
            payload["target_validation"] = destino_resultado.to_dict()
            payload["target_examples"] = exemplos_destino_branch()
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(_formatar_resultado_humano(resultado, destino_resultado))

    return 0 if resultado.valid and (destino_resultado is None or destino_resultado.valid) else 1


if __name__ == "__main__":
    raise SystemExit(main())
