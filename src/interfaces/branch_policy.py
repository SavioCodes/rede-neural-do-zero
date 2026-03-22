"""Validador leve para o padrao oficial de nomes de branch.

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
            "release/v2.2.5",
        ],
        "invalidas": [
            "feature/nova-coisa",
            "bugfix/erro-x",
            "Feat/maiuscula",
            "docs/wiki links",
            "release/2.2.5",
            "minha-branch",
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


def _formatar_resultado_humano(resultado: BranchPolicyResult) -> str:
    linhas = [
        f"Branch: {resultado.branch_name or '<vazia>'}",
        f"Status: {'valida' if resultado.valid else 'invalida'}",
        f"Categoria: {resultado.category}",
        f"Mensagem: {resultado.message}",
    ]
    exemplos = exemplos_branch()
    linhas.append("Exemplos validos: " + ", ".join(exemplos["validas"]))
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

    if args.json:
        print(json.dumps(resultado.to_dict(), indent=2, ensure_ascii=True))
    else:
        print(_formatar_resultado_humano(resultado))

    return 0 if resultado.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
