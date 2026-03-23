"""Resolve labels oficiais de pull request a partir das branches do projeto."""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict, dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """Representa um label oficial do repositorio."""

    name: str
    color: str
    description: str

    def to_dict(self) -> dict[str, str]:
        """Converte o label em um dicionario serializavel."""
        return asdict(self)


LABEL_DEFINITIONS: Final[dict[str, LabelDefinition]] = {
    "feat": LabelDefinition(
        name="feat",
        color="0E8A16",
        description="Pull requests que nascem de branches feat/*.",
    ),
    "fix": LabelDefinition(
        name="fix",
        color="FBCA04",
        description="Pull requests que nascem de branches fix/*.",
    ),
    "docs": LabelDefinition(
        name="docs",
        color="0075CA",
        description="Pull requests que nascem de branches docs/*.",
    ),
    "chore": LabelDefinition(
        name="chore",
        color="BFDADC",
        description="Pull requests que nascem de branches chore/*.",
    ),
    "hotfix": LabelDefinition(
        name="hotfix",
        color="D93F0B",
        description="Correcoes urgentes abertas a partir de branches hotfix/*.",
    ),
    "release": LabelDefinition(
        name="release",
        color="5319E7",
        description="Packaging, tags, releases, and publication flow.",
    ),
    "governance": LabelDefinition(
        name="governance",
        color="D4C5F9",
        description="Repository processes, templates, and contribution workflow.",
    ),
}
PREFIX_TO_LABEL: Final[dict[str, str]] = {
    "feat": "feat",
    "fix": "fix",
    "docs": "docs",
    "chore": "chore",
    "hotfix": "hotfix",
    "release": "release",
}
ROUTE_LABELS: Final[dict[tuple[str, str], tuple[str, ...]]] = {
    ("develop", "main"): ("release",),
    ("main", "develop"): ("governance",),
}


def categoria_branch(branch_name: str) -> str | None:
    """Resolve a categoria oficial da branch para aplicar labels consistentes."""
    try:
        modulo = importlib.import_module("src.interfaces.branch_policy")
    except ImportError:  # pragma: no cover - suporte ao uso standalone no workflow
        modulo = importlib.import_module("branch_policy")

    resultado = modulo.validar_nome_branch(branch_name)
    if not resultado.valid:
        return None

    if resultado.category == "permanente":
        return branch_name.strip()

    return resultado.category


def labels_para_pull_request(head_branch: str, base_branch: str | None = None) -> list[str]:
    """Retorna labels oficiais a partir da branch de origem e, opcionalmente, da base."""
    labels: list[str] = []
    categoria = categoria_branch(head_branch)

    if categoria is not None:
        label_prefixo = PREFIX_TO_LABEL.get(categoria)
        if label_prefixo is not None:
            labels.append(label_prefixo)

    if base_branch is not None:
        for label in ROUTE_LABELS.get((head_branch.strip(), base_branch.strip()), ()):
            if label not in labels:
                labels.append(label)

    return labels


def definicoes_para_labels(labels: list[str]) -> list[dict[str, str]]:
    """Retorna as definicoes completas dos labels usados pelo workflow."""
    definicoes: list[dict[str, str]] = []
    for label in labels:
        definicao = LABEL_DEFINITIONS.get(label)
        if definicao is not None:
            definicoes.append(definicao.to_dict())
    return definicoes


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser do script standalone."""
    parser = argparse.ArgumentParser(description="Resolve labels oficiais para pull requests.")
    parser.add_argument("--head", required=True, help="Branch de origem do pull request.")
    parser.add_argument(
        "--base",
        default=None,
        help="Branch de destino do pull request para considerar labels do fluxo.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emite um payload JSON pronto para automacoes.",
    )
    parser.add_argument(
        "--format",
        choices=("lines",),
        default=None,
        help="Opcionalmente imprime apenas uma label por linha.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint standalone para uso local e em workflows."""
    parser = build_parser()
    args = parser.parse_args(argv)

    labels = labels_para_pull_request(args.head, args.base)
    payload = {
        "head_branch": args.head.strip(),
        "base_branch": args.base.strip() if args.base else None,
        "labels": labels,
        "definitions": definicoes_para_labels(labels),
    }

    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    elif args.format == "lines":
        for label in labels:
            print(label)
    else:
        print(", ".join(labels))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
