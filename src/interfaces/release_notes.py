"""Extrai release notes oficiais a partir do CHANGELOG.md."""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

CHANGELOG_HEADER_RE = re.compile(r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>.+))?$")


@dataclass(slots=True)
class ReleaseNotesResult:
    """Representa um bloco de release notes pronto para automacao."""

    version: str
    tag_name: str
    title: str
    body: str
    source_path: str

    def to_dict(self) -> dict[str, str]:
        """Converte o resultado em dicionario serializavel."""
        return asdict(self)


def normalizar_versao(valor: str) -> str:
    """Remove prefixos como `v` e espacos extras."""
    versao = valor.strip()
    if versao.startswith("v"):
        versao = versao[1:]
    return versao


def carregar_versao_pyproject(path: str | Path = "pyproject.toml") -> str:
    """Le a versao oficial do pacote no pyproject."""
    dados = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    return str(dados["project"]["version"])


def extrair_secao_changelog(texto: str, version: str) -> str:
    """Extrai a secao do changelog correspondente a uma versao."""
    alvo = normalizar_versao(version)
    linhas = texto.splitlines()
    inicio: int | None = None
    for indice, linha in enumerate(linhas):
        match = CHANGELOG_HEADER_RE.match(linha)
        if match and normalizar_versao(match.group("version")) == alvo:
            inicio = indice
            break

    if inicio is None:
        raise ValueError(f"Nao encontrei a versao {alvo} em CHANGELOG.md.")

    fim = len(linhas)
    for indice in range(inicio + 1, len(linhas)):
        if CHANGELOG_HEADER_RE.match(linhas[indice]):
            fim = indice
            break

    bloco = "\n".join(linhas[inicio:fim]).strip()
    if not bloco:
        raise ValueError(f"A secao da versao {alvo} esta vazia no CHANGELOG.md.")
    return bloco


def construir_release_notes(
    version: str | None = None,
    changelog_path: str | Path = "CHANGELOG.md",
    pyproject_path: str | Path = "pyproject.toml",
) -> ReleaseNotesResult:
    """Monta o payload oficial de release notes para a versao desejada."""
    versao = normalizar_versao(version or carregar_versao_pyproject(pyproject_path))
    changelog = Path(changelog_path)
    corpo = extrair_secao_changelog(changelog.read_text(encoding="utf-8"), versao)
    tag_name = f"v{versao}"
    titulo = f"{tag_name} - Draft release from CHANGELOG"
    return ReleaseNotesResult(
        version=versao,
        tag_name=tag_name,
        title=titulo,
        body=corpo,
        source_path=str(changelog),
    )


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser standalone."""
    parser = argparse.ArgumentParser(description="Extrai release notes oficiais do CHANGELOG.")
    parser.add_argument(
        "--version",
        default=None,
        help="Versao ou tag desejada. Se omitido, usa a versao do pyproject.",
    )
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Caminho para o CHANGELOG.",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Caminho para o pyproject.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Arquivo opcional para salvar o corpo das release notes em Markdown.",
    )
    parser.add_argument("--json", action="store_true", help="Emite payload JSON completo.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint standalone."""
    parser = build_parser()
    args = parser.parse_args(argv)
    resultado = construir_release_notes(
        version=args.version,
        changelog_path=args.changelog,
        pyproject_path=args.pyproject,
    )

    if args.output:
        Path(args.output).write_text(resultado.body, encoding="utf-8")

    if args.json:
        print(json.dumps(resultado.to_dict(), indent=2, ensure_ascii=True))
    else:
        print(resultado.body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
