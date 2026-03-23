"""Verificacao leve de links internos da documentacao em Markdown."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(?P<title>.+?)\s*$")
MARKDOWN_LINK_RE = re.compile(r"!?\[[^\]]*\]\((?P<target>[^)]+)\)")
IGNORED_SCHEMES = ("http://", "https://", "mailto:", "tel:", "data:", "javascript:")


@dataclass(slots=True)
class LinkIssue:
    """Representa um problema encontrado em um link da documentacao."""

    source_path: str
    target: str
    message: str

    def to_dict(self) -> dict[str, str]:
        """Converte o problema em dicionario serializavel."""
        return asdict(self)


@dataclass(slots=True)
class DocsLinkCheckResult:
    """Resultado consolidado da verificacao de links da documentacao."""

    checked_files: list[str]
    issues: list[LinkIssue]

    @property
    def issue_count(self) -> int:
        """Quantidade de links invalidos encontrados."""
        return len(self.issues)

    def to_dict(self) -> dict[str, object]:
        """Converte o resultado em dicionario serializavel."""
        return {
            "checked_files": self.checked_files,
            "issue_count": self.issue_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _slugify_heading(texto: str) -> str:
    conteudo = re.sub(r"`([^`]*)`", r"\1", texto).strip().lower()
    conteudo = unicodedata.normalize("NFKD", conteudo).encode("ascii", "ignore").decode("ascii")
    conteudo = re.sub(r"[^\w\s-]", "", conteudo)
    conteudo = re.sub(r"[-\s]+", "-", conteudo).strip("-")
    return conteudo or "secao"


def extrair_anchors_markdown(texto: str) -> set[str]:
    """Extrai anchors aproximados de um arquivo Markdown usando o slug do heading."""
    anchors: set[str] = set()
    ocorrencias: dict[str, int] = {}
    em_bloco_codigo = False

    for linha in texto.splitlines():
        if linha.strip().startswith("```"):
            em_bloco_codigo = not em_bloco_codigo
            continue
        if em_bloco_codigo:
            continue
        match = HEADING_RE.match(linha)
        if not match:
            continue
        slug_base = _slugify_heading(match.group("title"))
        repeticao = ocorrencias.get(slug_base, 0)
        ocorrencias[slug_base] = repeticao + 1
        anchors.add(slug_base if repeticao == 0 else f"{slug_base}-{repeticao}")

    return anchors


def coletar_arquivos_markdown(caminhos: list[str | Path]) -> list[Path]:
    """Expande arquivos e diretorios em uma lista unica de arquivos Markdown."""
    arquivos: list[Path] = []
    vistos: set[Path] = set()

    for caminho in caminhos:
        path = Path(caminho)
        candidatos = [path]
        if path.is_dir():
            candidatos = sorted(path.rglob("*.md"))
        for candidato in candidatos:
            resolvido = candidato.resolve()
            if candidato.suffix.lower() != ".md" or resolvido in vistos:
                continue
            vistos.add(resolvido)
            arquivos.append(candidato)

    return arquivos


def _normalizar_alvo_bruto(raw_target: str) -> str:
    alvo = raw_target.strip().strip("<>").strip()
    if " " in alvo:
        alvo = alvo.split(" ", 1)[0]
    return alvo


def _resolver_alvo(source_path: Path, target: str) -> tuple[Path, str | None]:
    if "#" in target:
        destino, ancora = target.split("#", 1)
    else:
        destino, ancora = target, None
    if not destino:
        return source_path, ancora
    destino_path = Path(destino)
    if destino_path.is_absolute():
        return destino_path, ancora
    return (source_path.parent / destino_path).resolve(), ancora


def verificar_links_markdown(caminhos: list[str | Path]) -> DocsLinkCheckResult:
    """Verifica links internos entre arquivos Markdown do repositorio."""
    arquivos = coletar_arquivos_markdown(caminhos)
    cache_anchors: dict[Path, set[str]] = {}
    issues: list[LinkIssue] = []

    for arquivo in arquivos:
        texto = arquivo.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK_RE.finditer(texto):
            alvo = _normalizar_alvo_bruto(match.group("target"))
            if not alvo or alvo.startswith(IGNORED_SCHEMES):
                continue

            destino, ancora = _resolver_alvo(arquivo.resolve(), alvo)
            if not destino.exists():
                issues.append(
                    LinkIssue(
                        source_path=str(arquivo),
                        target=alvo,
                        message=f"Arquivo de destino inexistente: {destino}",
                    )
                )
                continue

            if ancora and destino.suffix.lower() == ".md":
                anchors = cache_anchors.setdefault(
                    destino,
                    extrair_anchors_markdown(destino.read_text(encoding="utf-8")),
                )
                if ancora not in anchors:
                    issues.append(
                        LinkIssue(
                            source_path=str(arquivo),
                            target=alvo,
                            message=f"Ancora '#{ancora}' nao encontrada em {destino}",
                        )
                    )

    return DocsLinkCheckResult(
        checked_files=[str(arquivo) for arquivo in arquivos],
        issues=issues,
    )


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser standalone usado localmente e em CI."""
    parser = argparse.ArgumentParser(description="Valida links internos da documentacao.")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["README.md", "docs"],
        help="Arquivos ou diretorios Markdown a verificar.",
    )
    parser.add_argument("--json", action="store_true", help="Emite o resultado em JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint standalone."""
    parser = build_parser()
    args = parser.parse_args(argv)
    resultado = verificar_links_markdown(args.paths)

    if args.json:
        print(json.dumps(resultado.to_dict(), indent=2, ensure_ascii=True))
    else:
        if resultado.issue_count == 0:
            print(
                "Links verificados com sucesso em "
                f"{len(resultado.checked_files)} arquivos Markdown."
            )
        else:
            for issue in resultado.issues:
                print(f"{issue.source_path}: {issue.message} ({issue.target})")

    return 0 if resultado.issue_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
