"""Valida o pacote de release do projeto antes de tags e publicacoes."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .release_notes import construir_release_notes, normalizar_versao

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
CHANGELOG_HEADER_RE = re.compile(r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>.+))?$")
VERSION_RE = re.compile(r'__version__\s*=\s*"(?P<version>[^"]+)"')


@dataclass(slots=True)
class ReleaseValidationCheck:
    """Representa um item do checklist oficial de release."""

    name: str
    ok: bool
    message: str
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Converte o item em dicionario serializavel."""
        return asdict(self)


@dataclass(slots=True)
class ReleaseValidationResult:
    """Consolida o status da release local antes de tags ou publicacao."""

    project_name: str
    pyproject_version: str
    src_version: str
    changelog_version: str
    release_notes_version: str
    expected_tag: str
    release_notes_title: str
    release_notes_body: str
    changelog_path: str
    checks: list[dict[str, Any]]
    ok: bool
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        """Converte o resultado em dicionario serializavel."""
        return asdict(self)


def _carregar_nome_projeto(pyproject_path: str | Path) -> str:
    import tomllib

    dados = tomllib.loads(Path(pyproject_path).read_text(encoding="utf-8"))
    return str(dados["project"]["name"])


def _carregar_versao_pyproject(pyproject_path: str | Path) -> str:
    import tomllib

    dados = tomllib.loads(Path(pyproject_path).read_text(encoding="utf-8"))
    return str(dados["project"]["version"])


def _carregar_versao_src_init(src_init_path: str | Path) -> str:
    texto = Path(src_init_path).read_text(encoding="utf-8")
    match = VERSION_RE.search(texto)
    if match is None:
        raise ValueError("Nao encontrei `__version__` em src/__init__.py.")
    return match.group("version")


def extrair_versao_topo_changelog(changelog_path: str | Path = "CHANGELOG.md") -> str:
    """Retorna a versao da primeira secao do changelog."""
    texto = Path(changelog_path).read_text(encoding="utf-8")
    for linha in texto.splitlines():
        match = CHANGELOG_HEADER_RE.match(linha.strip())
        if match:
            return normalizar_versao(match.group("version"))
    raise ValueError("Nao encontrei nenhuma secao versionada em CHANGELOG.md.")


def validar_release_local(
    pyproject_path: str | Path = "pyproject.toml",
    src_init_path: str | Path = "src/__init__.py",
    changelog_path: str | Path = "CHANGELOG.md",
) -> ReleaseValidationResult:
    """Valida se versao, changelog e release notes estao alinhados."""
    pyproject_version = normalizar_versao(_carregar_versao_pyproject(pyproject_path))
    src_version = normalizar_versao(_carregar_versao_src_init(src_init_path))
    changelog_version = extrair_versao_topo_changelog(changelog_path)
    expected_tag = f"v{pyproject_version}"
    release_notes_error: str | None = None
    try:
        release_notes = construir_release_notes(
            version=pyproject_version,
            changelog_path=changelog_path,
            pyproject_path=pyproject_path,
        )
        release_notes_version = release_notes.version
        release_notes_title = release_notes.title
        release_notes_body = release_notes.body
        release_notes_tag = release_notes.tag_name
    except ValueError as exc:
        release_notes_error = str(exc)
        release_notes_version = ""
        release_notes_title = ""
        release_notes_body = ""
        release_notes_tag = ""

    checks = [
        ReleaseValidationCheck(
            name="pyproject_semver",
            ok=bool(SEMVER_RE.match(pyproject_version)),
            message="A versao em pyproject.toml precisa seguir semver simples X.Y.Z.",
            expected="X.Y.Z",
            actual=pyproject_version,
        ),
        ReleaseValidationCheck(
            name="src_semver",
            ok=bool(SEMVER_RE.match(src_version)),
            message="A versao em src/__init__.py precisa seguir semver simples X.Y.Z.",
            expected="X.Y.Z",
            actual=src_version,
        ),
        ReleaseValidationCheck(
            name="pyproject_matches_src",
            ok=pyproject_version == src_version,
            message="`pyproject.toml` e `src/__init__.py` precisam usar a mesma versao.",
            expected=pyproject_version,
            actual=src_version,
        ),
        ReleaseValidationCheck(
            name="pyproject_matches_changelog_top",
            ok=pyproject_version == changelog_version,
            message=(
                "A primeira secao do CHANGELOG precisa corresponder a versao atual do pacote."
            ),
            expected=pyproject_version,
            actual=changelog_version,
        ),
        ReleaseValidationCheck(
            name="release_notes_matches_version",
            ok=release_notes_version == pyproject_version,
            message="As release notes extraidas precisam apontar para a versao atual.",
            expected=pyproject_version,
            actual=release_notes_version or release_notes_error,
        ),
        ReleaseValidationCheck(
            name="release_notes_tag_matches",
            ok=release_notes_tag == expected_tag,
            message="A tag esperada precisa bater com a versao atual do pacote.",
            expected=expected_tag,
            actual=release_notes_tag or release_notes_error,
        ),
        ReleaseValidationCheck(
            name="release_notes_title_mentions_tag",
            ok=release_notes_title.startswith(expected_tag),
            message="O titulo das release notes precisa comecar com a tag esperada.",
            expected=expected_tag,
            actual=release_notes_title or release_notes_error,
        ),
        ReleaseValidationCheck(
            name="release_notes_body_not_empty",
            ok=bool(release_notes_body.strip()),
            message="O corpo das release notes nao pode estar vazio.",
            expected="texto nao vazio",
            actual=(
                f"{len(release_notes_body.strip())} chars"
                if release_notes_body
                else release_notes_error
            ),
        ),
        ReleaseValidationCheck(
            name="release_notes_body_contains_header",
            ok=f"## [{pyproject_version}]" in release_notes_body,
            message="O corpo das release notes precisa conter o cabecalho da versao atual.",
            expected=f"## [{pyproject_version}]",
            actual=(
                release_notes_body.splitlines()[0]
                if release_notes_body
                else release_notes_error
            ),
        ),
    ]

    ok = all(item.ok for item in checks)
    if ok:
        next_step = (
            "Release local consistente. Rode a verificacao completa, revise o draft e so "
            "depois publique a release/tag."
        )
    else:
        next_step = (
            "Corrija versao, changelog ou release notes antes de criar tag ou publicar release."
        )

    return ReleaseValidationResult(
        project_name=_carregar_nome_projeto(pyproject_path),
        pyproject_version=pyproject_version,
        src_version=src_version,
        changelog_version=changelog_version,
        release_notes_version=release_notes_version,
        expected_tag=expected_tag,
        release_notes_title=release_notes_title,
        release_notes_body=release_notes_body,
        changelog_path=str(changelog_path),
        checks=[item.to_dict() for item in checks],
        ok=ok,
        next_step=next_step,
    )


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser standalone."""
    parser = argparse.ArgumentParser(description="Valida o pacote oficial de release.")
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--src-init", default="src/__init__.py")
    parser.add_argument("--changelog", default="CHANGELOG.md")
    parser.add_argument("--json", action="store_true", help="Emite um payload JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint standalone."""
    parser = build_parser()
    args = parser.parse_args(argv)
    resultado = validar_release_local(
        pyproject_path=args.pyproject,
        src_init_path=args.src_init,
        changelog_path=args.changelog,
    )
    if args.json:
        print(json.dumps(resultado.to_dict(), indent=2, ensure_ascii=True))
    else:
        for check in resultado.checks:
            status = "OK" if check["ok"] else "FAIL"
            print(f"[{status}] {check['name']}: {check['message']}")
        print(resultado.next_step)
    return 0 if resultado.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
