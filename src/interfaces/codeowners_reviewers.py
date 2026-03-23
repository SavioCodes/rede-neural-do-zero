"""Resolve reviewers a partir do CODEOWNERS para automacoes de pull request."""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CodeownersEntry:
    """Representa uma linha util do arquivo CODEOWNERS."""

    pattern: str
    owners: tuple[str, ...]


@dataclass(slots=True)
class ReviewerResolution:
    """Resultado serializavel da resolucao de reviewers."""

    files: list[str]
    user_reviewers: list[str]
    team_reviewers: list[str]
    matched_owners: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        """Converte o resultado em dicionario serializavel."""
        return asdict(self)


def _normalizar_path(path: str) -> str:
    return path.replace("\\", "/").strip().lstrip("./")


def parse_codeowners(texto: str) -> list[CodeownersEntry]:
    """Converte o texto do CODEOWNERS em entradas estruturadas."""
    entradas: list[CodeownersEntry] = []
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        partes = linha.split()
        if len(partes) < 2:
            continue
        entradas.append(CodeownersEntry(pattern=partes[0], owners=tuple(partes[1:])))
    return entradas


def carregar_codeowners(path: str | Path) -> list[CodeownersEntry]:
    """Carrega e faz parse do arquivo CODEOWNERS."""
    conteudo = Path(path).read_text(encoding="utf-8")
    return parse_codeowners(conteudo)


def pattern_matches(pattern: str, path: str) -> bool:
    """Faz um match simples compativel com os padroes usados neste repositorio."""
    caminho = _normalizar_path(path)
    bruto = pattern.strip()
    ancorado = bruto.startswith("/")
    padrao = bruto.lstrip("/")

    if not padrao:
        return False

    if padrao.endswith("/"):
        prefixo = padrao.rstrip("/")
        if ancorado:
            return caminho == prefixo or caminho.startswith(prefixo + "/")
        return caminho == prefixo or ("/" + prefixo + "/") in ("/" + caminho + "/")

    if ancorado:
        return fnmatch.fnmatch(caminho, padrao)

    if "/" not in padrao:
        return any(fnmatch.fnmatch(parte, padrao) for parte in caminho.split("/"))

    if fnmatch.fnmatch(caminho, padrao):
        return True

    partes = caminho.split("/")
    for indice in range(1, len(partes)):
        if fnmatch.fnmatch("/".join(partes[indice:]), padrao):
            return True
    return False


def owners_for_file(path: str, entries: list[CodeownersEntry]) -> tuple[str, ...]:
    """Resolve os owners finais de um arquivo seguindo a ultima regra que der match."""
    owners: tuple[str, ...] = ()
    for entrada in entries:
        if pattern_matches(entrada.pattern, path):
            owners = entrada.owners
    return owners


def _split_reviewers(
    owners: tuple[str, ...],
    excluded_users: set[str],
) -> tuple[list[str], list[str]]:
    usuarios: list[str] = []
    times: list[str] = []
    for owner in owners:
        normalizado = owner.strip().lstrip("@")
        if not normalizado:
            continue
        if "/" in normalizado:
            if normalizado not in times:
                times.append(normalizado)
            continue
        if normalizado not in excluded_users and normalizado not in usuarios:
            usuarios.append(normalizado)
    return usuarios, times


def resolve_reviewers(
    files: list[str],
    entries: list[CodeownersEntry],
    excluded_users: set[str] | None = None,
) -> ReviewerResolution:
    """Resolve reviewers de usuario e time a partir dos arquivos alterados."""
    excluded = {
        usuario.strip().lstrip("@")
        for usuario in (excluded_users or set())
        if usuario.strip()
    }
    usuarios: list[str] = []
    times: list[str] = []
    matched: dict[str, list[str]] = {}

    for arquivo in [_normalizar_path(item) for item in files if item.strip()]:
        owners = owners_for_file(arquivo, entries)
        matched[arquivo] = list(owners)
        novos_usuarios, novos_times = _split_reviewers(owners, excluded)
        for usuario in novos_usuarios:
            if usuario not in usuarios:
                usuarios.append(usuario)
        for time in novos_times:
            if time not in times:
                times.append(time)

    return ReviewerResolution(
        files=[_normalizar_path(item) for item in files if item.strip()],
        user_reviewers=usuarios,
        team_reviewers=times,
        matched_owners=matched,
    )


def _ler_lista_arquivos(path: str | Path) -> list[str]:
    conteudo = Path(path).read_text(encoding="utf-8")
    return [linha.strip() for linha in conteudo.splitlines() if linha.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser standalone usado em workflows."""
    parser = argparse.ArgumentParser(description="Resolve reviewers com base no CODEOWNERS.")
    parser.add_argument(
        "--codeowners",
        default=".github/CODEOWNERS",
        help="Caminho do arquivo CODEOWNERS.",
    )
    parser.add_argument(
        "--files",
        required=True,
        help="Arquivo texto com um caminho alterado por linha.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Usuario a excluir da resolucao. Pode ser repetido.",
    )
    parser.add_argument("--json", action="store_true", help="Emite payload JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entrypoint standalone."""
    parser = build_parser()
    args = parser.parse_args(argv)

    entradas = carregar_codeowners(args.codeowners)
    arquivos = _ler_lista_arquivos(args.files)
    reviewers = resolve_reviewers(arquivos, entradas, set(args.exclude))

    if args.json:
        print(json.dumps(reviewers.to_dict(), indent=2, ensure_ascii=True))
    else:
        print(", ".join(reviewers.user_reviewers + reviewers.team_reviewers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
