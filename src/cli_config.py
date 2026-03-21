"""Suporte a arquivos de configuracao para a CLI oficial."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import yaml


def carregar_arquivo_config(caminho: str | Path) -> dict[str, Any]:
    """Carrega configuracoes em JSON, TOML ou YAML."""
    caminho_path = Path(caminho)
    if not caminho_path.exists():
        raise FileNotFoundError(f"Arquivo de config nao encontrado: {caminho_path}")

    sufixo = caminho_path.suffix.lower()
    texto = caminho_path.read_text(encoding="utf-8")
    if sufixo == ".json":
        dados = json.loads(texto)
    elif sufixo == ".toml":
        dados = tomllib.loads(texto)
    elif sufixo in {".yaml", ".yml"}:
        dados = yaml.safe_load(texto) or {}
    else:
        raise ValueError("Use um arquivo .json, .toml, .yaml ou .yml para --config.")

    if not isinstance(dados, dict):
        raise ValueError("O arquivo de configuracao precisa representar um objeto/dicionario.")
    return dados


def resolver_config_comando(config: dict[str, Any], comando: str) -> dict[str, Any]:
    """Resolve a secao aplicavel de um arquivo de configuracao."""
    dados: dict[str, Any] = {}
    comum = config.get("common")
    if isinstance(comum, dict):
        dados.update(comum)

    secao_comando = config.get(comando)
    if isinstance(secao_comando, dict):
        dados.update(secao_comando)
        return dados

    ignorar = {"common", "train", "evaluate", "benchmark", "example", "resume", "verify"}
    dados.update({chave: valor for chave, valor in config.items() if chave not in ignorar})
    return dados


def _mapear_opcoes(parser: argparse.ArgumentParser) -> dict[str, str]:
    """Mapeia `--flag` para o atributo `dest` do parser."""
    mapa: dict[str, str] = {}
    for acao in parser._actions:
        for opcao in acao.option_strings:
            mapa[opcao] = acao.dest
    return mapa


def _destinos_explicitos(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    """Descobre quais argumentos foram passados explicitamente pelo usuario."""
    mapa = _mapear_opcoes(parser)
    destinos: set[str] = set()

    indice = 0
    while indice < len(argv):
        token = argv[indice]
        if token == "--":
            break
        if token.startswith("--"):
            nome = token.split("=", 1)[0]
            dest = mapa.get(nome)
            if dest:
                destinos.add(dest)
        indice += 1

    return destinos


def aplicar_config_cli(
    args: argparse.Namespace,
    parser_comando: argparse.ArgumentParser,
    argv_comando: list[str],
) -> tuple[argparse.Namespace, dict[str, Any]]:
    """Aplica um arquivo de config sem sobrescrever flags passadas explicitamente."""
    caminho = getattr(args, "config", None)
    if not caminho:
        return args, {}

    config = carregar_arquivo_config(caminho)
    valores = resolver_config_comando(config, str(args.command))
    destinos_explicitos = _destinos_explicitos(parser_comando, argv_comando)

    for chave, valor in valores.items():
        if hasattr(args, chave) and chave not in destinos_explicitos:
            setattr(args, chave, valor)

    return args, valores


def serializar_config_efetiva(args: argparse.Namespace) -> dict[str, Any]:
    """Converte o namespace final da CLI em um dicionario serializavel."""
    dados: dict[str, Any] = {}
    for chave, valor in vars(args).items():
        if chave == "func":
            continue
        if isinstance(valor, Path):
            dados[chave] = str(valor)
        else:
            dados[chave] = valor
    return dados


def argv_comando_atual(argv: list[str] | None = None) -> tuple[list[str], list[str]]:
    """Separa o argv em raiz e parte especifica do subcomando."""
    argumentos = list(sys.argv[1:] if argv is None else argv)
    if not argumentos:
        return [], []
    return argumentos, argumentos[1:]
