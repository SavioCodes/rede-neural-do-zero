#!/usr/bin/env python3
"""Exporta notebooks do projeto para paginas Markdown dentro de `docs/`."""

from __future__ import annotations

import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
GENERATED_DIR = ROOT / "docs" / "notebooks" / "generated"


def _slugify(nome: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", nome.lower()).strip("-")


def _extrair_titulo(cells: list[nbformat.NotebookNode], fallback: str) -> str:
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        for linha in cell.source.splitlines():
            if linha.strip().startswith("# "):
                return linha.strip()[2:].strip()
    return fallback


def _converter_markdown(
    cells: list[nbformat.NotebookNode], titulo: str, notebook_path: Path
) -> str:
    linhas = [
        f"# {titulo}",
        "",
        "> Pagina gerada automaticamente a partir do notebook oficial do projeto.",
        "",
        f"- Arquivo fonte: `{notebook_path.relative_to(ROOT).as_posix()}`",
        "",
    ]

    primeira_heading_consumida = False
    for indice, cell in enumerate(cells, start=1):
        if cell.cell_type == "markdown":
            conteudo = cell.source.strip()
            if not conteudo:
                continue
            if not primeira_heading_consumida and conteudo.startswith("# "):
                primeira_heading_consumida = True
                restante = "\n".join(conteudo.splitlines()[1:]).strip()
                if restante:
                    linhas.extend([restante, ""])
                continue
            linhas.extend([conteudo, ""])
            continue

        if cell.cell_type != "code":
            continue

        linhas.extend(
            [
                f"## Celula {indice}",
                "",
                "```python",
                cell.source.rstrip(),
                "```",
                "",
            ]
        )

        textos_saida: list[str] = []
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                textos_saida.append(str(output.get("text", "")).rstrip())
            elif "text" in output:
                textos_saida.append(str(output.get("text", "")).rstrip())
            elif "data" in output and "text/plain" in output["data"]:
                textos_saida.append(str(output["data"]["text/plain"]).rstrip())

        if textos_saida:
            linhas.extend(
                [
                    "<details>",
                    "<summary>Saida registrada</summary>",
                    "",
                    "```text",
                    "\n\n".join(textos_saida).rstrip(),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
            )

    return "\n".join(linhas).strip() + "\n"


def exportar_notebook(notebook_path: Path) -> Path:
    notebook = nbformat.read(notebook_path, as_version=4)
    titulo = _extrair_titulo(notebook.cells, notebook_path.stem)
    slug = _slugify(notebook_path.stem)
    destino = GENERATED_DIR / f"{slug}.md"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        _converter_markdown(notebook.cells, titulo, notebook_path),
        encoding="utf-8",
    )
    return destino


def gerar_indice(paginas: list[tuple[str, Path, Path]]) -> None:
    linhas = [
        "# Notebooks Gerados",
        "",
        (
            "Estas paginas sao geradas automaticamente a partir dos notebooks "
            "versionados em `notebooks/`."
        ),
        "",
    ]
    for titulo, notebook_path, pagina_path in paginas:
        linhas.append(
            "- "
            f"[{titulo}]({pagina_path.name}) "
            f"a partir de `{notebook_path.relative_to(ROOT).as_posix()}`"
        )

    indice = GENERATED_DIR / "index.md"
    indice.write_text("\n".join(linhas).strip() + "\n", encoding="utf-8")


def main() -> None:
    paginas: list[tuple[str, Path, Path]] = []
    for notebook_path in sorted(NOTEBOOKS_DIR.glob("*.ipynb")):
        notebook = nbformat.read(notebook_path, as_version=4)
        titulo = _extrair_titulo(notebook.cells, notebook_path.stem)
        destino = exportar_notebook(notebook_path)
        paginas.append((titulo, notebook_path, destino))
        print(f"Notebook exportado: {notebook_path.name} -> {destino.relative_to(ROOT)}")

    gerar_indice(paginas)
    print(f"Indice atualizado em: {(GENERATED_DIR / 'index.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
