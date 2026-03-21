#!/usr/bin/env python3
"""Valida notebooks do projeto sem depender de execucao completa."""

from __future__ import annotations

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"


def validar_notebook(caminho: Path) -> tuple[int, int]:
    """Carrega o notebook e compila celulas Python para detectar erros de sintaxe."""
    notebook = nbformat.read(caminho, as_version=4)
    markdown_cells = 0
    code_cells = 0

    if not notebook.cells:
        raise ValueError(f"O notebook {caminho.name} nao possui celulas.")

    for indice, cell in enumerate(notebook.cells, start=1):
        if cell.cell_type == "markdown":
            markdown_cells += 1
            continue
        if cell.cell_type != "code":
            continue

        code_cells += 1
        source = cell.source.rstrip()
        if not source:
            continue
        compile(source, f"{caminho.name}:cell_{indice}", "exec")

    if markdown_cells == 0:
        raise ValueError(f"O notebook {caminho.name} precisa ter pelo menos uma celula markdown.")

    return markdown_cells, code_cells


def main() -> None:
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit("Nenhum notebook encontrado para validar.")

    total_markdown = 0
    total_code = 0
    for notebook in notebooks:
        markdown_cells, code_cells = validar_notebook(notebook)
        total_markdown += markdown_cells
        total_code += code_cells
        print(
            f"Notebook valido: {notebook.name} " f"(markdown={markdown_cells}, code={code_cells})"
        )

    print(
        f"Validacao concluida: {len(notebooks)} notebooks, "
        f"{total_markdown} celulas markdown, {total_code} celulas de codigo."
    )


if __name__ == "__main__":
    main()
