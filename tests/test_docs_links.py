#!/usr/bin/env python3
"""Tests for internal documentation link checks."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.interfaces.docs_links import verificar_links_markdown  # noqa: E402


class TestDocsLinks(unittest.TestCase):
    """Cobertura do verificador leve de links Markdown."""

    def test_links_validos_entre_docs_e_readme(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio)
            readme = raiz / "README.md"
            docs = raiz / "docs"
            docs.mkdir()
            pagina_a = docs / "a.md"
            pagina_b = docs / "b.md"

            readme.write_text("# Inicio\n", encoding="utf-8")
            pagina_b.write_text("# Pagina B\n\n## Secao B\n", encoding="utf-8")
            pagina_a.write_text(
                "\n".join(
                    [
                        "# Pagina A",
                        "",
                        "[Leia a B](./b.md#secao-b)",
                        "[Voltar](../README.md)",
                        "[Externo](https://example.com)",
                    ]
                ),
                encoding="utf-8",
            )

            resultado = verificar_links_markdown([readme, docs])
            self.assertEqual(resultado.issue_count, 0)

    def test_detecta_arquivo_e_ancora_inexistentes(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            raiz = Path(diretorio)
            docs = raiz / "docs"
            docs.mkdir()
            pagina = docs / "a.md"
            destino = docs / "b.md"

            destino.write_text("# Pagina B\n", encoding="utf-8")
            pagina.write_text(
                "\n".join(
                    [
                        "# Pagina A",
                        "",
                        "[Sem ancora](./b.md#nao-existe)",
                        "[Sem arquivo](./c.md)",
                    ]
                ),
                encoding="utf-8",
            )

            resultado = verificar_links_markdown([docs])
            self.assertEqual(resultado.issue_count, 2)
            mensagens = [issue.message for issue in resultado.issues]
            self.assertTrue(any("Ancora" in mensagem for mensagem in mensagens))
            self.assertTrue(
                any("Arquivo de destino inexistente" in mensagem for mensagem in mensagens)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
