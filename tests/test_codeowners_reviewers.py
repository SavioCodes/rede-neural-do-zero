#!/usr/bin/env python3
"""Tests for CODEOWNERS reviewer resolution."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.interfaces.codeowners_reviewers import (  # noqa: E402
    carregar_codeowners,
    parse_codeowners,
    pattern_matches,
    resolve_reviewers,
)


class TestCodeownersReviewers(unittest.TestCase):
    """Cobertura da resolucao de reviewers baseada em CODEOWNERS."""

    def test_parse_codeowners_ignora_comentarios(self) -> None:
        entradas = parse_codeowners(
            """
            # comentario
            /docs/ @SavioCodes
            /src/core/ @SavioCodes
            """
        )
        self.assertEqual(len(entradas), 2)
        self.assertEqual(entradas[0].pattern, "/docs/")

    def test_pattern_matches_para_pastas_e_arquivos_ancorados(self) -> None:
        self.assertTrue(pattern_matches("/docs/", "docs/governance.md"))
        self.assertTrue(pattern_matches("/README.md", "README.md"))
        self.assertFalse(pattern_matches("/docs/", "src/docs/governance.md"))

    def test_resolve_reviewers_deduplica_usuarios(self) -> None:
        entradas = carregar_codeowners(Path(".github") / "CODEOWNERS")
        resultado = resolve_reviewers(
            ["docs/governance.md", "src/interfaces/cli.py"],
            entradas,
        )
        self.assertEqual(resultado.user_reviewers, ["SavioCodes"])
        self.assertEqual(resultado.team_reviewers, [])

    def test_excluir_autor_remove_usuario_da_lista(self) -> None:
        entradas = carregar_codeowners(Path(".github") / "CODEOWNERS")
        resultado = resolve_reviewers(
            ["docs/governance.md"],
            entradas,
            excluded_users={"SavioCodes"},
        )
        self.assertEqual(resultado.user_reviewers, [])

    def test_codeowners_cobre_areas_granulares_do_projeto(self) -> None:
        entradas = carregar_codeowners(Path(".github") / "CODEOWNERS")
        padroes = {entrada.pattern for entrada in entradas}
        self.assertIn("/docs/", padroes)
        self.assertIn("/roadmaps/", padroes)
        self.assertIn("/src/core/", padroes)
        self.assertIn("/src/data/", padroes)
        self.assertIn("/src/training/", padroes)
        self.assertIn("/src/workflows/", padroes)
        self.assertIn("/src/interfaces/", padroes)

    def test_script_standalone_emite_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as arquivo:
            arquivo.write("docs/governance.md\nsrc/interfaces/cli.py\n")
            caminho_arquivos = arquivo.name
        try:
            resultado = subprocess.run(
                [
                    sys.executable,
                    "src/interfaces/codeowners_reviewers.py",
                    "--codeowners",
                    ".github/CODEOWNERS",
                    "--files",
                    caminho_arquivos,
                    "--json",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn('"user_reviewers": [', resultado.stdout)
            self.assertIn('"SavioCodes"', resultado.stdout)
        finally:
            Path(caminho_arquivos).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
