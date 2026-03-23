#!/usr/bin/env python3
"""Tests for changelog-based release note extraction."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.interfaces.release_notes import (  # noqa: E402
    construir_release_notes,
    extrair_secao_changelog,
    normalizar_versao,
)


class TestReleaseNotes(unittest.TestCase):
    """Cobertura da extracao de release notes do changelog."""

    def test_normalizar_versao_remove_prefixo_v(self) -> None:
        self.assertEqual(normalizar_versao("v2.4.0"), "2.4.0")
        self.assertEqual(normalizar_versao("2.4.0"), "2.4.0")

    def test_extrair_secao_changelog_da_versao(self) -> None:
        changelog = """
# Changelog

## [2.4.0] - 2026-03-23

### Added

- item novo

## [2.3.0] - 2026-03-23

### Added

- item antigo
"""
        secao = extrair_secao_changelog(changelog, "v2.4.0")
        self.assertIn("## [2.4.0] - 2026-03-23", secao)
        self.assertNotIn("## [2.3.0]", secao)

    def test_construir_release_notes_le_pyproject_e_changelog(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz / "pyproject.toml").write_text(
                """
[project]
version = "2.4.0"
""".strip(),
                encoding="utf-8",
            )
            (raiz / "CHANGELOG.md").write_text(
                """
# Changelog

## [2.4.0] - 2026-03-23

### Added

- item novo
""".strip(),
                encoding="utf-8",
            )
            resultado = construir_release_notes(
                changelog_path=raiz / "CHANGELOG.md",
                pyproject_path=raiz / "pyproject.toml",
            )
            self.assertEqual(resultado.tag_name, "v2.4.0")
            self.assertIn("item novo", resultado.body)

    def test_script_standalone_emite_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        resultado = subprocess.run(
            [
                sys.executable,
                "src/interfaces/release_notes.py",
                "--version",
                "v2.3.0",
                "--json",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"tag_name": "v2.3.0"', resultado.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
