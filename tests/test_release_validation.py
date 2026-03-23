#!/usr/bin/env python3
"""Tests for release readiness validation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.interfaces.release_validation import (  # noqa: E402
    extrair_versao_topo_changelog,
    validar_release_local,
)


class TestReleaseValidation(unittest.TestCase):
    """Cobertura da validacao oficial de release."""

    def test_extrair_versao_topo_changelog(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            changelog = Path(pasta) / "CHANGELOG.md"
            changelog.write_text(
                "\n".join(
                    [
                        "# Changelog",
                        "",
                        "## [2.5.0] - 2026-03-23",
                        "",
                        "### Added",
                        "",
                        "- item novo",
                    ]
                ),
                encoding="utf-8",
            )
            self.assertEqual(extrair_versao_topo_changelog(changelog), "2.5.0")

    def test_validar_release_local_aprovado(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz / "pyproject.toml").write_text(
                '[project]\nname = "rede-neural-do-zero"\nversion = "2.5.0"\n',
                encoding="utf-8",
            )
            (raiz / "src").mkdir()
            (raiz / "src" / "__init__.py").write_text(
                '__version__ = "2.5.0"\n',
                encoding="utf-8",
            )
            (raiz / "CHANGELOG.md").write_text(
                "\n".join(
                    [
                        "# Changelog",
                        "",
                        "## [2.5.0] - 2026-03-23",
                        "",
                        "### Added",
                        "",
                        "- item novo",
                    ]
                ),
                encoding="utf-8",
            )
            resultado = validar_release_local(
                pyproject_path=raiz / "pyproject.toml",
                src_init_path=raiz / "src" / "__init__.py",
                changelog_path=raiz / "CHANGELOG.md",
            )
            self.assertTrue(resultado.ok)
            self.assertEqual(resultado.expected_tag, "v2.5.0")

    def test_validar_release_local_detecta_desalinhamento(self) -> None:
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz / "pyproject.toml").write_text(
                '[project]\nname = "rede-neural-do-zero"\nversion = "2.5.0"\n',
                encoding="utf-8",
            )
            (raiz / "src").mkdir()
            (raiz / "src" / "__init__.py").write_text(
                '__version__ = "2.4.9"\n',
                encoding="utf-8",
            )
            (raiz / "CHANGELOG.md").write_text(
                "\n".join(
                    [
                        "# Changelog",
                        "",
                        "## [2.4.9] - 2026-03-23",
                        "",
                        "### Added",
                        "",
                        "- item antigo",
                    ]
                ),
                encoding="utf-8",
            )
            resultado = validar_release_local(
                pyproject_path=raiz / "pyproject.toml",
                src_init_path=raiz / "src" / "__init__.py",
                changelog_path=raiz / "CHANGELOG.md",
            )
            self.assertFalse(resultado.ok)
            checks = {item["name"]: item for item in resultado.checks}
            self.assertFalse(checks["pyproject_matches_src"]["ok"])
            self.assertFalse(checks["pyproject_matches_changelog_top"]["ok"])

    def test_cli_release_check_emite_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        resultado = subprocess.run(
            [sys.executable, "-m", "rede_neural_do_zero", "release-check", "--json"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(resultado.stdout)
        self.assertIn("expected_tag", payload)
        self.assertIn("checks", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
