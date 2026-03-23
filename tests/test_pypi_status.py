#!/usr/bin/env python3
"""Tests for PyPI publication status helpers and CLI exposure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.interfaces.pypi_status import (  # noqa: E402
    carregar_nome_projeto,
    obter_status_pypi,
    trusted_publisher_config,
)


class TestPyPIStatus(unittest.TestCase):
    """Cobertura do fluxo oficial de Trusted Publisher e status do PyPI."""

    def test_carregar_nome_projeto_do_pyproject(self) -> None:
        self.assertEqual(carregar_nome_projeto(), "rede-neural-do-zero")

    def test_trusted_publisher_config_padrao(self) -> None:
        payload = trusted_publisher_config("rede-neural-do-zero")
        self.assertEqual(payload.owner, "SavioCodes")
        self.assertEqual(payload.repository, "rede-neural-do-zero")
        self.assertEqual(payload.workflow_filename, "publish.yml")
        self.assertEqual(payload.environment, "pypi")
        self.assertEqual(payload.workflow_path, ".github/workflows/publish.yml")

    @patch("src.interfaces.pypi_status.projeto_existe_no_pypi", return_value=False)
    def test_status_indica_pending_publisher_na_primeira_publicacao(self, mocked_exists) -> None:
        payload = obter_status_pypi()
        self.assertFalse(payload.package_exists_on_pypi)
        self.assertTrue(payload.requires_pending_publisher)
        self.assertEqual(payload.publish_workflow_path, ".github/workflows/publish.yml")
        self.assertIn("pending publisher", payload.next_step.lower())
        mocked_exists.assert_called_once_with("rede-neural-do-zero")

    @patch("src.interfaces.pypi_status.projeto_existe_no_pypi", return_value=True)
    def test_status_indica_release_quando_projeto_ja_existe(self, mocked_exists) -> None:
        payload = obter_status_pypi(project_name="rede-neural-do-zero")
        self.assertTrue(payload.package_exists_on_pypi)
        self.assertFalse(payload.requires_pending_publisher)
        self.assertIn("GitHub Release", payload.next_step)
        mocked_exists.assert_called_once_with("rede-neural-do-zero")

    @patch("src.interfaces.pypi_status.projeto_existe_no_pypi", return_value=False)
    def test_cli_pypi_status_emite_json(self, _mocked_exists) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        resultado = subprocess.run(
            [sys.executable, "-m", "rede_neural_do_zero", "pypi-status"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(resultado.stdout)
        self.assertEqual(payload["project_name"], "rede-neural-do-zero")
        self.assertEqual(payload["trusted_publisher"]["environment"], "pypi")


if __name__ == "__main__":
    unittest.main(verbosity=2)
