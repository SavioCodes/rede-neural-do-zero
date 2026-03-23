#!/usr/bin/env python3
"""Tests for automatic PR labels based on branch names."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.interfaces.branch_labels import (  # noqa: E402
    definicoes_para_labels,
    labels_para_pull_request,
)


class TestBranchLabels(unittest.TestCase):
    """Cobertura da logica de labels automaticas."""

    def test_labels_por_prefixo(self) -> None:
        cenarios = [
            ("feat/add-governance-page", "develop", ["feat"]),
            ("fix/adjust-verify-step", "develop", ["fix"]),
            ("docs/update-release-guide", "develop", ["docs"]),
            ("chore/reorganize-workflows", "develop", ["chore"]),
            ("hotfix/fix-main-sync", "main", ["hotfix"]),
            ("release/v2.3.0", "main", ["release"]),
        ]
        for head, base, esperado in cenarios:
            with self.subTest(head=head, base=base):
                self.assertEqual(labels_para_pull_request(head, base), esperado)

    def test_labels_por_fluxo_entre_branches_permanentes(self) -> None:
        self.assertEqual(labels_para_pull_request("develop", "main"), ["release"])
        self.assertEqual(labels_para_pull_request("main", "develop"), ["governance"])

    def test_branch_invalida_nao_recebe_label(self) -> None:
        self.assertEqual(labels_para_pull_request("feature/minha-branch", "develop"), [])

    def test_definicoes_somente_para_labels_existentes(self) -> None:
        definicoes = definicoes_para_labels(["feat", "release", "inexistente"])
        nomes = [item["name"] for item in definicoes]
        self.assertEqual(nomes, ["feat", "release"])

    def test_script_standalone_emite_json(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        resultado = subprocess.run(
            [
                sys.executable,
                "src/interfaces/branch_labels.py",
                "--head",
                "develop",
                "--base",
                "main",
                "--json",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn('"labels": [', resultado.stdout)
        self.assertIn('"release"', resultado.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
