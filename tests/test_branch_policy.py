#!/usr/bin/env python3
"""Tests for official branch naming policy."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.interfaces.branch_policy import (  # noqa: E402
    detectar_branch_atual,
    exemplos_branch,
    validar_nome_branch,
)


class TestBranchPolicy(unittest.TestCase):
    """Cobertura do padrao oficial de nomes de branch."""

    def test_validar_branches_permanentes(self) -> None:
        self.assertTrue(validar_nome_branch("main").valid)
        self.assertTrue(validar_nome_branch("develop").valid)

    def test_validar_branches_tematicas(self) -> None:
        for nome in [
            "feat/add-branch-policy",
            "fix/checkpoint-parser",
            "docs/update-branching-guide",
            "chore/reorganize-ci",
            "hotfix/fix-release-link",
            "release/v2.2.4",
        ]:
            with self.subTest(nome=nome):
                resultado = validar_nome_branch(nome)
                self.assertTrue(resultado.valid)

    def test_rejeitar_branches_fora_do_padrao(self) -> None:
        for nome in [
            "feature/minha-branch",
            "docs/wiki links",
            "release/2.2.4",
            "Feat/upper",
            "",
        ]:
            with self.subTest(nome=nome):
                resultado = validar_nome_branch(nome)
                self.assertFalse(resultado.valid)

    def test_exemplos_contam_validas_e_invalidas(self) -> None:
        exemplos = exemplos_branch()
        self.assertIn("feat/add-multiclass-report", exemplos["validas"])
        self.assertIn("feature/nova-coisa", exemplos["invalidas"])

    def test_detectar_branch_atual_por_ambiente(self) -> None:
        valor_branch = os.environ.get("BRANCH_NAME")
        valor_github_ref = os.environ.get("GITHUB_REF_NAME")
        os.environ["BRANCH_NAME"] = "docs/update-wiki"
        os.environ["GITHUB_REF_NAME"] = "main"
        try:
            self.assertEqual(detectar_branch_atual(), "docs/update-wiki")
        finally:
            if valor_branch is None:
                os.environ.pop("BRANCH_NAME", None)
            else:
                os.environ["BRANCH_NAME"] = valor_branch

            if valor_github_ref is None:
                os.environ.pop("GITHUB_REF_NAME", None)
            else:
                os.environ["GITHUB_REF_NAME"] = valor_github_ref

    def test_cli_check_branch(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        resultado = subprocess.run(
            [
                sys.executable,
                "-m",
                "rede_neural_do_zero",
                "check-branch",
                "--name",
                "feat/add-branch-policy",
            ],
            check=True,
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertIn('"valid": true', resultado.stdout)

    def test_standalone_branch_policy_script(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        resultado = subprocess.run(
            [
                sys.executable,
                "src/interfaces/branch_policy.py",
                "--name",
                "feature/invalida",
                "--json",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(resultado.returncode, 0)
        self.assertIn('"valid": false', resultado.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
