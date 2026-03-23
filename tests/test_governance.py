#!/usr/bin/env python3
"""Tests for governance helpers and repository status reporting."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.interfaces.governance import (  # noqa: E402
    EXPECTED_CODEOWNERS_PATTERNS,
    EXPECTED_WORKFLOWS,
    RepoContext,
    avaliar_regras_governanca,
    montar_pr_summary,
    obter_release_status,
    parse_github_remote_url,
)
from src.interfaces.pypi_status import PyPIStatusResult, TrustedPublisherConfig  # noqa: E402


class TestGovernanceHelpers(unittest.TestCase):
    """Cobertura da camada de governanca da CLI oficial."""

    def test_parse_github_remote_url_https_e_ssh(self) -> None:
        https_context = parse_github_remote_url(
            "https://github.com/SavioCodes/rede-neural-do-zero.git"
        )
        ssh_context = parse_github_remote_url("git@github.com:SavioCodes/rede-neural-do-zero.git")

        self.assertEqual(https_context, RepoContext("SavioCodes", "rede-neural-do-zero"))
        self.assertEqual(ssh_context, RepoContext("SavioCodes", "rede-neural-do-zero"))

    def test_avaliar_regras_governanca_aprovado(self) -> None:
        relatorio = {
            "repository": {
                "homepage": "https://saviocodes.github.io/rede-neural-do-zero/",
                "allow_squash_merge": True,
                "allow_merge_commit": False,
                "allow_rebase_merge": False,
            },
            "pages": {"enabled": True, "url": "https://saviocodes.github.io/rede-neural-do-zero/"},
            "workflows": [
                {"name": nome, "path": f".github/workflows/{nome.lower()}.yml", "state": "active"}
                for nome in sorted(EXPECTED_WORKFLOWS)
            ],
            "branches": {
                "main": {
                    "enabled": True,
                    "required_checks": ["CI / quality", "Branch Policy / branch_name"],
                    "require_code_owner_reviews": True,
                    "required_linear_history": True,
                    "require_conversation_resolution": True,
                },
                "develop": {
                    "enabled": True,
                    "required_checks": ["CI / quality", "Branch Policy / branch_name"],
                    "require_code_owner_reviews": True,
                    "required_linear_history": True,
                    "require_conversation_resolution": True,
                },
            },
            "rulesets": [
                {
                    "name": "Linear History Protected Branches",
                    "enforcement": "active",
                    "target": "branch",
                    "html_url": "https://example.com/ruleset",
                }
            ],
            "environments": ["pypi"],
            "codeowners": {"patterns": sorted(EXPECTED_CODEOWNERS_PATTERNS)},
            "expected_site_url": "https://saviocodes.github.io/rede-neural-do-zero/",
        }

        checks = avaliar_regras_governanca(relatorio)
        self.assertTrue(all(check.ok for check in checks))

    def test_avaliar_regras_governanca_detecta_pages_desligado(self) -> None:
        relatorio = {
            "repository": {
                "homepage": "https://errado.example.com",
                "allow_squash_merge": True,
                "allow_merge_commit": False,
                "allow_rebase_merge": False,
            },
            "pages": {"enabled": False, "url": None},
            "workflows": [],
            "branches": {
                "main": {
                    "enabled": False,
                    "required_checks": [],
                    "require_code_owner_reviews": False,
                    "required_linear_history": False,
                    "require_conversation_resolution": False,
                },
                "develop": {
                    "enabled": False,
                    "required_checks": [],
                    "require_code_owner_reviews": False,
                    "required_linear_history": False,
                    "require_conversation_resolution": False,
                },
            },
            "rulesets": [],
            "environments": [],
            "codeowners": {"patterns": []},
            "expected_site_url": "https://saviocodes.github.io/rede-neural-do-zero/",
        }

        checks = avaliar_regras_governanca(relatorio)
        pages_check = next(check for check in checks if check.name == "pages_enabled")
        self.assertFalse(pages_check.ok)

    def test_montar_pr_summary_resolve_labels_areas_e_reviewers(self) -> None:
        resumo = montar_pr_summary(
            "docs/update-governance-guide",
            "develop",
            ["docs/governance.md", "src/interfaces/cli.py"],
        )

        self.assertEqual(resumo.labels, ["docs"])
        self.assertIn("docs", resumo.areas_touched)
        self.assertIn("interfaces", resumo.areas_touched)
        self.assertEqual(resumo.user_reviewers, ["SavioCodes"])
        self.assertTrue(resumo.target_validation["valid"])

    @patch("src.interfaces.governance.obter_status_pypi")
    @patch("src.interfaces.governance._gh_api_optional")
    @patch("src.interfaces.governance._gh_api")
    @patch("src.interfaces.governance._ler_versao_src_init")
    @patch("src.interfaces.governance.carregar_versao_pyproject")
    @patch("src.interfaces.governance.detectar_repositorio")
    def test_obter_release_status_detecta_pending_publisher(
        self,
        mock_detectar_repositorio,
        mock_carregar_versao,
        mock_ler_versao_src,
        mock_gh_api,
        mock_gh_api_optional,
        mock_obter_status_pypi,
    ) -> None:
        mock_detectar_repositorio.return_value = RepoContext("SavioCodes", "rede-neural-do-zero")
        mock_carregar_versao.return_value = "2.5.0"
        mock_ler_versao_src.return_value = "2.5.0"
        mock_gh_api.return_value = [
            {"tag_name": "v2.5.0", "draft": True},
            {"tag_name": "v2.4.1", "draft": False},
        ]
        mock_gh_api_optional.return_value = {"html_url": "https://saviocodes.github.io/rede-neural-do-zero/"}
        mock_obter_status_pypi.return_value = PyPIStatusResult(
            project_name="rede-neural-do-zero",
            project_url="https://pypi.org/project/rede-neural-do-zero/",
            package_exists_on_pypi=False,
            requires_pending_publisher=True,
            publish_workflow_path=".github/workflows/publish.yml",
            trusted_publisher=TrustedPublisherConfig(
                project_name="rede-neural-do-zero",
                owner="SavioCodes",
                repository="rede-neural-do-zero",
                workflow_filename="publish.yml",
                environment="pypi",
            ),
            next_step="Criar um pending publisher no PyPI.",
        )

        resultado = obter_release_status()
        self.assertEqual(resultado.expected_tag, "v2.5.0")
        self.assertTrue(resultado.versions_match)
        self.assertFalse(resultado.ready_for_release)
        self.assertIn("pending publisher", resultado.next_step.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
