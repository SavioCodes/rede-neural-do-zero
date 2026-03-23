"""Camada oficial de governanca do repositorio para uso na CLI e em automacoes."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .branch_labels import labels_para_pull_request
from .branch_policy import (
    destinos_permitidos,
    detectar_branch_atual,
    validar_destino_pr,
    validar_nome_branch,
)
from .codeowners_reviewers import carregar_codeowners, resolve_reviewers
from .pypi_status import carregar_nome_projeto, obter_status_pypi
from .release_notes import carregar_versao_pyproject
from .release_validation import validar_release_local

VERSION_RE = re.compile(r'__version__\s*=\s*"(?P<version>[^"]+)"')
SITE_URL_RE = re.compile(r"^site_url:\s*(?P<site_url>\S+)\s*$", re.MULTILINE)
ROOT_VERSION_FILES = {"pyproject.toml", "src/__init__.py", "CHANGELOG.md"}
EXPECTED_WORKFLOWS = {
    "Branch Policy",
    "CI",
    "Codeowners Reviewers",
    "Docs",
    "Hotfix Sync",
    "PR Labels",
    "Publish",
    "Release Draft",
    "Release Readiness",
}
EXPECTED_CODEOWNERS_PATTERNS = {
    "/docs/",
    "/roadmaps/",
    "/src/core/",
    "/src/data/",
    "/src/training/",
    "/src/workflows/",
    "/src/interfaces/",
    "/notebooks/",
    "/scripts/",
}


@dataclass(frozen=True, slots=True)
class RepoContext:
    """Identifica o repositorio alvo das consultas de governanca."""

    owner: str
    repository: str

    @property
    def full_name(self) -> str:
        """Retorna o identificador `owner/repository`."""
        return f"{self.owner}/{self.repository}"

    def to_dict(self) -> dict[str, str]:
        """Converte o contexto em dicionario serializavel."""
        payload = asdict(self)
        payload["full_name"] = self.full_name
        return payload


@dataclass(slots=True)
class GovernanceCheck:
    """Representa uma verificacao oficial da governanca do repositorio."""

    name: str
    ok: bool
    message: str
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Converte a verificacao em dicionario serializavel."""
        return asdict(self)


@dataclass(slots=True)
class ReleaseStatusResult:
    """Resume o estado atual de release, versao e publicacao do projeto."""

    repository: str
    project_name: str
    local_version: str
    package_version: str
    versions_match: bool
    expected_tag: str
    current_release: dict[str, Any] | None
    latest_draft_release: dict[str, Any] | None
    latest_published_release: dict[str, Any] | None
    pages_enabled: bool
    pypi: dict[str, Any]
    ready_for_release: bool
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        """Converte o resultado em dicionario serializavel."""
        return asdict(self)


@dataclass(slots=True)
class PullRequestSummary:
    """Resumo local de um PR com base na branch, nos arquivos e no CODEOWNERS."""

    head_branch: str
    base_branch: str | None
    compare_range: str | None
    branch_validation: dict[str, Any]
    target_validation: dict[str, Any] | None
    changed_files: list[str]
    changed_file_count: int
    areas_touched: list[str]
    labels: list[str]
    user_reviewers: list[str]
    team_reviewers: list[str]
    matched_owners: dict[str, list[str]]
    release_related: bool

    def to_dict(self) -> dict[str, Any]:
        """Converte o resumo em dicionario serializavel."""
        return asdict(self)


def _executar_texto(comando: list[str]) -> str:
    resultado = subprocess.run(comando, check=True, capture_output=True, text=True)
    return resultado.stdout.strip()


def _executar_json(comando: list[str]) -> Any:
    saida = _executar_texto(comando)
    return json.loads(saida) if saida else None


def _git_ref_existe(ref: str) -> bool:
    resultado = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        capture_output=True,
        text=True,
    )
    return resultado.returncode == 0


def parse_github_remote_url(url: str) -> RepoContext | None:
    """Converte URLs comuns de remote GitHub em `owner/repository`."""
    valor = url.strip()
    padroes = [
        re.compile(r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?/?$"),
        re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?$"),
        re.compile(r"^ssh://git@github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+?)(?:\.git)?/?$"),
    ]
    for padrao in padroes:
        match = padrao.match(valor)
        if match:
            return RepoContext(match.group("owner"), match.group("repo"))
    return None


def detectar_repositorio(
    owner: str | None = None,
    repository: str | None = None,
    remote_name: str = "origin",
) -> RepoContext:
    """Resolve o repositorio atual pelo remote Git ou por argumentos explicitos."""
    if owner and repository:
        return RepoContext(owner, repository)
    if owner or repository:
        raise ValueError("Informe owner e repository juntos ou deixe ambos em branco.")

    remote_url = _executar_texto(["git", "config", "--get", f"remote.{remote_name}.url"])
    contexto = parse_github_remote_url(remote_url)
    if contexto is None:
        raise ValueError(f"Remote {remote_name!r} nao aponta para um repositorio GitHub valido.")
    return contexto


def _gh_api(contexto: RepoContext, endpoint: str) -> Any:
    caminho = f"repos/{contexto.full_name}"
    if endpoint:
        caminho = f"{caminho}/{endpoint.lstrip('/')}"
    return _executar_json(["gh", "api", caminho])


def _gh_api_optional(contexto: RepoContext, endpoint: str) -> Any | None:
    caminho = f"repos/{contexto.full_name}"
    if endpoint:
        caminho = f"{caminho}/{endpoint.lstrip('/')}"
    resultado = subprocess.run(["gh", "api", caminho], capture_output=True, text=True)
    if resultado.returncode == 0:
        saida = resultado.stdout.strip()
        return json.loads(saida) if saida else None
    stderr = (resultado.stderr or "").lower()
    if "not found" in stderr or "404" in stderr:
        return None
    raise subprocess.CalledProcessError(
        resultado.returncode,
        ["gh", "api", caminho],
        output=resultado.stdout,
        stderr=resultado.stderr,
    )


def _ler_site_url_mkdocs(path: str | Path = "mkdocs.yml") -> str:
    texto = Path(path).read_text(encoding="utf-8")
    match = SITE_URL_RE.search(texto)
    if match is None:
        raise ValueError("Nao encontrei `site_url` em mkdocs.yml.")
    return match.group("site_url")


def _ler_versao_src_init(path: str | Path = "src/__init__.py") -> str:
    texto = Path(path).read_text(encoding="utf-8")
    match = VERSION_RE.search(texto)
    if match is None:
        raise ValueError("Nao encontrei `__version__` em src/__init__.py.")
    return match.group("version")


def _resumir_branch_protection(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "required_checks": payload["required_status_checks"]["contexts"],
        "strict_status_checks": payload["required_status_checks"]["strict"],
        "required_approvals": payload["required_pull_request_reviews"][
            "required_approving_review_count"
        ],
        "require_code_owner_reviews": payload["required_pull_request_reviews"][
            "require_code_owner_reviews"
        ],
        "dismiss_stale_reviews": payload["required_pull_request_reviews"][
            "dismiss_stale_reviews"
        ],
        "require_conversation_resolution": payload["required_conversation_resolution"][
            "enabled"
        ],
        "required_linear_history": payload["required_linear_history"]["enabled"],
        "allow_force_pushes": payload["allow_force_pushes"]["enabled"],
        "allow_deletions": payload["allow_deletions"]["enabled"],
        "enforce_admins": payload["enforce_admins"]["enabled"],
    }


def _resumir_pages(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"enabled": False, "url": None, "build_type": None}
    return {
        "enabled": True,
        "url": payload.get("html_url"),
        "status": payload.get("status"),
        "build_type": payload.get("build_type"),
        "cname": payload.get("cname"),
        "source": payload.get("source"),
        "https_enforced": payload.get("https_enforced"),
    }


def _resumir_workflows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    workflows = payload.get("workflows", [])
    return [
        {
            "name": item["name"],
            "path": item["path"],
            "state": item["state"],
        }
        for item in workflows
    ]


def _resumir_rulesets(payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": item["name"],
            "target": item["target"],
            "enforcement": item["enforcement"],
            "html_url": item["_links"]["html"]["href"],
        }
        for item in payload
    ]


def _resumir_arquivos_comunidade() -> dict[str, bool]:
    return {
        "security": Path("SECURITY.md").exists(),
        "support": Path("SUPPORT.md").exists(),
        "funding": Path(".github/FUNDING.yml").exists(),
        "roadmap_index": Path("ROADMAP.md").exists(),
        "roadmap_directory": Path("roadmaps/README.md").exists(),
        "roadmap_template": Path("roadmaps/template.md").exists(),
        "onboarding": Path("docs/onboarding.md").exists(),
    }


def obter_governance_report(
    owner: str | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """Carrega um relatorio consolidado da governanca do repositorio no GitHub."""
    contexto = detectar_repositorio(owner, repository)
    repo = _gh_api(contexto, "")
    pages = _gh_api_optional(contexto, "pages")
    workflows = _resumir_workflows(_gh_api(contexto, "actions/workflows"))
    rulesets = _resumir_rulesets(_gh_api(contexto, "rulesets"))
    environments = _gh_api(contexto, "environments")
    branches = {
        branch: _resumir_branch_protection(
            _gh_api_optional(contexto, f"branches/{branch}/protection")
        )
        for branch in ("main", "develop")
    }
    codeowners_entries = carregar_codeowners(Path(".github") / "CODEOWNERS")

    return {
        "repository": {
            **contexto.to_dict(),
            "html_url": repo["html_url"],
            "default_branch": repo["default_branch"],
            "homepage": repo["homepage"],
            "allow_squash_merge": repo["allow_squash_merge"],
            "allow_merge_commit": repo["allow_merge_commit"],
            "allow_rebase_merge": repo["allow_rebase_merge"],
            "delete_branch_on_merge": repo["delete_branch_on_merge"],
            "allow_update_branch": repo["allow_update_branch"],
        },
        "pages": _resumir_pages(pages),
        "workflows": workflows,
        "rulesets": rulesets,
        "branches": branches,
        "environments": [item["name"] for item in environments.get("environments", [])],
        "codeowners": {
            "total_rules": len(codeowners_entries),
            "patterns": [entry.pattern for entry in codeowners_entries],
        },
        "community_files": _resumir_arquivos_comunidade(),
        "expected_site_url": _ler_site_url_mkdocs(),
    }


def avaliar_regras_governanca(relatorio: dict[str, Any]) -> list[GovernanceCheck]:
    """Avalia se o repositorio segue o conjunto oficial de regras esperado."""
    checks: list[GovernanceCheck] = []
    pages = relatorio["pages"]
    repo = relatorio["repository"]
    workflows = {item["name"]: item for item in relatorio["workflows"]}
    patterns = set(relatorio["codeowners"]["patterns"])
    community_files = relatorio["community_files"]

    checks.append(
        GovernanceCheck(
            name="pages_enabled",
            ok=bool(pages["enabled"]),
            message="GitHub Pages deve estar habilitado para publicar a documentacao oficial.",
            expected=True,
            actual=pages["enabled"],
        )
    )
    checks.append(
        GovernanceCheck(
            name="homepage_matches_site_url",
            ok=repo["homepage"] == relatorio["expected_site_url"],
            message="A homepage do repositorio deve apontar para a URL oficial do GitHub Pages.",
            expected=relatorio["expected_site_url"],
            actual=repo["homepage"],
        )
    )
    checks.append(
        GovernanceCheck(
            name="squash_merge_only",
            ok=repo["allow_squash_merge"]
            and not repo["allow_merge_commit"]
            and not repo["allow_rebase_merge"],
            message=(
                "O repositorio deve padronizar squash merge e "
                "bloquear merge commit/rebase merge."
            ),
            expected={
                "allow_squash_merge": True,
                "allow_merge_commit": False,
                "allow_rebase_merge": False,
            },
            actual={
                "allow_squash_merge": repo["allow_squash_merge"],
                "allow_merge_commit": repo["allow_merge_commit"],
                "allow_rebase_merge": repo["allow_rebase_merge"],
            },
        )
    )
    checks.append(
        GovernanceCheck(
            name="expected_workflows_active",
            ok=all(
                nome in workflows and workflows[nome].get("state") == "active"
                for nome in EXPECTED_WORKFLOWS
            ),
            message="Os workflows oficiais precisam existir e estar ativos.",
            expected=sorted(EXPECTED_WORKFLOWS),
            actual={nome: item.get("state") for nome, item in sorted(workflows.items())},
        )
    )

    for branch_name, branch in relatorio["branches"].items():
        checks.append(
            GovernanceCheck(
                name=f"{branch_name}_branch_protected",
                ok=bool(branch["enabled"]),
                message=f"A branch {branch_name} precisa estar protegida.",
                expected=True,
                actual=branch["enabled"],
            )
        )
        checks.append(
            GovernanceCheck(
                name=f"{branch_name}_required_checks",
                ok={"CI / quality", "Branch Policy / branch_name"}.issubset(
                    set(branch.get("required_checks", []))
                ),
                message=f"A branch {branch_name} precisa exigir CI e Branch Policy.",
                expected=["CI / quality", "Branch Policy / branch_name"],
                actual=branch.get("required_checks", []),
            )
        )
        checks.append(
            GovernanceCheck(
                name=f"{branch_name}_codeowners_reviews",
                ok=bool(branch.get("require_code_owner_reviews")),
                message=f"A branch {branch_name} precisa exigir review de CODEOWNERS.",
                expected=True,
                actual=branch.get("require_code_owner_reviews"),
            )
        )
        checks.append(
            GovernanceCheck(
                name=f"{branch_name}_linear_history",
                ok=bool(branch.get("required_linear_history")),
                message=f"A branch {branch_name} precisa exigir historico linear.",
                expected=True,
                actual=branch.get("required_linear_history"),
            )
        )
        checks.append(
            GovernanceCheck(
                name=f"{branch_name}_conversation_resolution",
                ok=bool(branch.get("require_conversation_resolution")),
                message=f"A branch {branch_name} precisa exigir resolucao de conversas.",
                expected=True,
                actual=branch.get("require_conversation_resolution"),
            )
        )

    checks.append(
        GovernanceCheck(
            name="ruleset_linear_history_active",
            ok=any(
                item["name"] == "Linear History Protected Branches"
                and item["enforcement"] == "active"
                for item in relatorio["rulesets"]
            ),
            message="O ruleset oficial de historico linear precisa estar ativo.",
            expected="Linear History Protected Branches",
            actual=relatorio["rulesets"],
        )
    )
    checks.append(
        GovernanceCheck(
            name="codeowners_granular_domains",
            ok=EXPECTED_CODEOWNERS_PATTERNS.issubset(patterns),
            message=(
                "O CODEOWNERS precisa cobrir docs, core, data, "
                "training, workflows e interfaces."
            ),
            expected=sorted(EXPECTED_CODEOWNERS_PATTERNS),
            actual=sorted(patterns),
        )
    )
    checks.append(
        GovernanceCheck(
            name="security_policy_present",
            ok=bool(community_files["security"]),
            message="O repositorio precisa ter um SECURITY.md oficial.",
            expected=True,
            actual=community_files["security"],
        )
    )
    checks.append(
        GovernanceCheck(
            name="support_policy_present",
            ok=bool(community_files["support"]),
            message="O repositorio precisa ter um SUPPORT.md oficial.",
            expected=True,
            actual=community_files["support"],
        )
    )
    checks.append(
        GovernanceCheck(
            name="versioned_roadmap_assets_present",
            ok=all(
                community_files[chave]
                for chave in ("roadmap_index", "roadmap_directory", "roadmap_template")
            ),
            message=(
                "O fluxo oficial de roadmap por versao precisa estar "
                "versionado no repositorio."
            ),
            expected={
                "roadmap_index": True,
                "roadmap_directory": True,
                "roadmap_template": True,
            },
            actual={
                "roadmap_index": community_files["roadmap_index"],
                "roadmap_directory": community_files["roadmap_directory"],
                "roadmap_template": community_files["roadmap_template"],
            },
        )
    )
    checks.append(
        GovernanceCheck(
            name="onboarding_doc_present",
            ok=bool(community_files["onboarding"]),
            message="O repositorio precisa ter um guia de onboarding oficial.",
            expected=True,
            actual=community_files["onboarding"],
        )
    )
    checks.append(
        GovernanceCheck(
            name="pypi_environment_present",
            ok="pypi" in relatorio["environments"],
            message="O environment `pypi` precisa existir para Trusted Publishing.",
            expected=True,
            actual=relatorio["environments"],
        )
    )

    return checks


def obter_rules_check(
    owner: str | None = None,
    repository: str | None = None,
) -> dict[str, Any]:
    """Executa o conjunto oficial de checagens de governanca."""
    relatorio = obter_governance_report(owner, repository)
    checks = avaliar_regras_governanca(relatorio)
    return {
        "repository": relatorio["repository"],
        "ok": all(check.ok for check in checks),
        "checks": [check.to_dict() for check in checks],
    }


def obter_release_status(
    owner: str | None = None,
    repository: str | None = None,
    pyproject_path: str | Path = "pyproject.toml",
    src_init_path: str | Path = "src/__init__.py",
) -> ReleaseStatusResult:
    """Consolida o estado de versao, draft release, Pages e PyPI."""
    contexto = detectar_repositorio(owner, repository)
    local_version = carregar_versao_pyproject(pyproject_path)
    package_version = _ler_versao_src_init(src_init_path)
    release_validation = validar_release_local(
        pyproject_path=pyproject_path,
        src_init_path=src_init_path,
    )
    expected_tag = f"v{local_version}"
    releases = _gh_api(contexto, "releases?per_page=10")
    current_release = next((item for item in releases if item["tag_name"] == expected_tag), None)
    latest_draft_release = next((item for item in releases if item["draft"]), None)
    latest_published_release = next((item for item in releases if not item["draft"]), None)
    pages = _gh_api_optional(contexto, "pages")
    pypi_status = obter_status_pypi(project_name=carregar_nome_projeto(pyproject_path))
    versions_match = local_version == package_version
    ready_for_release = (
        versions_match
        and pages is not None
        and current_release is not None
        and current_release.get("draft", False)
        and release_validation.ok
        and not pypi_status.requires_pending_publisher
    )

    if not versions_match:
        next_step = "Alinhe `pyproject.toml` e `src/__init__.py` antes de publicar."
    elif not release_validation.ok:
        next_step = "Corrija o checklist de release antes de publicar tag ou release."
    elif pages is None:
        next_step = "Ative o GitHub Pages oficial antes da proxima release publica."
    elif pypi_status.requires_pending_publisher:
        next_step = "Cadastre o pending publisher no PyPI antes de publicar a release final."
    elif current_release is None:
        next_step = "Espere o draft release ser criado ou crie a release correspondente."
    elif current_release.get("draft", False):
        next_step = "Revise o draft release atual e publique quando estiver pronto."
    else:
        next_step = "Fluxo de release sincronizado."

    return ReleaseStatusResult(
        repository=contexto.full_name,
        project_name=pypi_status.project_name,
        local_version=local_version,
        package_version=package_version,
        versions_match=versions_match,
        expected_tag=expected_tag,
        current_release=current_release,
        latest_draft_release=latest_draft_release,
        latest_published_release=latest_published_release,
        pages_enabled=pages is not None,
        pypi=pypi_status.to_dict(),
        ready_for_release=ready_for_release,
        next_step=next_step,
    )


def _listar_arquivos_alterados(
    base_branch: str | None,
    head_branch: str,
) -> tuple[list[str], str | None]:
    branch_atual = detectar_branch_atual() or ""
    head_ref = "HEAD"
    if head_branch != branch_atual and _git_ref_existe(head_branch):
        head_ref = head_branch
    if base_branch:
        compare_range = f"{base_branch}...{head_ref}"
    else:
        compare_range = None

    comando = ["git", "diff", "--name-only", "--diff-filter=ACMR"]
    if compare_range is not None:
        comando.append(compare_range)
    else:
        comando.extend(["HEAD~1", "HEAD"])

    saida = _executar_texto(comando)
    arquivos = [linha.strip() for linha in saida.splitlines() if linha.strip()]
    return arquivos, compare_range


def _categorizar_arquivo(path: str) -> str:
    valor = path.replace("\\", "/")
    if valor.startswith("docs/") or valor in {"README.md", "CHANGELOG.md", "CONTRIBUTING.md"}:
        return "docs"
    if valor.startswith(".github/"):
        return "github"
    if valor.startswith("notebooks/"):
        return "notebooks"
    if valor.startswith("scripts/"):
        return "scripts"
    if valor.startswith("configs/"):
        return "configs"
    if valor.startswith("experiments/"):
        return "experiments"
    if valor.startswith("rede_neural_do_zero/"):
        return "package"
    if valor.startswith("tests/"):
        return "tests"
    if valor.startswith("src/core/") or valor.startswith("src/rede_neural"):
        return "core"
    if valor.startswith("src/data/") or valor.startswith("src/utils.py"):
        return "data"
    if valor.startswith("src/training/") or valor.startswith("src/callbacks.py"):
        return "training"
    if valor.startswith("src/workflows/") or valor.startswith("src/benchmarking.py"):
        return "workflows"
    if valor.startswith("src/interfaces/") or valor.startswith("src/cli"):
        return "interfaces"
    return "root"


def montar_pr_summary(
    head_branch: str,
    base_branch: str | None,
    changed_files: list[str],
    excluded_users: set[str] | None = None,
) -> PullRequestSummary:
    """Monta o resumo local de um PR a partir da branch e dos arquivos alterados."""
    branch_validation = validar_nome_branch(head_branch)
    target_validation = (
        validar_destino_pr(head_branch, base_branch) if base_branch is not None else None
    )
    labels = labels_para_pull_request(head_branch, base_branch)
    reviewers = resolve_reviewers(
        changed_files,
        carregar_codeowners(Path(".github") / "CODEOWNERS"),
        excluded_users=excluded_users,
    )
    areas = sorted({_categorizar_arquivo(arquivo) for arquivo in changed_files})
    release_related = bool(
        set(changed_files) & ROOT_VERSION_FILES
        or "release" in labels
        or "hotfix" in labels
    )
    compare_range = f"{base_branch}...{head_branch}" if base_branch else None

    return PullRequestSummary(
        head_branch=head_branch,
        base_branch=base_branch,
        compare_range=compare_range,
        branch_validation=branch_validation.to_dict(),
        target_validation=target_validation.to_dict() if target_validation is not None else None,
        changed_files=changed_files,
        changed_file_count=len(changed_files),
        areas_touched=areas,
        labels=labels,
        user_reviewers=reviewers.user_reviewers,
        team_reviewers=reviewers.team_reviewers,
        matched_owners=reviewers.matched_owners,
        release_related=release_related,
    )


def obter_pr_summary(
    head_branch: str | None = None,
    base_branch: str | None = None,
) -> PullRequestSummary:
    """Resolve um resumo local do PR atual usando Git, branch policy e CODEOWNERS."""
    head = head_branch or detectar_branch_atual()
    if not head:
        raise ValueError("Nao foi possivel detectar a branch atual. Use --head.")
    base = base_branch
    if base is None:
        permitidos = destinos_permitidos(head)
        base = permitidos[0] if permitidos else None
    changed_files, compare_range = _listar_arquivos_alterados(base, head)
    resumo = montar_pr_summary(head, base, changed_files)
    resumo.compare_range = compare_range
    return resumo
