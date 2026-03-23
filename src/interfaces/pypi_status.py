"""Utilitarios para inspecionar o fluxo oficial de publicacao no PyPI."""

from __future__ import annotations

import tomllib
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class TrustedPublisherConfig:
    """Representa a configuracao esperada do Trusted Publisher no PyPI."""

    project_name: str
    owner: str
    repository: str
    workflow_filename: str
    environment: str

    @property
    def workflow_path(self) -> str:
        """Caminho relativo do workflow que publica no PyPI."""
        return f".github/workflows/{self.workflow_filename}"

    def to_dict(self) -> dict[str, str]:
        """Converte a configuracao em dicionario serializavel."""
        payload = asdict(self)
        payload["workflow_path"] = self.workflow_path
        return payload


@dataclass(slots=True)
class PyPIStatusResult:
    """Resume o estado atual do pacote e do fluxo de Trusted Publisher."""

    project_name: str
    project_url: str
    package_exists_on_pypi: bool
    requires_pending_publisher: bool
    publish_workflow_path: str
    trusted_publisher: TrustedPublisherConfig
    next_step: str

    def to_dict(self) -> dict[str, object]:
        """Converte o resultado em dicionario serializavel."""
        payload = asdict(self)
        payload["trusted_publisher"] = self.trusted_publisher.to_dict()
        return payload


def carregar_nome_projeto(pyproject_path: str | Path = "pyproject.toml") -> str:
    """Le o nome oficial do pacote no `pyproject.toml`."""
    dados = tomllib.loads(Path(pyproject_path).read_text(encoding="utf-8"))
    return str(dados["project"]["name"])


def trusted_publisher_config(
    project_name: str,
    owner: str = "SavioCodes",
    repository: str = "rede-neural-do-zero",
    workflow_filename: str = "publish.yml",
    environment: str = "pypi",
) -> TrustedPublisherConfig:
    """Monta a configuracao oficial do Trusted Publisher esperada pelo repositorio."""
    return TrustedPublisherConfig(
        project_name=project_name,
        owner=owner,
        repository=repository,
        workflow_filename=workflow_filename,
        environment=environment,
    )


def projeto_existe_no_pypi(project_name: str) -> bool:
    """Verifica se o projeto ja esta publicado no PyPI pelo endpoint JSON oficial."""
    url = f"https://pypi.org/pypi/{project_name}/json"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status == 200
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def obter_status_pypi(
    project_name: str | None = None,
    *,
    owner: str = "SavioCodes",
    repository: str = "rede-neural-do-zero",
    workflow_filename: str = "publish.yml",
    environment: str = "pypi",
    pyproject_path: str | Path = "pyproject.toml",
) -> PyPIStatusResult:
    """Resolve o estado atual do pacote no PyPI e a configuracao esperada do publisher."""
    nome_projeto = project_name or carregar_nome_projeto(pyproject_path)
    publisher = trusted_publisher_config(
        nome_projeto,
        owner=owner,
        repository=repository,
        workflow_filename=workflow_filename,
        environment=environment,
    )
    existe = projeto_existe_no_pypi(nome_projeto)
    next_step = (
        "Criar um pending publisher no PyPI com os mesmos dados "
        "do workflow antes da primeira release."
        if not existe
        else "Publicar uma GitHub Release para acionar o workflow oficial de upload no PyPI."
    )
    return PyPIStatusResult(
        project_name=nome_projeto,
        project_url=f"https://pypi.org/project/{nome_projeto}/",
        package_exists_on_pypi=existe,
        requires_pending_publisher=not existe,
        publish_workflow_path=publisher.workflow_path,
        trusted_publisher=publisher,
        next_step=next_step,
    )
