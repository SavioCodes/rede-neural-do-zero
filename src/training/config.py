"""Dataclasses para organizar configuracoes de modelo e treinamento."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .callbacks import Callback


@dataclass(slots=True)
class ModelConfig:
    """Configuracao declarativa do modelo."""

    arquitetura: list[int]
    ativacao: str = "sigmoid"
    inicializacao: str = "xavier"
    seed: Optional[int] = None
    funcao_custo: str = "binary_crossentropy"
    ativacao_saida: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Converte a configuracao em dicionario serializavel."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Reconstrui a configuracao a partir de um dicionario."""
        return cls(**data)


@dataclass(slots=True)
class TrainingConfig:
    """Configuracao declarativa do treinamento."""

    epochs: int = 1000
    taxa_aprendizado: float = 0.1
    verbose: bool = True
    paciencia: Optional[int] = None
    min_delta: float = 0.0
    restaurar_melhores_pesos: bool = True
    batch_size: Optional[int] = None
    otimizador: str = "sgd"
    embaralhar: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    l2_lambda: float = 0.0
    dropout: float = 0.0
    gradient_clip: Optional[float] = None
    callbacks: list[Callback] = field(default_factory=list)

    def to_dict(self, include_callbacks: bool = False) -> dict[str, Any]:
        """Converte a configuracao em dicionario serializavel."""
        data = asdict(self)
        if not include_callbacks:
            data["callbacks"] = []
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        """Reconstrui a configuracao a partir de um dicionario."""
        return cls(**data)
