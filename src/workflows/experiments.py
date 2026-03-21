"""Workflows reutilizaveis para CLI, scripts e exemplos do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..core.rede_neural import RedeNeural
from ..data.utils import DataUtils, MetricUtils
from ..training.config import ModelConfig, TrainingConfig


@dataclass(slots=True)
class DatasetBundle:
    """Agrupa dados, metadados e tipo de tarefa."""

    nome: str
    X: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any]

    @property
    def tipo_tarefa(self) -> str:
        return str(self.metadata["tipo_tarefa"])


def carregar_dataset(
    nome: str,
    seed: int = 42,
    samples: int = 240,
    normalizar: Optional[str] = "padrao",
) -> DatasetBundle:
    """Carrega um dataset sintetico ou real suportado pelo projeto."""
    nome_normalizado = nome.lower().strip()
    metadata: dict[str, Any]

    if nome_normalizado == "xor":
        X, y = DataUtils.gerar_xor_dataset()
        metadata = {"tipo_tarefa": "classificacao_binaria", "origem": "sintetico"}
    elif nome_normalizado in {"binario", "synthetic_binary", "classificacao"}:
        X, y = DataUtils.gerar_dataset_classificacao(n_samples=samples, random_state=seed)
        metadata = {"tipo_tarefa": "classificacao_binaria", "origem": "sintetico"}
    elif nome_normalizado in {"multiclasse", "synthetic_multiclass"}:
        X, y = DataUtils.gerar_dataset_multiclasse(n_samples=samples, random_state=seed)
        metadata = {"tipo_tarefa": "classificacao_multiclasse", "origem": "sintetico"}
    elif nome_normalizado in {"regressao", "synthetic_regression"}:
        X, y = DataUtils.gerar_dataset_regressao(n_samples=samples, random_state=seed)
        metadata = {"tipo_tarefa": "regressao", "origem": "sintetico"}
    elif nome_normalizado in DataUtils.listar_datasets_reais():
        X, y, metadata = DataUtils.carregar_dataset_real(nome_normalizado, normalizar=None)
        metadata = {**metadata, "origem": "real"}
    else:
        raise ValueError(
            f"Dataset '{nome}' nao reconhecido. "
            "Opcoes: xor, binario, multiclasse, regressao, iris, wine, diabetes."
        )

    if normalizar:
        X, params_normalizacao = DataUtils.normalizar_dados(X, metodo=normalizar)
        metadata = {**metadata, "normalizacao": params_normalizacao}

    return DatasetBundle(nome=nome_normalizado, X=X, y=y, metadata=metadata)


def dividir_treino_validacao_teste(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.2,
) -> dict[str, np.ndarray]:
    """Divide o dataset em treino, validacao e teste."""
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
        X,
        y,
        test_size=test_size,
        random_state=seed,
    )
    X_train, X_val, y_train, y_val = DataUtils.dividir_treino_teste(
        X_train,
        y_train,
        test_size=val_size,
        random_state=seed,
    )
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def criar_configs_padrao(
    dataset: DatasetBundle,
    seed: int,
    epochs: int,
    verbose: bool = False,
) -> tuple[ModelConfig, TrainingConfig]:
    """Escolhe uma configuracao padrao coerente com o dataset."""
    input_dim = int(dataset.X.shape[1])
    tipo = dataset.tipo_tarefa

    if tipo == "regressao":
        model_config = ModelConfig(
            arquitetura=[input_dim, 32, 16, 1],
            ativacao="relu",
            inicializacao="he",
            seed=seed,
            funcao_custo="mse",
            ativacao_saida="linear",
        )
        training_config = TrainingConfig(
            epochs=epochs,
            taxa_aprendizado=0.01,
            batch_size=32,
            otimizador="adam",
            paciencia=35,
            min_delta=1e-4,
            l2_lambda=1e-4,
            gradient_clip=1.0,
            verbose=verbose,
        )
        return model_config, training_config

    output_dim = 1
    funcao_custo = "binary_crossentropy"
    if tipo == "classificacao_multiclasse":
        output_dim = int(len(np.unique(dataset.y.reshape(-1))))
        funcao_custo = "categorical_crossentropy"

    arquitetura = [input_dim, 16, 12, output_dim] if output_dim > 1 else [input_dim, 12, 1]
    model_config = ModelConfig(
        arquitetura=arquitetura,
        ativacao="relu",
        inicializacao="he",
        seed=seed,
        funcao_custo=funcao_custo,
    )
    training_config = TrainingConfig(
        epochs=epochs,
        taxa_aprendizado=0.01,
        batch_size=16,
        otimizador="adam",
        paciencia=30,
        min_delta=1e-4,
        l2_lambda=1e-4,
        gradient_clip=1.0,
        verbose=verbose,
    )
    return model_config, training_config


def avaliar_modelo(rede: RedeNeural, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Avalia a rede com metricas apropriadas para a tarefa."""
    resultado = rede.avaliar(X, y)
    if resultado["tipo_problema"].startswith("regressao"):
        resultado["metricas_regressao"] = MetricUtils.metricas_regressao(y, resultado["predicoes"])
        return resultado

    if resultado["tipo_problema"] == "classificacao_multiclasse":
        resultado["metricas_classificacao"] = MetricUtils.metricas_classificacao(
            y,
            resultado["predicoes"],
        )
        return resultado

    resultado["metricas_classificacao"] = MetricUtils.precisao_recall_f1(y, resultado["predicoes"])
    return resultado
