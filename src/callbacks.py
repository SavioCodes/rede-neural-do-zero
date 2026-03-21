"""Callbacks reutilizaveis para acompanhar e controlar o treinamento."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional


class Callback:
    """Classe base para callbacks.

    Os hooks recebem um dicionario simples de `logs` para manter a API
    facil de estudar. Cada callback pode ler metricas, salvar artefatos
    ou sinalizar parada antecipada.
    """

    def __init__(self) -> None:
        self.model = None
        self.params: dict[str, Any] = {}

    def set_model(self, model: Any) -> None:
        self.model = model

    def set_params(self, params: dict[str, Any]) -> None:
        self.params = params

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Executado antes da primeira epoca."""

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """Executado ao final de cada epoca."""

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """Executado ao final do treinamento."""


class History(Callback):
    """Armazena logs de treinamento em memoria."""

    def __init__(self) -> None:
        super().__init__()
        self.history: dict[str, list[Any]] = {}

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        self.history = {"epoch": []}

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        logs = logs or {}
        self.history.setdefault("epoch", []).append(epoch + 1)
        for chave, valor in logs.items():
            self.history.setdefault(chave, []).append(valor)


class EarlyStopping(Callback):
    """Interrompe o treino quando a metrica monitorada para de melhorar."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self._melhores_pesos = None

    def _melhorou(self, valor_atual: float) -> bool:
        if self.mode == "max":
            return valor_atual > self.best_value + self.min_delta
        return valor_atual < self.best_value - self.min_delta

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        self.wait = 0
        self.stopped_epoch = 0
        self.best_epoch = 0
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self._melhores_pesos = None

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        logs = logs or {}
        valor_atual = logs.get(self.monitor)
        if valor_atual is None and self.monitor.startswith("val_"):
            valor_atual = logs.get("loss")
        if valor_atual is None or self.model is None:
            return

        valor_atual = float(valor_atual)
        if self._melhorou(valor_atual):
            self.best_value = valor_atual
            self.best_epoch = epoch + 1
            self.wait = 0
            if self.restore_best_weights:
                self._melhores_pesos = self.model._copiar_parametros()
            return

        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = epoch + 1
            self.model.stop_training = True
            self.model._motivo_parada = "early_stopping"

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        if self.model is None:
            return

        if self.restore_best_weights and self._melhores_pesos is not None:
            self.model._restaurar_parametros(*self._melhores_pesos)

        self.model._melhor_monitor_callback = self.best_value
        self.model._melhor_epoch_callback = self.best_epoch


class ModelCheckpoint(Callback):
    """Salva parametros ao longo do treinamento."""

    def __init__(
        self,
        caminho: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ) -> None:
        super().__init__()
        self.caminho = caminho
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.ultimo_caminho_salvo: Optional[str] = None

    def _melhorou(self, valor_atual: float) -> bool:
        if self.mode == "max":
            return valor_atual > self.best_value
        return valor_atual < self.best_value

    def _formatar_caminho(self, epoch: int, logs: dict[str, Any]) -> str:
        nome = self.caminho.format(
            epoch=epoch + 1,
            monitor=self.monitor,
            valor=logs.get(self.monitor, logs.get("loss", 0.0)),
        )
        return nome

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        logs = logs or {}
        if self.model is None:
            return

        valor_atual = logs.get(self.monitor)
        if valor_atual is None and self.monitor.startswith("val_"):
            valor_atual = logs.get("loss")
        if valor_atual is None:
            return

        valor_atual = float(valor_atual)
        if self.save_best_only and not self._melhorou(valor_atual):
            return

        self.best_value = valor_atual
        caminho = self._formatar_caminho(epoch, logs)
        self.model.salvar_parametros(caminho)
        self.ultimo_caminho_salvo = caminho


class CSVLogger(Callback):
    """Registra logs de epoca em um arquivo CSV."""

    def __init__(self, caminho: str, append: bool = False) -> None:
        super().__init__()
        self.caminho = Path(caminho)
        self.append = append
        self._cabecalho_escrito = False

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        if self.caminho.parent != Path("."):
            self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._cabecalho_escrito = (
            self.append and self.caminho.exists() and self.caminho.stat().st_size > 0
        )

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        logs = logs or {}
        linha = {"epoch": epoch + 1, **logs}
        modo = "a" if self.append or self._cabecalho_escrito else "w"

        with self.caminho.open(modo, newline="", encoding="utf-8") as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=list(linha.keys()))
            if not self._cabecalho_escrito:
                writer.writeheader()
                self._cabecalho_escrito = True
            writer.writerow(linha)
