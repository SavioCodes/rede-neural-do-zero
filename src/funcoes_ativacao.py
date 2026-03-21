"""Funcoes de ativacao usadas pela rede neural educacional."""

from __future__ import annotations

from typing import Callable

import numpy as np


class FuncoesAtivacao:
    """Colecao de funcoes de ativacao e suas derivadas.

    O objetivo desta classe nao e "esconder" a matematica, e sim deixar os
    blocos centrais da rede em um lugar facil de estudar e testar.
    """

    _FUNCOES = {
        "sigmoid": "sigmoid",
        "relu": "relu",
        "tanh": "tanh",
        "leaky_relu": "leaky_relu",
        "linear": "linear",
    }

    _DERIVADAS = {
        "sigmoid": "sigmoid_derivada",
        "relu": "relu_derivada",
        "tanh": "tanh_derivada",
        "leaky_relu": "leaky_relu_derivada",
        "linear": "linear_derivada",
    }

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Retorna valores entre 0 e 1.

        O clipping evita overflow numerico quando `x` fica muito grande em modulo.
        """
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    @staticmethod
    def sigmoid_derivada(x: np.ndarray) -> np.ndarray:
        """Derivada da sigmoid: s(x) * (1 - s(x))."""
        sigmoid_x = FuncoesAtivacao.sigmoid(x)
        return sigmoid_x * (1.0 - sigmoid_x)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit: mantem positivos e zera negativos."""
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivada(x: np.ndarray) -> np.ndarray:
        """Derivada da ReLU: 1 para positivos, 0 caso contrario."""
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tangente hiperbolica: saida entre -1 e 1."""
        return np.tanh(x)

    @staticmethod
    def tanh_derivada(x: np.ndarray) -> np.ndarray:
        """Derivada da tanh: 1 - tanh(x)^2."""
        tanh_x = np.tanh(x)
        return 1.0 - tanh_x**2

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Leaky ReLU: evita zerar totalmente o gradiente para valores negativos."""
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivada(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Derivada da Leaky ReLU."""
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Funcao identidade."""
        return x

    @staticmethod
    def linear_derivada(x: np.ndarray) -> np.ndarray:
        """Derivada da identidade."""
        return np.ones_like(x)

    def _resolver_nome(self, nome_funcao: str) -> str:
        """Normaliza o nome recebido pelo usuario."""
        return nome_funcao.lower().strip()

    def _obter_funcao(self, nome_funcao: str) -> Callable[[np.ndarray], np.ndarray]:
        """Resolve uma funcao de ativacao pelo nome."""
        nome_normalizado = self._resolver_nome(nome_funcao)
        if nome_normalizado not in self._FUNCOES:
            raise ValueError(
                f"Funcao de ativacao '{nome_funcao}' nao reconhecida. "
                f"Opcoes disponiveis: {self.listar_funcoes()}"
            )
        return getattr(self, self._FUNCOES[nome_normalizado])

    def _obter_derivada(self, nome_funcao: str) -> Callable[[np.ndarray], np.ndarray]:
        """Resolve a derivada de uma funcao de ativacao pelo nome."""
        nome_normalizado = self._resolver_nome(nome_funcao)
        if nome_normalizado not in self._DERIVADAS:
            raise ValueError(
                f"Derivada da funcao '{nome_funcao}' nao disponivel. "
                f"Opcoes disponiveis: {self.listar_funcoes()}"
            )
        return getattr(self, self._DERIVADAS[nome_normalizado])

    def aplicar(self, x: np.ndarray, nome_funcao: str) -> np.ndarray:
        """Aplica a funcao de ativacao escolhida."""
        return self._obter_funcao(nome_funcao)(x)

    def derivada(self, x: np.ndarray, nome_funcao: str) -> np.ndarray:
        """Aplica a derivada da funcao de ativacao escolhida."""
        return self._obter_derivada(nome_funcao)(x)

    @classmethod
    def listar_funcoes(cls) -> list[str]:
        """Lista as ativacoes suportadas pela implementacao."""
        return list(cls._FUNCOES.keys())

    @classmethod
    def info_funcao(cls, nome_funcao: str) -> str:
        """Retorna uma descricao curta e didatica sobre cada ativacao."""
        info = {
            "sigmoid": "Sigmoid: saida entre 0 e 1, muito usada para probabilidade.",
            "relu": "ReLU: simples e eficiente, comum em camadas ocultas.",
            "tanh": "Tanh: saida entre -1 e 1, centrada em zero.",
            "leaky_relu": (
                "Leaky ReLU: semelhante a ReLU, "
                "mas mantem gradiente pequeno nos negativos."
            ),
            "linear": "Linear: nao distorce a entrada, util em saidas de regressao.",
        }
        return info.get(nome_funcao.lower(), f"Funcao '{nome_funcao}' nao encontrada.")
