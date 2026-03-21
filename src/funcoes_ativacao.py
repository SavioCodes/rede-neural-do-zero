"""Activation functions used by the educational neural network."""

from __future__ import annotations

import numpy as np


class FuncoesAtivacao:
    """Collection of activation functions and their derivatives."""

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function with clipping for numerical stability."""
        x_clipped = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clipped))

    @staticmethod
    def sigmoid_derivada(x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the sigmoid function."""
        sigmoid_x = FuncoesAtivacao.sigmoid(x)
        return sigmoid_x * (1.0 - sigmoid_x)

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Compute the ReLU activation."""
        return np.maximum(0.0, x)

    @staticmethod
    def relu_derivada(x: np.ndarray) -> np.ndarray:
        """Compute the derivative of ReLU."""
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Compute the hyperbolic tangent activation."""
        return np.tanh(x)

    @staticmethod
    def tanh_derivada(x: np.ndarray) -> np.ndarray:
        """Compute the derivative of tanh."""
        tanh_x = np.tanh(x)
        return 1.0 - tanh_x**2

    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Compute the leaky ReLU activation."""
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivada(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """Compute the derivative of leaky ReLU."""
        return np.where(x > 0, 1.0, alpha)

    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Compute the identity activation."""
        return x

    @staticmethod
    def linear_derivada(x: np.ndarray) -> np.ndarray:
        """Compute the derivative of the identity activation."""
        return np.ones_like(x)

    def aplicar(self, x: np.ndarray, nome_funcao: str) -> np.ndarray:
        """Apply an activation function by name."""
        funcoes = {
            "sigmoid": self.sigmoid,
            "relu": self.relu,
            "tanh": self.tanh,
            "leaky_relu": self.leaky_relu,
            "linear": self.linear,
        }

        nome_normalizado = nome_funcao.lower()
        if nome_normalizado not in funcoes:
            raise ValueError(
                f"Funcao de ativacao '{nome_funcao}' nao reconhecida. "
                f"Opcoes disponiveis: {list(funcoes.keys())}"
            )

        return funcoes[nome_normalizado](x)

    def derivada(self, x: np.ndarray, nome_funcao: str) -> np.ndarray:
        """Compute the derivative of an activation function by name."""
        derivadas = {
            "sigmoid": self.sigmoid_derivada,
            "relu": self.relu_derivada,
            "tanh": self.tanh_derivada,
            "leaky_relu": self.leaky_relu_derivada,
            "linear": self.linear_derivada,
        }

        nome_normalizado = nome_funcao.lower()
        if nome_normalizado not in derivadas:
            raise ValueError(
                f"Derivada da funcao '{nome_funcao}' nao disponivel. "
                f"Opcoes disponiveis: {list(derivadas.keys())}"
            )

        return derivadas[nome_normalizado](x)

    @classmethod
    def listar_funcoes(cls) -> list[str]:
        """Return the list of supported activation names."""
        return ["sigmoid", "relu", "tanh", "leaky_relu", "linear"]

    @classmethod
    def info_funcao(cls, nome_funcao: str) -> str:
        """Return a short textual description for a given activation."""
        info = {
            "sigmoid": "Sigmoid: saida entre 0 e 1, comum em classificacao binaria.",
            "relu": "ReLU: eficiente e simples, mas pode sofrer com dead neurons.",
            "tanh": "Tanh: saida entre -1 e 1, zero-centered.",
            "leaky_relu": (
                "Leaky ReLU: versao da ReLU com gradiente pequeno para valores negativos."
            ),
            "linear": "Linear: identidade, util para regressao.",
        }
        return info.get(nome_funcao.lower(), f"Funcao '{nome_funcao}' nao encontrada.")
