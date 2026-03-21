"""Core neural network implementation built with NumPy only."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .funcoes_ativacao import FuncoesAtivacao


class RedeNeural:
    """Simple fully-connected neural network for educational experiments."""

    _INICIALIZACOES_VALIDAS = {"xavier", "he", "aleatorio"}

    def __init__(
        self,
        arquitetura: List[int],
        ativacao: str = "sigmoid",
        inicializacao: str = "xavier",
        seed: Optional[int] = None,
    ) -> None:
        self._validar_arquitetura(arquitetura)

        self.funcoes = FuncoesAtivacao()
        self.ativacao = self._validar_ativacao(ativacao)
        self.inicializacao = self._validar_inicializacao(inicializacao)
        self.seed = seed

        self.arquitetura = [int(neuronios) for neuronios in arquitetura]
        self.num_camadas = len(self.arquitetura)
        self._rng = np.random.default_rng(seed)

        self.historico_erro: list[float] = []
        self.historico_acuracia: list[float] = []
        self.historico_validacao_erro: list[float] = []
        self.historico_validacao_acuracia: list[float] = []

        self._inicializar_parametros(self.inicializacao)

    def _validar_arquitetura(self, arquitetura: List[int]) -> None:
        if len(arquitetura) < 2:
            raise ValueError("A arquitetura precisa ter pelo menos camada de entrada e saida.")

        if any(int(neuronios) <= 0 for neuronios in arquitetura):
            raise ValueError("Todos os tamanhos de camada devem ser inteiros positivos.")

    def _validar_ativacao(self, ativacao: str) -> str:
        ativacao_normalizada = ativacao.lower()
        if ativacao_normalizada not in self.funcoes.listar_funcoes():
            raise ValueError(
                f"Funcao de ativacao '{ativacao}' nao reconhecida. "
                f"Opcoes: {self.funcoes.listar_funcoes()}"
            )
        return ativacao_normalizada

    def _validar_inicializacao(self, inicializacao: str) -> str:
        inicializacao_normalizada = inicializacao.lower()
        if inicializacao_normalizada not in self._INICIALIZACOES_VALIDAS:
            raise ValueError(
                f"Inicializacao '{inicializacao}' nao reconhecida. "
                f"Opcoes: {sorted(self._INICIALIZACOES_VALIDAS)}"
            )
        return inicializacao_normalizada

    def _validar_entrada(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            X_array = X_array.reshape(1, -1)

        if X_array.ndim != 2:
            raise ValueError("Os dados de entrada devem ter formato 2D: (amostras, features).")

        if X_array.shape[1] != self.arquitetura[0]:
            raise ValueError(
                f"Esperadas {self.arquitetura[0]} features, mas recebido {X_array.shape[1]}."
            )

        if not np.all(np.isfinite(X_array)):
            raise ValueError("Os dados de entrada precisam conter apenas valores finitos.")

        return X_array

    def _validar_rotulos(self, y: np.ndarray, n_amostras: int) -> np.ndarray:
        y_array = np.asarray(y, dtype=float)
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)

        if y_array.ndim != 2:
            raise ValueError("Os rotulos devem ter formato 2D: (amostras, saidas).")

        if y_array.shape[0] != n_amostras:
            raise ValueError("X e y precisam ter a mesma quantidade de amostras.")

        if y_array.shape[1] != self.arquitetura[-1]:
            raise ValueError(
                f"Esperadas {self.arquitetura[-1]} saidas, mas recebido {y_array.shape[1]}."
            )

        if not np.all(np.isfinite(y_array)):
            raise ValueError("Os rotulos precisam conter apenas valores finitos.")

        return y_array

    def _validar_dados_validacao(
        self,
        validacao_X: Optional[np.ndarray],
        validacao_y: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if (validacao_X is None) != (validacao_y is None):
            raise ValueError("Forneca validacao_X e validacao_y juntos.")

        if validacao_X is None or validacao_y is None:
            return None, None

        validacao_X_array = self._validar_entrada(validacao_X)
        validacao_y_array = self._validar_rotulos(validacao_y, validacao_X_array.shape[0])
        return validacao_X_array, validacao_y_array

    def _inicializar_parametros(self, metodo: str) -> None:
        self.pesos = []
        self.biases = []

        for indice in range(self.num_camadas - 1):
            entrada_size = self.arquitetura[indice]
            saida_size = self.arquitetura[indice + 1]

            if metodo == "xavier":
                limite = np.sqrt(6.0 / (entrada_size + saida_size))
                peso = self._rng.uniform(-limite, limite, size=(entrada_size, saida_size))
            elif metodo == "he":
                peso = self._rng.standard_normal((entrada_size, saida_size)) * np.sqrt(
                    2.0 / entrada_size
                )
            else:
                peso = self._rng.standard_normal((entrada_size, saida_size)) * 0.1

            bias = np.zeros((1, saida_size))

            self.pesos.append(peso)
            self.biases.append(bias)

    def _forward_propagation(self, X: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
        ativacoes = [X]
        z_values = []

        for indice in range(self.num_camadas - 1):
            z = np.dot(ativacoes[indice], self.pesos[indice]) + self.biases[indice]
            z_values.append(z)

            if indice == self.num_camadas - 2:
                a = self.funcoes.sigmoid(z)
            else:
                a = self.funcoes.aplicar(z, self.ativacao)

            ativacoes.append(a)

        return ativacoes, z_values

    def _backward_propagation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ativacoes: List[np.ndarray],
        z_values: List[np.ndarray],
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        m = X.shape[0]
        gradientes_pesos: List[np.ndarray] = []
        gradientes_biases: List[np.ndarray] = []

        delta = ativacoes[-1] - y

        for indice in reversed(range(self.num_camadas - 1)):
            dW = np.dot(ativacoes[indice].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m

            gradientes_pesos.insert(0, dW)
            gradientes_biases.insert(0, db)

            if indice > 0:
                delta_z = self.funcoes.derivada(z_values[indice - 1], self.ativacao)
                delta = np.dot(delta, self.pesos[indice].T) * delta_z

        return gradientes_pesos, gradientes_biases

    def _calcular_erro(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def _calcular_acuracia(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        limiar: float = 0.5,
    ) -> float:
        predicoes_binarias = (y_pred >= limiar).astype(int)
        y_true_binarias = (y_true >= limiar).astype(int)
        return float(np.mean(predicoes_binarias == y_true_binarias) * 100)

    def treinar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        taxa_aprendizado: float = 0.1,
        verbose: bool = True,
        validacao_X: Optional[np.ndarray] = None,
        validacao_y: Optional[np.ndarray] = None,
    ) -> dict:
        if epochs <= 0:
            raise ValueError("epochs precisa ser maior que zero.")

        if taxa_aprendizado <= 0:
            raise ValueError("taxa_aprendizado precisa ser maior que zero.")

        X_array = self._validar_entrada(X)
        y_array = self._validar_rotulos(y, X_array.shape[0])
        validacao_X_array, validacao_y_array = self._validar_dados_validacao(
            validacao_X, validacao_y
        )

        self.historico_erro = []
        self.historico_acuracia = []
        self.historico_validacao_erro = []
        self.historico_validacao_acuracia = []

        intervalo_log = max(1, epochs // 10)

        for epoch in range(epochs):
            ativacoes, z_values = self._forward_propagation(X_array)
            grad_pesos, grad_biases = self._backward_propagation(
                X_array, y_array, ativacoes, z_values
            )

            for indice in range(len(self.pesos)):
                self.pesos[indice] -= taxa_aprendizado * grad_pesos[indice]
                self.biases[indice] -= taxa_aprendizado * grad_biases[indice]

            y_pred = ativacoes[-1]
            erro = self._calcular_erro(y_array, y_pred)
            acuracia = self._calcular_acuracia(y_array, y_pred)

            self.historico_erro.append(erro)
            self.historico_acuracia.append(acuracia)

            resumo_validacao = None
            if validacao_X_array is not None and validacao_y_array is not None:
                val_pred = self.prever(validacao_X_array)
                val_erro = self._calcular_erro(validacao_y_array, val_pred)
                val_acuracia = self._calcular_acuracia(validacao_y_array, val_pred)
                self.historico_validacao_erro.append(val_erro)
                self.historico_validacao_acuracia.append(val_acuracia)
                resumo_validacao = (val_erro, val_acuracia)

            if verbose and (epoch % intervalo_log == 0 or epoch == epochs - 1):
                print(f"Epoca {epoch:4d}: Erro = {erro:.4f}, Acuracia = {acuracia:.2f}%")
                if resumo_validacao is not None:
                    print(
                        "         Validacao: "
                        f"Erro = {resumo_validacao[0]:.4f}, "
                        f"Acuracia = {resumo_validacao[1]:.2f}%"
                    )

        resumo = {
            "erro_final": self.historico_erro[-1],
            "acuracia_final": self.historico_acuracia[-1],
            "epochs": epochs,
            "taxa_aprendizado": float(taxa_aprendizado),
        }

        if self.historico_validacao_erro:
            resumo["erro_validacao_final"] = self.historico_validacao_erro[-1]
            resumo["acuracia_validacao_final"] = self.historico_validacao_acuracia[-1]

        if verbose:
            print("\n" + "=" * 50)
            print("TREINAMENTO CONCLUIDO")
            print("=" * 50)
            print(f"Erro final: {resumo['erro_final']:.4f}")
            print(f"Acuracia final: {resumo['acuracia_final']:.2f}%")

        return resumo

    def prever(self, X: np.ndarray) -> np.ndarray:
        X_array = self._validar_entrada(X)
        ativacoes, _ = self._forward_propagation(X_array)
        return ativacoes[-1]

    def prever_classes(self, X: np.ndarray, limiar: float = 0.5) -> np.ndarray:
        if not 0 <= limiar <= 1:
            raise ValueError("limiar precisa estar entre 0 e 1.")
        return (self.prever(X) >= limiar).astype(int)

    def avaliar(self, X: np.ndarray, y: np.ndarray) -> dict:
        X_array = self._validar_entrada(X)
        y_array = self._validar_rotulos(y, X_array.shape[0])
        predicoes = self.prever(X_array)

        return {
            "erro": self._calcular_erro(y_array, predicoes),
            "acuracia": self._calcular_acuracia(y_array, predicoes),
            "predicoes": predicoes,
        }

    def obter_parametros(self) -> dict:
        return {
            "pesos": [peso.copy() for peso in self.pesos],
            "biases": [bias.copy() for bias in self.biases],
            "arquitetura": list(self.arquitetura),
            "ativacao": self.ativacao,
            "inicializacao": self.inicializacao,
            "seed": self.seed,
        }

    def salvar_parametros(self, caminho: str) -> None:
        caminho_arquivo = Path(caminho)
        if caminho_arquivo.parent != Path("."):
            caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

        pesos_obj = np.empty(len(self.pesos), dtype=object)
        pesos_obj[:] = self.pesos
        biases_obj = np.empty(len(self.biases), dtype=object)
        biases_obj[:] = self.biases

        parametros = {
            "pesos": pesos_obj,
            "biases": biases_obj,
            "arquitetura": np.array(self.arquitetura, dtype=int),
            "ativacao": np.array([self.ativacao], dtype=object),
            "inicializacao": np.array([self.inicializacao], dtype=object),
            "seed": np.array([self.seed], dtype=object),
        }
        np.savez(caminho_arquivo, **parametros)
        print(f"Parametros salvos em: {caminho_arquivo}")

    def carregar_parametros(self, caminho: str) -> None:
        dados = np.load(caminho, allow_pickle=True)

        self.pesos = [np.array(camada) for camada in dados["pesos"].tolist()]
        self.biases = [np.array(camada) for camada in dados["biases"].tolist()]
        self.arquitetura = [int(neuronios) for neuronios in dados["arquitetura"].tolist()]
        self.ativacao = str(dados["ativacao"].tolist()[0])
        self.inicializacao = str(
            dados["inicializacao"].tolist()[0]
            if "inicializacao" in dados.files
            else self.inicializacao
        )
        if "seed" in dados.files:
            seed_salva = dados["seed"].tolist()[0]
            self.seed = None if seed_salva is None else int(seed_salva)
        self.num_camadas = len(self.arquitetura)
        self._rng = np.random.default_rng(self.seed)
        self.historico_erro = []
        self.historico_acuracia = []
        self.historico_validacao_erro = []
        self.historico_validacao_acuracia = []
        print(f"Parametros carregados de: {caminho}")
