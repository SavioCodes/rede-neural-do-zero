"""Nucleo avancado da implementacao didatica de rede neural."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, List, Optional

import numpy as np

from .callbacks import Callback, EarlyStopping
from .config import ModelConfig, TrainingConfig
from .funcoes_ativacao import FuncoesAtivacao


class RedeNeural:
    """Rede neural simples voltada para estudo de classificacao e regressao."""

    _INICIALIZACOES_VALIDAS = {"xavier", "he", "aleatorio"}
    _FUNCOES_CUSTO_VALIDAS = {
        "binary_crossentropy",
        "categorical_crossentropy",
        "mse",
    }
    _OTIMIZADORES_VALIDOS = {"sgd", "adam"}
    _ATIVACOES_SAIDA_VALIDAS = {"sigmoid", "softmax", "linear"}

    def __init__(
        self,
        arquitetura: List[int],
        ativacao: str = "sigmoid",
        inicializacao: str = "xavier",
        seed: Optional[int] = None,
        funcao_custo: str = "binary_crossentropy",
        ativacao_saida: Optional[str] = None,
    ) -> None:
        """Constroi a rede e inicializa seus parametros."""
        self._validar_arquitetura(arquitetura)

        self.arquitetura = [int(neuronios) for neuronios in arquitetura]
        self.num_camadas = len(self.arquitetura)
        self.output_dim = self.arquitetura[-1]

        funcao_custo_resolvida = self._resolver_funcao_custo_inicial(
            funcao_custo,
            ativacao_saida,
        )

        self.funcoes = FuncoesAtivacao()
        self.ativacao = self._validar_ativacao(ativacao)
        self.inicializacao = self._validar_inicializacao(inicializacao)
        self.ativacao_saida = self._validar_ativacao_saida(ativacao_saida)
        self.funcao_custo = self._validar_funcao_custo(funcao_custo_resolvida)
        self._validar_combinacao_saida_e_perda()
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self.stop_training = False
        self._motivo_parada = "epochs_concluidas"
        self._melhor_monitor_callback = None
        self._melhor_epoch_callback = 0
        self._ultimo_l2_lambda = 0.0
        self._ultimo_estado_otimizador: Optional[dict[str, Any]] = None
        self._ultimo_config_treino: Optional[dict[str, Any]] = None
        self._ultima_epoca_treinada = 0
        self._total_atualizacoes_treinadas = 0
        self._epoca_atual_treino = 0
        self._estado_otimizador_em_uso: Optional[dict[str, Any]] = None
        self._config_treino_em_uso: Optional[dict[str, Any]] = None

        self._resetar_historicos()
        self._inicializar_parametros(self.inicializacao)

    @classmethod
    def from_config(cls, config: ModelConfig) -> "RedeNeural":
        """Cria uma rede a partir de `ModelConfig`."""
        return cls(
            arquitetura=config.arquitetura,
            ativacao=config.ativacao,
            inicializacao=config.inicializacao,
            seed=config.seed,
            funcao_custo=config.funcao_custo,
            ativacao_saida=config.ativacao_saida,
        )

    def _resolver_funcao_custo_inicial(
        self,
        funcao_custo: str,
        ativacao_saida: Optional[str],
    ) -> str:
        """Ajusta defaults para manter a API simples em classificacao multiclasse."""
        funcao_custo_normalizada = funcao_custo.lower()
        if (
            ativacao_saida is not None
            and ativacao_saida.lower() == "linear"
            and funcao_custo_normalizada == "binary_crossentropy"
        ):
            return "mse"
        if (
            self.output_dim > 1
            and funcao_custo_normalizada == "binary_crossentropy"
            and (ativacao_saida is None or ativacao_saida.lower() == "softmax")
        ):
            return "categorical_crossentropy"
        return funcao_custo_normalizada

    def _resetar_historicos(self) -> None:
        """Limpa o historico armazenado a cada novo treinamento."""
        self.historico_erro: list[float] = []
        self.historico_mse: list[float] = []
        self.historico_acuracia: list[float] = []
        self.historico_mae: list[float] = []
        self.historico_rmse: list[float] = []
        self.historico_r2: list[float] = []
        self.historico_validacao_erro: list[float] = []
        self.historico_validacao_mse: list[float] = []
        self.historico_validacao_acuracia: list[float] = []
        self.historico_validacao_mae: list[float] = []
        self.historico_validacao_rmse: list[float] = []
        self.historico_validacao_r2: list[float] = []

    def _definir_historicos(self, historicos: dict[str, list[float]]) -> None:
        """Restaura historicos previamente salvos em um checkpoint."""
        self.historico_erro = list(historicos.get("historico_erro", []))
        self.historico_mse = list(historicos.get("historico_mse", []))
        self.historico_acuracia = list(historicos.get("historico_acuracia", []))
        self.historico_mae = list(historicos.get("historico_mae", []))
        self.historico_rmse = list(historicos.get("historico_rmse", []))
        self.historico_r2 = list(historicos.get("historico_r2", []))
        self.historico_validacao_erro = list(historicos.get("historico_validacao_erro", []))
        self.historico_validacao_mse = list(historicos.get("historico_validacao_mse", []))
        self.historico_validacao_acuracia = list(historicos.get("historico_validacao_acuracia", []))
        self.historico_validacao_mae = list(historicos.get("historico_validacao_mae", []))
        self.historico_validacao_rmse = list(historicos.get("historico_validacao_rmse", []))
        self.historico_validacao_r2 = list(historicos.get("historico_validacao_r2", []))

    def _obter_historicos(self) -> dict[str, list[float]]:
        """Empacota historicos em um dicionario serializavel."""
        return {
            "historico_erro": list(self.historico_erro),
            "historico_mse": list(self.historico_mse),
            "historico_acuracia": list(self.historico_acuracia),
            "historico_mae": list(self.historico_mae),
            "historico_rmse": list(self.historico_rmse),
            "historico_r2": list(self.historico_r2),
            "historico_validacao_erro": list(self.historico_validacao_erro),
            "historico_validacao_mse": list(self.historico_validacao_mse),
            "historico_validacao_acuracia": list(self.historico_validacao_acuracia),
            "historico_validacao_mae": list(self.historico_validacao_mae),
            "historico_validacao_rmse": list(self.historico_validacao_rmse),
            "historico_validacao_r2": list(self.historico_validacao_r2),
        }

    def _copiar_estado_otimizador(
        self, estado_otimizador: Optional[dict[str, Any]]
    ) -> Optional[dict[str, Any]]:
        """Faz copia profunda do estado do otimizador para checkpoints seguros."""
        if estado_otimizador is None:
            return None
        return deepcopy(estado_otimizador)

    def _eh_regressao(self) -> bool:
        """Indica se a tarefa atual usa saida linear para regressao."""
        return self.ativacao_saida == "linear"

    def _validar_arquitetura(self, arquitetura: List[int]) -> None:
        """Garante que a arquitetura tenha pelo menos entrada e saida validas."""
        if len(arquitetura) < 2:
            raise ValueError("A arquitetura precisa ter pelo menos camada de entrada e saida.")
        if any(int(neuronios) <= 0 for neuronios in arquitetura):
            raise ValueError("Todos os tamanhos de camada devem ser inteiros positivos.")

    def _validar_ativacao(self, ativacao: str) -> str:
        """Normaliza e valida a ativacao usada nas camadas ocultas."""
        ativacao_normalizada = ativacao.lower()
        if ativacao_normalizada not in self.funcoes.listar_funcoes():
            raise ValueError(
                f"Funcao de ativacao '{ativacao}' nao reconhecida. "
                f"Opcoes: {self.funcoes.listar_funcoes()}"
            )
        return ativacao_normalizada

    def _validar_ativacao_saida(self, ativacao_saida: Optional[str]) -> str:
        """Escolhe a ativacao da camada de saida."""
        if ativacao_saida is None:
            return "sigmoid" if self.output_dim == 1 else "softmax"

        ativacao_normalizada = ativacao_saida.lower()
        if ativacao_normalizada not in self._ATIVACOES_SAIDA_VALIDAS:
            raise ValueError(
                f"Ativacao de saida '{ativacao_saida}' nao reconhecida. "
                f"Opcoes: {sorted(self._ATIVACOES_SAIDA_VALIDAS)}"
            )
        return ativacao_normalizada

    def _validar_inicializacao(self, inicializacao: str) -> str:
        """Normaliza e valida a estrategia de inicializacao escolhida."""
        inicializacao_normalizada = inicializacao.lower()
        if inicializacao_normalizada not in self._INICIALIZACOES_VALIDAS:
            raise ValueError(
                f"Inicializacao '{inicializacao}' nao reconhecida. "
                f"Opcoes: {sorted(self._INICIALIZACOES_VALIDAS)}"
            )
        return inicializacao_normalizada

    def _validar_funcao_custo(self, funcao_custo: str) -> str:
        """Normaliza e valida a funcao de custo usada no treinamento."""
        funcao_custo_normalizada = funcao_custo.lower()
        if funcao_custo_normalizada not in self._FUNCOES_CUSTO_VALIDAS:
            raise ValueError(
                f"Funcao de custo '{funcao_custo}' nao reconhecida. "
                f"Opcoes: {sorted(self._FUNCOES_CUSTO_VALIDAS)}"
            )
        return funcao_custo_normalizada

    def _validar_combinacao_saida_e_perda(self) -> None:
        """Garante que a camada de saida combine com a perda escolhida."""
        if self.output_dim == 1 and self.funcao_custo == "categorical_crossentropy":
            raise ValueError("categorical_crossentropy exige mais de uma saida.")
        if self.ativacao_saida == "softmax" and self.funcao_custo != "categorical_crossentropy":
            raise ValueError("Softmax deve ser usada com categorical_crossentropy.")
        if self.ativacao_saida == "linear" and self.funcao_custo != "mse":
            raise ValueError("Saida linear deve ser usada com mse em regressao.")

    def _validar_otimizador(self, otimizador: str) -> str:
        """Normaliza e valida o algoritmo de atualizacao dos parametros."""
        otimizador_normalizado = otimizador.lower()
        if otimizador_normalizado not in self._OTIMIZADORES_VALIDOS:
            raise ValueError(
                f"Otimizador '{otimizador}' nao reconhecido. "
                f"Opcoes: {sorted(self._OTIMIZADORES_VALIDOS)}"
            )
        return otimizador_normalizado

    def _normalizar_batch_size(self, batch_size: Optional[int], n_amostras: int) -> int:
        """Converte `batch_size=None` em batch completo e valida o valor informado."""
        if batch_size is None:
            return n_amostras
        if not isinstance(batch_size, (int, np.integer)):
            raise ValueError("batch_size precisa ser um inteiro positivo.")
        if int(batch_size) <= 0:
            raise ValueError("batch_size precisa ser maior que zero.")
        return min(int(batch_size), n_amostras)

    def _validar_hiperparametros_adam(
        self,
        beta1: float,
        beta2: float,
        epsilon: float,
    ) -> None:
        """Confere se os hiperparametros do Adam estao em faixas validas."""
        if not 0 < beta1 < 1:
            raise ValueError("beta1 precisa estar entre 0 e 1.")
        if not 0 < beta2 < 1:
            raise ValueError("beta2 precisa estar entre 0 e 1.")
        if epsilon <= 0:
            raise ValueError("epsilon precisa ser maior que zero.")

    def _validar_regularizacao(
        self,
        l2_lambda: float,
        dropout: float,
        gradient_clip: Optional[float],
    ) -> None:
        """Valida hiperparametros de regularizacao."""
        if l2_lambda < 0:
            raise ValueError("l2_lambda nao pode ser negativo.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout precisa estar entre 0 e 1.")
        if gradient_clip is not None and gradient_clip <= 0:
            raise ValueError("gradient_clip precisa ser maior que zero.")

    def _validar_limiar(self, limiar: float) -> None:
        """Confere se o limiar esta no intervalo usado por probabilidades."""
        if self._eh_regressao():
            raise ValueError("Limiar nao se aplica a modelos de regressao.")
        if not 0 <= limiar <= 1:
            raise ValueError("limiar precisa estar entre 0 e 1.")

    def _validar_entrada(self, X: np.ndarray) -> np.ndarray:
        """Converte a entrada para array 2D e checa compatibilidade com a rede."""
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

    def _one_hot_encode(self, y_indices: np.ndarray) -> np.ndarray:
        """Converte rotulos inteiros para one-hot."""
        y_indices_int = y_indices.astype(int).reshape(-1)
        if np.any(y_indices_int < 0) or np.any(y_indices_int >= self.output_dim):
            raise ValueError(
                f"Rotulos devem estar entre 0 e {self.output_dim - 1} para esta arquitetura."
            )
        one_hot = np.zeros((y_indices_int.shape[0], self.output_dim), dtype=float)
        one_hot[np.arange(y_indices_int.shape[0]), y_indices_int] = 1.0
        return one_hot

    def _validar_rotulos(self, y: np.ndarray, n_amostras: int) -> np.ndarray:
        """Garante que os rotulos tenham shape compativel com a camada de saida."""
        y_array = np.asarray(y, dtype=float)

        if self._eh_regressao():
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            if y_array.ndim != 2:
                raise ValueError("Rotulos de regressao devem ter formato 2D.")
        elif self.output_dim == 1:
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            if y_array.ndim != 2 or y_array.shape[1] != 1:
                raise ValueError("Rotulos binarios devem ter formato (amostras, 1).")
        else:
            if y_array.ndim == 1:
                y_array = self._one_hot_encode(y_array)
            elif y_array.ndim == 2 and y_array.shape[1] == 1:
                y_array = self._one_hot_encode(y_array.reshape(-1))
            elif y_array.ndim == 2 and y_array.shape[1] == self.output_dim:
                pass
            else:
                raise ValueError(
                    "Para multiclasse, use indices de classe ou matriz one-hot "
                    f"com {self.output_dim} colunas."
                )

        if y_array.ndim != 2:
            raise ValueError("Os rotulos devem ter formato 2D: (amostras, saidas).")
        if y_array.shape[0] != n_amostras:
            raise ValueError("X e y precisam ter a mesma quantidade de amostras.")
        if y_array.shape[1] != self.output_dim:
            raise ValueError(
                f"Esperadas {self.output_dim} saidas, mas recebido {y_array.shape[1]}."
            )
        if not np.all(np.isfinite(y_array)):
            raise ValueError("Os rotulos precisam conter apenas valores finitos.")
        if self.output_dim > 1 and self.funcao_custo == "categorical_crossentropy":
            soma_linhas = np.sum(y_array, axis=1)
            if not np.allclose(soma_linhas, 1.0, atol=1e-6):
                raise ValueError("Rotulos para categorical_crossentropy devem somar 1 por linha.")
        return y_array.astype(float)

    def _validar_dados_validacao(
        self,
        validacao_X: Optional[np.ndarray],
        validacao_y: Optional[np.ndarray],
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Valida o par de validacao, que precisa ser passado por completo."""
        if (validacao_X is None) != (validacao_y is None):
            raise ValueError("Forneca validacao_X e validacao_y juntos.")
        if validacao_X is None or validacao_y is None:
            return None, None

        validacao_X_array = self._validar_entrada(validacao_X)
        validacao_y_array = self._validar_rotulos(validacao_y, validacao_X_array.shape[0])
        return validacao_X_array, validacao_y_array

    def _inicializar_parametros(self, metodo: str) -> None:
        """Inicializa pesos e biases camada por camada."""
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

            self.pesos.append(peso)
            self.biases.append(np.zeros((1, saida_size)))

    def _aplicar_ativacao_saida(self, z: np.ndarray) -> np.ndarray:
        """Escolhe a ativacao final adequada para o problema."""
        if self.ativacao_saida == "softmax":
            return self.funcoes.softmax(z)
        if self.ativacao_saida == "linear":
            return self.funcoes.linear(z)
        return self.funcoes.sigmoid(z)

    def _aplicar_ativacao(self, indice_camada: int, z: np.ndarray) -> np.ndarray:
        """Escolhe a ativacao adequada para a camada atual."""
        if indice_camada == self.num_camadas - 2:
            return self._aplicar_ativacao_saida(z)
        return self.funcoes.aplicar(z, self.ativacao)

    def _forward_pass(
        self,
        X: np.ndarray,
        treino: bool = False,
        dropout: float = 0.0,
    ) -> tuple[List[np.ndarray], List[np.ndarray], list[Optional[np.ndarray]]]:
        """Executa o fluxo de ida com suporte opcional a dropout."""
        ativacoes = [X]
        z_values = []
        dropout_masks: list[Optional[np.ndarray]] = []
        keep_prob = 1.0 - dropout

        for indice in range(self.num_camadas - 1):
            z = np.dot(ativacoes[indice], self.pesos[indice]) + self.biases[indice]
            z_values.append(z)
            ativacao = self._aplicar_ativacao(indice, z)

            mascara = None
            eh_camada_oculta = indice < self.num_camadas - 2
            if treino and eh_camada_oculta and dropout > 0:
                mascara = (self._rng.random(ativacao.shape) < keep_prob).astype(float) / keep_prob
                ativacao = ativacao * mascara

            ativacoes.append(ativacao)
            dropout_masks.append(mascara)

        return ativacoes, z_values, dropout_masks

    def _forward_propagation(self, X: np.ndarray) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Executa o fluxo de ida pela rede."""
        ativacoes, z_values, _ = self._forward_pass(X, treino=False, dropout=0.0)
        return ativacoes, z_values

    def _calcular_delta_saida(
        self,
        y: np.ndarray,
        ativacao_saida: np.ndarray,
        z_saida: np.ndarray,
    ) -> np.ndarray:
        """Calcula o gradiente na camada de saida."""
        if self.funcao_custo in {"binary_crossentropy", "categorical_crossentropy"}:
            return ativacao_saida - y
        if self.ativacao_saida == "linear":
            return ativacao_saida - y
        if self.ativacao_saida == "sigmoid":
            return (ativacao_saida - y) * self.funcoes.sigmoid_derivada(z_saida)
        raise ValueError("Softmax com MSE nao e suportado nesta implementacao didatica.")

    def _backward_propagation(
        self,
        y: np.ndarray,
        ativacoes: List[np.ndarray],
        z_values: List[np.ndarray],
        dropout_masks: Optional[list[Optional[np.ndarray]]] = None,
        l2_lambda: float = 0.0,
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Calcula os gradientes por backpropagation."""
        m = y.shape[0]
        gradientes_pesos: List[np.ndarray] = []
        gradientes_biases: List[np.ndarray] = []
        masks = dropout_masks or [None] * len(z_values)

        delta = self._calcular_delta_saida(y, ativacoes[-1], z_values[-1])

        for indice in reversed(range(self.num_camadas - 1)):
            dW = np.dot(ativacoes[indice].T, delta) / m
            if l2_lambda > 0:
                dW += (l2_lambda / m) * self.pesos[indice]
            db = np.sum(delta, axis=0, keepdims=True) / m

            gradientes_pesos.insert(0, dW)
            gradientes_biases.insert(0, db)

            if indice > 0:
                delta = np.dot(delta, self.pesos[indice].T)
                mascara = masks[indice - 1]
                if mascara is not None:
                    delta = delta * mascara
                delta_z = self.funcoes.derivada(z_values[indice - 1], self.ativacao)
                delta = delta * delta_z

        return gradientes_pesos, gradientes_biases

    def _gerar_batches(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        embaralhar: bool,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Divide os dados em lotes menores para cada passo do otimizador."""
        n_amostras = X.shape[0]
        indices = np.arange(n_amostras)
        if embaralhar and n_amostras > 1:
            indices = self._rng.permutation(n_amostras)

        for inicio in range(0, n_amostras, batch_size):
            fim = inicio + batch_size
            indices_batch = indices[inicio:fim]
            yield X[indices_batch], y[indices_batch]

    def _inicializar_estado_otimizador(
        self,
        otimizador: str,
        beta1: float,
        beta2: float,
        epsilon: float,
    ) -> dict[str, Any]:
        """Prepara o estado interno usado por otimizadores adaptativos."""
        if otimizador == "sgd":
            return {"otimizador": otimizador, "passo": 0}

        return {
            "otimizador": otimizador,
            "passo": 0,
            "beta1": float(beta1),
            "beta2": float(beta2),
            "epsilon": float(epsilon),
            "m_pesos": [np.zeros_like(peso) for peso in self.pesos],
            "v_pesos": [np.zeros_like(peso) for peso in self.pesos],
            "m_biases": [np.zeros_like(bias) for bias in self.biases],
            "v_biases": [np.zeros_like(bias) for bias in self.biases],
        }

    def _atualizar_parametros_sgd(
        self,
        gradientes_pesos: List[np.ndarray],
        gradientes_biases: List[np.ndarray],
        taxa_aprendizado: float,
    ) -> None:
        """Aplica gradiente descendente tradicional em todas as camadas."""
        for indice in range(len(self.pesos)):
            self.pesos[indice] -= taxa_aprendizado * gradientes_pesos[indice]
            self.biases[indice] -= taxa_aprendizado * gradientes_biases[indice]

    def _atualizar_parametros_adam(
        self,
        gradientes_pesos: List[np.ndarray],
        gradientes_biases: List[np.ndarray],
        taxa_aprendizado: float,
        estado_otimizador: dict[str, Any],
    ) -> None:
        """Atualiza os parametros usando medias moveis dos gradientes."""
        estado_otimizador["passo"] = int(estado_otimizador["passo"]) + 1
        passo = int(estado_otimizador["passo"])
        beta1 = float(estado_otimizador["beta1"])
        beta2 = float(estado_otimizador["beta2"])
        epsilon = float(estado_otimizador["epsilon"])

        momentos_pesos = estado_otimizador["m_pesos"]
        velocidades_pesos = estado_otimizador["v_pesos"]
        momentos_biases = estado_otimizador["m_biases"]
        velocidades_biases = estado_otimizador["v_biases"]

        for indice in range(len(self.pesos)):
            momentos_pesos[indice] = (
                beta1 * momentos_pesos[indice] + (1 - beta1) * gradientes_pesos[indice]
            )
            velocidades_pesos[indice] = beta2 * velocidades_pesos[indice] + (1 - beta2) * (
                gradientes_pesos[indice] ** 2
            )
            momentos_biases[indice] = (
                beta1 * momentos_biases[indice] + (1 - beta1) * gradientes_biases[indice]
            )
            velocidades_biases[indice] = beta2 * velocidades_biases[indice] + (1 - beta2) * (
                gradientes_biases[indice] ** 2
            )

            m_peso_corrigido = momentos_pesos[indice] / (1 - beta1**passo)
            v_peso_corrigido = velocidades_pesos[indice] / (1 - beta2**passo)
            m_bias_corrigido = momentos_biases[indice] / (1 - beta1**passo)
            v_bias_corrigido = velocidades_biases[indice] / (1 - beta2**passo)

            self.pesos[indice] -= (
                taxa_aprendizado * m_peso_corrigido / (np.sqrt(v_peso_corrigido) + epsilon)
            )
            self.biases[indice] -= (
                taxa_aprendizado * m_bias_corrigido / (np.sqrt(v_bias_corrigido) + epsilon)
            )

    def _atualizar_parametros(
        self,
        gradientes_pesos: List[np.ndarray],
        gradientes_biases: List[np.ndarray],
        taxa_aprendizado: float,
        otimizador: str,
        estado_otimizador: dict[str, Any],
    ) -> None:
        """Encaminha a atualizacao para o otimizador escolhido."""
        if otimizador == "adam":
            self._atualizar_parametros_adam(
                gradientes_pesos,
                gradientes_biases,
                taxa_aprendizado,
                estado_otimizador,
            )
            return

        self._atualizar_parametros_sgd(gradientes_pesos, gradientes_biases, taxa_aprendizado)

    def _clip_gradientes_por_norma(
        self,
        gradientes_pesos: List[np.ndarray],
        gradientes_biases: List[np.ndarray],
        limite: Optional[float],
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Aplica gradient clipping por norma em cada tensor."""
        if limite is None:
            return gradientes_pesos, gradientes_biases

        gradientes_pesos_clipados = []
        gradientes_biases_clipados = []

        for gradiente in gradientes_pesos:
            norma = np.linalg.norm(gradiente)
            if norma > limite:
                gradiente = gradiente * (limite / (norma + 1e-12))
            gradientes_pesos_clipados.append(gradiente)

        for gradiente in gradientes_biases:
            norma = np.linalg.norm(gradiente)
            if norma > limite:
                gradiente = gradiente * (limite / (norma + 1e-12))
            gradientes_biases_clipados.append(gradiente)

        return gradientes_pesos_clipados, gradientes_biases_clipados

    def _calcular_regularizacao_l2(self, n_amostras: int, l2_lambda: float) -> float:
        """Calcula o termo extra de L2 usado na funcao objetivo."""
        if l2_lambda == 0:
            return 0.0
        soma_quadrados = sum(float(np.sum(peso**2)) for peso in self.pesos)
        return float((l2_lambda / (2 * n_amostras)) * soma_quadrados)

    def _calcular_mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula o erro quadratico medio."""
        return float(np.mean((y_true - y_pred) ** 2))

    def _calcular_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula o erro absoluto medio."""
        return float(np.mean(np.abs(y_true - y_pred)))

    def _calcular_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula a raiz do erro quadratico medio."""
        return float(np.sqrt(self._calcular_mse(y_true, y_pred)))

    def _calcular_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula o coeficiente de determinacao para regressao."""
        soma_quadrados_total = float(np.sum((y_true - np.mean(y_true, axis=0)) ** 2))
        if soma_quadrados_total == 0:
            return 0.0
        soma_quadrados_residuos = float(np.sum((y_true - y_pred) ** 2))
        return float(1 - soma_quadrados_residuos / soma_quadrados_total)

    def _calcular_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        l2_lambda: float = 0.0,
    ) -> float:
        """Calcula a perda configurada para o modelo."""
        if self.funcao_custo == "mse":
            base_loss = self._calcular_mse(y_true, y_pred)
        elif self.funcao_custo == "categorical_crossentropy":
            y_pred_seguro = np.clip(y_pred, 1e-10, 1.0)
            base_loss = float(-np.mean(np.sum(y_true * np.log(y_pred_seguro), axis=1)))
        else:
            y_pred_seguro = np.clip(y_pred, 1e-10, 1 - 1e-10)
            base_loss = float(
                -np.mean(y_true * np.log(y_pred_seguro) + (1 - y_true) * np.log(1 - y_pred_seguro))
            )

        return float(base_loss + self._calcular_regularizacao_l2(y_true.shape[0], l2_lambda))

    def _calcular_erro(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        l2_lambda: float = 0.0,
    ) -> float:
        """Mantem compatibilidade com o restante do projeto usando a perda atual."""
        return self._calcular_loss(y_true, y_pred, l2_lambda=l2_lambda)

    def _calcular_acuracia(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        limiar: float = 0.5,
    ) -> float:
        """Converte probabilidades em classes e calcula a acuracia percentual."""
        if self._eh_regressao():
            return float("nan")
        self._validar_limiar(limiar)
        if self.output_dim == 1:
            predicoes_binarias = (y_pred >= limiar).astype(int)
            y_true_binarias = (y_true >= limiar).astype(int)
            return float(np.mean(predicoes_binarias == y_true_binarias) * 100)

        classes_preditas = np.argmax(y_pred, axis=1)
        classes_reais = np.argmax(y_true, axis=1)
        return float(np.mean(classes_preditas == classes_reais) * 100)

    def _calcular_metricas_epoca(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        l2_lambda: float = 0.0,
    ) -> dict[str, float]:
        """Agrupa as metricas usadas ao final de cada epoca."""
        metricas = {
            "loss": self._calcular_loss(y_true, y_pred, l2_lambda=l2_lambda),
            "mse": self._calcular_mse(y_true, y_pred),
        }
        if self._eh_regressao():
            metricas.update(
                {
                    "mae": self._calcular_mae(y_true, y_pred),
                    "rmse": self._calcular_rmse(y_true, y_pred),
                    "r2": self._calcular_r2(y_true, y_pred),
                }
            )
        else:
            metricas["acuracia"] = self._calcular_acuracia(y_true, y_pred)
        return metricas

    def _avaliar_dataset_validado(
        self,
        X: np.ndarray,
        y: np.ndarray,
        l2_lambda: float = 0.0,
    ) -> tuple[dict[str, float], np.ndarray]:
        """Executa forward em dados ja validados e devolve metricas + predicoes."""
        ativacoes, _, _ = self._forward_pass(X, treino=False, dropout=0.0)
        predicoes = ativacoes[-1]
        return self._calcular_metricas_epoca(y, predicoes, l2_lambda=l2_lambda), predicoes

    def _copiar_parametros(self) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Cria um snapshot dos parametros atuais para restauracao futura."""
        return [peso.copy() for peso in self.pesos], [bias.copy() for bias in self.biases]

    def _restaurar_parametros(
        self,
        pesos: List[np.ndarray],
        biases: List[np.ndarray],
    ) -> None:
        """Restaura pesos e biases a partir de um snapshot."""
        self.pesos = [peso.copy() for peso in pesos]
        self.biases = [bias.copy() for bias in biases]

    def _tipo_problema(self) -> str:
        """Resume o tipo de tarefa suportada pelo modelo atual."""
        if self._eh_regressao():
            return "regressao" if self.output_dim == 1 else "regressao_multi_saida"
        if self.output_dim == 1:
            return "classificacao_binaria"
        if self.ativacao_saida == "softmax":
            return "classificacao_multiclasse"
        return "classificacao_multi_saida"

    def _obter_model_config_dict(self) -> dict[str, Any]:
        """Gera uma versao serializavel da configuracao atual do modelo."""
        return ModelConfig(
            arquitetura=list(self.arquitetura),
            ativacao=self.ativacao,
            inicializacao=self.inicializacao,
            seed=self.seed,
            funcao_custo=self.funcao_custo,
            ativacao_saida=self.ativacao_saida,
        ).to_dict()

    def _atualizar_estado_treino_em_uso(
        self,
        epoch_atual: int,
        total_atualizacoes: int,
        estado_otimizador: dict[str, Any],
        config_treino: dict[str, Any],
    ) -> None:
        """Mantem o estado mais recente acessivel para callbacks e checkpoint."""
        self._epoca_atual_treino = int(epoch_atual)
        self._total_atualizacoes_treinadas = int(total_atualizacoes)
        self._estado_otimizador_em_uso = estado_otimizador
        self._config_treino_em_uso = deepcopy(config_treino)

    def obter_estado_treinamento(self) -> dict[str, Any]:
        """Retorna o estado completo necessario para retomar o treinamento."""
        return {
            "model_config": self._obter_model_config_dict(),
            "training_config": deepcopy(self._ultimo_config_treino or self._config_treino_em_uso),
            "epoch": int(self._ultima_epoca_treinada or self._epoca_atual_treino),
            "total_atualizacoes": int(self._total_atualizacoes_treinadas),
            "optimizer_state": self._copiar_estado_otimizador(
                self._ultimo_estado_otimizador or self._estado_otimizador_em_uso
            ),
            "historicos": self._obter_historicos(),
            "rng_state": deepcopy(self._rng.bit_generator.state),
            "motivo_parada": self._motivo_parada,
            "melhor_monitor_callback": self._melhor_monitor_callback,
            "melhor_epoch_callback": self._melhor_epoch_callback,
            "ultimo_l2_lambda": self._ultimo_l2_lambda,
        }

    def _preparar_callbacks(
        self,
        callbacks: Optional[list[Callback]],
        paciencia: Optional[int],
        min_delta: float,
        restaurar_melhores_pesos: bool,
        possui_validacao: bool,
    ) -> list[Callback]:
        """Resolve a lista final de callbacks ativos."""
        callbacks_ativos = list(callbacks or [])
        if paciencia is not None and not any(
            isinstance(cb, EarlyStopping) for cb in callbacks_ativos
        ):
            callbacks_ativos.append(
                EarlyStopping(
                    monitor="val_loss" if possui_validacao else "loss",
                    patience=paciencia,
                    min_delta=min_delta,
                    restore_best_weights=restaurar_melhores_pesos,
                )
            )
        return callbacks_ativos

    def contar_parametros(self) -> int:
        """Conta quantos parametros treinaveis a rede possui."""
        return sum(peso.size + bias.size for peso, bias in zip(self.pesos, self.biases))

    def resumir_modelo(self) -> dict:
        """Retorna um resumo curto e legivel do modelo atual."""
        return {
            "arquitetura": list(self.arquitetura),
            "camadas_treinaveis": self.num_camadas - 1,
            "ativacao_oculta": self.ativacao,
            "ativacao_saida": self.ativacao_saida,
            "tipo_problema": self._tipo_problema(),
            "funcao_custo": self.funcao_custo,
            "inicializacao": self.inicializacao,
            "seed": self.seed,
            "parametros_treinaveis": self.contar_parametros(),
            "checkpoint_disponivel": self._ultimo_estado_otimizador is not None
            or self._estado_otimizador_em_uso is not None,
        }

    def treinar_com_config(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: TrainingConfig,
        validacao_X: Optional[np.ndarray] = None,
        validacao_y: Optional[np.ndarray] = None,
    ) -> dict:
        """Treina a rede usando um objeto `TrainingConfig`."""
        return self.treinar(
            X,
            y,
            epochs=config.epochs,
            taxa_aprendizado=config.taxa_aprendizado,
            verbose=config.verbose,
            validacao_X=validacao_X,
            validacao_y=validacao_y,
            paciencia=config.paciencia,
            min_delta=config.min_delta,
            restaurar_melhores_pesos=config.restaurar_melhores_pesos,
            batch_size=config.batch_size,
            otimizador=config.otimizador,
            embaralhar=config.embaralhar,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            l2_lambda=config.l2_lambda,
            dropout=config.dropout,
            gradient_clip=config.gradient_clip,
            callbacks=config.callbacks,
        )

    def treinar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        taxa_aprendizado: float = 0.1,
        verbose: bool = True,
        validacao_X: Optional[np.ndarray] = None,
        validacao_y: Optional[np.ndarray] = None,
        paciencia: Optional[int] = None,
        min_delta: float = 0.0,
        restaurar_melhores_pesos: bool = True,
        batch_size: Optional[int] = None,
        otimizador: str = "sgd",
        embaralhar: bool = True,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        l2_lambda: float = 0.0,
        dropout: float = 0.0,
        gradient_clip: Optional[float] = None,
        callbacks: Optional[list[Callback]] = None,
        _epoca_inicial: int = 0,
        _estado_otimizador_inicial: Optional[dict[str, Any]] = None,
        _historicos_iniciais: Optional[dict[str, list[float]]] = None,
    ) -> dict:
        """Treina a rede usando batch completo ou mini-batches."""
        if epochs <= 0:
            raise ValueError("epochs precisa ser maior que zero.")
        if taxa_aprendizado <= 0:
            raise ValueError("taxa_aprendizado precisa ser maior que zero.")
        if paciencia is not None and paciencia <= 0:
            raise ValueError("paciencia precisa ser maior que zero quando informada.")
        if min_delta < 0:
            raise ValueError("min_delta nao pode ser negativo.")

        X_array = self._validar_entrada(X)
        y_array = self._validar_rotulos(y, X_array.shape[0])
        batch_size_efetivo = self._normalizar_batch_size(batch_size, X_array.shape[0])
        otimizador_normalizado = self._validar_otimizador(otimizador)
        self._validar_hiperparametros_adam(beta1, beta2, epsilon)
        self._validar_regularizacao(l2_lambda, dropout, gradient_clip)
        validacao_X_array, validacao_y_array = self._validar_dados_validacao(
            validacao_X,
            validacao_y,
        )

        callbacks_ativos = self._preparar_callbacks(
            callbacks,
            paciencia,
            min_delta,
            restaurar_melhores_pesos,
            possui_validacao=validacao_X_array is not None,
        )

        if _historicos_iniciais is not None:
            self._definir_historicos(_historicos_iniciais)
        else:
            self._resetar_historicos()
        self.stop_training = False
        self._motivo_parada = "epochs_concluidas"
        self._melhor_monitor_callback = None
        self._melhor_epoch_callback = 0
        self._ultimo_l2_lambda = float(l2_lambda)

        estado_otimizador = (
            self._copiar_estado_otimizador(_estado_otimizador_inicial)
            if _estado_otimizador_inicial is not None
            else self._inicializar_estado_otimizador(
                otimizador_normalizado,
                beta1,
                beta2,
                epsilon,
            )
        )
        if estado_otimizador is None:
            raise ValueError("Falha ao inicializar o estado do otimizador.")
        if estado_otimizador.get("otimizador") != otimizador_normalizado:
            raise ValueError("O checkpoint usa um otimizador diferente do solicitado.")

        intervalo_log = max(1, epochs // 10)
        total_atualizacoes = int(estado_otimizador.get("passo", 0))
        fonte_monitoramento = "validacao" if validacao_X_array is not None else "treino"
        config_treino_dict: dict[str, Any] = {
            "epochs": int(epochs),
            "taxa_aprendizado": float(taxa_aprendizado),
            "verbose": bool(verbose),
            "paciencia": paciencia,
            "min_delta": float(min_delta),
            "restaurar_melhores_pesos": bool(restaurar_melhores_pesos),
            "batch_size": batch_size_efetivo,
            "otimizador": otimizador_normalizado,
            "embaralhar": bool(embaralhar),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "epsilon": float(epsilon),
            "l2_lambda": float(l2_lambda),
            "dropout": float(dropout),
            "gradient_clip": None if gradient_clip is None else float(gradient_clip),
            "callbacks": [],
        }

        params_callbacks = {
            "epochs": epochs + _epoca_inicial,
            "taxa_aprendizado": float(taxa_aprendizado),
            "batch_size": batch_size_efetivo,
            "otimizador": otimizador_normalizado,
            "fonte_monitoramento": fonte_monitoramento,
        }
        for callback in callbacks_ativos:
            callback.set_model(self)
            callback.set_params(params_callbacks)
            callback.on_train_begin(
                {"monitor": "val_loss" if validacao_X_array is not None else "loss"}
            )

        epocas_executadas_nesta_execucao = 0
        for epoch in range(epochs):
            for X_batch, y_batch in self._gerar_batches(
                X_array,
                y_array,
                batch_size=batch_size_efetivo,
                embaralhar=embaralhar,
            ):
                ativacoes, z_values, dropout_masks = self._forward_pass(
                    X_batch,
                    treino=True,
                    dropout=dropout,
                )
                grad_pesos, grad_biases = self._backward_propagation(
                    y_batch,
                    ativacoes,
                    z_values,
                    dropout_masks=dropout_masks,
                    l2_lambda=l2_lambda,
                )
                grad_pesos, grad_biases = self._clip_gradientes_por_norma(
                    grad_pesos,
                    grad_biases,
                    limite=gradient_clip,
                )
                self._atualizar_parametros(
                    grad_pesos,
                    grad_biases,
                    taxa_aprendizado,
                    otimizador_normalizado,
                    estado_otimizador,
                )
                total_atualizacoes += 1

            metricas_treino, _ = self._avaliar_dataset_validado(
                X_array,
                y_array,
                l2_lambda=l2_lambda,
            )
            self.historico_erro.append(metricas_treino["loss"])
            self.historico_mse.append(metricas_treino["mse"])
            if self._eh_regressao():
                self.historico_mae.append(metricas_treino["mae"])
                self.historico_rmse.append(metricas_treino["rmse"])
                self.historico_r2.append(metricas_treino["r2"])
            else:
                self.historico_acuracia.append(metricas_treino["acuracia"])

            logs = {
                "loss": metricas_treino["loss"],
                "mse": metricas_treino["mse"],
                "taxa_aprendizado": float(taxa_aprendizado),
            }
            if self._eh_regressao():
                logs.update(
                    {
                        "mae": metricas_treino["mae"],
                        "rmse": metricas_treino["rmse"],
                        "r2": metricas_treino["r2"],
                    }
                )
            else:
                logs["acuracia"] = metricas_treino["acuracia"]

            metricas_validacao = None
            if validacao_X_array is not None and validacao_y_array is not None:
                metricas_validacao, _ = self._avaliar_dataset_validado(
                    validacao_X_array,
                    validacao_y_array,
                    l2_lambda=l2_lambda,
                )
                self.historico_validacao_erro.append(metricas_validacao["loss"])
                self.historico_validacao_mse.append(metricas_validacao["mse"])
                logs.update(
                    {
                        "val_loss": metricas_validacao["loss"],
                        "val_mse": metricas_validacao["mse"],
                    }
                )
                if self._eh_regressao():
                    self.historico_validacao_mae.append(metricas_validacao["mae"])
                    self.historico_validacao_rmse.append(metricas_validacao["rmse"])
                    self.historico_validacao_r2.append(metricas_validacao["r2"])
                    logs.update(
                        {
                            "val_mae": metricas_validacao["mae"],
                            "val_rmse": metricas_validacao["rmse"],
                            "val_r2": metricas_validacao["r2"],
                        }
                    )
                else:
                    self.historico_validacao_acuracia.append(metricas_validacao["acuracia"])
                    logs["val_acuracia"] = metricas_validacao["acuracia"]

            epoca_global = _epoca_inicial + epoch + 1
            epocas_executadas_nesta_execucao += 1
            self._atualizar_estado_treino_em_uso(
                epoca_global,
                total_atualizacoes,
                estado_otimizador,
                config_treino_dict,
            )

            if verbose and (epoch == 0 or (epoch + 1) % intervalo_log == 0 or epoch == epochs - 1):
                print(
                    f"Epoca {epoca_global:4d}/{_epoca_inicial + epochs}: "
                    f"Loss = {metricas_treino['loss']:.4f}, "
                    f"MSE = {metricas_treino['mse']:.4f}"
                )
                if self._eh_regressao():
                    print(
                        "              Regressao: "
                        f"MAE = {metricas_treino['mae']:.4f}, "
                        f"RMSE = {metricas_treino['rmse']:.4f}, "
                        f"R2 = {metricas_treino['r2']:.4f}"
                    )
                else:
                    print(f"              Acuracia = {metricas_treino['acuracia']:.2f}%")
                if metricas_validacao is not None:
                    mensagem_validacao = (
                        "              Validacao: "
                        f"Loss = {metricas_validacao['loss']:.4f}, "
                        f"MSE = {metricas_validacao['mse']:.4f}"
                    )
                    if self._eh_regressao():
                        mensagem_validacao += (
                            f", MAE = {metricas_validacao['mae']:.4f}, "
                            f"R2 = {metricas_validacao['r2']:.4f}"
                        )
                    else:
                        mensagem_validacao += f", Acuracia = {metricas_validacao['acuracia']:.2f}%"
                    print(mensagem_validacao)
                if dropout > 0 or l2_lambda > 0 or gradient_clip is not None:
                    print(
                        "              Regularizacao: "
                        f"dropout={dropout:.2f}, "
                        f"l2={l2_lambda:.4f}, "
                        f"clip={gradient_clip}"
                    )

            for callback in callbacks_ativos:
                callback.on_epoch_end(epoca_global - 1, logs)

            if self.stop_training:
                break

        for callback in callbacks_ativos:
            callback.on_train_end({"motivo_parada": self._motivo_parada})

        resumo_treino_final, predicoes_treino = self._avaliar_dataset_validado(
            X_array,
            y_array,
            l2_lambda=l2_lambda,
        )
        resumo = {
            "erro_final": resumo_treino_final["loss"],
            "loss_final": resumo_treino_final["loss"],
            "mse_final": resumo_treino_final["mse"],
            "melhor_erro": min(self.historico_erro),
            "melhor_mse": min(self.historico_mse),
            "epochs_planejadas": _epoca_inicial + epochs,
            "epochs_solicitadas_nesta_execucao": epochs,
            "epoch_inicial": _epoca_inicial,
            "epocas_executadas": len(self.historico_erro),
            "epocas_nesta_execucao": epocas_executadas_nesta_execucao,
            "taxa_aprendizado": float(taxa_aprendizado),
            "batch_size": batch_size_efetivo,
            "otimizador": otimizador_normalizado,
            "embaralhar": bool(embaralhar),
            "total_atualizacoes": total_atualizacoes,
            "funcao_custo": self.funcao_custo,
            "ativacao_saida": self.ativacao_saida,
            "tipo_problema": self._tipo_problema(),
            "parametros_treinaveis": self.contar_parametros(),
            "motivo_parada": self._motivo_parada,
            "fonte_monitoramento": fonte_monitoramento,
            "melhor_loss_monitorado": (
                float(self._melhor_monitor_callback)
                if self._melhor_monitor_callback is not None
                else (
                    min(self.historico_validacao_erro)
                    if self.historico_validacao_erro
                    else min(self.historico_erro)
                )
            ),
            "melhor_epoch_monitorado": int(
                self._melhor_epoch_callback or np.argmin(self.historico_erro) + 1
            ),
            "early_stopping_ativado": any(isinstance(cb, EarlyStopping) for cb in callbacks_ativos),
            "callbacks": [callback.__class__.__name__ for callback in callbacks_ativos],
            "l2_lambda": float(l2_lambda),
            "dropout": float(dropout),
            "gradient_clip": None if gradient_clip is None else float(gradient_clip),
        }
        if self._eh_regressao():
            resumo["mae_final"] = resumo_treino_final["mae"]
            resumo["rmse_final"] = resumo_treino_final["rmse"]
            resumo["r2_final"] = resumo_treino_final["r2"]
            resumo["melhor_r2"] = (
                max(self.historico_r2) if self.historico_r2 else resumo_treino_final["r2"]
            )
            resumo["acuracia_final"] = None
            resumo["melhor_acuracia"] = None
        else:
            resumo["acuracia_final"] = resumo_treino_final["acuracia"]
            resumo["melhor_acuracia"] = max(self.historico_acuracia)

        if otimizador_normalizado == "adam":
            resumo["beta1"] = float(beta1)
            resumo["beta2"] = float(beta2)
            resumo["epsilon"] = float(epsilon)

        if validacao_X_array is not None and validacao_y_array is not None:
            resumo_validacao_final, _ = self._avaliar_dataset_validado(
                validacao_X_array,
                validacao_y_array,
                l2_lambda=l2_lambda,
            )
            resumo["erro_validacao_final"] = resumo_validacao_final["loss"]
            resumo["loss_validacao_final"] = resumo_validacao_final["loss"]
            resumo["mse_validacao_final"] = resumo_validacao_final["mse"]
            if self._eh_regressao():
                resumo["mae_validacao_final"] = resumo_validacao_final["mae"]
                resumo["rmse_validacao_final"] = resumo_validacao_final["rmse"]
                resumo["r2_validacao_final"] = resumo_validacao_final["r2"]
                resumo["acuracia_validacao_final"] = None
            else:
                resumo["acuracia_validacao_final"] = resumo_validacao_final["acuracia"]

        if self.output_dim > 1 and not self._eh_regressao():
            resumo["classes_treino_preditas"] = np.argmax(predicoes_treino, axis=1).tolist()

        self._ultimo_estado_otimizador = self._copiar_estado_otimizador(estado_otimizador)
        self._ultimo_config_treino = deepcopy(config_treino_dict)
        self._ultima_epoca_treinada = _epoca_inicial + epocas_executadas_nesta_execucao
        self._total_atualizacoes_treinadas = total_atualizacoes

        if verbose:
            print("\n" + "=" * 50)
            print("TREINAMENTO CONCLUIDO")
            print("=" * 50)
            print(f"Loss final: {resumo['loss_final']:.4f}")
            print(f"MSE final: {resumo['mse_final']:.4f}")
            if self._eh_regressao():
                print(f"MAE final: {resumo['mae_final']:.4f}")
                print(f"R2 final: {resumo['r2_final']:.4f}")
            else:
                print(f"Acuracia final: {resumo['acuracia_final']:.2f}%")
                print(f"Melhor acuracia: {resumo['melhor_acuracia']:.2f}%")
            print(f"Epocas executadas: {resumo['epocas_executadas']}")
            print(f"Otimizador: {resumo['otimizador']}")
            print(f"Batch size: {resumo['batch_size']}")
            print(f"Tipo de problema: {resumo['tipo_problema']}")
            print(f"Atualizacoes: {resumo['total_atualizacoes']}")
            print(f"Parametros treinaveis: {resumo['parametros_treinaveis']}")

        return resumo

    def prever(self, X: np.ndarray) -> np.ndarray:
        """Executa apenas o forward e retorna probabilidades."""
        X_array = self._validar_entrada(X)
        ativacoes, _, _ = self._forward_pass(X_array, treino=False, dropout=0.0)
        return ativacoes[-1]

    def prever_classes(
        self,
        X: np.ndarray,
        limiar: float = 0.5,
        one_hot: bool = False,
    ) -> np.ndarray:
        """Converte probabilidades em classes."""
        if self._eh_regressao():
            raise ValueError("prever_classes nao esta disponivel para regressao.")
        self._validar_limiar(limiar)
        predicoes = self.prever(X)

        if self.output_dim == 1:
            return (predicoes >= limiar).astype(int)

        indices = np.argmax(predicoes, axis=1)
        if one_hot:
            return self._one_hot_encode(indices)
        return indices.reshape(-1, 1)

    def avaliar(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Roda previsao e devolve metricas basicas do problema atual."""
        X_array = self._validar_entrada(X)
        y_array = self._validar_rotulos(y, X_array.shape[0])
        metricas, predicoes = self._avaliar_dataset_validado(
            X_array,
            y_array,
            l2_lambda=self._ultimo_l2_lambda,
        )

        resultado = {
            "erro": metricas["loss"],
            "loss": metricas["loss"],
            "mse": metricas["mse"],
            "funcao_custo": self.funcao_custo,
            "ativacao_saida": self.ativacao_saida,
            "tipo_problema": self._tipo_problema(),
            "predicoes": predicoes,
        }
        if self._eh_regressao():
            resultado["mae"] = metricas["mae"]
            resultado["rmse"] = metricas["rmse"]
            resultado["r2"] = metricas["r2"]
            resultado["acuracia"] = None
        else:
            resultado["acuracia"] = metricas["acuracia"]
        if self.output_dim > 1 and not self._eh_regressao():
            resultado["classes_preditas"] = np.argmax(predicoes, axis=1).reshape(-1, 1)
        return resultado

    def obter_parametros(self) -> dict:
        """Retorna uma copia dos parametros atuais da rede."""
        return {
            "pesos": [peso.copy() for peso in self.pesos],
            "biases": [bias.copy() for bias in self.biases],
            "arquitetura": list(self.arquitetura),
            "ativacao": self.ativacao,
            "ativacao_saida": self.ativacao_saida,
            "inicializacao": self.inicializacao,
            "funcao_custo": self.funcao_custo,
            "seed": self.seed,
        }

    def _criar_array_objeto(self, valores: list[Any]) -> np.ndarray:
        """Empacota listas heterogeneas para escrita em `.npz`."""
        array_obj = np.empty(len(valores), dtype=object)
        array_obj[:] = valores
        return array_obj

    def _aplicar_dados_modelo_carregados(self, dados: Any) -> None:
        """Restaura o estado estrutural da rede a partir de um arquivo salvo."""
        arquitetura_carregada = [int(neuronios) for neuronios in dados["arquitetura"].tolist()]
        self._validar_arquitetura(arquitetura_carregada)

        self.pesos = [np.array(camada) for camada in dados["pesos"].tolist()]
        self.biases = [np.array(camada) for camada in dados["biases"].tolist()]
        self.arquitetura = arquitetura_carregada
        self.num_camadas = len(self.arquitetura)
        self.output_dim = self.arquitetura[-1]

        self.ativacao = self._validar_ativacao(str(dados["ativacao"].tolist()[0]))
        if "ativacao_saida" in dados.files:
            self.ativacao_saida = self._validar_ativacao_saida(
                str(dados["ativacao_saida"].tolist()[0])
            )
        else:
            self.ativacao_saida = self._validar_ativacao_saida(None)

        if "inicializacao" in dados.files:
            self.inicializacao = self._validar_inicializacao(
                str(dados["inicializacao"].tolist()[0])
            )
        if "funcao_custo" in dados.files:
            self.funcao_custo = self._validar_funcao_custo(str(dados["funcao_custo"].tolist()[0]))
        if "seed" in dados.files:
            seed_salva = dados["seed"].tolist()[0]
            self.seed = None if seed_salva is None else int(seed_salva)

        self._validar_combinacao_saida_e_perda()
        self._rng = np.random.default_rng(self.seed)

    def salvar_parametros(self, caminho: str) -> None:
        """Salva pesos, biases e metadados em um arquivo `.npz`."""
        caminho_arquivo = Path(caminho)
        if caminho_arquivo.parent != Path("."):
            caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

        parametros = {
            "pesos": self._criar_array_objeto(self.pesos),
            "biases": self._criar_array_objeto(self.biases),
            "arquitetura": np.array(self.arquitetura, dtype=int),
            "ativacao": np.array([self.ativacao], dtype=object),
            "ativacao_saida": np.array([self.ativacao_saida], dtype=object),
            "inicializacao": np.array([self.inicializacao], dtype=object),
            "funcao_custo": np.array([self.funcao_custo], dtype=object),
            "seed": np.array([self.seed], dtype=object),
        }
        np.savez(caminho_arquivo, **parametros)  # type: ignore[arg-type]
        print(f"Parametros salvos em: {caminho_arquivo}")

    def carregar_parametros(self, caminho: str) -> None:
        """Carrega os parametros salvos anteriormente com `salvar_parametros`."""
        dados = np.load(caminho, allow_pickle=True)
        self._aplicar_dados_modelo_carregados(dados)
        self._resetar_historicos()
        self._ultimo_estado_otimizador = None
        self._ultimo_config_treino = None
        self._ultima_epoca_treinada = 0
        self._total_atualizacoes_treinadas = 0
        self._estado_otimizador_em_uso = None
        self._config_treino_em_uso = None
        print(f"Parametros carregados de: {caminho}")

    def salvar_checkpoint(self, caminho: str) -> None:
        """Salva pesos, config, historicos e estado do otimizador para resume."""
        caminho_arquivo = Path(caminho)
        if caminho_arquivo.parent != Path("."):
            caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

        estado = self.obter_estado_treinamento()
        parametros = {
            "pesos": self._criar_array_objeto(self.pesos),
            "biases": self._criar_array_objeto(self.biases),
            "arquitetura": np.array(self.arquitetura, dtype=int),
            "ativacao": np.array([self.ativacao], dtype=object),
            "ativacao_saida": np.array([self.ativacao_saida], dtype=object),
            "inicializacao": np.array([self.inicializacao], dtype=object),
            "funcao_custo": np.array([self.funcao_custo], dtype=object),
            "seed": np.array([self.seed], dtype=object),
            "model_config": np.array([estado["model_config"]], dtype=object),
            "training_config": np.array([estado["training_config"]], dtype=object),
            "optimizer_state": np.array([estado["optimizer_state"]], dtype=object),
            "historicos": np.array([estado["historicos"]], dtype=object),
            "epoch": np.array([estado["epoch"]], dtype=int),
            "total_atualizacoes": np.array([estado["total_atualizacoes"]], dtype=int),
            "rng_state": np.array([estado["rng_state"]], dtype=object),
            "motivo_parada": np.array([estado["motivo_parada"]], dtype=object),
            "melhor_monitor_callback": np.array([estado["melhor_monitor_callback"]], dtype=object),
            "melhor_epoch_callback": np.array([estado["melhor_epoch_callback"]], dtype=int),
            "ultimo_l2_lambda": np.array([estado["ultimo_l2_lambda"]], dtype=float),
        }
        np.savez(caminho_arquivo, **parametros)  # type: ignore[arg-type]
        print(f"Checkpoint salvo em: {caminho_arquivo}")

    def carregar_checkpoint(self, caminho: str) -> dict[str, Any]:
        """Carrega um checkpoint completo salvo por `salvar_checkpoint`."""
        dados = np.load(caminho, allow_pickle=True)
        self._aplicar_dados_modelo_carregados(dados)

        historicos = (
            dados["historicos"].tolist()[0]
            if "historicos" in dados.files
            else self._obter_historicos()
        )
        self._definir_historicos(historicos)

        self._ultimo_config_treino = (
            deepcopy(dados["training_config"].tolist()[0])
            if "training_config" in dados.files
            else None
        )
        self._ultimo_estado_otimizador = (
            deepcopy(dados["optimizer_state"].tolist()[0])
            if "optimizer_state" in dados.files
            else None
        )
        self._ultima_epoca_treinada = int(dados["epoch"][0]) if "epoch" in dados.files else 0
        self._total_atualizacoes_treinadas = (
            int(dados["total_atualizacoes"][0]) if "total_atualizacoes" in dados.files else 0
        )
        self._motivo_parada = (
            str(dados["motivo_parada"].tolist()[0])
            if "motivo_parada" in dados.files
            else "checkpoint_carregado"
        )
        self._melhor_monitor_callback = (
            dados["melhor_monitor_callback"].tolist()[0]
            if "melhor_monitor_callback" in dados.files
            else None
        )
        self._melhor_epoch_callback = (
            int(dados["melhor_epoch_callback"][0]) if "melhor_epoch_callback" in dados.files else 0
        )
        self._ultimo_l2_lambda = (
            float(dados["ultimo_l2_lambda"][0]) if "ultimo_l2_lambda" in dados.files else 0.0
        )
        if "rng_state" in dados.files:
            self._rng.bit_generator.state = deepcopy(dados["rng_state"].tolist()[0])

        resumo = {
            "epoch": self._ultima_epoca_treinada,
            "total_atualizacoes": self._total_atualizacoes_treinadas,
            "tipo_problema": self._tipo_problema(),
            "training_config": deepcopy(self._ultimo_config_treino),
        }
        print(f"Checkpoint carregado de: {caminho}")
        return resumo

    def retomar_treinamento(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs_adicionais: int,
        validacao_X: Optional[np.ndarray] = None,
        validacao_y: Optional[np.ndarray] = None,
        callbacks: Optional[list[Callback]] = None,
        verbose: Optional[bool] = None,
        **overrides: Any,
    ) -> dict:
        """Retoma o treinamento usando o ultimo checkpoint carregado ou salvo."""
        if self._ultimo_config_treino is None or self._ultimo_estado_otimizador is None:
            raise ValueError(
                "Nao ha estado de treinamento salvo. "
                "Use carregar_checkpoint() ou treine antes de retomar."
            )

        config = deepcopy(self._ultimo_config_treino)
        config.update(overrides)
        config["epochs"] = int(epochs_adicionais)
        if verbose is not None:
            config["verbose"] = bool(verbose)
        if callbacks is not None:
            config["callbacks"] = callbacks

        config_obj = TrainingConfig.from_dict(config)
        return self.treinar(
            X,
            y,
            epochs=config_obj.epochs,
            taxa_aprendizado=config_obj.taxa_aprendizado,
            verbose=config_obj.verbose,
            validacao_X=validacao_X,
            validacao_y=validacao_y,
            paciencia=config_obj.paciencia,
            min_delta=config_obj.min_delta,
            restaurar_melhores_pesos=config_obj.restaurar_melhores_pesos,
            batch_size=config_obj.batch_size,
            otimizador=config_obj.otimizador,
            embaralhar=config_obj.embaralhar,
            beta1=config_obj.beta1,
            beta2=config_obj.beta2,
            epsilon=config_obj.epsilon,
            l2_lambda=config_obj.l2_lambda,
            dropout=config_obj.dropout,
            gradient_clip=config_obj.gradient_clip,
            callbacks=config_obj.callbacks,
            _epoca_inicial=self._ultima_epoca_treinada,
            _estado_otimizador_inicial=self._ultimo_estado_otimizador,
            _historicos_iniciais=self._obter_historicos(),
        )
