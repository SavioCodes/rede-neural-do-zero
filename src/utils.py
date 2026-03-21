"""Utility helpers for datasets, metrics, plots, and simple file IO."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


class DataUtils:
    """Helpers for generating, normalizing, and splitting datasets."""

    @staticmethod
    def _garantir_array_2d(X: np.ndarray, nome: str) -> np.ndarray:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError(f"{nome} deve ter formato 2D: (amostras, features).")
        if X_array.shape[0] == 0:
            raise ValueError(f"{nome} precisa ter pelo menos uma amostra.")
        if not np.all(np.isfinite(X_array)):
            raise ValueError(f"{nome} precisa conter apenas valores finitos.")
        return X_array

    @staticmethod
    def gerar_xor_dataset() -> Tuple[np.ndarray, np.ndarray]:
        """Return the classic XOR dataset."""
        X = np.array(
            [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ],
            dtype=float,
        )
        y = np.array([[0], [1], [1], [0]], dtype=float)
        return X, y

    @staticmethod
    def gerar_dataset_classificacao(
        n_samples: int = 1000,
        n_features: int = 2,
        noise: float = 0.1,
        random_state: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a simple synthetic binary classification dataset."""
        if n_samples < 2:
            raise ValueError("n_samples precisa ser pelo menos 2.")
        if n_features < 2:
            raise ValueError("n_features precisa ser pelo menos 2.")
        if noise < 0:
            raise ValueError("noise nao pode ser negativo.")

        rng = np.random.default_rng(random_state)
        n_class0 = n_samples // 2
        n_class1 = n_samples - n_class0

        media_classe0 = np.full(n_features, -1.0)
        media_classe1 = np.full(n_features, 1.0)

        # Criamos duas "nuvens" gaussianas com centros diferentes.
        # Isso produz um problema de classificacao simples, mas suficiente
        # para demonstrar aprendizado, normalizacao e avaliacao.
        cov_classe0 = np.eye(n_features)
        cov_classe1 = np.eye(n_features)
        cov_classe0[0, 1] = cov_classe0[1, 0] = 0.5
        cov_classe1[0, 1] = cov_classe1[1, 0] = -0.5

        X_class0 = rng.multivariate_normal(media_classe0, cov_classe0, size=n_class0)
        X_class1 = rng.multivariate_normal(media_classe1, cov_classe1, size=n_class1)
        y_class0 = np.zeros((n_class0, 1))
        y_class1 = np.ones((n_class1, 1))

        X = np.vstack([X_class0, X_class1])
        y = np.vstack([y_class0, y_class1])

        if noise:
            X = X + rng.normal(0.0, noise, size=X.shape)

        # Embaralhar evita que todas as amostras da mesma classe fiquem juntas.
        indices = rng.permutation(X.shape[0])
        return X[indices], y[indices]

    @staticmethod
    def normalizar_dados(X: np.ndarray, metodo: str = "padrao") -> Tuple[np.ndarray, dict]:
        """Normalize a 2D dataset using standard, min-max, or robust scaling."""
        X_array = DataUtils._garantir_array_2d(X, "X")

        if metodo == "padrao":
            media = np.mean(X_array, axis=0)
            desvio = np.std(X_array, axis=0)
            desvio_seguro = np.where(desvio == 0, 1.0, desvio)
            X_norm = (X_array - media) / desvio_seguro
            params = {"media": media, "desvio": desvio, "metodo": "padrao"}
        elif metodo == "minmax":
            minimo = np.min(X_array, axis=0)
            maximo = np.max(X_array, axis=0)
            faixa = maximo - minimo
            faixa_segura = np.where(faixa == 0, 1.0, faixa)
            X_norm = (X_array - minimo) / faixa_segura
            params = {"minimo": minimo, "maximo": maximo, "metodo": "minmax"}
        elif metodo == "robusto":
            mediana = np.median(X_array, axis=0)
            iqr = np.percentile(X_array, 75, axis=0) - np.percentile(X_array, 25, axis=0)
            iqr_seguro = np.where(iqr == 0, 1.0, iqr)
            X_norm = (X_array - mediana) / iqr_seguro
            params = {"mediana": mediana, "iqr": iqr, "metodo": "robusto"}
        else:
            raise ValueError(
                f"Metodo '{metodo}' nao reconhecido. Use 'padrao', 'minmax' ou 'robusto'."
            )

        return X_norm, params

    @staticmethod
    def aplicar_normalizacao(X: np.ndarray, params: dict) -> np.ndarray:
        """Apply a previously fitted normalization configuration."""
        if "metodo" not in params:
            raise ValueError("params precisa conter a chave 'metodo'.")

        X_array = DataUtils._garantir_array_2d(X, "X")
        metodo = params["metodo"]

        if metodo == "padrao":
            desvio_seguro = np.where(params["desvio"] == 0, 1.0, params["desvio"])
            return (X_array - params["media"]) / desvio_seguro
        if metodo == "minmax":
            faixa = params["maximo"] - params["minimo"]
            faixa_segura = np.where(faixa == 0, 1.0, faixa)
            return (X_array - params["minimo"]) / faixa_segura
        if metodo == "robusto":
            iqr_seguro = np.where(params["iqr"] == 0, 1.0, params["iqr"])
            return (X_array - params["mediana"]) / iqr_seguro

        raise ValueError(f"Metodo '{metodo}' nao reconhecido nos parametros salvos.")

    @staticmethod
    def dividir_treino_teste(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: Optional[int] = 42,
    ) -> Tuple[np.ndarray, ...]:
        """Split aligned feature and label arrays into train and test sets."""
        if not 0 < test_size < 1:
            raise ValueError("test_size precisa estar entre 0 e 1.")

        X_array = DataUtils._garantir_array_2d(X, "X")
        y_array = DataUtils._garantir_array_2d(y, "y")

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X e y precisam ter a mesma quantidade de amostras.")

        n_samples = X_array.shape[0]
        n_test = int(round(n_samples * test_size))
        n_test = min(max(n_test, 1), n_samples - 1)

        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        return (
            X_array[train_indices],
            X_array[test_indices],
            y_array[train_indices],
            y_array[test_indices],
        )


class VisualizationUtils:
    """Helpers for plotting training history and 2D classification data."""

    @staticmethod
    def _resolver_caminho_saida(salvar: Optional[str]) -> Optional[Path]:
        if not salvar:
            return None

        caminho_saida = Path(salvar)
        if caminho_saida.parent != Path("."):
            caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        return caminho_saida

    @staticmethod
    def plotar_historico_treinamento(
        historico_erro: list,
        historico_acuracia: list,
        salvar: Optional[str] = None,
    ) -> None:
        """Plot training loss and accuracy history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(historico_erro, "b-", linewidth=2)
        ax1.set_title("Erro durante o treinamento")
        ax1.set_xlabel("Epoca")
        ax1.set_ylabel("Erro")
        ax1.grid(True, alpha=0.3)

        ax2.plot(historico_acuracia, "g-", linewidth=2)
        ax2.set_title("Acuracia durante o treinamento")
        ax2.set_xlabel("Epoca")
        ax2.set_ylabel("Acuracia (%)")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        caminho_saida = VisualizationUtils._resolver_caminho_saida(salvar)
        if caminho_saida:
            plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
            print(f"Grafico salvo em: {caminho_saida}")

        plt.show()
        plt.close(fig)

    @staticmethod
    def plotar_dados_classificacao(
        X: np.ndarray,
        y: np.ndarray,
        titulo: str = "Dataset de classificacao",
        salvar: Optional[str] = None,
    ) -> None:
        """Plot a binary classification dataset using the first two features."""
        X_array = DataUtils._garantir_array_2d(X, "X")
        y_array = DataUtils._garantir_array_2d(y, "y")

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X e y precisam ter a mesma quantidade de amostras.")

        if X_array.shape[1] != 2:
            print(
                "Aviso: plotagem disponivel apenas para dados 2D. "
                "Usando as duas primeiras features."
            )
            X_array = X_array[:, :2]

        plt.figure(figsize=(8, 6))

        classe0 = X_array[y_array.ravel() == 0]
        classe1 = X_array[y_array.ravel() == 1]

        plt.scatter(classe0[:, 0], classe0[:, 1], c="red", alpha=0.6, label="Classe 0", s=50)
        plt.scatter(classe1[:, 0], classe1[:, 1], c="blue", alpha=0.6, label="Classe 1", s=50)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(titulo)
        plt.legend()
        plt.grid(True, alpha=0.3)

        caminho_saida = VisualizationUtils._resolver_caminho_saida(salvar)
        if caminho_saida:
            plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
            print(f"Grafico salvo em: {caminho_saida}")

        plt.show()
        plt.close()

    @staticmethod
    def plotar_fronteira_decisao(
        rede_neural,
        X: np.ndarray,
        y: np.ndarray,
        resolucao: int = 100,
        titulo: str = "Fronteira de decisao",
        salvar: Optional[str] = None,
    ) -> None:
        """Plot the decision surface for a trained network on 2D data."""
        X_array = DataUtils._garantir_array_2d(X, "X")
        y_array = DataUtils._garantir_array_2d(y, "y")

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X e y precisam ter a mesma quantidade de amostras.")

        if X_array.shape[1] != 2:
            print(
                "Aviso: fronteira disponivel apenas para dados 2D. "
                "Usando as duas primeiras features."
            )
            X_array = X_array[:, :2]

        x_min, x_max = X_array[:, 0].min() - 1, X_array[:, 0].max() + 1
        y_min, y_max = X_array[:, 1].min() - 1, X_array[:, 1].max() + 1

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolucao),
            np.linspace(y_min, y_max, resolucao),
        )

        grade_pontos = np.c_[xx.ravel(), yy.ravel()]
        Z = rede_neural.prever(grade_pontos).reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap="RdYlBu")
        plt.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles="--", linewidths=2)

        classe0 = X_array[y_array.ravel() == 0]
        classe1 = X_array[y_array.ravel() == 1]

        plt.scatter(
            classe0[:, 0],
            classe0[:, 1],
            c="red",
            alpha=0.8,
            label="Classe 0",
            s=60,
            edgecolors="black",
        )
        plt.scatter(
            classe1[:, 0],
            classe1[:, 1],
            c="blue",
            alpha=0.8,
            label="Classe 1",
            s=60,
            edgecolors="black",
        )

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title(titulo)
        plt.legend()
        plt.colorbar(label="Probabilidade")

        caminho_saida = VisualizationUtils._resolver_caminho_saida(salvar)
        if caminho_saida:
            plt.savefig(caminho_saida, dpi=300, bbox_inches="tight")
            print(f"Grafico salvo em: {caminho_saida}")

        plt.show()
        plt.close()


class FileUtils:
    """Helpers for small CSV-based experiment artifacts."""

    @staticmethod
    def salvar_csv(dados: dict, caminho: str) -> None:
        """Persist a dictionary of equal-length columns to CSV."""
        if not dados:
            raise ValueError("dados nao pode ser vazio.")

        fieldnames = list(dados.keys())
        comprimentos = [len(valores) for valores in dados.values()]
        if len(set(comprimentos)) != 1:
            raise ValueError("Todas as colunas precisam ter o mesmo numero de linhas.")

        caminho_arquivo = Path(caminho)
        if caminho_arquivo.parent != Path("."):
            caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

        with caminho_arquivo.open("w", newline="", encoding="utf-8") as arquivo:
            writer = csv.DictWriter(arquivo, fieldnames=fieldnames)
            writer.writeheader()

            for indice in range(comprimentos[0]):
                row = {chave: valores[indice] for chave, valores in dados.items()}
                writer.writerow(row)

        print(f"Dados salvos em: {caminho_arquivo}")

    @staticmethod
    def carregar_csv(caminho: str) -> dict:
        """Load a CSV file into a dictionary of columns."""
        caminho_arquivo = Path(caminho)
        dados: dict[str, list] = {}

        with caminho_arquivo.open("r", encoding="utf-8") as arquivo:
            reader = csv.DictReader(arquivo)
            if reader.fieldnames is None:
                raise ValueError("O arquivo CSV precisa conter cabecalho.")

            for field in reader.fieldnames:
                dados[field] = []

            for row in reader:
                for field, value in row.items():
                    try:
                        dados[field].append(float(value))
                    except (TypeError, ValueError):
                        dados[field].append(value)

        print(f"Dados carregados de: {caminho_arquivo}")
        return dados


class MetricUtils:
    """Helpers for binary classification metrics."""

    @staticmethod
    def _validar_threshold(limiar: float) -> None:
        if not 0 <= limiar <= 1:
            raise ValueError("limiar precisa estar entre 0 e 1.")

    @staticmethod
    def matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray, limiar: float = 0.5) -> np.ndarray:
        """Compute a 2x2 confusion matrix for binary classification."""
        MetricUtils._validar_threshold(limiar)
        y_true_array = DataUtils._garantir_array_2d(y_true, "y_true")
        y_pred_array = DataUtils._garantir_array_2d(y_pred, "y_pred")

        if y_true_array.shape[0] != y_pred_array.shape[0]:
            raise ValueError("y_true e y_pred precisam ter a mesma quantidade de amostras.")

        y_pred_bin = (y_pred_array >= limiar).astype(int).ravel()
        y_true_bin = y_true_array.ravel().astype(int)

        # A matriz final fica no formato:
        # [[TN, FP],
        #  [FN, TP]]
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

        return np.array([[tn, fp], [fn, tp]])

    @staticmethod
    def precisao_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, limiar: float = 0.5) -> dict:
        """Compute precision, recall, and F1-score from predicted probabilities."""
        cm = MetricUtils.matriz_confusao(y_true, y_pred, limiar)
        tn, fp, fn, tp = cm.ravel()

        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_accuracy = (recall + especificidade) / 2
        f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0.0

        return {
            "precisao": float(precisao),
            "recall": float(recall),
            "especificidade": float(especificidade),
            "balanced_accuracy": float(balanced_accuracy),
            "f1_score": float(f1),
            "matriz_confusao": cm,
        }
