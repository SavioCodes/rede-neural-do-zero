"""Utility helpers for datasets, metrics, plots, and simple file IO."""

from __future__ import annotations

import csv
from importlib import resources
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


class DataUtils:
    """Helpers for generating, normalizing, and splitting datasets."""

    _DATASETS_REAIS: dict[str, dict[str, object]] = {
        "iris": {
            "tipo_tarefa": "classificacao_multiclasse",
            "class_names": ["setosa", "versicolor", "virginica"],
            "target_name": "species",
        },
        "wine": {
            "tipo_tarefa": "classificacao_multiclasse",
            "class_names": ["class_0", "class_1", "class_2"],
            "target_name": "wine_class",
        },
        "diabetes": {
            "tipo_tarefa": "regressao",
            "class_names": None,
            "target_name": "disease_progression",
        },
    }

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

        indices = rng.permutation(X.shape[0])
        return X[indices], y[indices]

    @staticmethod
    def gerar_dataset_multiclasse(
        n_samples: int = 600,
        n_features: int = 2,
        n_classes: int = 3,
        noise: float = 0.12,
        random_state: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a synthetic multi-class dataset using Gaussian blobs."""
        if n_samples < n_classes:
            raise ValueError("n_samples precisa ser pelo menos igual ao numero de classes.")
        if n_features < 2:
            raise ValueError("n_features precisa ser pelo menos 2.")
        if n_classes < 3:
            raise ValueError("n_classes precisa ser pelo menos 3.")
        if noise < 0:
            raise ValueError("noise nao pode ser negativo.")

        rng = np.random.default_rng(random_state)
        tamanhos = [n_samples // n_classes] * n_classes
        for indice in range(n_samples % n_classes):
            tamanhos[indice] += 1

        angulos = np.linspace(0, 2 * np.pi, n_classes, endpoint=False)
        blobs_X = []
        blobs_y = []

        for classe, (angulo, tamanho) in enumerate(zip(angulos, tamanhos)):
            centro = np.zeros(n_features, dtype=float)
            centro[0] = 3.0 * np.cos(angulo)
            centro[1] = 3.0 * np.sin(angulo)
            if n_features > 2:
                centro[2:] = np.linspace(-1.0, 1.0, n_features - 2) + classe * 0.2

            cov = np.eye(n_features) * (0.55 + noise)
            if n_features >= 2:
                cov[0, 1] = cov[1, 0] = ((-1) ** classe) * 0.15

            X_classe = rng.multivariate_normal(centro, cov, size=tamanho)
            if noise:
                X_classe = X_classe + rng.normal(0.0, noise, size=X_classe.shape)

            blobs_X.append(X_classe)
            blobs_y.append(np.full((tamanho, 1), classe, dtype=float))

        X = np.vstack(blobs_X)
        y = np.vstack(blobs_y)
        indices = rng.permutation(X.shape[0])
        return X[indices], y[indices]

    @staticmethod
    def gerar_dataset_regressao(
        n_samples: int = 240,
        n_features: int = 3,
        noise: float = 0.15,
        random_state: Optional[int] = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a smooth synthetic regression dataset with mild nonlinearity."""
        if n_samples < 2:
            raise ValueError("n_samples precisa ser pelo menos 2.")
        if n_features < 1:
            raise ValueError("n_features precisa ser pelo menos 1.")
        if noise < 0:
            raise ValueError("noise nao pode ser negativo.")

        rng = np.random.default_rng(random_state)
        X = rng.uniform(-2.5, 2.5, size=(n_samples, n_features))
        pesos = np.linspace(0.8, 1.6, n_features)
        base = X @ pesos
        componente_nonlinear = 0.8 * np.sin(1.5 * X[:, 0])
        if n_features > 1:
            componente_nonlinear += -0.35 * (X[:, 1] ** 2)
        if n_features > 2:
            componente_nonlinear += 0.25 * X[:, 2] ** 3

        y = base + componente_nonlinear
        if noise:
            y = y + rng.normal(0.0, noise, size=n_samples)

        return X.astype(float), y.reshape(-1, 1).astype(float)

    @staticmethod
    def listar_datasets_reais() -> list[str]:
        """List packaged real datasets available in the project."""
        return sorted(DataUtils._DATASETS_REAIS.keys())

    @staticmethod
    def _ler_dataset_csv(nome: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Load a packaged CSV dataset shipped with the project."""
        nome_normalizado = nome.lower().strip()
        if nome_normalizado not in DataUtils._DATASETS_REAIS:
            raise ValueError(
                f"Dataset real '{nome}' nao encontrado. "
                f"Opcoes: {DataUtils.listar_datasets_reais()}"
            )

        caminho = resources.files("src.datasets").joinpath(f"{nome_normalizado}.csv")
        with caminho.open("r", encoding="utf-8") as arquivo:
            reader = csv.DictReader(arquivo)
            if reader.fieldnames is None or "target" not in reader.fieldnames:
                raise ValueError("O dataset precisa conter cabecalho e coluna 'target'.")

            feature_names = [campo for campo in reader.fieldnames if campo != "target"]
            X_rows: list[list[float]] = []
            y_rows: list[float] = []
            for row in reader:
                X_rows.append([float(row[campo]) for campo in feature_names])
                y_rows.append(float(row["target"]))

        X = np.asarray(X_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float).reshape(-1, 1)
        metadata: dict[str, object] = {
            "nome": nome_normalizado,
            "feature_names": feature_names,
            **DataUtils._DATASETS_REAIS[nome_normalizado],
        }
        return X, y, metadata

    @staticmethod
    def carregar_dataset_real(
        nome: str,
        normalizar: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Load a packaged real dataset and optionally normalize its features."""
        X, y, metadata = DataUtils._ler_dataset_csv(nome)
        if normalizar:
            X, params = DataUtils.normalizar_dados(X, metodo=normalizar)
            metadata = {**metadata, "normalizacao": params}
        return X, y, metadata

    @staticmethod
    def carregar_dataset_iris(
        normalizar: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Load the packaged Iris dataset."""
        return DataUtils.carregar_dataset_real("iris", normalizar=normalizar)

    @staticmethod
    def carregar_dataset_wine(
        normalizar: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Load the packaged Wine dataset."""
        return DataUtils.carregar_dataset_real("wine", normalizar=normalizar)

    @staticmethod
    def carregar_dataset_diabetes(
        normalizar: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        """Load the packaged Diabetes regression dataset."""
        return DataUtils.carregar_dataset_real("diabetes", normalizar=normalizar)

    @staticmethod
    def one_hot_encode(y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
        """Convert integer labels to one-hot encoding."""
        y_array = np.asarray(y)
        if y_array.ndim == 2 and y_array.shape[1] > 1:
            return np.asarray(y_array, dtype=float)

        y_indices = np.asarray(y_array).reshape(-1).astype(int)
        if y_indices.size == 0:
            raise ValueError("y precisa ter pelo menos uma amostra.")
        if np.any(y_indices < 0):
            raise ValueError("Rotulos negativos nao sao suportados em one-hot.")

        total_classes = int(np.max(y_indices)) + 1 if n_classes is None else int(n_classes)
        if total_classes <= int(np.max(y_indices)):
            raise ValueError("n_classes precisa ser maior que o maior rotulo presente.")

        one_hot = np.zeros((y_indices.shape[0], total_classes), dtype=float)
        one_hot[np.arange(y_indices.shape[0]), y_indices] = 1.0
        return one_hot

    @staticmethod
    def decodificar_one_hot(y: np.ndarray) -> np.ndarray:
        """Convert one-hot arrays or logits into class indices."""
        y_array = np.asarray(y)
        if y_array.ndim == 1:
            return y_array.astype(int).reshape(-1, 1)
        if y_array.ndim != 2:
            raise ValueError("y precisa ter formato 1D ou 2D para decodificar.")
        if y_array.shape[1] == 1:
            return y_array.astype(int)
        return np.argmax(y_array, axis=1).reshape(-1, 1)

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
    """Helpers for plotting training history and classification data."""

    @staticmethod
    def _resolver_caminho_saida(salvar: Optional[str]) -> Optional[Path]:
        if not salvar:
            return None

        caminho_saida = Path(salvar)
        if caminho_saida.parent != Path("."):
            caminho_saida.parent.mkdir(parents=True, exist_ok=True)
        return caminho_saida

    @staticmethod
    def _finalizar_figura(fig, salvar: Optional[str], mostrar: bool) -> None:
        caminho_saida = VisualizationUtils._resolver_caminho_saida(salvar)
        if caminho_saida:
            fig.savefig(caminho_saida, dpi=300, bbox_inches="tight")
            print(f"Grafico salvo em: {caminho_saida}")

        if mostrar:
            plt.show()
        plt.close(fig)

    @staticmethod
    def plotar_historico_treinamento(
        historico_erro: list,
        historico_acuracia: list,
        historico_validacao_erro: Optional[list] = None,
        historico_validacao_acuracia: Optional[list] = None,
        salvar: Optional[str] = None,
        mostrar: bool = True,
    ) -> None:
        """Plot training and optional validation history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        epocas = np.arange(1, len(historico_erro) + 1)

        ax1.plot(epocas, historico_erro, "b-", linewidth=2, label="treino")
        if historico_validacao_erro:
            ax1.plot(epocas, historico_validacao_erro, "r--", linewidth=2, label="validacao")
        ax1.set_title("Loss durante o treinamento")
        ax1.set_xlabel("Epoca")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(epocas, historico_acuracia, "g-", linewidth=2, label="treino")
        if historico_validacao_acuracia:
            ax2.plot(
                epocas,
                historico_validacao_acuracia,
                color="orange",
                linestyle="--",
                linewidth=2,
                label="validacao",
            )
        ax2.set_title("Acuracia durante o treinamento")
        ax2.set_xlabel("Epoca")
        ax2.set_ylabel("Acuracia (%)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        VisualizationUtils._finalizar_figura(fig, salvar, mostrar)

    @staticmethod
    def plotar_dados_classificacao(
        X: np.ndarray,
        y: np.ndarray,
        titulo: str = "Dataset de classificacao",
        salvar: Optional[str] = None,
        mostrar: bool = True,
    ) -> None:
        """Plot a classification dataset using the first two features."""
        X_array = DataUtils._garantir_array_2d(X, "X")
        y_indices = MetricUtils._converter_em_indices(y, limiar=0.5, is_prediction=False)

        if X_array.shape[0] != y_indices.shape[0]:
            raise ValueError("X e y precisam ter a mesma quantidade de amostras.")

        if X_array.shape[1] < 2:
            raise ValueError("Plotagem de classificacao precisa de pelo menos duas features.")
        if X_array.shape[1] > 2:
            print(
                "Aviso: plotagem disponivel apenas para dados 2D. "
                "Usando as duas primeiras features."
            )
            X_array = X_array[:, :2]

        classes = np.unique(y_indices)
        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap("tab10", len(classes))

        for indice, classe in enumerate(classes):
            pontos = X_array[y_indices == classe]
            ax.scatter(
                pontos[:, 0],
                pontos[:, 1],
                alpha=0.7,
                s=50,
                color=cmap(indice),
                label=f"Classe {int(classe)}",
            )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(titulo)
        ax.legend()
        ax.grid(True, alpha=0.3)

        VisualizationUtils._finalizar_figura(fig, salvar, mostrar)

    @staticmethod
    def plotar_fronteira_decisao(
        rede_neural,
        X: np.ndarray,
        y: np.ndarray,
        resolucao: int = 100,
        titulo: str = "Fronteira de decisao",
        salvar: Optional[str] = None,
        mostrar: bool = True,
    ) -> None:
        """Plot the decision surface for a trained network on 2D data."""
        X_array = DataUtils._garantir_array_2d(X, "X")
        y_indices = MetricUtils._converter_em_indices(y, limiar=0.5, is_prediction=False)

        if X_array.shape[0] != y_indices.shape[0]:
            raise ValueError("X e y precisam ter a mesma quantidade de amostras.")
        if X_array.shape[1] < 2:
            raise ValueError("Fronteira de decisao exige pelo menos duas features.")
        if X_array.shape[1] > 2:
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
        predicoes = rede_neural.prever(grade_pontos)

        fig, ax = plt.subplots(figsize=(10, 8))
        if predicoes.shape[1] == 1:
            Z = predicoes.reshape(xx.shape)
            ax.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap="RdYlBu")
            ax.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles="--", linewidths=2)
        else:
            Z = np.argmax(predicoes, axis=1).reshape(xx.shape)
            ax.contourf(
                xx,
                yy,
                Z,
                levels=np.arange(predicoes.shape[1] + 1) - 0.5,
                alpha=0.3,
                cmap="tab10",
            )

        classes = np.unique(y_indices)
        cmap = plt.get_cmap("tab10", len(classes))
        for indice, classe in enumerate(classes):
            pontos = X_array[y_indices == classe]
            ax.scatter(
                pontos[:, 0],
                pontos[:, 1],
                alpha=0.8,
                s=60,
                color=cmap(indice),
                label=f"Classe {int(classe)}",
                edgecolors="black",
            )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(titulo)
        ax.legend()

        VisualizationUtils._finalizar_figura(fig, salvar, mostrar)

    @staticmethod
    def plotar_matriz_confusao(
        matriz: np.ndarray,
        labels: Optional[Sequence[str]] = None,
        titulo: str = "Matriz de confusao",
        salvar: Optional[str] = None,
        mostrar: bool = True,
    ) -> None:
        """Plot a confusion matrix with numeric annotations."""
        matriz_array = np.asarray(matriz)
        if matriz_array.ndim != 2 or matriz_array.shape[0] != matriz_array.shape[1]:
            raise ValueError("A matriz de confusao precisa ser quadrada.")

        total_classes = matriz_array.shape[0]
        rotulos = (
            list(labels) if labels is not None else [str(indice) for indice in range(total_classes)]
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        imagem = ax.imshow(matriz_array, cmap="Blues")
        plt.colorbar(imagem, ax=ax)

        ax.set_title(titulo)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.set_xticks(np.arange(total_classes))
        ax.set_yticks(np.arange(total_classes))
        ax.set_xticklabels(rotulos)
        ax.set_yticklabels(rotulos)

        for linha in range(total_classes):
            for coluna in range(total_classes):
                ax.text(
                    coluna,
                    linha,
                    str(int(matriz_array[linha, coluna])),
                    ha="center",
                    va="center",
                    color="black",
                )

        fig.tight_layout()
        VisualizationUtils._finalizar_figura(fig, salvar, mostrar)

    @staticmethod
    def plotar_regressao(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        titulo: str = "Valores reais vs previstos",
        salvar: Optional[str] = None,
        mostrar: bool = True,
    ) -> None:
        """Plot regression predictions against true values."""
        y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_true_array.shape[0] != y_pred_array.shape[0]:
            raise ValueError("y_true e y_pred precisam ter a mesma quantidade de amostras.")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true_array, y_pred_array, alpha=0.7, edgecolors="black")

        minimo = float(min(np.min(y_true_array), np.min(y_pred_array)))
        maximo = float(max(np.max(y_true_array), np.max(y_pred_array)))
        ax.plot([minimo, maximo], [minimo, maximo], linestyle="--", color="black", linewidth=2)

        ax.set_title(titulo)
        ax.set_xlabel("Valor real")
        ax.set_ylabel("Valor previsto")
        ax.grid(True, alpha=0.3)

        VisualizationUtils._finalizar_figura(fig, salvar, mostrar)


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
    def salvar_linhas_csv(linhas: list[dict[str, Any]], caminho: str) -> None:
        """Persiste uma lista de dicionarios, preenchendo colunas ausentes com vazio."""
        if not linhas:
            raise ValueError("linhas nao pode ser vazio.")

        fieldnames: list[str] = []
        for linha in linhas:
            for chave in linha.keys():
                if chave not in fieldnames:
                    fieldnames.append(chave)

        dados = {chave: [linha.get(chave, "") for linha in linhas] for chave in fieldnames}
        FileUtils.salvar_csv(dados, caminho)

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
    """Helpers for binary and multi-class classification metrics."""

    @staticmethod
    def _validar_threshold(limiar: float) -> None:
        if not 0 <= limiar <= 1:
            raise ValueError("limiar precisa estar entre 0 e 1.")

    @staticmethod
    def _converter_em_indices(
        y: np.ndarray,
        limiar: float = 0.5,
        is_prediction: bool = False,
    ) -> np.ndarray:
        y_array = np.asarray(y)
        if y_array.ndim == 1:
            return y_array.astype(int)
        if y_array.ndim != 2:
            raise ValueError("y precisa ter formato 1D ou 2D.")
        if y_array.shape[1] == 1:
            valores = y_array.reshape(-1)
            if is_prediction and np.all((valores >= 0) & (valores <= 1)):
                return (valores >= limiar).astype(int)
            return valores.astype(int)
        return np.argmax(y_array, axis=1).astype(int)

    @staticmethod
    def matriz_confusao(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        limiar: float = 0.5,
        labels: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Compute a confusion matrix for binary or multi-class classification."""
        MetricUtils._validar_threshold(limiar)
        y_true_indices = MetricUtils._converter_em_indices(
            y_true, limiar=limiar, is_prediction=False
        )
        y_pred_indices = MetricUtils._converter_em_indices(
            y_pred, limiar=limiar, is_prediction=True
        )

        if y_true_indices.shape[0] != y_pred_indices.shape[0]:
            raise ValueError("y_true e y_pred precisam ter a mesma quantidade de amostras.")

        classes = (
            np.array(labels, dtype=int)
            if labels is not None
            else np.unique(np.concatenate([y_true_indices, y_pred_indices]))
        )
        mapa = {classe: indice for indice, classe in enumerate(classes.tolist())}
        matriz = np.zeros((len(classes), len(classes)), dtype=int)

        for classe_real, classe_predita in zip(y_true_indices, y_pred_indices):
            matriz[mapa[int(classe_real)], mapa[int(classe_predita)]] += 1

        return matriz

    @staticmethod
    def metricas_classificacao(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        limiar: float = 0.5,
        labels: Optional[Sequence[int]] = None,
    ) -> dict:
        """Compute generic classification metrics for binary or multi-class tasks."""
        cm = MetricUtils.matriz_confusao(y_true, y_pred, limiar=limiar, labels=labels)
        suportes = cm.sum(axis=1)
        total = cm.sum()
        acuracia = float(np.trace(cm) / total) if total > 0 else 0.0

        precisao_por_classe = []
        recall_por_classe = []
        f1_por_classe = []

        for indice in range(cm.shape[0]):
            tp = cm[indice, indice]
            fp = cm[:, indice].sum() - tp
            fn = cm[indice, :].sum() - tp

            precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0.0

            precisao_por_classe.append(float(precisao))
            recall_por_classe.append(float(recall))
            f1_por_classe.append(float(f1))

        suportes_array = suportes.astype(float)
        peso_total = suportes_array.sum() if suportes_array.sum() > 0 else 1.0

        return {
            "acuracia": acuracia,
            "precision_macro": float(np.mean(precisao_por_classe)),
            "recall_macro": float(np.mean(recall_por_classe)),
            "f1_macro": float(np.mean(f1_por_classe)),
            "precision_weighted": float(np.average(precisao_por_classe, weights=suportes_array)),
            "recall_weighted": float(np.average(recall_por_classe, weights=suportes_array)),
            "f1_weighted": float(np.average(f1_por_classe, weights=suportes_array)),
            "supports": suportes.tolist(),
            "matriz_confusao": cm,
            "labels": list(range(cm.shape[0])) if labels is None else list(labels),
            "n_amostras": int(peso_total),
        }

    @staticmethod
    def metricas_regressao(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute core regression metrics for educational experiments."""
        y_true_array = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred_array = np.asarray(y_pred, dtype=float).reshape(-1)
        if y_true_array.shape[0] != y_pred_array.shape[0]:
            raise ValueError("y_true e y_pred precisam ter a mesma quantidade de amostras.")
        if y_true_array.shape[0] == 0:
            raise ValueError("As metricas de regressao exigem pelo menos uma amostra.")

        residuos = y_true_array - y_pred_array
        mse = float(np.mean(residuos**2))
        mae = float(np.mean(np.abs(residuos)))
        rmse = float(np.sqrt(mse))

        soma_quadrados_total = float(np.sum((y_true_array - np.mean(y_true_array)) ** 2))
        soma_quadrados_residuos = float(np.sum(residuos**2))
        r2 = (
            0.0
            if soma_quadrados_total == 0
            else float(1 - soma_quadrados_residuos / soma_quadrados_total)
        )

        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n_amostras": int(y_true_array.shape[0]),
        }

    @staticmethod
    def precisao_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, limiar: float = 0.5) -> dict:
        """Compute classic binary precision, recall, and F1-score."""
        cm = MetricUtils.matriz_confusao(y_true, y_pred, limiar)
        if cm.shape != (2, 2):
            raise ValueError(
                "precisao_recall_f1 e focada em classificacao binaria. "
                "Use metricas_classificacao para multiclasse."
            )

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
