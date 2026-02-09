"""
UtilitÃ¡rios para manipulaÃ§Ã£o de dados, visualizaÃ§Ã£o e mÃ©tricas
Autor: SÃ¡vio (https://github.com/SavioCodes)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import csv
import os


class DataUtils:
    """UtilitÃ¡rios para manipulaÃ§Ã£o e geraÃ§Ã£o de dados."""
    
    @staticmethod
    def gerar_xor_dataset() -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera o dataset clÃ¡ssico XOR.
        
        Returns:
            tuple: (X, y) onde X sÃ£o as entradas e y sÃ£o as saÃ­das
        """
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        
        y = np.array([
            [0],
            [1],
            [1],
            [0]
        ])
        
        return X, y
    
    @staticmethod
    def gerar_dataset_classificacao(n_samples: int = 1000, n_features: int = 2, 
                                   noise: float = 0.1, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera um dataset sintÃ©tico para classificaÃ§Ã£o binÃ¡ria.
        
        Args:
            n_samples: NÃºmero de amostras
            n_features: NÃºmero de features
            noise: NÃ­vel de ruÃ­do
            random_state: Seed para reprodutibilidade
            
        Returns:
            tuple: (X, y) dataset gerado
        """
        np.random.seed(random_state)
        
        # Gerar duas classes
        n_per_class = n_samples // 2
        
        # Classe 0: centrada em (-1, -1)
        X_class0 = np.random.multivariate_normal(
            mean=[-1, -1], 
            cov=[[1, 0.5], [0.5, 1]], 
            size=n_per_class
        )
        y_class0 = np.zeros((n_per_class, 1))
        
        # Classe 1: centrada em (1, 1)
        X_class1 = np.random.multivariate_normal(
            mean=[1, 1], 
            cov=[[1, -0.5], [-0.5, 1]], 
            size=n_per_class
        )
        y_class1 = np.ones((n_per_class, 1))
        
        # Combinar dados
        X = np.vstack([X_class0, X_class1])
        y = np.vstack([y_class0, y_class1])
        
        # Adicionar ruÃ­do
        X += np.random.normal(0, noise, X.shape)
        
        # Embaralhar
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    @staticmethod
    def normalizar_dados(X: np.ndarray, metodo: str = 'padrao') -> Tuple[np.ndarray, dict]:
        """
        Normaliza os dados usando diferentes mÃ©todos.
        
        Args:
            X: Dados de entrada
            metodo: 'padrao' (z-score), 'minmax', ou 'robusto'
            
        Returns:
            tuple: (dados_normalizados, parametros_normalizacao)
        """
        if metodo == 'padrao':
            media = np.mean(X, axis=0)
            desvio = np.std(X, axis=0)
            desvio_seguro = np.where(desvio == 0, 1.0, desvio)
            X_norm = (X - media) / desvio_seguro
            params = {'media': media, 'desvio': desvio, 'metodo': 'padrao'}
            
        elif metodo == 'minmax':
            minimo = np.min(X, axis=0)
            maximo = np.max(X, axis=0)
            faixa = maximo - minimo
            faixa_segura = np.where(faixa == 0, 1.0, faixa)
            X_norm = (X - minimo) / faixa_segura
            params = {'minimo': minimo, 'maximo': maximo, 'metodo': 'minmax'}
            
        elif metodo == 'robusto':
            mediana = np.median(X, axis=0)
            iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
            iqr_seguro = np.where(iqr == 0, 1.0, iqr)
            X_norm = (X - mediana) / iqr_seguro
            params = {'mediana': mediana, 'iqr': iqr, 'metodo': 'robusto'}
            
        else:
            raise ValueError(f"MÃ©todo '{metodo}' nÃ£o reconhecido. Use 'padrao', 'minmax' ou 'robusto'.")
        
        return X_norm, params
    
    @staticmethod
    def aplicar_normalizacao(X: np.ndarray, params: dict) -> np.ndarray:
        """
        Aplica normalizaÃ§Ã£o usando parÃ¢metros salvos.
        
        Args:
            X: Dados para normalizar
            params: ParÃ¢metros de normalizaÃ§Ã£o salvos
            
        Returns:
            np.ndarray: Dados normalizados
        """
        metodo = params['metodo']
        
        if metodo == 'padrao':
            desvio_seguro = np.where(params['desvio'] == 0, 1.0, params['desvio'])
            return (X - params['media']) / desvio_seguro
        elif metodo == 'minmax':
            faixa = params['maximo'] - params['minimo']
            faixa_segura = np.where(faixa == 0, 1.0, faixa)
            return (X - params['minimo']) / faixa_segura
        elif metodo == 'robusto':
            iqr_seguro = np.where(params['iqr'] == 0, 1.0, params['iqr'])
            return (X - params['mediana']) / iqr_seguro
    
    @staticmethod
    def dividir_treino_teste(X: np.ndarray, y: np.ndarray, 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, ...]:
        """
        Divide os dados em treino e teste.
        
        Args:
            X: Features
            y: Labels
            test_size: ProporÃ§Ã£o para teste (0.0 a 1.0)
            random_state: Seed para reprodutibilidade
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        np.random.seed(random_state)
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        
        # Indices aleatÃ³rios
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


class VisualizationUtils:
    """UtilitÃ¡rios para visualizaÃ§Ã£o de dados e resultados."""
    
    @staticmethod
    def plotar_historico_treinamento(historico_erro: list, historico_acuracia: list, 
                                   salvar: Optional[str] = None):
        """
        Plota o histÃ³rico de erro e acurÃ¡cia durante o treinamento.
        
        Args:
            historico_erro: Lista com erros por Ã©poca
            historico_acuracia: Lista com acurÃ¡cias por Ã©poca
            salvar: Caminho para salvar o grÃ¡fico (opcional)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Erro
        ax1.plot(historico_erro, 'b-', linewidth=2)
        ax1.set_title('Erro durante o Treinamento')
        ax1.set_xlabel('Ã‰poca')
        ax1.set_ylabel('Erro QuadrÃ¡tico MÃ©dio')
        ax1.grid(True, alpha=0.3)
        
        # AcurÃ¡cia
        ax2.plot(historico_acuracia, 'g-', linewidth=2)
        ax2.set_title('AcurÃ¡cia durante o Treinamento')
        ax2.set_xlabel('Ã‰poca')
        ax2.set_ylabel('AcurÃ¡cia (%)')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if salvar:
            plt.savefig(salvar, dpi=300, bbox_inches='tight')
            print(f"GrÃ¡fico salvo em: {salvar}")
        
        plt.show()
    
    @staticmethod
    def plotar_dados_classificacao(X: np.ndarray, y: np.ndarray, titulo: str = "Dataset de ClassificaÃ§Ã£o",
                                  salvar: Optional[str] = None):
        """
        Plota dados de classificaÃ§Ã£o binÃ¡ria em 2D.
        
        Args:
            X: Features (deve ter 2 colunas)
            y: Labels
            titulo: TÃ­tulo do grÃ¡fico
            salvar: Caminho para salvar (opcional)
        """
        if X.shape[1] != 2:
            print("Aviso: Plotagem disponÃ­vel apenas para dados 2D. Usando as duas primeiras features.")
            X = X[:, :2]
        
        plt.figure(figsize=(8, 6))
        
        # Separar por classe
        classe0 = X[y.ravel() == 0]
        classe1 = X[y.ravel() == 1]
        
        # Plotar
        plt.scatter(classe0[:, 0], classe0[:, 1], c='red', alpha=0.6, label='Classe 0', s=50)
        plt.scatter(classe1[:, 0], classe1[:, 1], c='blue', alpha=0.6, label='Classe 1', s=50)
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(titulo)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if salvar:
            plt.savefig(salvar, dpi=300, bbox_inches='tight')
            print(f"GrÃ¡fico salvo em: {salvar}")
        
        plt.show()
    
    @staticmethod
    def plotar_fronteira_decisao(rede_neural, X: np.ndarray, y: np.ndarray, 
                               resolucao: int = 100, titulo: str = "Fronteira de DecisÃ£o",
                               salvar: Optional[str] = None):
        """
        Plota a fronteira de decisÃ£o da rede neural.
        
        Args:
            rede_neural: Rede neural treinada
            X: Dados de entrada (2D)
            y: Labels
            resolucao: ResoluÃ§Ã£o da grade
            titulo: TÃ­tulo do grÃ¡fico
            salvar: Caminho para salvar (opcional)
        """
        if X.shape[1] != 2:
            print("Aviso: Fronteira disponÃ­vel apenas para dados 2D. Usando as duas primeiras features.")
            X = X[:, :2]
        
        # Criar grade
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolucao),
            np.linspace(y_min, y_max, resolucao)
        )
        
        # PrediÃ§Ãµes na grade
        grade_pontos = np.c_[xx.ravel(), yy.ravel()]
        Z = rede_neural.prever(grade_pontos)
        Z = Z.reshape(xx.shape)
        
        # Plotar
        plt.figure(figsize=(10, 8))
        
        # Contorno da fronteira
        plt.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdYlBu')
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        # Dados
        classe0 = X[y.ravel() == 0]
        classe1 = X[y.ravel() == 1]
        
        plt.scatter(classe0[:, 0], classe0[:, 1], c='red', alpha=0.8, label='Classe 0', s=60, edgecolors='black')
        plt.scatter(classe1[:, 0], classe1[:, 1], c='blue', alpha=0.8, label='Classe 1', s=60, edgecolors='black')
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(titulo)
        plt.legend()
        plt.colorbar(label='Probabilidade')
        
        if salvar:
            plt.savefig(salvar, dpi=300, bbox_inches='tight')
            print(f"GrÃ¡fico salvo em: {salvar}")
        
        plt.show()


class FileUtils:
    """UtilitÃ¡rios para manipulaÃ§Ã£o de arquivos."""
    
    @staticmethod
    def salvar_csv(dados: dict, caminho: str):
        """
        Salva dados em formato CSV.
        
        Args:
            dados: DicionÃ¡rio com os dados
            caminho: Caminho do arquivo
        """
        # Criar diretÃ³rio se nÃ£o existir
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        
        with open(caminho, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=dados.keys())
            writer.writeheader()
            
            # Assumir que todos os valores sÃ£o listas do mesmo tamanho
            n_rows = len(list(dados.values())[0])
            for i in range(n_rows):
                row = {key: values[i] for key, values in dados.items()}
                writer.writerow(row)
        
        print(f"Dados salvos em: {caminho}")
    
    @staticmethod
    def carregar_csv(caminho: str) -> dict:
        """
        Carrega dados de um arquivo CSV.
        
        Args:
            caminho: Caminho do arquivo
            
        Returns:
            dict: Dados carregados
        """
        dados = {}
        
        with open(caminho, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Inicializar listas
            for field in reader.fieldnames:
                dados[field] = []
            
            # Ler dados
            for row in reader:
                for field, value in row.items():
                    try:
                        # Tentar converter para float
                        dados[field].append(float(value))
                    except ValueError:
                        # Se nÃ£o conseguir, manter como string
                        dados[field].append(value)
        
        print(f"Dados carregados de: {caminho}")
        return dados


class MetricUtils:
    """UtilitÃ¡rios para cÃ¡lculo de mÃ©tricas adicionais."""
    
    @staticmethod
    def matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray, limiar: float = 0.5) -> np.ndarray:
        """
        Calcula a matriz de confusÃ£o para classificaÃ§Ã£o binÃ¡ria.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: PrediÃ§Ãµes (probabilidades)
            limiar: Limiar para classificaÃ§Ã£o binÃ¡ria
            
        Returns:
            np.ndarray: Matriz de confusÃ£o 2x2
        """
        y_pred_bin = (y_pred >= limiar).astype(int).ravel()
        y_true_bin = y_true.ravel().astype(int)
        
        # Calcular componentes da matriz
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))  # True Positives
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))  # True Negatives
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))  # False Positives
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))  # False Negatives
        
        return np.array([[tn, fp], [fn, tp]])
    
    @staticmethod
    def precisao_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, limiar: float = 0.5) -> dict:
        """
        Calcula precisÃ£o, recall e F1-score.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: PrediÃ§Ãµes (probabilidades)
            limiar: Limiar para classificaÃ§Ã£o binÃ¡ria
            
        Returns:
            dict: MÃ©tricas calculadas
        """
        cm = MetricUtils.matriz_confusao(y_true, y_pred, limiar)
        tn, fp, fn, tp = cm.ravel()
        
        # Evitar divisÃ£o por zero
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
        
        return {
            'precisao': precisao,
            'recall': recall,
            'f1_score': f1,
            'matriz_confusao': cm
        }
