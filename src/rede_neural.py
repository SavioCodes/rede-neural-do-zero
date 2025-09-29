"""
Implementação de uma Rede Neural Artificial do zero
Autor: Sávio (https://github.com/SavioCodes)
"""

import numpy as np
from typing import List, Optional
from .funcoes_ativacao import FuncoesAtivacao


class RedeNeural:
    """
    Rede Neural Artificial implementada do zero com NumPy.
    
    Suporta múltiplas camadas totalmente conectadas, várias funções
    de ativação e treinamento via backpropagation.
    """
    
    def __init__(self, arquitetura: List[int], ativacao: str = 'sigmoid', 
                 inicializacao: str = 'xavier'):
        """
        Inicializa a rede neural.
        
        Args:
            arquitetura: Lista com o número de neurônios em cada camada
                        Ex: [2, 4, 3, 1] = entrada(2), oculta1(4), oculta2(3), saída(1)
            ativacao: Função de ativação ('sigmoid', 'relu', 'tanh')
            inicializacao: Método de inicialização dos pesos ('xavier', 'he', 'aleatorio')
        """
        self.arquitetura = arquitetura
        self.num_camadas = len(arquitetura)
        self.ativacao = ativacao
        self.funcoes = FuncoesAtivacao()
        
        # Histórico de treinamento
        self.historico_erro = []
        self.historico_acuracia = []
        
        # Inicializar pesos e biases
        self._inicializar_parametros(inicializacao)
    
    def _inicializar_parametros(self, metodo: str):
        """Inicializa pesos e biases da rede neural."""
        self.pesos = []
        self.biases = []
        
        for i in range(self.num_camadas - 1):
            entrada_size = self.arquitetura[i]
            saida_size = self.arquitetura[i + 1]
            
            if metodo == 'xavier':
                # Inicialização Xavier/Glorot
                limite = np.sqrt(6.0 / (entrada_size + saida_size))
                peso = np.random.uniform(-limite, limite, (entrada_size, saida_size))
            elif metodo == 'he':
                # Inicialização He (boa para ReLU)
                peso = np.random.randn(entrada_size, saida_size) * np.sqrt(2.0 / entrada_size)
            else:
                # Inicialização aleatória simples
                peso = np.random.randn(entrada_size, saida_size) * 0.1
            
            bias = np.zeros((1, saida_size))
            
            self.pesos.append(peso)
            self.biases.append(bias)
    
    def _forward_propagation(self, X: np.ndarray) -> tuple:
        """
        Executa a propagação direta (forward propagation).
        
        Args:
            X: Dados de entrada (m, n_features)
            
        Returns:
            tuple: (ativacoes, z_values) - ativações e valores pré-ativação de cada camada
        """
        ativacoes = [X]  # A primeira ativação é a própria entrada
        z_values = []
        
        for i in range(self.num_camadas - 1):
            # Calcular z = W * a + b
            z = np.dot(ativacoes[i], self.pesos[i]) + self.biases[i]
            z_values.append(z)
            
            # Aplicar função de ativação
            if i == self.num_camadas - 2:  # Última camada (saída)
                # Para a camada de saída, usar sempre sigmoid para classificação binária
                a = self.funcoes.sigmoid(z)
            else:  # Camadas ocultas
                a = self.funcoes.aplicar(z, self.ativacao)
            
            ativacoes.append(a)
        
        return ativacoes, z_values
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, 
                            ativacoes: List[np.ndarray], 
                            z_values: List[np.ndarray]) -> tuple:
        """
        Executa a retropropagação (backpropagation).
        
        Args:
            X: Dados de entrada
            y: Labels verdadeiros
            ativacoes: Ativações de cada camada
            z_values: Valores pré-ativação de cada camada
            
        Returns:
            tuple: (gradientes_pesos, gradientes_biases)
        """
        m = X.shape[0]
        gradientes_pesos = []
        gradientes_biases = []
        
        # Erro da última camada (saída)
        delta = ativacoes[-1] - y
        
        # Backpropagation das camadas (de trás para frente)
        for i in reversed(range(self.num_camadas - 1)):
            # Gradientes para pesos e biases da camada atual
            dW = np.dot(ativacoes[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            gradientes_pesos.insert(0, dW)
            gradientes_biases.insert(0, db)
            
            # Calcular delta para a camada anterior (se não for a primeira)
            if i > 0:
                if i == self.num_camadas - 2:  # Vindo da camada de saída
                    delta_z = self.funcoes.sigmoid_derivada(z_values[i - 1])
                else:  # Camadas ocultas
                    delta_z = self.funcoes.derivada(z_values[i - 1], self.ativacao)
                
                delta = np.dot(delta, self.pesos[i].T) * delta_z
        
        return gradientes_pesos, gradientes_biases
    
    def _calcular_erro(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula o erro quadrático médio."""
        return np.mean((y_true - y_pred) ** 2)
    
    def _calcular_acuracia(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          limiar: float = 0.5) -> float:
        """Calcula a acurácia para classificação binária."""
        predicoes_binarias = (y_pred >= limiar).astype(int)
        y_true_binarias = (y_true >= limiar).astype(int)
        return np.mean(predicoes_binarias == y_true_binarias) * 100
    
    def treinar(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
               taxa_aprendizado: float = 0.1, verbose: bool = True, 
               validacao_X: Optional[np.ndarray] = None, 
               validacao_y: Optional[np.ndarray] = None):
        """
        Treina a rede neural usando gradiente descendente.
        
        Args:
            X: Dados de treino (m, n_features)
            y: Labels de treino (m, 1)
            epochs: Número de épocas de treinamento
            taxa_aprendizado: Taxa de aprendizado
            verbose: Se True, imprime progresso
            validacao_X: Dados de validação (opcional)
            validacao_y: Labels de validação (opcional)
        """
        # Resetar histórico
        self.historico_erro = []
        self.historico_acuracia = []
        
        for epoch in range(epochs):
            # Forward propagation
            ativacoes, z_values = self._forward_propagation(X)
            
            # Backward propagation
            grad_pesos, grad_biases = self._backward_propagation(X, y, ativacoes, z_values)
            
            # Atualizar pesos e biases
            for i in range(len(self.pesos)):
                self.pesos[i] -= taxa_aprendizado * grad_pesos[i]
                self.biases[i] -= taxa_aprendizado * grad_biases[i]
            
            # Calcular métricas
            y_pred = ativacoes[-1]
            erro = self._calcular_erro(y, y_pred)
            acuracia = self._calcular_acuracia(y, y_pred)
            
            self.historico_erro.append(erro)
            self.historico_acuracia.append(acuracia)
            
            # Imprimir progresso
            if verbose and epoch % (epochs // 10) == 0:
                print(f"Época {epoch:4d}: Erro = {erro:.4f}, Acurácia = {acuracia:.2f}%")
                
                # Se tiver dados de validação, calcular métricas também
                if validacao_X is not None and validacao_y is not None:
                    val_pred = self.prever(validacao_X)
                    val_erro = self._calcular_erro(validacao_y, val_pred)
                    val_acuracia = self._calcular_acuracia(validacao_y, val_pred)
                    print(f"         Validação: Erro = {val_erro:.4f}, Acurácia = {val_acuracia:.2f}%")
        
        # Resultados finais
        if verbose:
            print("\n" + "="*50)
            print("TREINAMENTO CONCLUÍDO!")
            print("="*50)
            print(f"Erro final: {self.historico_erro[-1]:.4f}")
            print(f"Acurácia final: {self.historico_acuracia[-1]:.2f}%")
    
    def prever(self, X: np.ndarray) -> np.ndarray:
        """
        Faz predições usando a rede neural treinada.
        
        Args:
            X: Dados de entrada (m, n_features)
            
        Returns:
            np.ndarray: Predições (m, 1)
        """
        ativacoes, _ = self._forward_propagation(X)
        return ativacoes[-1]
    
    def avaliar(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Avalia a rede neural em um conjunto de dados.
        
        Args:
            X: Dados de teste
            y: Labels verdadeiros
            
        Returns:
            dict: Métricas de avaliação
        """
        predicoes = self.prever(X)
        erro = self._calcular_erro(y, predicoes)
        acuracia = self._calcular_acuracia(y, predicoes)
        
        return {
            'erro': erro,
            'acuracia': acuracia,
            'predicoes': predicoes
        }
    
    def obter_parametros(self) -> dict:
        """Retorna os parâmetros atuais da rede (pesos e biases)."""
        return {
            'pesos': self.pesos.copy(),
            'biases': self.biases.copy(),
            'arquitetura': self.arquitetura,
            'ativacao': self.ativacao
        }
    
    def salvar_parametros(self, caminho: str):
        """Salva os parâmetros da rede em um arquivo."""
        parametros = self.obter_parametros()
        np.savez(caminho, **parametros)
        print(f"Parâmetros salvos em: {caminho}")
    
    def carregar_parametros(self, caminho: str):
        """Carrega parâmetros salvos de um arquivo."""
        dados = np.load(caminho, allow_pickle=True)
        self.pesos = dados['pesos'].tolist()
        self.biases = dados['biases'].tolist()
        self.arquitetura = dados['arquitetura'].tolist()
        self.ativacao = str(dados['ativacao'])
        print(f"Parâmetros carregados de: {caminho}")
