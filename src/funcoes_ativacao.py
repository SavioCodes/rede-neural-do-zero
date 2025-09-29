"""
Funções de Ativação para Redes Neurais
Autor: Sávio (https://github.com/SavioCodes)
"""

import numpy as np


class FuncoesAtivacao:
    """
    Classe contendo implementações das principais funções de ativação
    e suas derivadas para uso em redes neurais.
    """
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """
        Função Sigmoid: σ(x) = 1 / (1 + e^(-x))
        
        Características:
        - Saída entre 0 e 1
        - Suave e diferenciável
        - Pode sofrer de vanishing gradient
        
        Args:
            x: Array de entrada
            
        Returns:
            np.ndarray: Valores após aplicação da sigmoid
        """
        # Clip para evitar overflow
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    @staticmethod
    def sigmoid_derivada(x: np.ndarray) -> np.ndarray:
        """
        Derivada da função Sigmoid: σ'(x) = σ(x) * (1 - σ(x))
        
        Args:
            x: Array de entrada (valores pré-ativação)
            
        Returns:
            np.ndarray: Derivada da sigmoid
        """
        sigmoid_x = FuncoesAtivacao.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """
        Função ReLU (Rectified Linear Unit): f(x) = max(0, x)
        
        Características:
        - Saída entre 0 e +∞
        - Computacionalmente eficiente
        - Pode sofrer de dead neurons
        - Boa para deep networks
        
        Args:
            x: Array de entrada
            
        Returns:
            np.ndarray: Valores após aplicação da ReLU
        """
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivada(x: np.ndarray) -> np.ndarray:
        """
        Derivada da função ReLU: f'(x) = 1 se x > 0, senão 0
        
        Args:
            x: Array de entrada (valores pré-ativação)
            
        Returns:
            np.ndarray: Derivada da ReLU
        """
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """
        Função Tangente Hiperbólica: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Características:
        - Saída entre -1 e 1
        - Zero-centered (melhor que sigmoid para algumas aplicações)
        - Ainda pode sofrer de vanishing gradient
        
        Args:
            x: Array de entrada
            
        Returns:
            np.ndarray: Valores após aplicação da tanh
        """
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivada(x: np.ndarray) -> np.ndarray:
        """
        Derivada da função Tanh: tanh'(x) = 1 - tanh²(x)
        
        Args:
            x: Array de entrada (valores pré-ativação)
            
        Returns:
            np.ndarray: Derivada da tanh
        """
        tanh_x = np.tanh(x)
        return 1 - tanh_x ** 2
    
    @staticmethod
    def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Função Leaky ReLU: f(x) = x se x > 0, senão α*x
        
        Características:
        - Resolve o problema de dead neurons da ReLU
        - Permite gradiente pequeno para valores negativos
        
        Args:
            x: Array de entrada
            alpha: Coeficiente para valores negativos
            
        Returns:
            np.ndarray: Valores após aplicação da Leaky ReLU
        """
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivada(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
        """
        Derivada da função Leaky ReLU.
        
        Args:
            x: Array de entrada (valores pré-ativação)
            alpha: Coeficiente para valores negativos
            
        Returns:
            np.ndarray: Derivada da Leaky ReLU
        """
        return np.where(x > 0, 1, alpha)
    
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """
        Função Linear (identidade): f(x) = x
        
        Usada principalmente na camada de saída para regressão.
        
        Args:
            x: Array de entrada
            
        Returns:
            np.ndarray: Valores inalterados
        """
        return x
    
    @staticmethod
    def linear_derivada(x: np.ndarray) -> np.ndarray:
        """
        Derivada da função Linear: f'(x) = 1
        
        Args:
            x: Array de entrada (valores pré-ativação)
            
        Returns:
            np.ndarray: Array de uns
        """
        return np.ones_like(x)
    
    def aplicar(self, x: np.ndarray, nome_funcao: str) -> np.ndarray:
        """
        Aplica uma função de ativação pelo nome.
        
        Args:
            x: Array de entrada
            nome_funcao: Nome da função ('sigmoid', 'relu', 'tanh', 'leaky_relu', 'linear')
            
        Returns:
            np.ndarray: Resultado da função de ativação
        """
        funcoes = {
            'sigmoid': self.sigmoid,
            'relu': self.relu,
            'tanh': self.tanh,
            'leaky_relu': self.leaky_relu,
            'linear': self.linear
        }
        
        if nome_funcao.lower() not in funcoes:
            raise ValueError(f"Função de ativação '{nome_funcao}' não reconhecida. "
                           f"Opções disponíveis: {list(funcoes.keys())}")
        
        return funcoes[nome_funcao.lower()](x)
    
    def derivada(self, x: np.ndarray, nome_funcao: str) -> np.ndarray:
        """
        Calcula a derivada de uma função de ativação pelo nome.
        
        Args:
            x: Array de entrada (valores pré-ativação)
            nome_funcao: Nome da função ('sigmoid', 'relu', 'tanh', 'leaky_relu', 'linear')
            
        Returns:
            np.ndarray: Derivada da função de ativação
        """
        derivadas = {
            'sigmoid': self.sigmoid_derivada,
            'relu': self.relu_derivada,
            'tanh': self.tanh_derivada,
            'leaky_relu': self.leaky_relu_derivada,
            'linear': self.linear_derivada
        }
        
        if nome_funcao.lower() not in derivadas:
            raise ValueError(f"Derivada da função '{nome_funcao}' não disponível. "
                           f"Opções disponíveis: {list(derivadas.keys())}")
        
        return derivadas[nome_funcao.lower()](x)
    
    @classmethod
    def listar_funcoes(cls) -> list:
        """
        Retorna uma lista com todas as funções de ativação disponíveis.
        
        Returns:
            list: Lista com nomes das funções disponíveis
        """
        return ['sigmoid', 'relu', 'tanh', 'leaky_relu', 'linear']
    
    @classmethod
    def info_funcao(cls, nome_funcao: str) -> str:
        """
        Retorna informações sobre uma função de ativação específica.
        
        Args:
            nome_funcao: Nome da função
            
        Returns:
            str: Informações sobre a função
        """
        info = {
            'sigmoid': "Sigmoid: σ(x) = 1/(1+e^(-x)). Saída: [0,1]. Boa para classificação binária.",
            'relu': "ReLU: f(x) = max(0,x). Saída: [0,+∞]. Eficiente, mas pode ter dead neurons.",
            'tanh': "Tanh: tanh(x). Saída: [-1,1]. Zero-centered, melhor que sigmoid às vezes.",
            'leaky_relu': "Leaky ReLU: f(x) = x se x>0, senão αx. Resolve dead neurons da ReLU.",
            'linear': "Linear: f(x) = x. Usada principalmente para regressão na saída."
        }
        
        return info.get(nome_funcao.lower(), f"Função '{nome_funcao}' não encontrada.")
