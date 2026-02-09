"""
ImplementaÃ§Ã£o de uma Rede Neural Artificial do zero
Autor: SÃ¡vio (https://github.com/SavioCodes)
"""

import numpy as np
from typing import List, Optional
from .funcoes_ativacao import FuncoesAtivacao


class RedeNeural:
    """
    Rede Neural Artificial implementada do zero com NumPy.
    
    Suporta mÃºltiplas camadas totalmente conectadas, vÃ¡rias funÃ§Ãµes
    de ativaÃ§Ã£o e treinamento via backpropagation.
    """
    
    def __init__(self, arquitetura: List[int], ativacao: str = 'sigmoid', 
                 inicializacao: str = 'xavier'):
        """
        Inicializa a rede neural.
        
        Args:
            arquitetura: Lista com o nÃºmero de neurÃ´nios em cada camada
                        Ex: [2, 4, 3, 1] = entrada(2), oculta1(4), oculta2(3), saÃ­da(1)
            ativacao: FunÃ§Ã£o de ativaÃ§Ã£o ('sigmoid', 'relu', 'tanh')
            inicializacao: MÃ©todo de inicializaÃ§Ã£o dos pesos ('xavier', 'he', 'aleatorio')
        """
        self.arquitetura = arquitetura
        self.num_camadas = len(arquitetura)
        self.ativacao = ativacao
        self.funcoes = FuncoesAtivacao()
        
        # HistÃ³rico de treinamento
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
                # InicializaÃ§Ã£o Xavier/Glorot
                limite = np.sqrt(6.0 / (entrada_size + saida_size))
                peso = np.random.uniform(-limite, limite, (entrada_size, saida_size))
            elif metodo == 'he':
                # InicializaÃ§Ã£o He (boa para ReLU)
                peso = np.random.randn(entrada_size, saida_size) * np.sqrt(2.0 / entrada_size)
            else:
                # InicializaÃ§Ã£o aleatÃ³ria simples
                peso = np.random.randn(entrada_size, saida_size) * 0.1
            
            bias = np.zeros((1, saida_size))
            
            self.pesos.append(peso)
            self.biases.append(bias)
    
    def _forward_propagation(self, X: np.ndarray) -> tuple:
        """
        Executa a propagaÃ§Ã£o direta (forward propagation).
        
        Args:
            X: Dados de entrada (m, n_features)
            
        Returns:
            tuple: (ativacoes, z_values) - ativaÃ§Ãµes e valores prÃ©-ativaÃ§Ã£o de cada camada
        """
        ativacoes = [X]  # A primeira ativaÃ§Ã£o Ã© a prÃ³pria entrada
        z_values = []
        
        for i in range(self.num_camadas - 1):
            # Calcular z = W * a + b
            z = np.dot(ativacoes[i], self.pesos[i]) + self.biases[i]
            z_values.append(z)
            
            # Aplicar funÃ§Ã£o de ativaÃ§Ã£o
            if i == self.num_camadas - 2:  # Ãšltima camada (saÃ­da)
                # Para a camada de saÃ­da, usar sempre sigmoid para classificaÃ§Ã£o binÃ¡ria
                a = self.funcoes.sigmoid(z)
            else:  # Camadas ocultas
                a = self.funcoes.aplicar(z, self.ativacao)
            
            ativacoes.append(a)
        
        return ativacoes, z_values
    
    def _backward_propagation(self, X: np.ndarray, y: np.ndarray, 
                            ativacoes: List[np.ndarray], 
                            z_values: List[np.ndarray]) -> tuple:
        """
        Executa a retropropagaÃ§Ã£o (backpropagation).
        
        Args:
            X: Dados de entrada
            y: Labels verdadeiros
            ativacoes: AtivaÃ§Ãµes de cada camada
            z_values: Valores prÃ©-ativaÃ§Ã£o de cada camada
            
        Returns:
            tuple: (gradientes_pesos, gradientes_biases)
        """
        m = X.shape[0]
        gradientes_pesos = []
        gradientes_biases = []
        
        # Erro da Ãºltima camada (saÃ­da)
        delta = ativacoes[-1] - y
        
        # Backpropagation das camadas (de trÃ¡s para frente)
        for i in reversed(range(self.num_camadas - 1)):
            # Gradientes para pesos e biases da camada atual
            dW = np.dot(ativacoes[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            gradientes_pesos.insert(0, dW)
            gradientes_biases.insert(0, db)
            
            # Calcular delta para a camada anterior (se nÃ£o for a primeira)
            if i > 0:
                if i == self.num_camadas - 2:  # Vindo da camada de saÃ­da
                    delta_z = self.funcoes.sigmoid_derivada(z_values[i - 1])
                else:  # Camadas ocultas
                    delta_z = self.funcoes.derivada(z_values[i - 1], self.ativacao)
                
                delta = np.dot(delta, self.pesos[i].T) * delta_z
        
        return gradientes_pesos, gradientes_biases
    
    def _calcular_erro(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula o erro quadrÃ¡tico mÃ©dio."""
        return np.mean((y_true - y_pred) ** 2)
    
    def _calcular_acuracia(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          limiar: float = 0.5) -> float:
        """Calcula a acurÃ¡cia para classificaÃ§Ã£o binÃ¡ria."""
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
            epochs: NÃºmero de Ã©pocas de treinamento
            taxa_aprendizado: Taxa de aprendizado
            verbose: Se True, imprime progresso
            validacao_X: Dados de validaÃ§Ã£o (opcional)
            validacao_y: Labels de validaÃ§Ã£o (opcional)
        """
        # Resetar histÃ³rico
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
            
            # Calcular mÃ©tricas
            y_pred = ativacoes[-1]
            erro = self._calcular_erro(y, y_pred)
            acuracia = self._calcular_acuracia(y, y_pred)
            
            self.historico_erro.append(erro)
            self.historico_acuracia.append(acuracia)
            
            # Imprimir progresso
            if verbose and epoch % (epochs // 10) == 0:
                print(f"Ã‰poca {epoch:4d}: Erro = {erro:.4f}, AcurÃ¡cia = {acuracia:.2f}%")
                
                # Se tiver dados de validaÃ§Ã£o, calcular mÃ©tricas tambÃ©m
                if validacao_X is not None and validacao_y is not None:
                    val_pred = self.prever(validacao_X)
                    val_erro = self._calcular_erro(validacao_y, val_pred)
                    val_acuracia = self._calcular_acuracia(validacao_y, val_pred)
                    print(f"         ValidaÃ§Ã£o: Erro = {val_erro:.4f}, AcurÃ¡cia = {val_acuracia:.2f}%")
        
        # Resultados finais
        if verbose:
            print("\n" + "="*50)
            print("TREINAMENTO CONCLUÃDO!")
            print("="*50)
            print(f"Erro final: {self.historico_erro[-1]:.4f}")
            print(f"AcurÃ¡cia final: {self.historico_acuracia[-1]:.2f}%")
    
    def prever(self, X: np.ndarray) -> np.ndarray:
        """
        Faz prediÃ§Ãµes usando a rede neural treinada.
        
        Args:
            X: Dados de entrada (m, n_features)
            
        Returns:
            np.ndarray: PrediÃ§Ãµes (m, 1)
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
            dict: MÃ©tricas de avaliaÃ§Ã£o
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
        """Retorna os parÃ¢metros atuais da rede (pesos e biases)."""
        return {
            'pesos': self.pesos.copy(),
            'biases': self.biases.copy(),
            'arquitetura': self.arquitetura,
            'ativacao': self.ativacao
        }
    
    def salvar_parametros(self, caminho: str):
        """Salva os parametros da rede em um arquivo."""
        pesos_obj = np.empty(len(self.pesos), dtype=object)
        pesos_obj[:] = self.pesos
        biases_obj = np.empty(len(self.biases), dtype=object)
        biases_obj[:] = self.biases

        parametros = {
            'pesos': pesos_obj,
            'biases': biases_obj,
            'arquitetura': np.array(self.arquitetura, dtype=int),
            'ativacao': np.array([self.ativacao], dtype=object),
        }
        np.savez(caminho, **parametros)
        print(f"Parametros salvos em: {caminho}")

    def carregar_parametros(self, caminho: str):
        """Carrega parametros salvos de um arquivo."""
        dados = np.load(caminho, allow_pickle=True)
        self.pesos = [np.array(camada) for camada in dados['pesos'].tolist()]
        self.biases = [np.array(camada) for camada in dados['biases'].tolist()]
        self.arquitetura = [int(n) for n in dados['arquitetura'].tolist()]
        self.ativacao = str(dados['ativacao'].tolist()[0])
        self.num_camadas = len(self.arquitetura)
        print(f"Parametros carregados de: {caminho}")
