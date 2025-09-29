#!/usr/bin/env python3
"""
Testes unitários para a classe RedeNeural
Autor: Sávio (https://github.com/SavioCodes)
"""

import sys
import os
import unittest
import numpy as np

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rede_neural import RedeNeural
from src.utils import DataUtils


class TestRedeNeural(unittest.TestCase):
    """Testes para a classe RedeNeural."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        self.arquitetura_simples = [2, 3, 1]
        self.rede = RedeNeural(self.arquitetura_simples, ativacao='sigmoid')
        
        # Dados de teste simples
        self.X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y_test = np.array([[0], [1], [1], [0]])  # XOR
    
    def test_inicializacao_rede(self):
        """Testa se a rede é inicializada corretamente."""
        self.assertEqual(self.rede.arquitetura, self.arquitetura_simples)
        self.assertEqual(self.rede.num_camadas, 3)
        self.assertEqual(self.rede.ativacao, 'sigmoid')
        self.assertEqual(len(self.rede.pesos), 2)  # 3 camadas = 2 conexões
        self.assertEqual(len(self.rede.biases), 2)
    
    def test_formato_pesos(self):
        """Testa se os pesos têm o formato correto."""
        # Primeira camada: (2, 3)
        self.assertEqual(self.rede.pesos[0].shape, (2, 3))
        # Segunda camada: (3, 1) 
        self.assertEqual(self.rede.pesos[1].shape, (3, 1))
        
        # Biases devem ter formato (1, n_neurons)
        self.assertEqual(self.rede.biases[0].shape, (1, 3))
        self.assertEqual(self.rede.biases[1].shape, (1, 1))
    
    def test_forward_propagation(self):
        """Testa a propagação direta."""
        ativacoes, z_values = self.rede._forward_propagation(self.X_test)
        
        # Deve haver 3 ativações (entrada + 2 camadas)
        self.assertEqual(len(ativacoes), 3)
        self.assertEqual(len(z_values), 2)
        
        # Formatos das ativações
        self.assertEqual(ativacoes[0].shape, (4, 2))  # Entrada
        self.assertEqual(ativacoes[1].shape, (4, 3))  # Camada oculta
        self.assertEqual(ativacoes[2].shape, (4, 1))  # Saída
    
    def test_predicao(self):
        """Testa se a predição funciona."""
        predicoes = self.rede.prever(self.X_test)
        
        # Deve retornar predições com formato correto
        self.assertEqual(predicoes.shape, (4, 1))
        
        # Valores devem estar entre 0 e 1 (sigmoid)
        self.assertTrue(np.all(predicoes >= 0))
        self.assertTrue(np.all(predicoes <= 1))
    
    def test_treinamento_basico(self):
        """Testa se o treinamento executa sem erro."""
        erro_inicial = self.rede._calcular_erro(self.y_test, self.rede.prever(self.X_test))
        
        # Treinar por poucas épocas
        self.rede.treinar(self.X_test, self.y_test, epochs=100, taxa_aprendizado=0.5, verbose=False)
        
        erro_final = self.rede._calcular_erro(self.y_test, self.rede.prever(self.X_test))
        
        # O erro deve diminuir (ou pelo menos não piorar muito)
        self.assertLessEqual(erro_final, erro_inicial + 0.1)
    
    def test_xor_aprendizado(self):
        """Testa se a rede consegue aprender XOR."""
        # Usar arquitetura maior para garantir capacidade
        rede_xor = RedeNeural([2, 4, 1], ativacao='sigmoid')
        
        # Treinar mais intensivamente
        rede_xor.treinar(self.X_test, self.y_test, epochs=2000, taxa_aprendizado=0.5, verbose=False)
        
        # Verificar se consegue classificar corretamente
        predicoes = rede_xor.prever(self.X_test)
        acuracia = rede_xor._calcular_acuracia(self.y_test, predicoes)
        
        # Deve conseguir pelo menos 75% de acurácia no XOR
        self.assertGreater(acuracia, 75.0)
    
    def test_diferentes_ativacoes(self):
        """Testa diferentes funções de ativação."""
        ativacoes = ['sigmoid', 'relu', 'tanh']
        
        for ativacao in ativacoes:
            with self.subTest(ativacao=ativacao):
                rede = RedeNeural([2, 4, 1], ativacao=ativacao)
                
                # Deve conseguir fazer predição
                predicoes = rede.prever(self.X_test)
                self.assertEqual(predicoes.shape, (4, 1))
                
                # Deve conseguir treinar sem erro
                try:
                    rede.treinar(self.X_test, self.y_test, epochs=10, verbose=False)
                except Exception as e:
                    self.fail(f"Treinamento falhou com ativação {ativacao}: {e}")
    
    def test_diferentes_inicializacoes(self):
        """Testa diferentes métodos de inicialização."""
        inicializacoes = ['xavier', 'he', 'aleatorio']
        
        for init in inicializacoes:
            with self.subTest(inicializacao=init):
                rede = RedeNeural([2, 3, 1], inicializacao=init)
                
                # Pesos não devem ser zero
                self.assertFalse(np.all(rede.pesos[0] == 0))
                self.assertFalse(np.all(rede.pesos[1] == 0))
                
                # Deve conseguir treinar
                try:
                    rede.treinar(self.X_test, self.y_test, epochs=10, verbose=False)
                except Exception as e:
                    self.fail(f"Treinamento falhou com inicialização {init}: {e}")
    
    def test_avaliacao(self):
        """Testa a função de avaliação."""
        resultado = self.rede.avaliar(self.X_test, self.y_test)
        
        # Deve retornar dicionário com métricas
        self.assertIn('erro', resultado)
        self.assertIn('acuracia', resultado)
        self.assertIn('predicoes', resultado)
        
        # Métricas devem ser números válidos
        self.assertIsInstance(resultado['erro'], float)
        self.assertIsInstance(resultado['acuracia'], float)
        self.assertEqual(resultado['predicoes'].shape, (4, 1))
    
    def test_historico_treinamento(self):
        """Testa se o histórico é salvo corretamente."""
        self.rede.treinar(self.X_test, self.y_test, epochs=10, verbose=False)
        
        # Histórico deve ter 10 entradas
        self.assertEqual(len(self.rede.historico_erro), 10)
        self.assertEqual(len(self.rede.historico_acuracia), 10)
        
        # Todos os valores devem ser números
        self.assertTrue(all(isinstance(x, float) for x in self.rede.historico_erro))
        self.assertTrue(all(isinstance(x, float) for x in self.rede.historico_acuracia))
    
    def test_salvar_carregar_parametros(self):
        """Testa salvar e carregar parâmetros."""
        # Treinar um pouco para ter parâmetros únicos
        self.rede.treinar(self.X_test, self.y_test, epochs=50, verbose=False)
        
        # Fazer predição original
        pred_original = self.rede.prever(self.X_test)
        
        # Salvar parâmetros
        caminho = 'test_modelo.npz'
        self.rede.salvar_parametros(caminho)
        
        # Criar nova rede e carregar
        rede_nova = RedeNeural(self.arquitetura_simples)
        rede_nova.carregar_parametros(caminho)
        
        # Predições devem ser idênticas
        pred_carregada = rede_nova.prever(self.X_test)
        np.testing.assert_array_almost_equal(pred_original, pred_carregada, decimal=10)
        
        # Limpar arquivo de teste
        if os.path.exists(caminho):
            os.remove(caminho)
    
    def test_arquiteturas_profundas(self):
        """Testa redes com múltiplas camadas ocultas."""
        arquiteturas = [
            [2, 4, 3, 1],
            [2, 8, 4, 2, 1],
            [2, 10, 8, 6, 4, 1]
        ]
        
        for arq in arquiteturas:
            with self.subTest(arquitetura=arq):
                rede = RedeNeural(arq)
                
                # Deve conseguir fazer predição
                pred = rede.prever(self.X_test)
                self.assertEqual(pred.shape, (4, 1))
                
                # Deve conseguir treinar
                try:
                    rede.treinar(self.X_test, self.y_test, epochs=10, verbose=False)
                except Exception as e:
                    self.fail(f"Falha com arquitetura {arq}: {e}")
    
    def test_validacao_durante_treinamento(self):
        """Testa treinamento com dados de validação."""
        # Dividir dados em treino e validação
        X_train = self.X_test[:3]
        y_train = self.y_test[:3]
        X_val = self.X_test[3:]
        y_val = self.y_test[3:]
        
        # Treinar com validação
        try:
            self.rede.treinar(
                X_train, y_train, 
                epochs=10, 
                validacao_X=X_val, 
                validacao_y=y_val,
                verbose=False
            )
        except Exception as e:
            self.fail(f"Treinamento com validação falhou: {e}")
    
    def test_edge_cases(self):
        """Testa casos extremos."""
        # Entrada com apenas uma amostra
        X_single = np.array([[0.5, 0.5]])
        y_single = np.array([[1]])
        
        pred = self.rede.prever(X_single)
        self.assertEqual(pred.shape, (1, 1))
        
        # Treinar com uma amostra
        self.rede.treinar(X_single, y_single, epochs=1, verbose=False)
        
        # Entrada com valores extremos
        X_extreme = np.array([[-1000, 1000], [0, 0]])
        pred_extreme = self.rede.prever(X_extreme)
        
        # Não deve gerar NaN ou infinito
        self.assertFalse(np.any(np.isnan(pred_extreme)))
        self.assertFalse(np.any(np.isinf(pred_extreme)))


class TestIntegracao(unittest.TestCase):
    """Testes de integração com utilitários."""
    
    def test_xor_completo(self):
        """Teste completo com dataset XOR."""
        # Usar função utilitária
        X, y = DataUtils.gerar_xor_dataset()
        
        # Criar e treinar rede
        rede = RedeNeural([2, 4, 1], ativacao='sigmoid')
        rede.treinar(X, y, epochs=1000, taxa_aprendizado=0.5, verbose=False)
        
        # Verificar performance
        resultado = rede.avaliar(X, y)
        self.assertGreater(resultado['acuracia'], 50.0)  # Pelo menos melhor que aleatório
    
    def test_dataset_sintetico(self):
        """Testa com dataset sintético mais complexo."""
        # Gerar dados
        X, y = DataUtils.gerar_dataset_classificacao(n_samples=100, noise=0.1)
        
        # Normalizar
        X_norm, _ = DataUtils.normalizar_dados(X)
        
        # Dividir
        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X_norm, y)
        
        # Treinar
        rede = RedeNeural([2, 8, 1], ativacao='relu')
        rede.treinar(X_train, y_train, epochs=500, taxa_aprendizado=0.01, verbose=False)
        
        # Avaliar
        resultado = rede.avaliar(X_test, y_test)
        self.assertGreater(resultado['acuracia'], 60.0)  # Performance razoável


if __name__ == '__main__':
    print("🧪 EXECUTANDO TESTES DA REDE NEURAL")
    print("=" * 40)
    
    # Executar testes
    unittest.main(verbosity=2)
