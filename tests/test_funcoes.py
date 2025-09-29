#!/usr/bin/env python3
"""
Testes unitários para funções de ativação e utilitários
Autor: Sávio (https://github.com/SavioCodes)
"""

import sys
import os
import unittest
import numpy as np

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.funcoes_ativacao import FuncoesAtivacao
from src.utils import DataUtils, MetricUtils


class TestFuncoesAtivacao(unittest.TestCase):
    """Testes para as funções de ativação."""
    
    def setUp(self):
        """Configuração inicial."""
        self.funcoes = FuncoesAtivacao()
        self.x_test = np.array([-2, -1, 0, 1, 2])
    
    def test_sigmoid(self):
        """Testa função sigmoid."""
        resultado = self.funcoes.sigmoid(self.x_test)
        
        # Deve estar entre 0 e 1
        self.assertTrue(np.all(resultado >= 0))
        self.assertTrue(np.all(resultado <= 1))
        
        # sigmoid(0) = 0.5
        self.assertAlmostEqual(self.funcoes.sigmoid(np.array([0]))[0], 0.5, places=5)
        
        # Teste de valores extremos
        self.assertAlmostEqual(self.funcoes.sigmoid(np.array([1000]))[0], 1.0, places=3)
        self.assertAlmostEqual(self.funcoes.sigmoid(np.array([-1000]))[0], 0.0, places=3)
    
    def test_sigmoid_derivada(self):
        """Testa derivada da sigmoid."""
        # Teste analítico: σ'(x) = σ(x) * (1 - σ(x))
        x = np.array([0, 1, -1])
        sigmoid_x = self.funcoes.sigmoid(x)
        derivada_esperada = sigmoid_x * (1 - sigmoid_x)
        derivada_calculada = self.funcoes.sigmoid_derivada(x)
        
        np.testing.assert_array_almost_equal(derivada_calculada, derivada_esperada)
    
    def test_relu(self):
        """Testa função ReLU."""
        resultado = self.funcoes.relu(self.x_test)
        esperado = np.array([0, 0, 0, 1, 2])
        
        np.testing.assert_array_equal(resultado, esperado)
        
        # Teste com valores negativos grandes
        x_neg = np.array([-1000, -1])
        resultado_neg = self.funcoes.relu(x_neg)
        np.testing.assert_array_equal(resultado_neg, np.array([0, 0]))
    
    def test_relu_derivada(self):
        """Testa derivada da ReLU."""
        resultado = self.funcoes.relu_derivada(self.x_test)
        esperado = np.array([0, 0, 0, 1, 1])  # 0 para x<=0, 1 para x>0
        
        np.testing.assert_array_equal(resultado, esperado)
    
    def test_tanh(self):
        """Testa função tanh."""
        resultado = self.funcoes.tanh(self.x_test)
        
        # Deve estar entre -1 e 1
        self.assertTrue(np.all(resultado >= -1))
        self.assertTrue(np.all(resultado <= 1))
        
        # tanh(0) = 0
        self.assertAlmostEqual(self.funcoes.tanh(np.array([0]))[0], 0.0, places=5)
        
        # Comparar com numpy
        np.testing.assert_array_almost_equal(resultado, np.tanh(self.x_test))
    
    def test_tanh_derivada(self):
        """Testa derivada da tanh."""
        # Teste analítico: tanh'(x) = 1 - tanh²(x)
        x = np.array([0, 1, -1])
        tanh_x = np.tanh(x)
        derivada_esperada = 1 - tanh_x ** 2
        derivada_calculada = self.funcoes.tanh_derivada(x)
        
        np.testing.assert_array_almost_equal(derivada_calculada, derivada_esperada)
    
    def test_leaky_relu(self):
        """Testa função Leaky ReLU."""
        alpha = 0.1
        resultado = self.funcoes.leaky_relu(self.x_test, alpha)
        
        # Para x > 0, deve ser igual a x
        indices_pos = self.x_test > 0
        np.testing.assert_array_equal(resultado[indices_pos], self.x_test[indices_pos])
        
        # Para x <= 0, deve ser alpha * x
        indices_neg = self.x_test <= 0
        np.testing.assert_array_almost_equal(resultado[indices_neg], alpha * self.x_test[indices_neg])
    
    def test_linear(self):
        """Testa função linear."""
        resultado = self.funcoes.linear(self.x_test)
        np.testing.assert_array_equal(resultado, self.x_test)
    
    def test_linear_derivada(self):
        """Testa derivada da função linear."""
        resultado = self.funcoes.linear_derivada(self.x_test)
        esperado = np.ones_like(self.x_test)
        np.testing.assert_array_equal(resultado, esperado)
    
    def test_aplicar_funcao(self):
        """Testa aplicação de função por nome."""
        funcoes_teste = ['sigmoid', 'relu', 'tanh']
        
        for nome in funcoes_teste:
            resultado = self.funcoes.aplicar(self.x_test, nome)
            self.assertEqual(resultado.shape, self.x_test.shape)
    
    def test_aplicar_derivada(self):
        """Testa cálculo de derivada por nome."""
        funcoes_teste = ['sigmoid', 'relu', 'tanh']
        
        for nome in funcoes_teste:
            resultado = self.funcoes.derivada(self.x_test, nome)
            self.assertEqual(resultado.shape, self.x_test.shape)
    
    def test_funcao_inexistente(self):
        """Testa erro para função inexistente."""
        with self.assertRaises(ValueError):
            self.funcoes.aplicar(self.x_test, 'funcao_inexistente')
        
        with self.assertRaises(ValueError):
            self.funcoes.derivada(self.x_test, 'funcao_inexistente')
    
    def test_listar_funcoes(self):
        """Testa listagem de funções disponíveis."""
        funcoes = self.funcoes.listar_funcoes()
        self.assertIsInstance(funcoes, list)
        self.assertIn('sigmoid', funcoes)
        self.assertIn('relu', funcoes)
        self.assertIn('tanh', funcoes)
    
    def test_info_funcao(self):
        """Testa informações sobre funções."""
        info_sigmoid = self.funcoes.info_funcao('sigmoid')
        self.assertIsInstance(info_sigmoid, str)
        self.assertIn('sigmoid', info_sigmoid.lower())


class TestDataUtils(unittest.TestCase):
    """Testes para utilitários de dados."""
    
    def test_gerar_xor_dataset(self):
        """Testa geração do dataset XOR."""
        X, y = DataUtils.gerar_xor_dataset()
        
        # Formato correto
        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(y.shape, (4, 1))
        
        # Valores corretos
        esperado_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        esperado_y = np.array([[0], [1], [1], [0]])
        
        np.testing.assert_array_equal(X, esperado_X)
        np.testing.assert_array_equal(y, esperado_y)
    
    def test_gerar_dataset_classificacao(self):
        """Testa geração de dataset de classificação."""
        n_samples = 100
        n_features = 2
        
        X, y = DataUtils.gerar_dataset_classificacao(n_samples, n_features)
        
        # Formato correto
        self.assertEqual(X.shape, (n_samples, n_features))
        self.assertEqual(y.shape, (n_samples, 1))
        
        # Classes balanceadas (aproximadamente)
        n_classe0 = np.sum(y == 0)
        n_classe1 = np.sum(y == 1)
        self.assertAlmostEqual(n_classe0, n_samples // 2, delta=5)
        self.assertAlmostEqual(n_classe1, n_samples // 2, delta=5)
    
    def test_normalizar_dados_padrao(self):
        """Testa normalização z-score."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_norm, params = DataUtils.normalizar_dados(X, metodo='padrao')
        
        # Média deve ser próxima de 0
        np.testing.assert_array_almost_equal(np.mean(X_norm, axis=0), [0, 0], decimal=10)
        
        # Desvio deve ser próximo de 1
        np.testing.assert_array_almost_equal(np.std(X_norm, axis=0), [1, 1], decimal=10)
        
        # Parâmetros salvos
        self.assertIn('media', params)
        self.assertIn('desvio', params)
        self.assertEqual(params['metodo'], 'padrao')
    
    def test_normalizar_dados_minmax(self):
        """Testa normalização min-max."""
        X = np.array([[1, 10], [2, 20], [3, 30]])
        X_norm, params = DataUtils.normalizar_dados(X, metodo='minmax')
        
        # Valores devem estar entre 0 e 1
        self.assertTrue(np.all(X_norm >= 0))
        self.assertTrue(np.all(X_norm <= 1))
        
        # Mínimo deve ser 0, máximo deve ser 1
        np.testing.assert_array_almost_equal(np.min(X_norm, axis=0), [0, 0])
        np.testing.assert_array_almost_equal(np.max(X_norm, axis=0), [1, 1])
    
    def test_aplicar_normalizacao(self):
        """Testa aplicação de normalização com parâmetros salvos."""
        X_treino = np.array([[1, 2], [3, 4], [5, 6]])
        X_teste = np.array([[2, 3], [4, 5]])
        
        # Normalizar dados de treino
        X_treino_norm, params = DataUtils.normalizar_dados(X_treino)
        
        # Aplicar mesma normalização aos dados de teste
        X_teste_norm = DataUtils.aplicar_normalizacao(X_teste, params)
        
        # Formato correto
        self.assertEqual(X_teste_norm.shape, X_teste.shape)
        
        # Não deve gerar NaN
        self.assertFalse(np.any(np.isnan(X_teste_norm)))
    
    def test_dividir_treino_teste(self):
        """Testa divisão treino/teste."""
        X = np.random.rand(100, 3)
        y = np.random.rand(100, 1)
        
        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X, y, test_size=0.2)
        
        # Tamanhos corretos
        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
        
        # Features preservadas
        self.assertEqual(X_train.shape[1], 3)
        self.assertEqual(X_test.shape[1], 3)


class TestMetricUtils(unittest.TestCase):
    """Testes para utilitários de métricas."""
    
    def setUp(self):
        """Configuração inicial."""
        # Dados de teste perfeitos
        self.y_true = np.array([[0], [1], [1], [0], [1]])
        self.y_pred_perfect = np.array([[0.1], [0.9], [0.8], [0.2], [0.7]])
        
        # Dados de teste com erros
        self.y_pred_errors = np.array([[0.6], [0.3], [0.9], [0.8], [0.2]])
    
    def test_matriz_confusao(self):
        """Testa cálculo da matriz de confusão."""
        cm = MetricUtils.matriz_confusao(self.y_true, self.y_pred_perfect)
        
        # Deve ser matriz 2x2
        self.assertEqual(cm.shape, (2, 2))
        
        # Para predições perfeitas, diagonal principal deve ter valores
        self.assertGreater(cm[0, 0] + cm[1, 1], 0)  # TP + TN > 0
    
    def test_precisao_recall_f1(self):
        """Testa cálculo de precisão, recall e F1."""
        metricas = MetricUtils.precisao_recall_f1(self.y_true, self.y_pred_perfect)
        
        # Deve conter todas as métricas
        self.assertIn('precisao', metricas)
        self.assertIn('recall', metricas)
        self.assertIn('f1_score', metricas)
        self.assertIn('matriz_confusao', metricas)
        
        # Valores devem estar entre 0 e 1
        self.assertTrue(0 <= metricas['precisao'] <= 1)
        self.assertTrue(0 <= metricas['recall'] <= 1)
        self.assertTrue(0 <= metricas['f1_score'] <= 1)
    
    def test_edge_cases_metricas(self):
        """Testa casos extremos para métricas."""
        # Todas predições como classe 0
        y_pred_all_zero = np.array([[0.1], [0.2], [0.3], [0.1], [0.4]])
        metricas = MetricUtils.precisao_recall_f1(self.y_true, y_pred_all_zero)
        
        # Não deve gerar erro ou NaN
        self.assertFalse(np.isnan(metricas['precisao']))
        self.assertFalse(np.isnan(metricas['recall']))
        self.assertFalse(np.isnan(metricas['f1_score']))
        
        # Todas predições como classe 1
        y_pred_all_one = np.array([[0.9], [0.8], [0.7], [0.9], [0.8]])
        metricas = MetricUtils.precisao_recall_f1(self.y_true, y_pred_all_one)
        
        self.assertFalse(np.isnan(metricas['precisao']))
        self.assertFalse(np.isnan(metricas['recall']))
        self.assertFalse(np.isnan(metricas['f1_score']))
    
    def test_diferentes_limiares(self):
        """Testa métricas com diferentes limiares."""
        limiares = [0.3, 0.5, 0.7]
        
        for limiar in limiares:
            metricas = MetricUtils.precisao_recall_f1(self.y_true, self.y_pred_perfect, limiar)
            
            # Deve funcionar para todos os limiares
            self.assertIsInstance(metricas['precisao'], float)
            self.assertIsInstance(metricas['recall'], float)
            self.assertIsInstance(metricas['f1_score'], float)


class TestIntegracaoUtils(unittest.TestCase):
    """Testes de integração entre utilitários."""
    
    def test_pipeline_completo(self):
        """Testa pipeline completo de dados."""
        # 1. Gerar dados
        X, y = DataUtils.gerar_dataset_classificacao(n_samples=200)
        
        # 2. Normalizar
        X_norm, norm_params = DataUtils.normalizar_dados(X)
        
        # 3. Dividir
        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X_norm, y)
        
        # 4. Simular predições (aleatórias para teste)
        np.random.seed(42)
        y_pred = np.random.rand(len(y_test), 1)
        
        # 5. Calcular métricas
        metricas = MetricUtils.precisao_recall_f1(y_test, y_pred)
        
        # Pipeline deve executar sem erro
        self.assertIsInstance(metricas['f1_score'], float)
        self.assertFalse(np.isnan(metricas['f1_score']))


if __name__ == '__main__':
    print("🧪 EXECUTANDO TESTES DAS FUNÇÕES E UTILITÁRIOS")
    print("=" * 50)
    
    # Executar testes
    unittest.main(verbosity=2)
