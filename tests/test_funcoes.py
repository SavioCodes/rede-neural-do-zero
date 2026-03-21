#!/usr/bin/env python3
"""Tests for activation functions and utility helpers."""

import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.funcoes_ativacao import FuncoesAtivacao
from src.utils import DataUtils, FileUtils, MetricUtils


class TestFuncoesAtivacao(unittest.TestCase):
    """Tests for activation functions."""

    def setUp(self) -> None:
        self.funcoes = FuncoesAtivacao()
        self.x_test = np.array([-2, -1, 0, 1, 2], dtype=float)

    def test_sigmoid(self) -> None:
        resultado = self.funcoes.sigmoid(self.x_test)

        self.assertTrue(np.all(resultado >= 0))
        self.assertTrue(np.all(resultado <= 1))
        self.assertAlmostEqual(self.funcoes.sigmoid(np.array([0.0]))[0], 0.5, places=5)
        self.assertAlmostEqual(self.funcoes.sigmoid(np.array([1000.0]))[0], 1.0, places=3)
        self.assertAlmostEqual(self.funcoes.sigmoid(np.array([-1000.0]))[0], 0.0, places=3)

    def test_sigmoid_derivada(self) -> None:
        x = np.array([0.0, 1.0, -1.0])
        sigmoid_x = self.funcoes.sigmoid(x)
        derivada_esperada = sigmoid_x * (1 - sigmoid_x)
        derivada_calculada = self.funcoes.sigmoid_derivada(x)

        np.testing.assert_array_almost_equal(derivada_calculada, derivada_esperada)

    def test_relu(self) -> None:
        resultado = self.funcoes.relu(self.x_test)
        esperado = np.array([0, 0, 0, 1, 2], dtype=float)

        np.testing.assert_array_equal(resultado, esperado)
        np.testing.assert_array_equal(
            self.funcoes.relu(np.array([-1000.0, -1.0])),
            np.array([0, 0], dtype=float),
        )

    def test_relu_derivada(self) -> None:
        resultado = self.funcoes.relu_derivada(self.x_test)
        esperado = np.array([0, 0, 0, 1, 1], dtype=float)
        np.testing.assert_array_equal(resultado, esperado)

    def test_tanh(self) -> None:
        resultado = self.funcoes.tanh(self.x_test)

        self.assertTrue(np.all(resultado >= -1))
        self.assertTrue(np.all(resultado <= 1))
        self.assertAlmostEqual(self.funcoes.tanh(np.array([0.0]))[0], 0.0, places=5)
        np.testing.assert_array_almost_equal(resultado, np.tanh(self.x_test))

    def test_tanh_derivada(self) -> None:
        x = np.array([0.0, 1.0, -1.0])
        derivada_esperada = 1 - np.tanh(x) ** 2
        derivada_calculada = self.funcoes.tanh_derivada(x)
        np.testing.assert_array_almost_equal(derivada_calculada, derivada_esperada)

    def test_leaky_relu(self) -> None:
        alpha = 0.1
        resultado = self.funcoes.leaky_relu(self.x_test, alpha)

        positivos = self.x_test > 0
        negativos = self.x_test <= 0
        np.testing.assert_array_equal(resultado[positivos], self.x_test[positivos])
        np.testing.assert_array_almost_equal(resultado[negativos], alpha * self.x_test[negativos])

    def test_linear(self) -> None:
        np.testing.assert_array_equal(self.funcoes.linear(self.x_test), self.x_test)

    def test_linear_derivada(self) -> None:
        np.testing.assert_array_equal(
            self.funcoes.linear_derivada(self.x_test),
            np.ones_like(self.x_test),
        )

    def test_aplicar_funcao(self) -> None:
        for nome in ["sigmoid", "relu", "tanh", "leaky_relu", "linear"]:
            resultado = self.funcoes.aplicar(self.x_test, nome)
            self.assertEqual(resultado.shape, self.x_test.shape)

    def test_aplicar_funcao_com_nome_formatado(self) -> None:
        resultado = self.funcoes.aplicar(self.x_test, "  ReLU ")
        np.testing.assert_array_equal(resultado, self.funcoes.relu(self.x_test))

    def test_aplicar_derivada(self) -> None:
        for nome in ["sigmoid", "relu", "tanh", "leaky_relu", "linear"]:
            resultado = self.funcoes.derivada(self.x_test, nome)
            self.assertEqual(resultado.shape, self.x_test.shape)

    def test_funcao_inexistente(self) -> None:
        with self.assertRaises(ValueError):
            self.funcoes.aplicar(self.x_test, "funcao_inexistente")

        with self.assertRaises(ValueError):
            self.funcoes.derivada(self.x_test, "funcao_inexistente")

    def test_listar_funcoes(self) -> None:
        funcoes = self.funcoes.listar_funcoes()
        self.assertIsInstance(funcoes, list)
        self.assertIn("sigmoid", funcoes)
        self.assertIn("relu", funcoes)
        self.assertIn("tanh", funcoes)

    def test_info_funcao(self) -> None:
        info_sigmoid = self.funcoes.info_funcao("sigmoid")
        self.assertIsInstance(info_sigmoid, str)
        self.assertIn("sigmoid", info_sigmoid.lower())


class TestDataUtils(unittest.TestCase):
    """Tests for data utilities."""

    def test_gerar_xor_dataset(self) -> None:
        X, y = DataUtils.gerar_xor_dataset()
        esperado_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        esperado_y = np.array([[0], [1], [1], [0]], dtype=float)

        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(y.shape, (4, 1))
        np.testing.assert_array_equal(X, esperado_X)
        np.testing.assert_array_equal(y, esperado_y)

    def test_gerar_dataset_classificacao(self) -> None:
        X, y = DataUtils.gerar_dataset_classificacao(n_samples=100, n_features=2, random_state=123)

        self.assertEqual(X.shape, (100, 2))
        self.assertEqual(y.shape, (100, 1))
        n_classe0 = np.sum(y == 0)
        n_classe1 = np.sum(y == 1)
        self.assertAlmostEqual(n_classe0, 50, delta=5)
        self.assertAlmostEqual(n_classe1, 50, delta=5)

    def test_gerar_dataset_classificacao_multi_feature_e_reproduzivel(self) -> None:
        X1, y1 = DataUtils.gerar_dataset_classificacao(n_samples=101, n_features=4, random_state=7)
        X2, y2 = DataUtils.gerar_dataset_classificacao(n_samples=101, n_features=4, random_state=7)

        self.assertEqual(X1.shape, (101, 4))
        self.assertEqual(y1.shape, (101, 1))
        self.assertEqual(int(np.sum(y1 == 1)), 51)
        np.testing.assert_array_almost_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_gerar_dataset_classificacao_parametros_invalidos(self) -> None:
        with self.assertRaises(ValueError):
            DataUtils.gerar_dataset_classificacao(n_samples=1)

        with self.assertRaises(ValueError):
            DataUtils.gerar_dataset_classificacao(n_features=1)

        with self.assertRaises(ValueError):
            DataUtils.gerar_dataset_classificacao(noise=-0.1)

    def test_normalizar_dados_padrao(self) -> None:
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        X_norm, params = DataUtils.normalizar_dados(X, metodo="padrao")

        np.testing.assert_array_almost_equal(np.mean(X_norm, axis=0), [0, 0], decimal=10)
        np.testing.assert_array_almost_equal(np.std(X_norm, axis=0), [1, 1], decimal=10)
        self.assertIn("media", params)
        self.assertIn("desvio", params)
        self.assertEqual(params["metodo"], "padrao")

    def test_normalizar_dados_minmax(self) -> None:
        X = np.array([[1, 10], [2, 20], [3, 30]], dtype=float)
        X_norm, _ = DataUtils.normalizar_dados(X, metodo="minmax")

        self.assertTrue(np.all(X_norm >= 0))
        self.assertTrue(np.all(X_norm <= 1))
        np.testing.assert_array_almost_equal(np.min(X_norm, axis=0), [0, 0])
        np.testing.assert_array_almost_equal(np.max(X_norm, axis=0), [1, 1])

    def test_aplicar_normalizacao(self) -> None:
        X_treino = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        X_teste = np.array([[2, 3], [4, 5]], dtype=float)

        _, params = DataUtils.normalizar_dados(X_treino)
        X_teste_norm = DataUtils.aplicar_normalizacao(X_teste, params)

        self.assertEqual(X_teste_norm.shape, X_teste.shape)
        self.assertFalse(np.any(np.isnan(X_teste_norm)))

    def test_aplicar_normalizacao_invalida(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=float)

        with self.assertRaises(ValueError):
            DataUtils.aplicar_normalizacao(X, {})

        with self.assertRaises(ValueError):
            DataUtils.aplicar_normalizacao(X, {"metodo": "desconhecido"})

    def test_dividir_treino_teste(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.random((100, 3))
        y = rng.random((100, 1))

        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
            X, y, test_size=0.2, random_state=10
        )

        self.assertEqual(X_train.shape[0], 80)
        self.assertEqual(X_test.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_test.shape[0], 20)
        self.assertEqual(X_train.shape[1], 3)
        self.assertEqual(X_test.shape[1], 3)

    def test_dividir_treino_teste_parametros_invalidos(self) -> None:
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([[0], [1]], dtype=float)

        with self.assertRaises(ValueError):
            DataUtils.dividir_treino_teste(X, y, test_size=0)

        with self.assertRaises(ValueError):
            DataUtils.dividir_treino_teste(X, y, test_size=1)

        with self.assertRaises(ValueError):
            DataUtils.dividir_treino_teste(X, y[:1], test_size=0.5)


class TestMetricUtils(unittest.TestCase):
    """Tests for classification metrics."""

    def setUp(self) -> None:
        self.y_true = np.array([[0], [1], [1], [0], [1]], dtype=float)
        self.y_pred_perfect = np.array([[0.1], [0.9], [0.8], [0.2], [0.7]], dtype=float)
        self.y_pred_errors = np.array([[0.6], [0.3], [0.9], [0.8], [0.2]], dtype=float)

    def test_matriz_confusao(self) -> None:
        cm = MetricUtils.matriz_confusao(self.y_true, self.y_pred_perfect)
        self.assertEqual(cm.shape, (2, 2))
        self.assertGreater(cm[0, 0] + cm[1, 1], 0)

    def test_precisao_recall_f1(self) -> None:
        metricas = MetricUtils.precisao_recall_f1(self.y_true, self.y_pred_perfect)

        self.assertIn("precisao", metricas)
        self.assertIn("recall", metricas)
        self.assertIn("especificidade", metricas)
        self.assertIn("balanced_accuracy", metricas)
        self.assertIn("f1_score", metricas)
        self.assertIn("matriz_confusao", metricas)
        self.assertTrue(0 <= metricas["precisao"] <= 1)
        self.assertTrue(0 <= metricas["recall"] <= 1)
        self.assertTrue(0 <= metricas["especificidade"] <= 1)
        self.assertTrue(0 <= metricas["balanced_accuracy"] <= 1)
        self.assertTrue(0 <= metricas["f1_score"] <= 1)
        self.assertIsInstance(metricas["precisao"], float)

    def test_edge_cases_metricas(self) -> None:
        y_pred_all_zero = np.array([[0.1], [0.2], [0.3], [0.1], [0.4]], dtype=float)
        metricas_zero = MetricUtils.precisao_recall_f1(self.y_true, y_pred_all_zero)
        self.assertFalse(np.isnan(metricas_zero["precisao"]))
        self.assertFalse(np.isnan(metricas_zero["recall"]))
        self.assertFalse(np.isnan(metricas_zero["f1_score"]))

        y_pred_all_one = np.array([[0.9], [0.8], [0.7], [0.9], [0.8]], dtype=float)
        metricas_one = MetricUtils.precisao_recall_f1(self.y_true, y_pred_all_one)
        self.assertFalse(np.isnan(metricas_one["precisao"]))
        self.assertFalse(np.isnan(metricas_one["recall"]))
        self.assertFalse(np.isnan(metricas_one["f1_score"]))

    def test_diferentes_limiares(self) -> None:
        for limiar in [0.3, 0.5, 0.7]:
            metricas = MetricUtils.precisao_recall_f1(self.y_true, self.y_pred_perfect, limiar)
            self.assertIsInstance(metricas["precisao"], float)
            self.assertIsInstance(metricas["recall"], float)
            self.assertIsInstance(metricas["f1_score"], float)

    def test_limiar_invalido_lanca_erro(self) -> None:
        with self.assertRaises(ValueError):
            MetricUtils.matriz_confusao(self.y_true, self.y_pred_perfect, limiar=1.5)

        with self.assertRaises(ValueError):
            MetricUtils.precisao_recall_f1(self.y_true, self.y_pred_perfect, limiar=-0.1)


class TestFileUtils(unittest.TestCase):
    """Tests for file helpers."""

    def test_salvar_e_carregar_csv_sem_subpasta(self) -> None:
        dados = {"epoca": [1, 2], "erro": [0.5, 0.2]}

        with tempfile.TemporaryDirectory() as diretorio:
            caminho = os.path.join(diretorio, "metricas.csv")
            FileUtils.salvar_csv(dados, caminho)
            carregado = FileUtils.carregar_csv(caminho)

        self.assertEqual(carregado["epoca"], [1.0, 2.0])
        self.assertEqual(carregado["erro"], [0.5, 0.2])

    def test_salvar_csv_com_colunas_inconsistentes_lanca_erro(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            caminho = os.path.join(diretorio, "metricas.csv")
            with self.assertRaises(ValueError):
                FileUtils.salvar_csv({"epoca": [1, 2], "erro": [0.5]}, caminho)


class TestIntegracaoUtils(unittest.TestCase):
    """Integration tests across utility helpers."""

    def test_pipeline_completo(self) -> None:
        X, y = DataUtils.gerar_dataset_classificacao(n_samples=200, random_state=5)
        X_norm, _ = DataUtils.normalizar_dados(X)
        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X_norm, y, random_state=5)

        rng = np.random.default_rng(42)
        y_pred = rng.random((len(y_test), 1))
        metricas = MetricUtils.precisao_recall_f1(y_test, y_pred)

        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(y_train.shape[1], y_test.shape[1])
        self.assertIsInstance(metricas["f1_score"], float)
        self.assertFalse(np.isnan(metricas["f1_score"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
