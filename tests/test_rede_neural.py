#!/usr/bin/env python3
"""Tests for the core neural network implementation."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.rede_neural import RedeNeural
from src.utils import DataUtils


class TestRedeNeural(unittest.TestCase):
    """Unit tests for RedeNeural."""

    def setUp(self) -> None:
        self.arquitetura_simples = [2, 3, 1]
        self.rede = RedeNeural(self.arquitetura_simples, ativacao="sigmoid", seed=123)

        self.X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        self.y_test = np.array([[0], [1], [1], [0]], dtype=float)

    def test_inicializacao_rede(self) -> None:
        self.assertEqual(self.rede.arquitetura, self.arquitetura_simples)
        self.assertEqual(self.rede.num_camadas, 3)
        self.assertEqual(self.rede.ativacao, "sigmoid")
        self.assertEqual(self.rede.inicializacao, "xavier")
        self.assertEqual(self.rede.seed, 123)
        self.assertEqual(len(self.rede.pesos), 2)
        self.assertEqual(len(self.rede.biases), 2)

    def test_formato_pesos(self) -> None:
        self.assertEqual(self.rede.pesos[0].shape, (2, 3))
        self.assertEqual(self.rede.pesos[1].shape, (3, 1))
        self.assertEqual(self.rede.biases[0].shape, (1, 3))
        self.assertEqual(self.rede.biases[1].shape, (1, 1))

    def test_forward_propagation(self) -> None:
        ativacoes, z_values = self.rede._forward_propagation(self.X_test)

        self.assertEqual(len(ativacoes), 3)
        self.assertEqual(len(z_values), 2)
        self.assertEqual(ativacoes[0].shape, (4, 2))
        self.assertEqual(ativacoes[1].shape, (4, 3))
        self.assertEqual(ativacoes[2].shape, (4, 1))

    def test_predicao(self) -> None:
        predicoes = self.rede.prever(self.X_test)

        self.assertEqual(predicoes.shape, (4, 1))
        self.assertTrue(np.all(predicoes >= 0))
        self.assertTrue(np.all(predicoes <= 1))

    def test_prever_classes(self) -> None:
        classes = self.rede.prever_classes(self.X_test)

        self.assertEqual(classes.shape, (4, 1))
        self.assertTrue(np.all(np.isin(classes, [0, 1])))

    def test_treinamento_basico(self) -> None:
        erro_inicial = self.rede._calcular_erro(self.y_test, self.rede.prever(self.X_test))
        self.rede.treinar(self.X_test, self.y_test, epochs=100, taxa_aprendizado=0.5, verbose=False)
        erro_final = self.rede._calcular_erro(self.y_test, self.rede.prever(self.X_test))

        self.assertLessEqual(erro_final, erro_inicial + 0.1)

    def test_xor_aprendizado(self) -> None:
        rede_xor = RedeNeural([2, 4, 1], ativacao="sigmoid", seed=7)
        rede_xor.treinar(self.X_test, self.y_test, epochs=2000, taxa_aprendizado=0.5, verbose=False)

        predicoes = rede_xor.prever(self.X_test)
        acuracia = rede_xor._calcular_acuracia(self.y_test, predicoes)
        self.assertGreater(acuracia, 75.0)

    def test_diferentes_ativacoes(self) -> None:
        for ativacao in ["sigmoid", "relu", "tanh", "leaky_relu", "linear"]:
            with self.subTest(ativacao=ativacao):
                rede = RedeNeural([2, 4, 1], ativacao=ativacao, seed=10)
                predicoes = rede.prever(self.X_test)
                self.assertEqual(predicoes.shape, (4, 1))

                try:
                    rede.treinar(self.X_test, self.y_test, epochs=10, verbose=False)
                except Exception as exc:  # pragma: no cover - explicit assertion message
                    self.fail(f"Treinamento falhou com ativacao {ativacao}: {exc}")

    def test_diferentes_inicializacoes(self) -> None:
        for inicializacao in ["xavier", "he", "aleatorio"]:
            with self.subTest(inicializacao=inicializacao):
                rede = RedeNeural([2, 3, 1], inicializacao=inicializacao, seed=99)
                self.assertFalse(np.all(rede.pesos[0] == 0))
                self.assertFalse(np.all(rede.pesos[1] == 0))

                try:
                    rede.treinar(self.X_test, self.y_test, epochs=10, verbose=False)
                except Exception as exc:  # pragma: no cover - explicit assertion message
                    self.fail(f"Treinamento falhou com inicializacao {inicializacao}: {exc}")

    def test_avaliacao(self) -> None:
        resultado = self.rede.avaliar(self.X_test, self.y_test)

        self.assertIn("erro", resultado)
        self.assertIn("acuracia", resultado)
        self.assertIn("predicoes", resultado)
        self.assertIsInstance(resultado["erro"], float)
        self.assertIsInstance(resultado["acuracia"], float)
        self.assertEqual(resultado["predicoes"].shape, (4, 1))

    def test_historico_treinamento(self) -> None:
        self.rede.treinar(self.X_test, self.y_test, epochs=10, verbose=False)

        self.assertEqual(len(self.rede.historico_erro), 10)
        self.assertEqual(len(self.rede.historico_acuracia), 10)
        self.assertTrue(all(isinstance(x, float) for x in self.rede.historico_erro))
        self.assertTrue(all(isinstance(x, float) for x in self.rede.historico_acuracia))

    def test_treinar_retorna_resumo_e_historico_validacao(self) -> None:
        resumo = self.rede.treinar(
            self.X_test[:3],
            self.y_test[:3],
            epochs=3,
            verbose=False,
            validacao_X=self.X_test[3:],
            validacao_y=self.y_test[3:],
        )

        self.assertIn("erro_final", resumo)
        self.assertIn("acuracia_final", resumo)
        self.assertIn("erro_validacao_final", resumo)
        self.assertEqual(len(self.rede.historico_validacao_erro), 3)
        self.assertEqual(len(self.rede.historico_validacao_acuracia), 3)

    def test_salvar_carregar_parametros(self) -> None:
        self.rede.treinar(self.X_test, self.y_test, epochs=50, verbose=False)
        pred_original = self.rede.prever(self.X_test)

        caminho = "test_modelo.npz"
        self.rede.salvar_parametros(caminho)

        rede_nova = RedeNeural(self.arquitetura_simples)
        rede_nova.carregar_parametros(caminho)
        pred_carregada = rede_nova.prever(self.X_test)

        np.testing.assert_array_almost_equal(pred_original, pred_carregada, decimal=10)
        self.assertEqual(rede_nova.inicializacao, self.rede.inicializacao)
        self.assertEqual(rede_nova.seed, self.rede.seed)

        if os.path.exists(caminho):
            os.remove(caminho)

    def test_seed_reproduz_parametros(self) -> None:
        rede_a = RedeNeural([2, 4, 1], ativacao="relu", inicializacao="he", seed=99)
        rede_b = RedeNeural([2, 4, 1], ativacao="relu", inicializacao="he", seed=99)
        rede_c = RedeNeural([2, 4, 1], ativacao="relu", inicializacao="he", seed=100)

        for pesos_a, pesos_b in zip(rede_a.pesos, rede_b.pesos):
            np.testing.assert_array_almost_equal(pesos_a, pesos_b)

        self.assertTrue(
            any(not np.allclose(pesos_a, pesos_c) for pesos_a, pesos_c in zip(rede_a.pesos, rede_c.pesos))
        )

    def test_arquiteturas_profundas(self) -> None:
        arquiteturas = [
            [2, 4, 3, 1],
            [2, 8, 4, 2, 1],
            [2, 10, 8, 6, 4, 1],
        ]

        for arquitetura in arquiteturas:
            with self.subTest(arquitetura=arquitetura):
                rede = RedeNeural(arquitetura, seed=3)
                predicoes = rede.prever(self.X_test)
                self.assertEqual(predicoes.shape, (4, 1))

                try:
                    rede.treinar(self.X_test, self.y_test, epochs=10, verbose=False)
                except Exception as exc:  # pragma: no cover - explicit assertion message
                    self.fail(f"Falha com arquitetura {arquitetura}: {exc}")

    def test_validacao_durante_treinamento(self) -> None:
        X_train = self.X_test[:3]
        y_train = self.y_test[:3]
        X_val = self.X_test[3:]
        y_val = self.y_test[3:]

        try:
            self.rede.treinar(
                X_train,
                y_train,
                epochs=10,
                validacao_X=X_val,
                validacao_y=y_val,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - explicit assertion message
            self.fail(f"Treinamento com validacao falhou: {exc}")

    def test_treinamento_verbose_com_poucas_epocas(self) -> None:
        try:
            self.rede.treinar(self.X_test, self.y_test, epochs=3, verbose=True)
        except Exception as exc:  # pragma: no cover - explicit assertion message
            self.fail(f"Treinamento verbose com poucas epocas falhou: {exc}")

    def test_validacao_parcial_lanca_erro(self) -> None:
        with self.assertRaises(ValueError):
            self.rede.treinar(self.X_test, self.y_test, epochs=5, validacao_X=self.X_test, verbose=False)

    def test_entrada_invalida_lanca_erro(self) -> None:
        with self.assertRaises(ValueError):
            self.rede.prever(np.array([1.0, 2.0, 3.0]))

        with self.assertRaises(ValueError):
            self.rede.treinar(self.X_test, self.y_test, epochs=0, verbose=False)

        with self.assertRaises(ValueError):
            self.rede.treinar(self.X_test, self.y_test, epochs=5, taxa_aprendizado=0, verbose=False)

    def test_edge_cases(self) -> None:
        X_single = np.array([[0.5, 0.5]], dtype=float)
        y_single = np.array([[1]], dtype=float)

        pred = self.rede.prever(X_single)
        self.assertEqual(pred.shape, (1, 1))

        self.rede.treinar(X_single, y_single, epochs=1, verbose=False)

        X_extreme = np.array([[-1000, 1000], [0, 0]], dtype=float)
        pred_extreme = self.rede.prever(X_extreme)
        self.assertFalse(np.any(np.isnan(pred_extreme)))
        self.assertFalse(np.any(np.isinf(pred_extreme)))


class TestIntegracao(unittest.TestCase):
    """Integration tests using the utility helpers."""

    def test_xor_completo(self) -> None:
        X, y = DataUtils.gerar_xor_dataset()
        rede = RedeNeural([2, 4, 1], ativacao="sigmoid", seed=11)
        rede.treinar(X, y, epochs=1000, taxa_aprendizado=0.5, verbose=False)

        resultado = rede.avaliar(X, y)
        self.assertGreater(resultado["acuracia"], 50.0)

    def test_dataset_sintetico(self) -> None:
        X, y = DataUtils.gerar_dataset_classificacao(n_samples=100, noise=0.1)
        X_norm, _ = DataUtils.normalizar_dados(X)
        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X_norm, y)

        rede = RedeNeural([2, 8, 1], ativacao="relu", seed=21)
        rede.treinar(X_train, y_train, epochs=500, taxa_aprendizado=0.01, verbose=False)

        resultado = rede.avaliar(X_test, y_test)
        self.assertGreater(resultado["acuracia"], 60.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
