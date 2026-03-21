#!/usr/bin/env python3
"""Advanced tests for configs, callbacks, multiclass, and visualization."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import (  # noqa: E402
    CSVLogger,
    DataUtils,
    EarlyStopping,
    History,
    MetricUtils,
    ModelCheckpoint,
    ModelConfig,
    RedeNeural,
    TrainingConfig,
    VisualizationUtils,
)


class TestConfigsAndCallbacks(unittest.TestCase):
    """Tests focused on configs and callback orchestration."""

    def setUp(self) -> None:
        self.X, self.y = DataUtils.gerar_xor_dataset()

    def test_model_config_and_training_config(self) -> None:
        model_config = ModelConfig(
            arquitetura=[2, 4, 1],
            ativacao="sigmoid",
            inicializacao="xavier",
            seed=42,
        )
        train_config = TrainingConfig(
            epochs=60,
            taxa_aprendizado=0.05,
            batch_size=2,
            otimizador="adam",
            embaralhar=False,
        )

        rede = RedeNeural.from_config(model_config)
        resumo = rede.treinar_com_config(self.X, self.y, train_config)

        self.assertEqual(rede.seed, 42)
        self.assertEqual(resumo["batch_size"], 2)
        self.assertEqual(resumo["otimizador"], "adam")

    def test_callbacks_salvam_logs_e_checkpoint(self) -> None:
        history = History()

        with tempfile.TemporaryDirectory() as diretorio:
            csv_logger = CSVLogger(os.path.join(diretorio, "treino.csv"))
            checkpoint = ModelCheckpoint(
                os.path.join(diretorio, "melhor_modelo.npz"),
                monitor="loss",
                save_best_only=True,
            )

            rede = RedeNeural([2, 4, 1], ativacao="sigmoid", seed=10)
            resumo = rede.treinar(
                self.X,
                self.y,
                epochs=40,
                taxa_aprendizado=0.05,
                batch_size=2,
                otimizador="adam",
                embaralhar=False,
                callbacks=[history, csv_logger, checkpoint],
                verbose=False,
            )

            self.assertIn("History", resumo["callbacks"])
            self.assertEqual(len(history.history["epoch"]), resumo["epocas_executadas"])
            self.assertTrue(os.path.exists(os.path.join(diretorio, "treino.csv")))
            self.assertTrue(os.path.exists(os.path.join(diretorio, "melhor_modelo.npz")))

    def test_early_stopping_callback_explicito(self) -> None:
        early_stopping = EarlyStopping(
            monitor="loss",
            patience=2,
            min_delta=1e9,
            restore_best_weights=True,
        )
        rede = RedeNeural([2, 4, 1], ativacao="sigmoid", seed=20)
        resumo = rede.treinar(
            self.X,
            self.y,
            epochs=20,
            taxa_aprendizado=0.05,
            batch_size=2,
            otimizador="adam",
            embaralhar=False,
            callbacks=[early_stopping],
            verbose=False,
        )

        self.assertEqual(resumo["motivo_parada"], "early_stopping")
        self.assertLess(resumo["epocas_executadas"], 20)
        self.assertGreaterEqual(rede._melhor_epoch_callback, 1)

    def test_model_checkpoint_com_estado_completo(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            checkpoint = ModelCheckpoint(
                os.path.join(diretorio, "checkpoint-epoca-{epoch}.npz"),
                monitor="loss",
                save_best_only=False,
                save_training_state=True,
            )
            rede = RedeNeural([2, 4, 1], ativacao="sigmoid", seed=30)
            rede.treinar(
                self.X,
                self.y,
                epochs=4,
                taxa_aprendizado=0.05,
                batch_size=2,
                otimizador="adam",
                callbacks=[checkpoint],
                verbose=False,
            )

            self.assertIsNotNone(checkpoint.ultimo_caminho_salvo)
            self.assertTrue(os.path.exists(checkpoint.ultimo_caminho_salvo))


class TestMulticlassAndRegularization(unittest.TestCase):
    """Tests for multiclass support and regularization options."""

    def setUp(self) -> None:
        X, y = DataUtils.gerar_dataset_multiclasse(n_samples=180, random_state=7)
        X_norm, _ = DataUtils.normalizar_dados(X)
        self.X_train, self.X_test, self.y_train, self.y_test = DataUtils.dividir_treino_teste(
            X_norm,
            y,
            test_size=0.25,
            random_state=7,
        )

    def test_treinamento_multiclasse_com_indices(self) -> None:
        rede = RedeNeural(
            [2, 16, 12, 3],
            ativacao="relu",
            inicializacao="he",
            seed=7,
            funcao_custo="categorical_crossentropy",
        )
        resumo = rede.treinar(
            self.X_train,
            self.y_train,
            epochs=150,
            taxa_aprendizado=0.01,
            batch_size=16,
            otimizador="adam",
            dropout=0.1,
            l2_lambda=1e-3,
            gradient_clip=1.0,
            verbose=False,
        )
        resultado = rede.avaliar(self.X_test, self.y_test)
        classes = rede.prever_classes(self.X_test)
        classes_one_hot = rede.prever_classes(self.X_test, one_hot=True)

        self.assertEqual(resumo["tipo_problema"], "classificacao_multiclasse")
        self.assertEqual(resultado["predicoes"].shape[1], 3)
        self.assertEqual(classes.shape[1], 1)
        self.assertEqual(classes_one_hot.shape[1], 3)
        self.assertGreater(resultado["acuracia"], 70.0)

    def test_treinamento_multiclasse_com_one_hot(self) -> None:
        y_train_one_hot = DataUtils.one_hot_encode(self.y_train, n_classes=3)
        y_test_one_hot = DataUtils.one_hot_encode(self.y_test, n_classes=3)

        rede = RedeNeural(
            [2, 12, 3],
            ativacao="tanh",
            inicializacao="xavier",
            seed=9,
            funcao_custo="categorical_crossentropy",
        )
        rede.treinar(
            self.X_train,
            y_train_one_hot,
            epochs=120,
            taxa_aprendizado=0.01,
            batch_size=12,
            otimizador="adam",
            verbose=False,
        )
        resultado = rede.avaliar(self.X_test, y_test_one_hot)
        self.assertGreater(resultado["acuracia"], 65.0)

    def test_metricas_multiclasse(self) -> None:
        y_true = np.array([[0], [1], [2], [1], [0], [2]], dtype=float)
        y_pred = np.array(
            [
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7],
                [0.2, 0.6, 0.2],
                [0.7, 0.2, 0.1],
                [0.2, 0.5, 0.3],
            ]
        )
        metricas = MetricUtils.metricas_classificacao(y_true, y_pred)

        self.assertEqual(metricas["matriz_confusao"].shape, (3, 3))
        self.assertIn("f1_macro", metricas)
        self.assertGreater(metricas["acuracia"], 0.6)

    def test_visualizacoes_geram_arquivos(self) -> None:
        rede = RedeNeural(
            [2, 10, 3],
            ativacao="relu",
            inicializacao="he",
            seed=3,
            funcao_custo="categorical_crossentropy",
        )
        rede.treinar(
            self.X_train,
            self.y_train,
            epochs=40,
            taxa_aprendizado=0.01,
            batch_size=16,
            otimizador="adam",
            verbose=False,
        )
        metricas = MetricUtils.metricas_classificacao(self.y_test, rede.prever(self.X_test))

        with tempfile.TemporaryDirectory() as diretorio:
            historico = os.path.join(diretorio, "historico.png")
            fronteira = os.path.join(diretorio, "fronteira.png")
            matriz = os.path.join(diretorio, "matriz.png")

            VisualizationUtils.plotar_historico_treinamento(
                rede.historico_erro,
                rede.historico_acuracia,
                salvar=historico,
                mostrar=False,
            )
            VisualizationUtils.plotar_fronteira_decisao(
                rede,
                self.X_test,
                self.y_test,
                salvar=fronteira,
                mostrar=False,
            )
            VisualizationUtils.plotar_matriz_confusao(
                metricas["matriz_confusao"],
                labels=["0", "1", "2"],
                salvar=matriz,
                mostrar=False,
            )

            self.assertTrue(os.path.exists(historico))
            self.assertTrue(os.path.exists(fronteira))
            self.assertTrue(os.path.exists(matriz))


if __name__ == "__main__":
    unittest.main(verbosity=2)
