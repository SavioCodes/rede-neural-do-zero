#!/usr/bin/env python3
"""Tests for regression support, packaged datasets, checkpoints, and CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import DataUtils, RedeNeural  # noqa: E402
from src.benchmarking import executar_benchmark  # noqa: E402
from src.utils import MetricUtils  # noqa: E402


class TestRegressaoEDatasets(unittest.TestCase):
    """Cobertura das partes novas: regressao e datasets reais."""

    def test_carregar_datasets_reais(self) -> None:
        X_iris, y_iris, meta_iris = DataUtils.carregar_dataset_iris(normalizar="padrao")
        X_wine, y_wine, meta_wine = DataUtils.carregar_dataset_wine()
        X_diabetes, y_diabetes, meta_diabetes = DataUtils.carregar_dataset_diabetes()

        self.assertEqual(X_iris.shape[0], 150)
        self.assertEqual(y_iris.shape[1], 1)
        self.assertEqual(meta_iris["tipo_tarefa"], "classificacao_multiclasse")
        self.assertEqual(X_wine.shape[0], 178)
        self.assertEqual(meta_wine["target_name"], "wine_class")
        self.assertEqual(X_diabetes.shape[1], 10)
        self.assertEqual(meta_diabetes["tipo_tarefa"], "regressao")

    def test_metricas_regressao(self) -> None:
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[1.1], [1.9], [3.2], [3.8]])
        metricas = MetricUtils.metricas_regressao(y_true, y_pred)

        self.assertIn("mse", metricas)
        self.assertIn("mae", metricas)
        self.assertIn("rmse", metricas)
        self.assertIn("r2", metricas)
        self.assertGreater(metricas["r2"], 0.9)

    def test_treinamento_regressao(self) -> None:
        X, y = DataUtils.gerar_dataset_regressao(n_samples=220, random_state=7, noise=0.08)
        X_norm, _ = DataUtils.normalizar_dados(X)
        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
            X_norm,
            y,
            test_size=0.25,
            random_state=7,
        )

        rede = RedeNeural(
            [X.shape[1], 24, 12, 1],
            ativacao="relu",
            inicializacao="he",
            seed=7,
            funcao_custo="mse",
            ativacao_saida="linear",
        )
        resumo = rede.treinar(
            X_train,
            y_train,
            epochs=180,
            taxa_aprendizado=0.01,
            batch_size=32,
            otimizador="adam",
            verbose=False,
        )
        avaliacao = rede.avaliar(X_test, y_test)

        self.assertEqual(resumo["tipo_problema"], "regressao")
        self.assertIsNone(avaliacao["acuracia"])
        self.assertGreater(avaliacao["r2"], 0.75)

    def test_checkpoint_completo_e_resume(self) -> None:
        X, y = DataUtils.gerar_dataset_regressao(n_samples=160, random_state=11, noise=0.08)
        X_norm, _ = DataUtils.normalizar_dados(X)
        X_train, X_val, y_train, y_val = DataUtils.dividir_treino_teste(
            X_norm,
            y,
            test_size=0.25,
            random_state=11,
        )

        rede = RedeNeural(
            [X.shape[1], 20, 10, 1],
            ativacao="relu",
            inicializacao="he",
            seed=11,
            funcao_custo="mse",
            ativacao_saida="linear",
        )
        resumo_inicial = rede.treinar(
            X_train,
            y_train,
            epochs=40,
            taxa_aprendizado=0.01,
            batch_size=16,
            otimizador="adam",
            validacao_X=X_val,
            validacao_y=y_val,
            verbose=False,
        )
        predicoes_antes = rede.prever(X_val[:8])

        with tempfile.TemporaryDirectory() as diretorio:
            checkpoint = Path(diretorio) / "modelo-checkpoint.npz"
            rede.salvar_checkpoint(str(checkpoint))

            rede_carregada = RedeNeural(
                [X.shape[1], 1],
                ativacao="relu",
                seed=1,
                funcao_custo="mse",
                ativacao_saida="linear",
            )
            info = rede_carregada.carregar_checkpoint(str(checkpoint))
            predicoes_carregadas = rede_carregada.prever(X_val[:8])
            np.testing.assert_allclose(predicoes_antes, predicoes_carregadas, atol=1e-10)

            resumo_resume = rede_carregada.retomar_treinamento(
                X_train,
                y_train,
                epochs_adicionais=12,
                validacao_X=X_val,
                validacao_y=y_val,
                verbose=False,
            )

        self.assertEqual(info["epoch"], resumo_inicial["epocas_executadas"])
        self.assertEqual(resumo_resume["epoch_inicial"], resumo_inicial["epocas_executadas"])
        self.assertEqual(resumo_resume["epocas_nesta_execucao"], 12)
        self.assertEqual(
            resumo_resume["epocas_executadas"], resumo_inicial["epocas_executadas"] + 12
        )


class TestBenchmarkECli(unittest.TestCase):
    """Cobertura da camada de benchmark e da CLI oficial."""

    def test_benchmark_multi_seed(self) -> None:
        relatorio = executar_benchmark("diabetes", amostras=442, seeds=[3, 5], epochs=30)

        self.assertEqual(relatorio["dataset"], "diabetes")
        self.assertEqual(relatorio["tipo_tarefa"], "regressao")
        self.assertEqual(len(relatorio["seeds"]), 2)
        self.assertTrue(relatorio["summary"])
        self.assertIn("ranking", relatorio["summary"][0])

    def test_cli_train_gera_artefatos(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            save_dir = Path(diretorio) / "cli"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src",
                    "train",
                    "--dataset",
                    "xor",
                    "--epochs",
                    "20",
                    "--save-dir",
                    str(save_dir),
                    "--no-plots",
                    "--no-verbose",
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            summary_path = save_dir / "train-summary.json"
            checkpoint_path = save_dir / "model-checkpoint.npz"
            self.assertTrue(summary_path.exists())
            self.assertTrue(checkpoint_path.exists())

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"], "xor")
            self.assertIn("training", payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
