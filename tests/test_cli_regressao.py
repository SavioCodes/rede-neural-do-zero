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
from src.benchmarking import (  # noqa: E402
    executar_benchmark,
    executar_suite_benchmark,
    gerar_relatorio_markdown,
    parse_datasets,
)
from src.cli import build_parser  # noqa: E402
from src.cli_config import aplicar_config_cli  # noqa: E402
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
        self.assertTrue(relatorio["leaderboard"])

    def test_benchmark_suite_e_markdown(self) -> None:
        relatorio = executar_suite_benchmark(
            ["iris", "diabetes"],
            amostras=150,
            seeds=[2, 4],
            epochs=25,
        )

        markdown = gerar_relatorio_markdown(relatorio)
        self.assertTrue(relatorio["suite"])
        self.assertEqual(relatorio["datasets"], ["iris", "diabetes"])
        self.assertEqual(len(relatorio["leaderboard"]), 2)
        self.assertIn("## Dataset `iris`", markdown)
        self.assertIn("## Dataset `diabetes`", markdown)

    def test_parse_datasets_remove_duplicatas(self) -> None:
        self.assertEqual(parse_datasets("iris, wine, iris"), ["iris", "wine"])

    def test_config_cli_aplica_yaml_sem_sobrescrever_flag_explicita(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            config_path = Path(diretorio) / "train.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "common:",
                        "  no_plots: true",
                        "train:",
                        "  dataset: iris",
                        "  epochs: 180",
                        "  save_dir: experiments/runs/config-test",
                    ]
                ),
                encoding="utf-8",
            )

            parser, parsers_por_comando = build_parser()
            args = parser.parse_args(
                ["train", "--config", str(config_path), "--epochs", "25", "--no-verbose"]
            )
            args, valores = aplicar_config_cli(
                args,
                parsers_por_comando["train"],
                ["--config", str(config_path), "--epochs", "25", "--no-verbose"],
            )

            self.assertEqual(args.dataset, "iris")
            self.assertEqual(args.epochs, 25)
            self.assertEqual(args.save_dir, "experiments/runs/config-test")
            self.assertTrue(args.no_plots)
            self.assertIn("dataset", valores)

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

    def test_cli_train_com_config_gera_config_efetiva(self) -> None:
        with tempfile.TemporaryDirectory() as diretorio:
            save_dir = Path(diretorio) / "cli-config"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src",
                    "train",
                    "--config",
                    "configs/train/iris.yaml",
                    "--save-dir",
                    str(save_dir),
                ],
                check=True,
                cwd=Path(__file__).resolve().parents[1],
            )

            effective_config = save_dir / "effective-config.json"
            payload = json.loads(effective_config.read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset"], "iris")
            self.assertEqual(payload["save_dir"], str(save_dir))


if __name__ == "__main__":
    unittest.main(verbosity=2)
