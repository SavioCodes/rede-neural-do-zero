"""Helpers para benchmark multi-seed e suites de experimentos."""

from __future__ import annotations

import statistics
from typing import Any

from .config import ModelConfig, TrainingConfig
from .experiments import (
    DatasetBundle,
    avaliar_modelo,
    carregar_dataset,
    dividir_treino_validacao_teste,
)
from .rede_neural import RedeNeural


def parse_seeds(texto: str) -> list[int]:
    """Converte `42,52,62` em lista de inteiros."""
    seeds = [int(parte.strip()) for parte in texto.split(",") if parte.strip()]
    if not seeds:
        raise ValueError("Forneca pelo menos uma seed em --seeds.")
    return seeds


def parse_datasets(texto: str) -> list[str]:
    """Converte `iris,wine` em lista de datasets sem duplicatas."""
    datasets: list[str] = []
    for parte in texto.split(","):
        nome = parte.strip().lower()
        if nome and nome not in datasets:
            datasets.append(nome)
    if not datasets:
        raise ValueError("Forneca pelo menos um dataset em --datasets.")
    return datasets


def nome_dataset_padrao(modo: str) -> str:
    """Resolve qual dataset usar por modo de benchmark."""
    return {
        "binario": "binario",
        "multiclasse": "iris",
        "regressao": "diabetes",
    }[modo]


def metrica_principal_para_tarefa(tipo_tarefa: str) -> str:
    """Resolve a metrica principal esperada para cada tipo de tarefa."""
    return "r2" if tipo_tarefa == "regressao" else "acuracia"


def configuracoes_para_dataset(
    dataset: DatasetBundle,
    seed: int,
    epochs: int,
) -> list[tuple[str, ModelConfig, TrainingConfig]]:
    """Monta configuracoes de modelo para comparar em benchmark."""
    tipo = dataset.tipo_tarefa
    input_dim = int(dataset.X.shape[1])

    if tipo == "regressao":
        return [
            (
                "relu_adam",
                ModelConfig(
                    arquitetura=[input_dim, 32, 16, 1],
                    ativacao="relu",
                    inicializacao="he",
                    seed=seed,
                    funcao_custo="mse",
                    ativacao_saida="linear",
                ),
                TrainingConfig(
                    epochs=epochs,
                    taxa_aprendizado=0.01,
                    batch_size=32,
                    otimizador="adam",
                    paciencia=35,
                    l2_lambda=1e-4,
                    gradient_clip=1.0,
                    verbose=False,
                ),
            ),
            (
                "tanh_adam_reg",
                ModelConfig(
                    arquitetura=[input_dim, 24, 12, 1],
                    ativacao="tanh",
                    inicializacao="xavier",
                    seed=seed + 1,
                    funcao_custo="mse",
                    ativacao_saida="linear",
                ),
                TrainingConfig(
                    epochs=epochs,
                    taxa_aprendizado=0.008,
                    batch_size=24,
                    otimizador="adam",
                    paciencia=35,
                    l2_lambda=1e-3,
                    verbose=False,
                ),
            ),
        ]

    output_dim = (
        1 if tipo == "classificacao_binaria" else int(len(set(dataset.y.reshape(-1).tolist())))
    )
    funcao_custo = "binary_crossentropy" if output_dim == 1 else "categorical_crossentropy"

    configuracoes = [
        (
            "relu_adam",
            ModelConfig(
                arquitetura=(
                    [input_dim, 16, 12, output_dim] if output_dim > 1 else [input_dim, 10, 1]
                ),
                ativacao="relu",
                inicializacao="he",
                seed=seed,
                funcao_custo=funcao_custo,
            ),
            TrainingConfig(
                epochs=epochs,
                taxa_aprendizado=0.01,
                batch_size=16,
                otimizador="adam",
                paciencia=25,
                gradient_clip=1.0,
                verbose=False,
            ),
        ),
        (
            "tanh_adam_reg",
            ModelConfig(
                arquitetura=[input_dim, 12, output_dim] if output_dim > 1 else [input_dim, 8, 1],
                ativacao="tanh",
                inicializacao="xavier",
                seed=seed + 1,
                funcao_custo=funcao_custo,
            ),
            TrainingConfig(
                epochs=epochs,
                taxa_aprendizado=0.01,
                batch_size=12 if output_dim > 1 else 16,
                otimizador="adam",
                paciencia=25,
                l2_lambda=1e-3,
                dropout=0.1 if output_dim > 1 else 0.0,
                verbose=False,
            ),
        ),
    ]

    if output_dim == 1:
        configuracoes.append(
            (
                "sigmoid_sgd",
                ModelConfig(
                    arquitetura=[input_dim, 8, 1],
                    ativacao="sigmoid",
                    inicializacao="xavier",
                    seed=seed + 2,
                    funcao_custo="binary_crossentropy",
                ),
                TrainingConfig(
                    epochs=epochs,
                    taxa_aprendizado=0.1,
                    batch_size=None,
                    otimizador="sgd",
                    verbose=False,
                ),
            )
        )

    return configuracoes


def _linha_resultado(
    nome_configuracao: str,
    dataset: DatasetBundle,
    seed: int,
    resumo: dict[str, Any],
    avaliacao: dict[str, Any],
) -> dict[str, Any]:
    linha = {
        "nome": nome_configuracao,
        "dataset": dataset.nome,
        "modo": dataset.tipo_tarefa,
        "seed": seed,
        "loss": float(avaliacao["loss"]),
        "mse": float(avaliacao["mse"]),
        "epochs": int(resumo["epocas_executadas"]),
        "otimizador": resumo["otimizador"],
        "batch_size": resumo["batch_size"],
    }

    if dataset.tipo_tarefa == "regressao":
        linha["mae"] = float(avaliacao["mae"])
        linha["rmse"] = float(avaliacao["rmse"])
        linha["r2"] = float(avaliacao["r2"])
    elif dataset.tipo_tarefa == "classificacao_multiclasse":
        metricas = avaliacao["metricas_classificacao"]
        linha["acuracia"] = float(avaliacao["acuracia"])
        linha["f1_macro"] = float(metricas["f1_macro"])
    else:
        metricas = avaliacao["metricas_classificacao"]
        linha["acuracia"] = float(avaliacao["acuracia"])
        linha["f1"] = float(metricas["f1_score"])

    return linha


def agregar_resultados(resultados: list[dict[str, Any]], tipo_tarefa: str) -> list[dict[str, Any]]:
    """Calcula medias, desvios e ranking por configuracao."""
    agrupados: dict[str, list[dict[str, Any]]] = {}
    for linha in resultados:
        agrupados.setdefault(str(linha["nome"]), []).append(linha)

    metrica_principal = metrica_principal_para_tarefa(tipo_tarefa)
    agregados = []
    for nome, linhas in agrupados.items():
        principal = [float(linha[metrica_principal]) for linha in linhas]
        linha_agregada = {
            "nome": nome,
            "dataset": linhas[0]["dataset"],
            "modo": linhas[0]["modo"],
            "runs": len(linhas),
            f"{metrica_principal}_media": statistics.mean(principal),
            f"{metrica_principal}_desvio": (
                statistics.stdev(principal) if len(principal) > 1 else 0.0
            ),
            "loss_media": statistics.mean(float(linha["loss"]) for linha in linhas),
            "mse_media": statistics.mean(float(linha["mse"]) for linha in linhas),
        }
        if tipo_tarefa == "regressao":
            linha_agregada["mae_media"] = statistics.mean(float(linha["mae"]) for linha in linhas)
            linha_agregada["rmse_media"] = statistics.mean(float(linha["rmse"]) for linha in linhas)
        elif tipo_tarefa == "classificacao_multiclasse":
            linha_agregada["f1_macro_media"] = statistics.mean(
                float(linha["f1_macro"]) for linha in linhas
            )
        else:
            linha_agregada["f1_media"] = statistics.mean(float(linha["f1"]) for linha in linhas)
        agregados.append(linha_agregada)

    agregados.sort(key=lambda linha: float(linha[f"{metrica_principal}_media"]), reverse=True)
    for indice, linha in enumerate(agregados, start=1):
        linha["ranking"] = indice
        linha["metrica_principal"] = metrica_principal
    return agregados


def _vencedor_relatorio(relatorio: dict[str, Any]) -> dict[str, Any]:
    """Extrai uma linha curta com o melhor experimento do relatorio."""
    resumo = relatorio["summary"]
    if not resumo:
        return {
            "dataset": relatorio["dataset"],
            "modo": relatorio["tipo_tarefa"],
            "nome": "sem_resultados",
            "metrica_principal": metrica_principal_para_tarefa(relatorio["tipo_tarefa"]),
            "score": 0.0,
        }
    vencedor = resumo[0]
    metrica = str(vencedor["metrica_principal"])
    return {
        "dataset": relatorio["dataset"],
        "modo": relatorio["tipo_tarefa"],
        "nome": vencedor["nome"],
        "metrica_principal": metrica,
        "score": float(vencedor[f"{metrica}_media"]),
        "runs": vencedor["runs"],
    }


def executar_benchmark(
    dataset_nome: str,
    amostras: int,
    seeds: list[int],
    epochs: int,
) -> dict[str, Any]:
    """Executa benchmark com varias seeds e gera relatorio agregado."""
    dataset_base = carregar_dataset(dataset_nome, seed=seeds[0], samples=amostras)
    resultados: list[dict[str, Any]] = []

    for seed in seeds:
        dataset = carregar_dataset(dataset_nome, seed=seed, samples=amostras)
        splits = dividir_treino_validacao_teste(dataset.X, dataset.y, seed=seed)

        for nome, model_config, train_config in configuracoes_para_dataset(dataset, seed, epochs):
            rede = RedeNeural.from_config(model_config)
            resumo = rede.treinar_com_config(
                splits["X_train"],
                splits["y_train"],
                train_config,
                validacao_X=splits["X_val"],
                validacao_y=splits["y_val"],
            )
            avaliacao = avaliar_modelo(rede, splits["X_test"], splits["y_test"])
            resultados.append(_linha_resultado(nome, dataset, seed, resumo, avaliacao))

    summary = agregar_resultados(resultados, dataset_base.tipo_tarefa)
    relatorio = {
        "suite": False,
        "dataset": dataset_base.nome,
        "datasets": [dataset_base.nome],
        "tipo_tarefa": dataset_base.tipo_tarefa,
        "seeds": seeds,
        "raw_results": resultados,
        "summary": summary,
    }
    relatorio["leaderboard"] = [_vencedor_relatorio(relatorio)]
    return relatorio


def executar_suite_benchmark(
    dataset_nomes: list[str],
    amostras: int,
    seeds: list[int],
    epochs: int,
) -> dict[str, Any]:
    """Executa benchmarks para varios datasets e concatena os resultados."""
    reports = [
        executar_benchmark(nome, amostras=amostras, seeds=seeds, epochs=epochs)
        for nome in dataset_nomes
    ]
    raw_results = [linha for report in reports for linha in report["raw_results"]]
    summary = [linha for report in reports for linha in report["summary"]]
    leaderboard = [_vencedor_relatorio(report) for report in reports]

    return {
        "suite": True,
        "dataset": "suite",
        "datasets": dataset_nomes,
        "tipo_tarefa": (
            "misto"
            if len({report["tipo_tarefa"] for report in reports}) > 1
            else reports[0]["tipo_tarefa"]
        ),
        "seeds": seeds,
        "reports": reports,
        "raw_results": raw_results,
        "summary": summary,
        "leaderboard": leaderboard,
    }


def _formatar_numero(valor: Any) -> str:
    """Formata valores numericos para relatorios humanos."""
    if isinstance(valor, float):
        return f"{valor:.4f}"
    return str(valor)


def _tabela_markdown(colunas: list[str], linhas: list[dict[str, Any]]) -> str:
    """Monta uma tabela Markdown simples."""
    cabecalho = "| " + " | ".join(colunas) + " |"
    separador = "| " + " | ".join("---" for _ in colunas) + " |"
    corpo = [
        "| " + " | ".join(_formatar_numero(linha.get(coluna, "")) for coluna in colunas) + " |"
        for linha in linhas
    ]
    return "\n".join([cabecalho, separador, *corpo]) if corpo else "\n".join([cabecalho, separador])


def _renderizar_relatorio_individual(relatorio: dict[str, Any]) -> str:
    """Renderiza a secao Markdown de um benchmark individual."""
    resumo = relatorio["summary"]
    colunas = [
        "ranking",
        "nome",
        "runs",
        f"{metrica_principal_para_tarefa(relatorio['tipo_tarefa'])}_media",
        f"{metrica_principal_para_tarefa(relatorio['tipo_tarefa'])}_desvio",
        "loss_media",
        "mse_media",
    ]
    if relatorio["tipo_tarefa"] == "regressao":
        colunas.extend(["mae_media", "rmse_media"])
    elif relatorio["tipo_tarefa"] == "classificacao_multiclasse":
        colunas.append("f1_macro_media")
    else:
        colunas.append("f1_media")

    linhas = [
        f"## Dataset `{relatorio['dataset']}`",
        "",
        f"- Tipo de tarefa: `{relatorio['tipo_tarefa']}`",
        f"- Seeds: `{', '.join(str(seed) for seed in relatorio['seeds'])}`",
        f"- Total de execucoes: `{len(relatorio['raw_results'])}`",
        "",
        _tabela_markdown(colunas, resumo),
    ]
    return "\n".join(linhas)


def gerar_relatorio_markdown(relatorio: dict[str, Any]) -> str:
    """Converte um relatorio de benchmark em Markdown legivel para docs ou artefatos."""
    if relatorio.get("suite"):
        linhas = [
            "# Relatorio de Benchmark",
            "",
            "Esta suite executa varios datasets em sequencia. "
            "Os scores entre tarefas diferentes nao sao diretamente comparaveis, "
            "entao o foco e observar o melhor modelo dentro de cada dataset.",
            "",
            "## Vencedores por dataset",
            "",
            _tabela_markdown(
                ["dataset", "modo", "nome", "metrica_principal", "score", "runs"],
                relatorio["leaderboard"],
            ),
            "",
        ]
        for report in relatorio["reports"]:
            linhas.extend([_renderizar_relatorio_individual(report), ""])
        return "\n".join(linhas).strip() + "\n"

    return _renderizar_relatorio_individual(relatorio).strip() + "\n"
