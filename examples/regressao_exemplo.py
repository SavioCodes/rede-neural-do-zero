#!/usr/bin/env python3
"""Exemplo de regressao com dataset real Diabetes."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import RedeNeural, VisualizationUtils  # noqa: E402
from src.experiments import (
    avaliar_modelo,
    carregar_dataset,
    criar_configs_padrao,
    dividir_treino_validacao_teste,
)  # noqa: E402


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Tipo nao serializavel: {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina a rede em regressao no dataset Diabetes")
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=Path, default=Path("results/diabetes"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    dataset = carregar_dataset("diabetes", seed=args.seed, samples=442)
    splits = dividir_treino_validacao_teste(dataset.X, dataset.y, seed=args.seed)
    model_config, train_config = criar_configs_padrao(
        dataset, seed=args.seed, epochs=args.epochs, verbose=False
    )

    rede = RedeNeural.from_config(model_config)
    resumo = rede.treinar_com_config(
        splits["X_train"],
        splits["y_train"],
        train_config,
        validacao_X=splits["X_val"],
        validacao_y=splits["y_val"],
    )
    avaliacao = avaliar_modelo(rede, splits["X_test"], splits["y_test"])

    args.save_dir.mkdir(parents=True, exist_ok=True)
    (args.save_dir / "summary.json").write_text(
        json.dumps({"training": resumo, "evaluation": avaliacao}, indent=2, default=_json_default),
        encoding="utf-8",
    )

    if not args.no_plots:
        VisualizationUtils.plotar_regressao(
            splits["y_test"],
            avaliacao["predicoes"],
            salvar=str(args.save_dir / "regression.png"),
            mostrar=False,
        )

    print(
        json.dumps({"training": resumo, "evaluation": avaliacao}, indent=2, default=_json_default)
    )


if __name__ == "__main__":
    main()
