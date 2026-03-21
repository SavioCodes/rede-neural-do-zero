# Rede Neural do Zero

<div class="hero">
  <div class="hero-copy">
    <p class="eyebrow">NumPy + didatica + engenharia real</p>
    <h1>Uma rede neural do zero que ensina, roda e publica.</h1>
    <p class="hero-text">
      Este projeto foi evoluido para servir ao mesmo tempo como material de estudo,
      portfolio tecnico e pacote Python utilizavel via CLI.
    </p>
    <div class="hero-actions">
      <a class="md-button md-button--primary" href="./getting-started/">Comecar agora</a>
      <a class="md-button" href="./cli/">Ver CLI</a>
      <a class="md-button" href="./notebooks/">Abrir notebooks</a>
      <a class="md-button" href="./project/">Ver projeto oficial</a>
    </div>
  </div>
</div>

## O que o projeto entrega

<div class="grid cards" markdown>

- :material-brain:
  **Rede do zero**

  Forward, backward, inicializacao de pesos, regularizacao, mini-batch, `SGD` e `Adam`.

- :material-shape:
  **Multiclasse e regressao**

  `softmax` com `categorical_crossentropy` para classificacao e saida linear com `mse` para regressao.

- :material-database:
  **Datasets reais pequenos**

  Iris, Wine e Diabetes empacotados junto do projeto para estudo sem depender de download externo.

- :material-restore:
  **Checkpoint completo**

  Salva pesos, biases, historico, estado do otimizador, epoca e config para retomar o treino.

- :material-console:
  **CLI oficial**

  `train`, `resume`, `evaluate`, `benchmark`, `example`, `build-docs`, `build-package` e `verify`.

- :material-file-cog:
  **Configs versionadas**

  Arquivos em JSON, TOML e YAML para repetir treinos, benchmarks e avaliacoes sem depender de flags manuais.

- :material-notebook:
  **Material didatico**

  Notebooks guiados, teoria, tutorial, referencia de API e docs web navegavel com MkDocs Material.

</div>

## Fluxo recomendado

1. Instale o projeto com `pip install -e .[dev]`.
2. Rode um exemplo com `python -m src example --config configs/example/wine.json`.
3. Treine com `python -m src train --config configs/train/iris.yaml`.
4. Compare configuracoes com `python -m src benchmark --config configs/benchmark/suite.yaml`.
5. Valide tudo com `python -m src verify --build-package`.

## Destaques da versao atual

- pacote instalavel com entrypoint `rede-neural-do-zero`
- suporte a classificacao binaria, multiclasse e regressao
- benchmark com multiplas `seeds`, media, desvio e ranking
- suite multi-dataset com leaderboard e relatorio Markdown
- referencia de API gerada a partir do codigo com `mkdocstrings`
- notebooks exportados para paginas da documentacao
- manifests versionados para experimentos reproduziveis
- CI, type-check, cobertura e workflow de publicacao
