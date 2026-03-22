# rede-neural-do-zero

[![CI](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-MkDocs%20Material-00897B)](https://saviocodes.github.io/rede-neural-do-zero/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

Implementacao educacional de rede neural do zero com NumPy, organizada como pacote instalavel com CLI oficial, configs versionadas, notebooks, docs navegavel, datasets reais pequenos e checkpoints completos de treino.

## O que o projeto cobre

- forward e backward propagation
- classificacao binaria, multiclasse e regressao
- `sigmoid`, `relu`, `tanh`, `leaky_relu`, `linear` e `softmax`
- `binary_crossentropy`, `categorical_crossentropy` e `mse`
- mini-batch, `SGD` e `Adam`
- `L2`, `dropout` e `gradient clipping`
- `EarlyStopping`, `History`, `CSVLogger` e `ModelCheckpoint`
- salvar e retomar treino completo
- benchmark com multiplas `seeds`, media, desvio e ranking
- suite multi-dataset com leaderboard e relatorio Markdown
- datasets reais empacotados: Iris, Wine e Diabetes
- CLI com `train`, `resume`, `evaluate`, `benchmark`, `example`, `check-branch`, `build-docs`, `build-package` e `verify`
- configs oficiais em JSON, TOML e YAML
- docs web com MkDocs Material e referencia de API automatica
- notebooks didaticos para estudo
- changelog, roadmap, templates e fluxo de contribuicao

## Instalacao

### Desenvolvimento

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### Como pacote

Instalacao local:

```bash
python -m pip install .
```

Instalacao editavel:

```bash
python -m pip install -e ".[dev]"
```

Quando publicado no PyPI:

```bash
python -m pip install rede-neural-do-zero
```

## Uso rapido

### Pela CLI oficial

```bash
python -m rede_neural_do_zero train --config configs/train/iris.yaml
python -m rede_neural_do_zero resume --checkpoint experiments/runs/iris-baseline/model-checkpoint.npz --dataset iris --epochs 40
python -m rede_neural_do_zero evaluate --config configs/evaluate/diabetes.toml
python -m rede_neural_do_zero benchmark --config configs/benchmark/suite.yaml
python -m rede_neural_do_zero example --config configs/example/wine.json
python -m rede_neural_do_zero check-branch --name feat/add-branch-policy
python -m rede_neural_do_zero verify --build-package
```

### Importando como biblioteca

```python
from rede_neural_do_zero import DataUtils, RedeNeural

X, y, meta = DataUtils.carregar_dataset_iris(normalizar="padrao")

rede = RedeNeural(
    [X.shape[1], 16, 12, 3],
    ativacao="relu",
    inicializacao="he",
    seed=42,
    funcao_custo="categorical_crossentropy",
)

resumo = rede.treinar(
    X,
    y,
    epochs=160,
    taxa_aprendizado=0.01,
    batch_size=16,
    otimizador="adam",
    verbose=False,
)

print(resumo["acuracia_final"])
```

## CLI oficial

Sem instalar script global:

```bash
python -m rede_neural_do_zero --help
```

Compatibilidade:

- `python -m src` continua funcionando para nao quebrar material antigo.
- `python -m rede_neural_do_zero` e o caminho oficial recomendado daqui para frente.

Comandos principais:

```bash
python -m rede_neural_do_zero train --config configs/train/iris.yaml
python -m rede_neural_do_zero resume --checkpoint experiments/runs/iris-baseline/model-checkpoint.npz --dataset iris --epochs 40
python -m rede_neural_do_zero evaluate --config configs/evaluate/diabetes.toml
python -m rede_neural_do_zero benchmark --datasets iris,wine,diabetes --seeds 42,52,62
python -m rede_neural_do_zero example --config configs/example/wine.json
python -m rede_neural_do_zero check-branch --name docs/update-wiki-links
python -m rede_neural_do_zero build-docs --strict
python -m rede_neural_do_zero build-package --check
python -m rede_neural_do_zero verify --build-package
```

Depois de instalar o pacote, tambem funciona:

```bash
rede-neural-do-zero --help
rnz --help
```

## Datasets disponiveis

### Sinteticos

- `xor`
- `binario`
- `multiclasse`
- `regressao`

### Reais empacotados

- `iris`
- `wine`
- `diabetes`

## Notebooks

Os notebooks ficam em `notebooks/`:

- `01_forward_backward.ipynb`
- `02_softmax_dropout_adam.ipynb`
- `03_datasets_reais_e_matriz_confusao.ipynb`

## Estrutura do repositorio

- `src/`: implementacao principal organizada por `core`, `training`, `data`, `workflows` e `interfaces`
- `rede_neural_do_zero/`: wrapper publico do pacote instalado
- `configs/`: configuracoes versionadas para a CLI
- `examples/`: exemplos pequenos e guiados
- `scripts/`: automacoes auxiliares de benchmark, avaliacao e docs
- `docs/`: fonte da documentacao web
- `tests/`: suite automatizada
- `experiments/manifests/`: manifests reproduziveis

Mapa completo:

- [Estrutura do repositorio](./docs/project-structure.md)

## Documentacao

- [Landing page e docs web](https://saviocodes.github.io/rede-neural-do-zero/)
- [Wiki no GitHub](https://github.com/SavioCodes/rede-neural-do-zero/wiki)
- [Referencia de API](./docs/api/index.md)
- [Teoria](./docs/teoria.md)
- [Algoritmos](./docs/algoritmos.md)
- [Tutorial](./docs/tutorial.md)
- [CLI](./docs/cli.md)
- [Publicacao PyPI](./docs/publishing.md)
- [Projeto Oficial](./docs/project.md)
- [Estrutura do Repositorio](./docs/project-structure.md)
- [Modelo de Branches](./docs/branching.md)

## Build e publicacao

Build local:

```bash
python -m rede_neural_do_zero build-package --check
```

O repositorio inclui:

- workflow de CI
- workflow de docs com GitHub Pages
- workflow de publicacao no PyPI via Trusted Publishing
- tags e releases oficiais no GitHub

## Qualidade

```bash
python -m rede_neural_do_zero verify --build-package
```

## Projeto oficial

- Branch estavel: `main`
- Branch de integracao: `develop`
- Prefixos recomendados: `feat/*`, `fix/*`, `docs/*`, `chore/*`, `hotfix/*`, `release/*`
- Releases: <https://github.com/SavioCodes/rede-neural-do-zero/releases>
- Tags: <https://github.com/SavioCodes/rede-neural-do-zero/tags>
- Issues: <https://github.com/SavioCodes/rede-neural-do-zero/issues>
- Wiki: <https://github.com/SavioCodes/rede-neural-do-zero/wiki>
- Contribuicao: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Roadmap: [ROADMAP.md](./ROADMAP.md)
- Changelog: [CHANGELOG.md](./CHANGELOG.md)

## Licenca

MIT. Veja [LICENSE](./LICENSE).

