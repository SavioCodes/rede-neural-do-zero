# rede-neural-do-zero

[![CI](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml/badge.svg)](https://github.com/SavioCodes/rede-neural-do-zero/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-MkDocs%20Material-00897B)](https://saviocodes.github.io/rede-neural-do-zero/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)

Projeto educacional de rede neural do zero com NumPy, organizado como pacote Python com CLI oficial, datasets pequenos, notebooks, benchmark, checkpoints completos e documentacao navegavel.

## O que voce encontra aqui

- implementacao didatica de forward e backward propagation
- classificacao binaria, multiclasse e regressao
- `SGD`, `Adam`, `dropout`, `L2` e `gradient clipping`
- CLI oficial para treinar, avaliar, retomar treino e rodar benchmark
- configs versionadas em JSON, TOML e YAML
- datasets reais pequenos: Iris, Wine e Diabetes
- notebooks, wiki, docs web e referencia de API
- testes, changelog, roadmap, templates e fluxo de release

## Comece por aqui

### 1. Instale o ambiente

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

No PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

### 2. Verifique se esta tudo certo

```bash
python -m rede_neural_do_zero verify --build-package
```

### 3. Rode alguma coisa util

```bash
python -m rede_neural_do_zero example --config configs/example/wine.json
python -m rede_neural_do_zero train --config configs/train/iris.yaml
python -m rede_neural_do_zero evaluate --config configs/evaluate/diabetes.toml
```

## Interface oficial

O caminho oficial do projeto hoje e:

```bash
python -m rede_neural_do_zero
```

Depois de instalar o pacote, os aliases tambem funcionam:

```bash
rede-neural-do-zero --help
rnz --help
```

`python -m src` ainda existe por compatibilidade com material antigo, mas nao e mais o caminho recomendado para uso novo.

## Exemplo rapido em Python

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

print(meta["target_names"])
print(resumo["acuracia_final"])
```

## Como o repositorio esta organizado

Pastas que importam para entender o projeto:

- `src/`: implementacao interna organizada por dominio (`core`, `data`, `training`, `workflows`, `interfaces`)
- `rede_neural_do_zero/`: wrapper publico do pacote instalado
- `configs/`: arquivos prontos para a CLI
- `examples/`: exemplos pequenos e guiados
- `docs/`: fonte da documentacao do site
- `tests/`: suite automatizada
- `experiments/manifests/`: experimentos reproduziveis

Pastas e arquivos que sao artefatos locais e podem ser apagados quando aparecerem:

- `site/`
- `dist/`
- `results/`
- `logs/`
- `.coverage`
- `*.egg-info/`

Para limpar isso de uma vez:

```bash
make clean
```

Se voce quer um mapa mais detalhado:

- [Estrutura do repositorio](./docs/project-structure.md)
- [FAQ sobre a estrutura](./docs/faq.md)

## Documentacao

- [Docs web](https://saviocodes.github.io/rede-neural-do-zero/)
- [Wiki](https://github.com/SavioCodes/rede-neural-do-zero/wiki)
- [Comecando](./docs/getting-started.md)
- [CLI](./docs/cli.md)
- [Datasets](./docs/datasets.md)
- [Tutorial](./docs/tutorial.md)
- [Teoria](./docs/teoria.md)
- [FAQ](./docs/faq.md)

## Projeto oficial

- Branch estavel: `main`
- Branch de integracao: `develop`
- Prefixos recomendados: `feat/*`, `fix/*`, `docs/*`, `chore/*`, `hotfix/*`, `release/*`
- Fluxo de PR recomendado: `feat/fix/docs/chore -> develop`, `hotfix/release -> main`, `develop -> main`, `main -> develop`
- Protecao oficial ativa em `main` e `develop` com checks obrigatorios
- `CODEOWNERS` oficial cobrindo docs, nucleo, CLI e workflows
- Labels de PR aplicadas automaticamente pelo prefixo da branch
- Hotfix em `main` agora gera PR automatico de sincronizacao para `develop`
- Release PR tem template oficial separado
- Issues: <https://github.com/SavioCodes/rede-neural-do-zero/issues>
- Releases: <https://github.com/SavioCodes/rede-neural-do-zero/releases>
- Tags: <https://github.com/SavioCodes/rede-neural-do-zero/tags>
- Wiki: <https://github.com/SavioCodes/rede-neural-do-zero/wiki>
- Contribuicao: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Roadmap: [ROADMAP.md](./ROADMAP.md)
- Changelog: [CHANGELOG.md](./CHANGELOG.md)
- Governanca: [docs/governance.md](./docs/governance.md)

## Licenca

MIT. Veja [LICENSE](./LICENSE).
