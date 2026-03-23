# Rede Neural do Zero

Projeto educacional de rede neural do zero com NumPy, pensado para estudo, portfolio tecnico e uso como pacote Python com CLI.

## Comece sem se perder

Se voce esta chegando agora, siga esta ordem:

1. [Instalar e rodar](./getting-started.md)
2. [Ver a CLI oficial](./cli.md)
3. [Entender a estrutura do repositorio](./project-structure.md)
4. [Ler a FAQ](./faq.md)

## O que o projeto entrega

<div class="grid cards" markdown>

- :material-brain:
  **Rede do zero**

  Forward, backward, inicializacao de pesos, regularizacao, mini-batch, `SGD` e `Adam`.

- :material-shape:
  **Classificacao e regressao**

  Binario, multiclasse com `softmax` e regressao com saida linear.

- :material-database:
  **Datasets pequenos**

  XOR, binario, multiclasse, regressao, Iris, Wine e Diabetes.

- :material-console:
  **CLI oficial**

  `train`, `resume`, `evaluate`, `benchmark`, `example`, `check-branch`, `build-docs`, `build-package` e `verify`.

- :material-notebook:
  **Material didatico**

  Notebooks, tutorial, teoria, wiki e referencia de API.

- :material-file-check:
  **Projeto organizado**

  Configs versionadas, manifests, testes, changelog, roadmap e releases.

</div>

## Fluxo recomendado

```bash
python -m pip install -e ".[dev]"
python -m rede_neural_do_zero verify --build-package
python -m rede_neural_do_zero example --config configs/example/wine.json
python -m rede_neural_do_zero train --config configs/train/iris.yaml
```

## Navegacao rapida

- Quer usar o projeto: [CLI oficial](./cli.md)
- Quer entender dados e tarefas: [Datasets](./datasets.md)
- Quer aprender a teoria: [Teoria](./teoria.md)
- Quer ver exemplos guiados: [Notebooks](./notebooks.md)
- Quer contribuir: [Projeto oficial](./project.md)

## Atalhos oficiais

<div class="grid cards quick-links" markdown>

- :material-web:
  **Docs publicadas**

  Navegue pela versao online oficial do projeto.

  [Abrir site](https://saviocodes.github.io/rede-neural-do-zero/)

- :material-book-open-page-variant:
  **Wiki**

  Veja paginas rapidas de onboarding, uso e organizacao.

  [Abrir wiki](https://github.com/SavioCodes/rede-neural-do-zero/wiki)

- :material-tag-outline:
  **Releases**

  Acompanhe versoes, tags e notas oficiais de publicacao.

  [Abrir releases](https://github.com/SavioCodes/rede-neural-do-zero/releases)

- :material-api:
  **API**

  Entre direto na referencia tecnica da CLI, modelos e utilitarios.

  [Abrir API](./api/index.md)

</div>
