# Publicacao no PyPI

O projeto foi organizado para funcionar como pacote instalavel de verdade, com build, verificacao e workflow de publicacao.

## Build local

```bash
python -m rede_neural_do_zero build-package --check
```

Arquivos esperados:

- `dist/rede_neural_do_zero-<versao>-py3-none-any.whl`
- `dist/rede_neural_do_zero-<versao>.tar.gz`

## Instalacao local do wheel

```bash
python -m pip install dist/rede_neural_do_zero-*.whl
```

## Publicacao automatizada

O repositorio inclui workflow de publicacao no GitHub Actions baseado em Trusted Publishing do PyPI.

Fluxo sugerido:

1. criar ou atualizar a versao em `pyproject.toml`
2. atualizar `src/__init__.py` e `CHANGELOG.md`
3. validar com `python -m rede_neural_do_zero verify --build-package`
4. commitar a mudanca
5. criar uma tag, por exemplo `v2.2.3`
6. publicar a release no GitHub
7. deixar o workflow publicar o pacote no PyPI

## Links oficiais

- [Releases](https://github.com/SavioCodes/rede-neural-do-zero/releases)
- [Tags](https://github.com/SavioCodes/rede-neural-do-zero/tags)

## Sobre os gates temporarios

Enquanto o repositorio ainda nao estiver com Pages e Trusted Publishing ativados externamente, os workflows ficam protegidos por variaveis de ambiente:

- `GITHUB_PAGES_ENABLED`
- `PYPI_PUBLISH_ENABLED`

Assim o projeto continua com automacao oficial sem falhar por falta de configuracao no GitHub ou no PyPI.

## Instalacao via pip

Quando a publicacao estiver ativa no PyPI:

```bash
python -m pip install rede-neural-do-zero
```

Ou em modo desenvolvimento:

```bash
python -m pip install -e ".[dev]"
```

