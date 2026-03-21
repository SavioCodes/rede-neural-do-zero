# Publicacao no PyPI

O projeto foi organizado para funcionar como pacote instalavel de verdade, com build, verificacao e workflow de publicacao.

## Build local

```bash
python -m build
python -m twine check dist/*
```

Arquivos esperados:

- `dist/rede_neural_do_zero-<versao>-py3-none-any.whl`
- `dist/rede_neural_do_zero-<versao>.tar.gz`

## Instalacao local do wheel

```bash
python -m pip install dist/rede_neural_do_zero-*.whl
```

## Publicacao automatizada

O repositório inclui workflow de publicacao no GitHub Actions baseado em Trusted Publishing do PyPI.

Fluxo sugerido:

1. criar ou atualizar a versao em `pyproject.toml`
2. commitar a mudanca
3. criar uma tag, por exemplo `v2.0.0`
4. publicar a release no GitHub
5. deixar o workflow publicar o pacote no PyPI

## Instalacao via pip

Quando a publicacao estiver ativa no PyPI:

```bash
python -m pip install rede-neural-do-zero
```

Ou em modo desenvolvimento:

```bash
python -m pip install -e ".[dev]"
```
