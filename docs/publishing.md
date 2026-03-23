# Publicacao no PyPI

Esta pagina descreve o fluxo oficial de publicacao do projeto no PyPI usando Trusted Publisher com GitHub Actions.

## Estado oficial do repositorio

- nome do pacote: `rede-neural-do-zero`
- workflow de publicacao: `.github/workflows/publish.yml`
- environment do GitHub Actions: `pypi`
- action de upload: `pypa/gh-action-pypi-publish@release/v1`
- validacao pos-upload: instalacao limpa com `pip install rede-neural-do-zero==<versao>` e teste da CLI `rede-neural-do-zero --help`

## Build local antes da release

```bash
python -m rede_neural_do_zero verify --build-package
```

Arquivos esperados em `dist/`:

- `rede_neural_do_zero-<versao>-py3-none-any.whl`
- `rede_neural_do_zero-<versao>.tar.gz`

## Conferindo o status do PyPI

Use a CLI oficial para checar se o projeto ja existe no PyPI e qual configuracao do Trusted Publisher deve ser cadastrada:

```bash
python -m rede_neural_do_zero pypi-status
```

O comando imprime um JSON com:

- URL esperada do projeto no PyPI
- se o pacote ja existe ou nao
- se ainda precisa de `pending publisher`
- owner, repository, workflow e environment esperados pelo upload oficial

## Configuracao unica no PyPI

Na primeira publicacao, o projeto ainda nao existe no PyPI. Nesse caso, e obrigatorio criar um `pending publisher` em:

- <https://pypi.org/manage/account/publishing/>

Preencha exatamente assim:

- `PyPI project name`: `rede-neural-do-zero`
- `Owner`: `SavioCodes`
- `Repository name`: `rede-neural-do-zero`
- `Workflow name`: `publish.yml`
- `Environment name`: `pypi`

Se algum desses campos nao bater com o workflow real, o upload vai falhar com erros como `invalid-pending-publisher` ou `invalid-publisher`.

## Fluxo oficial fim a fim

1. Atualize `pyproject.toml`, `src/__init__.py` e `CHANGELOG.md`.
2. Rode `python -m rede_neural_do_zero verify --build-package`.
3. Revise o draft de release notes gerado a partir do `CHANGELOG.md`.
4. Confirme no PyPI que o `pending publisher` ja foi cadastrado com os campos corretos.
5. Publique a release no GitHub.
6. O workflow `Publish` vai:
   - validar notebooks
   - gerar wheel e sdist
   - publicar no PyPI via Trusted Publisher
   - criar uma venv limpa
   - rodar `pip install rede-neural-do-zero==<versao>`
   - validar import e a CLI oficial

## Teste manual de instalacao limpa

Depois da primeira publicacao bem-sucedida, o smoke test manual fica assim:

```bash
python -m venv .venv-pypi-smoke
source .venv-pypi-smoke/bin/activate
python -m pip install --upgrade pip
python -m pip install rede-neural-do-zero
rede-neural-do-zero --help
```

No PowerShell:

```powershell
python -m venv .venv-pypi-smoke
.\.venv-pypi-smoke\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install rede-neural-do-zero
rede-neural-do-zero --help
```

## Release notes locais

```bash
python -m rede_neural_do_zero release-notes --version v2.4.1
python -m rede_neural_do_zero release-notes --json --output logs/release-notes.md
```

## Troubleshooting rapido

- `invalid-pending-publisher` ou `invalid-publisher`: confira owner, repository, nome do workflow e `environment` cadastrados no PyPI.
- `Non-user identities cannot create new projects`: o nome do projeto no `pending publisher` nao bateu com o `project.name` do `pyproject.toml`.
- workflow `Publish` pulando ou sem permissao: confira se o job esta rodando no environment `pypi` e se o repositório usa o workflow `.github/workflows/publish.yml`.

## Links oficiais

- [Projeto no PyPI](https://pypi.org/project/rede-neural-do-zero/)
- [Releases](https://github.com/SavioCodes/rede-neural-do-zero/releases)
- [Tags](https://github.com/SavioCodes/rede-neural-do-zero/tags)
