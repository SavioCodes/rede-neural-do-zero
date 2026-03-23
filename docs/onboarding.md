# Onboarding

Esta pagina e o ponto de entrada para quem quer contribuir sem se perder no repositorio.

## Caminho mais seguro para comecar

1. leia o [README](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/README.md)
2. rode `python -m rede_neural_do_zero verify --build-package`
3. consulte a [CLI oficial](./cli.md) e a [estrutura do repositorio](./project-structure.md)
4. escolha uma issue com label `good first issue`, `help wanted` ou `onboarding`
5. abra uma branch curta a partir de `develop`

## Labels para quem esta chegando agora

- `good first issue`: tarefas pequenas, com baixo risco e boa para primeira contribuicao
- `help wanted`: tarefas abertas para colaboracao
- `onboarding`: tarefas que ajudam a entender a base do projeto ou a documentacao
- `docs`: ajustes em guias, exemplos, wiki e documentacao web

## Fluxo recomendado

```bash
python -m rede_neural_do_zero check-branch --name docs/improve-onboarding --target develop
python -m rede_neural_do_zero verify --build-package
python -m rede_neural_do_zero pr-summary --head docs/improve-onboarding --base develop
```

## Onde pedir ajuda

- [SUPPORT.md](https://github.com/SavioCodes/rede-neural-do-zero/blob/main/SUPPORT.md)
- [Issues](https://github.com/SavioCodes/rede-neural-do-zero/issues)
- [Wiki](https://github.com/SavioCodes/rede-neural-do-zero/wiki)
- [Docs publicadas](https://saviocodes.github.io/rede-neural-do-zero/)

## Se a contribuicao tocar release ou governanca

Antes de abrir PR:

- atualize docs e exemplos afetados
- rode `python -m rede_neural_do_zero release-check`
- confira o roadmap da versao atual
- use o template de release PR quando a mudanca promover versao
