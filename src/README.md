# Estrutura de `src`

`src/` concentra a implementacao interna do projeto.

O pacote publico recomendado para quem instala a biblioteca e `rede_neural_do_zero/`, mas o trabalho real do repositorio acontece aqui.

## Subpastas principais

- `core/`: implementacao principal da rede neural, ativacoes e fluxo numerico
- `training/`: callbacks e dataclasses de configuracao de treino/modelo
- `data/`: geracao de datasets sinteticos, metricas, plots e utilitarios de arquivos
- `workflows/`: orquestracao de treino, avaliacao e benchmark
- `interfaces/`: CLI oficial e suporte a arquivos de configuracao
- `datasets/`: arquivos CSV empacotados com datasets reais pequenos

## Como ler essa pasta sem se perder

- comece por `core/` se quiser entender a rede neural
- va para `interfaces/` se quiser entender a CLI
- olhe `workflows/` para ver treino, avaliacao e benchmark completos
- use `data/` para datasets sinteticos, metricas e plots

Os modulos antigos como `src.rede_neural`, `src.utils` e `src.cli` continuam existindo como camadas de compatibilidade para nao quebrar exemplos, testes ou usuarios antigos.
