# Estrutura de `src`

O codigo-fonte agora esta organizado por responsabilidade:

- `core/`: implementacao principal da rede neural, ativacoes e fluxo numerico.
- `training/`: callbacks e dataclasses de configuracao de treino/modelo.
- `data/`: geracao de datasets sinteticos, metricas, plots e utilitarios de arquivos.
- `workflows/`: orquestracao de treino, avaliacao e benchmark.
- `interfaces/`: CLI oficial e suporte a arquivos de configuracao.
- `datasets/`: arquivos CSV empacotados com datasets reais pequenos.

Os modulos antigos como `src.rede_neural`, `src.utils` e `src.cli` continuam existindo como camadas de compatibilidade para nao quebrar exemplos, testes ou usuarios antigos.
