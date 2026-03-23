# Security Policy

## Versoes com suporte

| Versao | Suporte |
| --- | --- |
| 2.5.x | Sim |
| 2.4.x | Correcoes criticas apenas |
| anteriores | Nao |

## Como reportar uma vulnerabilidade

Nao abra um issue publico com detalhes exploraveis.

Fluxo preferido:

1. use o canal privado de seguranca do GitHub, quando disponivel, pelo repositorio
2. descreva impacto, passos de reproducao e, se possivel, uma sugestao de mitigacao
3. aguarde confirmacao antes de divulgar publicamente

## O que esperamos no relato

- versao afetada
- arquivo, comando ou fluxo impactado
- impacto esperado
- prova de conceito minima, sem expor segredo real

## Escopo tipico deste projeto

- execucao arbitraria por arquivos de configuracao ou checkpoints
- exposicao indevida de dados sensiveis em logs, docs ou exemplos
- falhas na publicacao automatica, supply chain ou workflow de release
- bypass das regras oficiais de branch, PR ou release
