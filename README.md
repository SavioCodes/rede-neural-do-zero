# rede-neural-do-zero

## Visao Geral
Projeto educacional de rede neural implementada do zero em Python.

## Status do Projeto
| Item | Valor |
|:--|:--|
| Maturidade | Educacional funcional |
| Tipo | Projeto de estudo |
| Ultima atualizacao relevante | 2026-02 |

## Stack
| Camada | Tecnologias |
|:--|:--|
| Runtime | Python 3.8+ |
| Computacao numerica | NumPy |
| Apoio analitico | pandas, matplotlib |
| Testes | unittest/pytest |

## Estrutura
- `src/`: implementacao da rede e utilitarios.
- `tests/`: testes automatizados.
- `examples/`: exemplos praticos.
- `docs/`: materiais de apoio.

## Como Executar
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Testes
```bash
pytest -q
```

## CI
Workflow padronizado em `.github/workflows/ci.yml`.

## Deploy
Projeto educacional, sem URL publica fixa.

## Roadmap
- aumentar cobertura de testes de treinamento
- adicionar benchmarks de desempenho
- expandir exemplos didaticos

## Licenca
MIT (`LICENSE`).