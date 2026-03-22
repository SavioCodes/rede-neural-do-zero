# Suite de testes

Os testes estao separados por area de responsabilidade:

- `test_funcoes.py`: ativacoes, utilitarios, metricas e IO simples.
- `test_rede_neural.py`: comportamento principal da rede, treino e validacoes.
- `test_avancado.py`: fluxos mais ricos com callbacks, configs e multiclasse.
- `test_cli_regressao.py`: CLI, regressao, checkpoints e datasets empacotados.
- `test_branch_policy.py`: padrao oficial de nomes de branch, CLI e automacao.

Antes de publicar mudancas grandes, rode:

```bash
python -m rede_neural_do_zero verify --build-package
```
