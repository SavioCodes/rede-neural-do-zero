# Suite de testes

Os testes estao separados por area de responsabilidade para facilitar manutencao e leitura.

- `test_funcoes.py`: ativacoes, utilitarios, metricas e IO simples
- `test_rede_neural.py`: comportamento principal da rede, treino e validacoes
- `test_avancado.py`: fluxos mais ricos com callbacks, configs e multiclasse
- `test_cli_regressao.py`: CLI, regressao, checkpoints e datasets empacotados
- `test_branch_policy.py`: padrao oficial de nomes de branch, CLI e automacao

Antes de publicar mudancas grandes, rode:

```bash
python -m rede_neural_do_zero verify --build-package
```

Se voce quiser validar so uma area:

- nucleo da rede: `python -m pytest -q tests/test_rede_neural.py`
- funcoes, datasets e metricas: `python -m pytest -q tests/test_funcoes.py`
- callbacks, configs e fluxos ricos: `python -m pytest -q tests/test_avancado.py`
- CLI e checkpoints: `python -m pytest -q tests/test_cli_regressao.py`
