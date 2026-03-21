# Forward e Backward Passo a Passo

> Pagina gerada automaticamente a partir do notebook oficial do projeto.

- Arquivo fonte: `notebooks/01_forward_backward.ipynb`

Este notebook mostra como a rede processa a entrada, gera ativacoes e propaga o gradiente de volta.

## Celula 3

```python
import numpy as np
from rede_neural_do_zero import DataUtils, RedeNeural

X, y = DataUtils.gerar_xor_dataset()
rede = RedeNeural([2, 4, 1], ativacao="sigmoid", seed=42)
ativacoes, z_values = rede._forward_propagation(X)
print("Shapes das ativacoes:", [a.shape for a in ativacoes])
print("Saida inicial:\n", ativacoes[-1])
```

## Celula 4

```python
ativacoes_treino, z_values_treino, _ = rede._forward_pass(X, treino=True, dropout=0.0)
grad_pesos, grad_biases = rede._backward_propagation(y, ativacoes_treino, z_values_treino)
print("Shape dW:", [g.shape for g in grad_pesos])
print("Shape db:", [g.shape for g in grad_biases])
```

Experimente alterar a arquitetura ou a ativacao oculta para observar como os gradientes mudam.
