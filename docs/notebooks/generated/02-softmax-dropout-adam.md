# Softmax, Dropout, Adam e Resume

> Pagina gerada automaticamente a partir do notebook oficial do projeto.

- Arquivo fonte: `notebooks/02_softmax_dropout_adam.ipynb`

Aqui treinamos um problema multiclasse com regularizacao e salvamos um checkpoint completo.

## Celula 3

```python
from pathlib import Path
from rede_neural_do_zero import DataUtils, RedeNeural

X, y = DataUtils.gerar_dataset_multiclasse(n_samples=240, random_state=42)
X, _ = DataUtils.normalizar_dados(X)
X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X, y, test_size=0.25, random_state=42)
rede = RedeNeural([2, 16, 12, 3], ativacao="relu", inicializacao="he", seed=42, funcao_custo="categorical_crossentropy")
resumo = rede.treinar(X_train, y_train, epochs=80, taxa_aprendizado=0.01, batch_size=16, otimizador="adam", dropout=0.1, l2_lambda=1e-3, gradient_clip=1.0, verbose=False)
print(resumo)
```

## Celula 4

```python
checkpoint = Path("../results/notebook-checkpoint.npz")
rede.salvar_checkpoint(str(checkpoint))
rede_nova = RedeNeural([2, 1], ativacao="relu", seed=1, funcao_custo="mse", ativacao_saida="linear")
info = rede_nova.carregar_checkpoint(str(checkpoint))
print(info)
resumo_resume = rede_nova.retomar_treinamento(X_train, y_train, epochs_adicionais=20, verbose=False)
print(resumo_resume["epocas_executadas"], resumo_resume["epocas_nesta_execucao"])
```

Observe como o checkpoint carrega pesos, epoca, historico e estado do Adam para continuar o treino.
