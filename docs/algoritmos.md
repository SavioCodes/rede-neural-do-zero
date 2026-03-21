# Detalhes dos Algoritmos

Este documento descreve os algoritmos que realmente aparecem na implementacao atual do projeto.

## Forward propagation

Para cada camada:

```text
z = a_anterior @ W + b
a = ativacao(z)
```

Fluxo adotado:

1. a entrada `X` vira a primeira ativacao
2. cada camada calcula `z`
3. camadas ocultas usam a ativacao configurada
4. a camada final usa `sigmoid` para binario ou `softmax` para multiclasse

## Backpropagation

Depois do forward, o modelo calcula gradientes para todos os pesos e biases.

Estrutura geral:

```text
delta_saida = gradiente_da_loss_na_saida
dW = a_anterior.T @ delta / m
db = soma(delta) / m
delta_anterior = (delta @ W.T) * derivada_ativacao(z_anterior)
```

Pontos importantes desta implementacao:

- o codigo pode trabalhar em batch completo ou mini-batches
- os gradientes sao calculados do fim para o inicio
- o gradiente das camadas ocultas depende da derivada da ativacao escolhida
- o gradiente da saida muda conforme a combinacao `ativacao_saida + funcao_custo`

## Camada de Saida

### Binario

```text
saida = sigmoid(z)
loss = binary_crossentropy ou mse
```

### Multiclasse

```text
saida = softmax(z)
loss = categorical_crossentropy
```

Quando usamos `softmax` com `categorical_crossentropy`, o gradiente final fica especialmente limpo:

```text
delta_saida = y_pred - y_true
```

## Atualizacao de Parametros

Depois do backpropagation, a rede aplica um passo de otimizacao.

### SGD

```text
W = W - learning_rate * dW
b = b - learning_rate * db
```

### Adam

```text
m = beta1 * m + (1 - beta1) * gradiente
v = beta2 * v + (1 - beta2) * gradiente^2
param = param - learning_rate * m_corrigido / (sqrt(v_corrigido) + epsilon)
```

Na pratica, o `Adam` costuma acelerar o aprendizado e estabilizar treinos em mini-batch.

## Regularizacao

### L2

O termo de regularizacao soma o quadrado dos pesos:

```text
loss_total = loss_base + lambda / (2m) * soma(W^2)
```

Isso penaliza pesos muito grandes e ajuda a reduzir overfitting.

### Dropout

Durante o treino, o modelo aplica mascaras aleatorias nas ativacoes ocultas:

```text
a_dropout = a * mascara / keep_prob
```

Aqui usamos a versao invertida do dropout, entao a escala ja fica corrigida no treino.

### Gradient clipping

Antes da atualizacao, cada gradiente pode ser limitado por norma:

```text
if ||g|| > limite:
    g = g * limite / ||g||
```

Isso ajuda a estabilizar treinos em arquiteturas mais profundas ou agressivas.

## Callbacks

O treinamento suporta callbacks reutilizaveis.

### `EarlyStopping`

- monitora `loss` ou `val_loss`
- acompanha paciencia e `min_delta`
- pode restaurar os melhores pesos

### `History`

- guarda os logs de epoca em memoria

### `CSVLogger`

- salva logs de treino em CSV

### `ModelCheckpoint`

- salva os pesos quando a metrica monitorada melhora

## Treinamento

O metodo `treinar()` executa:

1. validacao de `X`, `y` e dados de validacao
2. divisao opcional em mini-batches
3. forward propagation
4. backpropagation
5. regularizacao e gradient clipping
6. atualizacao de parametros com `sgd` ou `adam`
7. registro de historico de treino
8. execucao de callbacks
9. registro opcional de historico de validacao

O metodo retorna um resumo com as metricas finais e metadados importantes do treino.

## Configs Declarativas

O projeto tambem oferece:

- `ModelConfig`
- `TrainingConfig`

Essas dataclasses deixam os experimentos mais organizados, mais legiveis e mais faceis de comparar em benchmark e exemplos.

## Normalizacao de dados

### Padrao

```text
X_norm = (X - media) / desvio
```

### Min-max

```text
X_norm = (X - minimo) / (maximo - minimo)
```

### Robusta

```text
X_norm = (X - mediana) / IQR
```

Todas as variacoes tratam divisao por zero de forma segura.

## Metricas de classificacao

O modulo `MetricUtils` cobre:

- matriz de confusao binaria
- matriz de confusao multiclasse
- precisao, recall, especificidade e F1 no caso binario
- accuracy, precision/recall/F1 macro e weighted no caso generico

## Persistencia de parametros

O metodo `salvar_parametros()` grava:

- pesos
- biases
- arquitetura
- ativacao oculta
- ativacao de saida
- metodo de inicializacao
- funcao de custo
- seed

## Avaliacao e benchmark

### `src.workflows.evaluation.run_evaluation`

Executa um fluxo deterministico com dataset, split treino/validacao/teste e gate minimo de score para validacao local ou CI.

### `src.workflows.benchmarking`

Executa comparacoes entre configuracoes, agrega medias e desvios por `seed`, suporta suites multi-dataset e gera relatorios em JSON, CSV e Markdown.
