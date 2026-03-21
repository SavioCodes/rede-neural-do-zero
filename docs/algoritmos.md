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
4. a camada final usa sigmoid

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
- o gradiente da saida muda conforme a funcao de custo selecionada

## Atualizacao de parametros

Depois do backpropagation, a rede aplica um passo de otimizacao.

### SGD

No modo mais simples, o codigo faz:

```text
W = W - learning_rate * dW
b = b - learning_rate * db
```

Esse fluxo e direto e bom para estudar a ideia central de gradiente descendente.

### Adam

Quando `otimizador="adam"`, a implementacao mantem duas medias moveis:

```text
m = beta1 * m + (1 - beta1) * gradiente
v = beta2 * v + (1 - beta2) * gradiente^2
```

Depois, aplica a correcao de vies e atualiza os parametros:

```text
param = param - learning_rate * m_corrigido / (sqrt(v_corrigido) + epsilon)
```

Na pratica, isso costuma acelerar o aprendizado e estabilizar treinos em mini-batch.

## Funcoes de custo

O modelo suporta duas perdas:

### Binary cross-entropy

Usada por padrao, porque combina melhor com a saida sigmoide para classificacao binaria.

```text
loss = -media(y * log(p) + (1 - y) * log(1 - p))
```

Nesse caso, o gradiente da saida fica simples:

```text
delta_saida = y_pred - y_true
```

### MSE

Mantida como opcao didatica para comparacao.

```text
loss = media((y_true - y_pred)^2)
```

Aqui o gradiente da camada de saida ainda precisa multiplicar pela derivada da sigmoid.

## Treinamento

O metodo `treinar()` executa:

1. validacao de `X`, `y` e dados de validacao
2. divisao opcional em mini-batches
3. forward propagation
4. backward propagation
5. atualizacao de parametros com `sgd` ou `adam`
6. registro de historico de treino
7. registro opcional de historico de validacao

O metodo retorna um resumo com as metricas finais do treinamento.

### Early stopping

Quando `paciencia` e informada, o treino monitora a perda de validacao.
Se nao houver validacao, monitora a perda de treino.

Isso permite:

- interromper treinos que pararam de melhorar
- reduzir overfitting em exemplos com validacao
- restaurar os melhores pesos observados durante o processo

### Batch size

O parametro `batch_size` controla quantas amostras entram em cada atualizacao:

- `None` usa todas as amostras de uma vez
- valores menores ativam mini-batch training
- lotes menores costumam combinar bem com `Adam`

## Inicializacao

As opcoes de inicializacao disponiveis sao:

### Xavier

```text
limite = sqrt(6 / (fan_in + fan_out))
W ~ Uniform(-limite, limite)
```

### He

```text
W ~ Normal(0, sqrt(2 / fan_in))
```

### Aleatoria simples

```text
W ~ Normal(0, 0.1)
```

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

## Split treino/teste

O split:

- valida `test_size`
- garante que haja pelo menos 1 amostra em treino e em teste
- usa `default_rng(random_state)` para manter reprodutibilidade

## Metricas de classificacao

O modulo `MetricUtils` calcula:

### Matriz de confusao

```text
[[TN, FP],
 [FN, TP]]
```

### Precisao, recall, especificidade e F1-score

As metricas sao derivadas diretamente da matriz de confusao e protegidas contra divisao por zero.

## Persistencia de parametros

O metodo `salvar_parametros()` grava:

- pesos
- biases
- arquitetura
- ativacao
- metodo de inicializacao
- funcao de custo
- seed

O metodo `carregar_parametros()` reconstrui o estado do modelo a partir do arquivo `.npz`.

## Avaliacao deterministica

O script `scripts/evaluate.py` executa um fluxo padronizado:

1. gera dataset sintetico
2. normaliza os dados
3. divide treino, validacao e teste
4. treina uma arquitetura fixa com seed fixa e early stopping
5. calcula metricas
6. salva resumo em JSON e historico em JSONL

Isso permite:

- usar o repositorio em CI
- comparar mudancas sem flutuar tanto entre execucoes
- manter uma trilha simples de experimentos

## O que nao esta implementado ainda

Para manter a documentacao honesta, vale registrar o que ainda nao faz parte do codigo:

- regularizacao L1/L2
- dropout
- multi-class classification

Esses temas sao boas extensoes futuras, mas nao sao apresentados aqui como recursos prontos.
