# Fundamentos Teoricos

Este documento resume a base conceitual por tras da implementacao deste repositorio.

## O que uma rede neural faz

Uma rede neural recebe um vetor de entrada, aplica transformacoes lineares e funcoes nao lineares em camadas sucessivas, e produz uma saida.

Neste projeto, a saida final pode representar:

- uma probabilidade binaria
- uma distribuicao de probabilidades entre multiplas classes

## Componentes principais

### Neuronio artificial

Cada neuronio calcula:

```text
z = x . w + b
a = f(z)
```

Onde:

- `x` e o vetor de entrada
- `w` sao os pesos aprendidos
- `b` e o bias
- `f` e a funcao de ativacao

### Camadas

- camada de entrada: recebe as features
- camadas ocultas: aprendem representacoes intermediarias
- camada de saida: produz a previsao final

## Forward propagation

No forward propagation, os dados passam camada por camada ate a saida:

```text
z(l) = a(l-1) @ W(l) + b(l)
a(l) = f(z(l))
```

Neste projeto:

- as camadas ocultas usam a ativacao escolhida no construtor
- a camada final usa `sigmoid` ou `softmax`

## Backpropagation

Backpropagation e o processo usado para calcular como cada peso contribuiu para o erro final.
Com esses gradientes, o modelo atualiza seus parametros por gradiente descendente.

Em alto nivel:

1. faz o forward
2. calcula a perda na saida
3. propaga esse erro para tras
4. atualiza pesos e biases

## Funcoes de ativacao

### Sigmoid

```text
sigmoid(x) = 1 / (1 + e^(-x))
```

Boa para saida binaria, pois produz valores entre 0 e 1.

### Softmax

Transforma logits em probabilidades que somam 1 por amostra.
Ela e a escolha natural para classificacao multiclasse exclusiva.

### ReLU

```text
relu(x) = max(0, x)
```

Simples e eficiente, muito usada em camadas ocultas.

### Tanh

Tem saida entre -1 e 1 e pode ajudar quando dados centrados em zero fazem sentido.

### Leaky ReLU

Uma variacao da ReLU que mantem gradiente pequeno para valores negativos.

### Linear

Usada como identidade. Neste repositorio ela fica disponivel para experimentos didaticos.

## Inicializacao de pesos

Inicializar bem os pesos ajuda a evitar treinamento instavel.

### Xavier

Boa escolha para sigmoid e tanh.

### He

Boa escolha para ReLU e variantes.

### Aleatoria simples

Util como baseline, mas normalmente menos robusta.

## Funcoes de custo

### Binary cross-entropy

E a perda padrao para classificacao binaria com saida sigmoide.

### Categorical cross-entropy

E a perda mais natural para classificacao multiclasse com softmax.

### MSE

Tambem esta disponivel como opcao didatica para comparacoes.

## Otimizacao

O projeto suporta dois modos principais:

- `SGD`
- `Adam`

`SGD` ajuda a entender a atualizacao basica por gradiente.
`Adam` adiciona medias moveis dos gradientes e costuma convergir melhor em muitos cenarios praticos.

## Regularizacao

### L2

Penaliza pesos muito grandes e ajuda a suavizar a complexidade do modelo.

### Dropout

Desativa aleatoriamente parte das ativacoes ocultas durante o treino para reduzir co-adaptacao.

### Gradient clipping

Limita a norma dos gradientes para evitar atualizacoes explosivas.

## Callbacks

Do ponto de vista didatico, callbacks ajudam a separar responsabilidades:

- `EarlyStopping` cuida da parada antecipada
- `History` armazena logs
- `CSVLogger` salva o historico
- `ModelCheckpoint` salva os melhores pesos

Esse desenho deixa o loop de treino mais organizado e mais proximo de bibliotecas reais.

## Multiclasse e one-hot

Para tarefas multiclasse, o repositório aceita:

- rotulos inteiros, como `0`, `1`, `2`
- rotulos em one-hot

Internamente, a rede converte o formato quando necessario para manter a API pratica.

## Normalizacao de dados

Antes de treinar, normalizar os dados costuma ajudar bastante.

O projeto oferece:

- `padrao`: z-score
- `minmax`: escala para o intervalo `[0, 1]`
- `robusto`: usa mediana e IQR

## Reprodutibilidade

Do ponto de vista didatico, reproducibilidade e parte da explicacao.
Por isso o repositorio usa seeds explicitas e evita depender do estado global do gerador aleatorio do NumPy quando isso nao e necessario.

## Limites intencionais

O projeto nao tenta competir com frameworks de producao.
Ele foi desenhado para ser:

- pequeno o bastante para estudar
- previsivel o bastante para testar
- organizado o bastante para evoluir

## Referencias sugeridas

- Michael Nielsen, Neural Networks and Deep Learning
- Ian Goodfellow, Yoshua Bengio e Aaron Courville, Deep Learning
- Andrej Karpathy, micrograd e materiais introdutorios sobre backpropagation
