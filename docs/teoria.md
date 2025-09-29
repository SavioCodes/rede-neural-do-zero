# 📚 Fundamentos Teóricos das Redes Neurais

Este documento explica os conceitos teóricos por trás da implementação da rede neural.

## 🧠 Introdução às Redes Neurais

### O que são Redes Neurais?

As redes neurais artificiais são modelos computacionais inspirados no funcionamento do cérebro humano. Elas são compostas por unidades de processamento simples (neurônios artificiais) que trabalham em conjunto para resolver problemas complexos.

### Componentes Básicos

#### 1. Neurônio Artificial (Perceptron)
```
Entradas: x₁, x₂, ..., xₙ
Pesos: w₁, w₂, ..., wₙ
Bias: b
Saída: y = f(∑(xᵢ × wᵢ) + b)
```

#### 2. Função de Ativação
Introduz não-linearidade no modelo:
- **Sigmoid**: σ(x) = 1/(1 + e⁻ˣ)
- **ReLU**: f(x) = max(0, x)
- **Tanh**: f(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)

## 🔄 Forward Propagation (Propagação Direta)

### Processo
1. Os dados de entrada fluem da primeira camada até a saída
2. Cada neurônio calcula uma combinação linear das entradas
3. Aplica uma função de ativação
4. Passa o resultado para a próxima camada

### Matemática
Para uma camada l:
```
z⁽ˡ⁾ = W⁽ˡ⁾ × a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = f(z⁽ˡ⁾)
```

Onde:
- `z⁽ˡ⁾`: Soma ponderada da camada l
- `W⁽ˡ⁾`: Matriz de pesos da camada l
- `a⁽ˡ⁻¹⁾`: Ativações da camada anterior
- `b⁽ˡ⁾`: Vetor de bias da camada l
- `f`: Função de ativação

## ⬅️ Backpropagation (Retropropagação)

### Objetivo
Calcular os gradientes da função de custo em relação aos pesos e biases, permitindo atualizar os parâmetros para minimizar o erro.

### Algoritmo
1. **Calcular erro da saída**: δ⁽ᴸ⁾ = (a⁽ᴸ⁾ - y) ⊙ f'(z⁽ᴸ⁾)
2. **Propagar erro para trás**: δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ × δ⁽ˡ⁺¹⁾ ⊙ f'(z⁽ˡ⁾)
3. **Calcular gradientes**:
   - ∂C/∂W⁽ˡ⁾ = δ⁽ˡ⁾ × (a⁽ˡ⁻¹⁾)ᵀ
   - ∂C/∂b⁽ˡ⁾ = δ⁽ˡ⁾

### Notação
- `⊙`: Produto elemento por elemento (Hadamard)
- `L`: Índice da última camada
- `δ⁽ˡ⁾`: Erro da camada l
- `f'`: Derivada da função de ativação

## 🎯 Função de Custo

### Erro Quadrático Médio (MSE)
```
C = (1/2m) × ∑ᵢ(yᵢ - ŷᵢ)²
```

### Entropia Cruzada (Cross-Entropy)
```
C = -(1/m) × ∑ᵢ[yᵢ × log(ŷᵢ) + (1-yᵢ) × log(1-ŷᵢ)]
```

## 🔧 Gradiente Descendente

### Atualização dos Parâmetros
```
W⁽ˡ⁾ := W⁽ˡ⁾ - α × ∂C/∂W⁽ˡ⁾
b⁽ˡ⁾ := b⁽ˡ⁾ - α × ∂C/∂b⁽ˡ⁾
```

Onde `α` é a taxa de aprendizado (learning rate).

### Tipos de Gradiente Descendente
1. **Batch**: Usa todo o dataset
2. **Stochastic (SGD)**: Usa uma amostra por vez
3. **Mini-batch**: Usa pequenos grupos de amostras

## ⚖️ Inicialização de Pesos

### Importância
A inicialização adequada dos pesos é crucial para:
- Evitar vanishing/exploding gradients
- Garantir convergência
- Acelerar o treinamento

### Métodos

#### 1. Inicialização Xavier/Glorot
```
W ~ Uniform(-√(6/(nᵢₙ + nₒᵤₜ)), √(6/(nᵢₙ + nₒᵤₜ)))
```
Ideal para sigmoid e tanh.

#### 2. Inicialização He
```
W ~ Normal(0, √(2/nᵢₙ))
```
Ideal para ReLU.

## 🎪 Funções de Ativação Detalhadas

### 1. Sigmoid
**Função**: σ(x) = 1/(1 + e⁻ˣ)
**Derivada**: σ'(x) = σ(x) × (1 - σ(x))
**Características**:
- Saída entre 0 e 1
- Interpretável como probabilidade
- Problema: vanishing gradient

### 2. ReLU (Rectified Linear Unit)
**Função**: f(x) = max(0, x)
**Derivada**: f'(x) = 1 se x > 0, senão 0
**Características**:
- Computacionalmente eficiente
- Sem saturação para valores positivos
- Problema: dead neurons

### 3. Tanh
**Função**: tanh(x) = (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)
**Derivada**: tanh'(x) = 1 - tanh²(x)
**Características**:
- Saída entre -1 e 1
- Zero-centered
- Similar ao sigmoid, mas melhor

## 📊 Métricas de Avaliação

### Classificação Binária

#### Matriz de Confusão
```
             Predito
Real    |  0  |  1  |
--------|-----|-----|
   0    | TN  | FP  |
   1    | FN  | TP  |
```

#### Métricas Derivadas
- **Acurácia**: (TP + TN) / (TP + TN + FP + FN)
- **Precisão**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 × (Precisão × Recall) / (Precisão + Recall)

## 🚫 Problemas Comuns

### 1. Overfitting
**Sintomas**: Boa performance no treino, ruim no teste
**Soluções**:
- Regularização (L1, L2)
- Dropout
- Early stopping
- Mais dados

### 2. Underfitting
**Sintomas**: Performance ruim em treino e teste
**Soluções**:
- Modelo mais complexo
- Mais features
- Menos regularização

### 3. Vanishing Gradient
**Sintomas**: Gradientes muito pequenos nas primeiras camadas
**Soluções**:
- ReLU em vez de sigmoid
- Inicialização adequada
- Batch normalization

### 4. Exploding Gradient
**Sintomas**: Gradientes muito grandes
**Soluções**:
- Gradient clipping
- Taxa de aprendizado menor
- Inicialização adequada

## 📖 Referências Teóricas

1. **Rosenblatt, F. (1958)** - "The Perceptron: A Probabilistic Model"
2. **Rumelhart, D. E. et al. (1986)** - "Learning representations by back-propagating errors"
3. **Glorot, X. & Bengio, Y. (2010)** - "Understanding the difficulty of training deep feedforward neural networks"
4. **He, K. et al. (2015)** - "Delving Deep into Rectifiers"

## 🎓 Para Aprofundar

### Livros
- "Neural Networks and Deep Learning" - Michael Nielsen
- "Deep Learning" - Ian Goodfellow, Yoshua Bengio, Aaron Courville
- "Pattern Recognition and Machine Learning" - Christopher Bishop

### Cursos Online
- CS231n: Convolutional Neural Networks (Stanford)
- Deep Learning Specialization (Coursera)
- Fast.ai Practical Deep Learning

### Implementações de Referência
- Numpy-based implementations
- Micrograd (Andrej Karpathy)
- Neural Networks from Scratch (Harrison Kinsley)

---
**Autor**: [Sávio](https://github.com/SavioCodes)
**Última atualização**: Janeiro 2025
