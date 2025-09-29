# 🔬 Detalhes dos Algoritmos

Este documento fornece uma visão detalhada dos algoritmos implementados na rede neural.

## 🧮 Algoritmo de Forward Propagation

### Pseudocódigo
```
função forward_propagation(X):
    ativações = [X]  // Primeira ativação é a entrada
    z_values = []
    
    para cada camada l de 1 até L:
        // Calcular soma ponderada
        z = ativações[l-1] × W[l] + b[l]
        z_values.adicionar(z)
        
        // Aplicar função de ativação
        a = função_ativação(z)
        ativações.adicionar(a)
    
    retornar ativações, z_values
```

### Implementação Python Simplificada
```python
def forward_propagation(self, X):
    ativacoes = [X]
    z_values = []
    
    for i in range(self.num_camadas - 1):
        z = np.dot(ativacoes[i], self.pesos[i]) + self.biases[i]
        z_values.append(z)
        a = self.funcao_ativacao(z)
        ativacoes.append(a)
    
    return ativacoes, z_values
```

### Complexidade
- **Temporal**: O(n × m × h), onde n = amostras, m = features, h = neurônios
- **Espacial**: O(n × h) para armazenar ativações

## ⬅️ Algoritmo de Backpropagation

### Pseudocódigo Detalhado
```
função backpropagation(X, y, ativações, z_values):
    m = número_de_amostras
    gradientes_W = []
    gradientes_b = []
    
    // Erro da última camada
    delta = (ativações[-1] - y)
    
    // Backpropagation das camadas (de trás para frente)
    para l de L-1 até 0:
        // Gradientes da camada atual
        dW = (1/m) × ativações[l]ᵀ × delta
        db = (1/m) × soma(delta, axis=0)
        
        gradientes_W.inserir_no_início(dW)
        gradientes_b.inserir_no_início(db)
        
        // Delta para camada anterior (se não for a primeira)
        se l > 0:
            delta = (delta × W[l]ᵀ) ⊙ derivada_ativação(z_values[l-1])
    
    retornar gradientes_W, gradientes_b
```

### Derivação Matemática

#### Regra da Cadeia
Para a camada de saída L:
```
∂C/∂W⁽ᴸ⁾ = ∂C/∂a⁽ᴸ⁾ × ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ × ∂z⁽ᴸ⁾/∂W⁽ᴸ⁾
```

Onde:
- `∂C/∂a⁽ᴸ⁾ = a⁽ᴸ⁾ - y` (para MSE)
- `∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = f'(z⁽ᴸ⁾)` (derivada da ativação)
- `∂z⁽ᴸ⁾/∂W⁽ᴸ⁾ = a⁽ᴸ⁻¹⁾` (entrada da camada)

#### Para Camadas Ocultas
```
∂C/∂W⁽ˡ⁾ = δ⁽ˡ⁾ × (a⁽ˡ⁻¹⁾)ᵀ
```

Onde `δ⁽ˡ⁾` é propagado de:
```
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ × δ⁽ˡ⁺¹⁾ ⊙ f'(z⁽ˡ⁾)
```

### Complexidade
- **Temporal**: O(n × m × h) - mesma que forward prop
- **Espacial**: O(h²) para armazenar gradientes

## 🎯 Algoritmo de Treinamento

### Gradiente Descendente Completo
```
função treinar(X, y, epochs, learning_rate):
    inicializar_pesos_aleatoriamente()
    
    para epoch de 1 até epochs:
        // Forward pass
        ativações, z_values = forward_propagation(X)
        
        // Calcular erro
        erro = calcular_erro(y, ativações[-1])
        
        // Backward pass
        grad_W, grad_b = backpropagation(X, y, ativações, z_values)
        
        // Atualizar parâmetros
        para cada camada l:
            W[l] = W[l] - learning_rate × grad_W[l]
            b[l] = b[l] - learning_rate × grad_b[l]
        
        // Logging opcional
        se epoch % log_interval == 0:
            imprimir(f"Época {epoch}: Erro = {erro}")
```

### Variações do Gradiente Descendente

#### 1. Batch Gradient Descent
- Usa todo o dataset para cada update
- Convergência estável, mas lenta
- Implementação padrão neste projeto

#### 2. Stochastic Gradient Descent (SGD)
```python
def sgd_treinar(self, X, y, epochs, learning_rate):
    for epoch in range(epochs):
        # Embaralhar dados
        indices = np.random.permutation(len(X))
        
        for i in indices:
            # Uma amostra por vez
            x_sample = X[i:i+1]
            y_sample = y[i:i+1]
            
            # Forward + backward + update
            self._update_weights(x_sample, y_sample, learning_rate)
```

#### 3. Mini-batch Gradient Descent
```python
def minibatch_treinar(self, X, y, epochs, learning_rate, batch_size=32):
    for epoch in range(epochs):
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            self._update_weights(batch_X, batch_y, learning_rate)
```

## 🎲 Algoritmos de Inicialização

### 1. Inicialização Xavier/Glorot
```python
def inicializacao_xavier(entrada_size, saida_size):
    limite = np.sqrt(6.0 / (entrada_size + saida_size))
    return np.random.uniform(-limite, limite, (entrada_size, saida_size))
```

**Justificativa**: Mantém a variância das ativações constante através das camadas.

### 2. Inicialização He
```python
def inicializacao_he(entrada_size, saida_size):
    return np.random.randn(entrada_size, saida_size) * np.sqrt(2.0 / entrada_size)
```

**Justificativa**: Adequada para ReLU, considera apenas as conexões de entrada.

### 3. Inicialização Zero (❌ Não recomendada)
```python
# PROBLEMA: Todos os neurônios aprendem a mesma função
W = np.zeros((entrada_size, saida_size))
```

## 📊 Algoritmos de Avaliação

### Cálculo de Métricas
```python
def calcular_metricas(y_true, y_pred, limiar=0.5):
    # Converter probabilidades para classes
    y_pred_bin = (y_pred >= limiar).astype(int)
    
    # Matriz de confusão
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    tn = np.sum((y_true == 0) & (y_pred_bin == 0))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))
    
    # Métricas derivadas
    acuracia = (tp + tn) / (tp + tn + fp + fn)
    precisao = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0
    
    return {
        'acuracia': acuracia,
        'precisao': precisao,
        'recall': recall,
        'f1_score': f1
    }
```

## 🔧 Algoritmos de Normalização

### 1. Z-Score (Padronização)
```python
def normalizacao_zscore(X):
    media = np.mean(X, axis=0)
    desvio = np.std(X, axis=0)
    X_norm = (X - media) / (desvio + 1e-8)  # Evitar divisão por zero
    return X_norm, {'media': media, 'desvio': desvio}
```

### 2. Min-Max Scaling
```python
def normalizacao_minmax(X):
    minimo = np.min(X, axis=0)
    maximo = np.max(X, axis=0)
    X_norm = (X - minimo) / (maximo - minimo + 1e-8)
    return X_norm, {'min': minimo, 'max': maximo}
```

### 3. Normalização Robusta
```python
def normalizacao_robusta(X):
    mediana = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    X_norm = (X - mediana) / (iqr + 1e-8)
    return X_norm, {'mediana': mediana, 'iqr': iqr}
```

## ⚡ Otimizações Implementadas

### 1. Clipping de Gradientes (Previne Exploding)
```python
def clip_gradientes(gradientes, max_norm=5.0):
    for i, grad in enumerate(gradientes):
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            gradientes[i] = grad * (max_norm / norm)
    return gradientes
```

### 2. Estabilização Numérica na Sigmoid
```python
def sigmoid_estavel(x):
    # Evita overflow clippando valores extremos
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))
```

### 3. Detecção de Convergência
```python
def convergiu(historico_erro, janela=50, tolerancia=1e-6):
    if len(historico_erro) < janela:
        return False
    
    erro_recente = historico_erro[-janela:]
    variacao = np.std(erro_recente)
    
    return variacao < tolerancia
```

## 📈 Algoritmos de Análise

### 1. Análise de Gradientes
```python
def analisar_gradientes(self):
    normas_grad = []
    for grad in self.gradientes_recentes:
        norma = np.linalg.norm(grad)
        normas_grad.append(norma)
    
    return {
        'media': np.mean(normas_grad),
        'desvio': np.std(normas_grad),
        'max': np.max(normas_grad),
        'min': np.min(normas_grad)
    }
```

### 2. Análise de Ativações
```python
def analisar_ativacoes(self, X):
    ativacoes, _ = self.forward_propagation(X)
    
    estatisticas = []
    for i, ativacao in enumerate(ativacoes[1:]):  # Pular entrada
        stats = {
            'camada': i + 1,
            'media': np.mean(ativacao),
            'desvio': np.std(ativacao),
            'zeros': np.sum(ativacao == 0) / ativacao.size,  # Para ReLU
            'saturacao': np.sum(ativacao > 0.99) / ativacao.size  # Para sigmoid
        }
        estatisticas.append(stats)
    
    return estatisticas
```

## 🚀 Algoritmos de Aceleração (Futuras Implementações)

### 1. Adam Optimizer
```python
def adam_update(self, gradientes, m, v, t, lr=0.001, beta1=0.9, beta2=0.999):
    """
    m: primeira estimativa de momento
    v: segunda estimativa de momento  
    t: passo de tempo
    """
    for i, grad in enumerate(gradientes):
        # Atualizar momentos
        m[i] = beta1 * m[i] + (1 - beta1) * grad
        v[i] = beta2 * v[i] + (1 - beta2) * (grad ** 2)
        
        # Correção de bias
        m_corrected = m[i] / (1 - beta1 ** t)
        v_corrected = v[i] / (1 - beta2 ** t)
        
        # Atualização de parâmetros
        self.parametros[i] -= lr * m_corrected / (np.sqrt(v_corrected) + 1e-8)
```

### 2. Learning Rate Scheduling
```python
def schedule_learning_rate(epoch, lr_inicial, metodo='step'):
    if metodo == 'step':
        return lr_inicial * (0.1 ** (epoch // 30))
    elif metodo == 'exponential':
        return lr_inicial * (0.95 ** epoch)
    elif metodo == 'cosine':
        return lr_inicial * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
```

---
**Autor**: [Sávio](https://github.com/SavioCodes)
**Última atualização**: Janeiro 2025
