# 📁 Dados (Data)

Esta pasta contém os datasets utilizados nos exemplos da rede neural.

## 📊 Datasets Disponíveis

### 1. `xor_dataset.csv`
Dataset clássico do problema XOR (OR Exclusivo).

**Formato:**
- `x1`: Primeira entrada (0 ou 1)
- `x2`: Segunda entrada (0 ou 1)
- `y`: Saída esperada (0 ou 1)

**Características:**
- 4 amostras
- 2 features de entrada
- 1 target binário
- Problema não-linearmente separável

### 2. `mnist_sample.csv` (Planejado)
Amostra do dataset MNIST com dígitos manuscritos.

**Características planejadas:**
- 1000 amostras (100 de cada dígito 0-9)
- 784 features (pixels 28x28)
- 10 classes (dígitos 0-9)

## 🔧 Como Usar

### Carregar XOR
```python
from src.utils import DataUtils
X, y = DataUtils.gerar_xor_dataset()
```

### Gerar Dataset Sintético
```python
from src.utils import DataUtils
X, y = DataUtils.gerar_dataset_classificacao(n_samples=1000)
```

### Carregar CSV Personalizado
```python
from src.utils import FileUtils
dados = FileUtils.carregar_csv('data/meu_dataset.csv')
```

## 📈 Criando Seus Próprios Datasets

### Formato CSV Recomendado
```csv
feature1,feature2,feature3,target
1.2,3.4,5.6,0
2.1,4.3,6.5,1
...
```

### Dicas:
1. **Normalização**: Sempre normalize suas features
2. **Separação**: Divida em treino/validação/teste
3. **Balanceamento**: Verifique se as classes estão balanceadas
4. **Ruído**: Considere adicionar ruído para robustez

## 🎯 Datasets Recomendados para Teste

### Classificação Binária:
- Ionosphere
- Breast Cancer Wisconsin
- Sonar
- Heart Disease

### Classificação Multi-classe:
- Iris
- Wine
- Digits
- CIFAR-10 (simplificado)

### Regressão:
- Boston Housing
- California Housing
- Diabetes

---
Autor: [Sávio](https://github.com/SavioCodes)
