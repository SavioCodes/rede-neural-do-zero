# 🧠 Rede Neural do Zero

Uma implementação completa de uma rede neural artificial desenvolvida do zero em Python, sem uso de frameworks como TensorFlow ou PyTorch. Projeto educacional para entender os fundamentos do deep learning.

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Teoria e Fundamentos](#teoria-e-fundamentos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Exemplos](#exemplos)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Funcionalidades](#funcionalidades)
- [Contribuindo](#contribuindo)
- [Licença](#licença)
- [Créditos](#créditos)

## 🎯 Sobre o Projeto

Este projeto implementa uma rede neural artificial completamente do zero usando apenas NumPy e bibliotecas básicas do Python. O objetivo é fornecer uma compreensão clara e detalhada de como funcionam as redes neurais internamente, incluindo:

- Forward propagation (propagação direta)
- Backpropagation (retropropagação)
- Funções de ativação
- Otimização de pesos e biases
- Treinamento e validação

## 📚 Teoria e Fundamentos

### O que é uma Rede Neural?

Uma rede neural artificial é um modelo computacional inspirado no funcionamento do cérebro humano. Ela é composta por neurônios artificiais (nós) organizados em camadas:

1. **Camada de Entrada**: Recebe os dados de input
2. **Camadas Ocultas**: Processam as informações
3. **Camada de Saída**: Produz o resultado final

### Forward Propagation

O processo onde os dados fluem da entrada para a saída:

```
z = W·x + b
a = f(z)
```

Onde:
- `z` é a soma ponderada
- `W` são os pesos
- `x` são as entradas
- `b` é o bias
- `f` é a função de ativação
- `a` é a saída ativada

### Backpropagation

Algoritmo usado para treinar a rede, calculando gradientes e atualizando pesos:

```
∂L/∂W = ∂L/∂a · ∂a/∂z · ∂z/∂W
```

### Funções de Ativação Implementadas

1. **Sigmoid**: `σ(x) = 1/(1 + e^(-x))`
2. **ReLU**: `f(x) = max(0, x)`
3. **Tanh**: `f(x) = (e^x - e^(-x))/(e^x + e^(-x))`

## 🚀 Instalação

### Pré-requisitos

- Python 3.7+
- pip

### Instalação das Dependências

```bash
# Clone o repositório
git clone https://github.com/SavioCodes/rede-neural-do-zero.git
cd rede-neural-do-zero

# Instale as dependências
pip install -r requirements.txt
```

## 💻 Uso

### Uso Básico

```python
from src.rede_neural import RedeNeural
import numpy as np

# Criar uma rede neural com 2 entradas, 4 neurônios ocultos e 1 saída
rede = RedeNeural([2, 4, 1])

# Dados de exemplo (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Treinar a rede
rede.treinar(X, y, epochs=1000, taxa_aprendizado=0.1)

# Fazer predições
predicoes = rede.prever(X)
print("Predições:", predicoes)
```

### Executar Exemplo Completo

```bash
python examples/exemplo.py
```

## 📊 Exemplos

### Problema XOR

O exemplo clássico do XOR demonstra a capacidade da rede de aprender padrões não-lineares:

```
Entrada: [0, 0] → Saída Esperada: 0
Entrada: [0, 1] → Saída Esperada: 1
Entrada: [1, 0] → Saída Esperada: 1
Entrada: [1, 1] → Saída Esperada: 0
```

**Resultados Esperados:**
```
Época 0: Erro = 0.2500
Época 100: Erro = 0.0847
Época 500: Erro = 0.0234
Época 1000: Erro = 0.0089

Predições Finais:
[0, 0] → 0.05 (≈ 0)
[0, 1] → 0.94 (≈ 1)
[1, 0] → 0.96 (≈ 1)
[1, 1] → 0.07 (≈ 0)

Acurácia: 97.5%
```

### Classificação com Dataset Personalizado

```python
# Exemplo com dados de classificação binária
from src.utils import gerar_dataset_classificacao

X, y = gerar_dataset_classificacao(n_samples=1000, n_features=2)
rede = RedeNeural([2, 8, 4, 1], ativacao='relu')
rede.treinar(X, y, epochs=2000, taxa_aprendizado=0.01)
```

## 📁 Estrutura do Projeto

```
rede-neural-do-zero/
├── src/
│   ├── rede_neural.py      # Implementação principal da rede neural
│   ├── funcoes_ativacao.py # Funções de ativação e suas derivadas
│   └── utils.py            # Utilitários para dados e visualização
├── data/
│   ├── xor_dataset.csv     # Dataset XOR
│   └── mnist_sample.csv    # Amostra do MNIST
├── examples/
│   ├── exemplo.py          # Exemplo principal de uso
│   ├── xor_exemplo.py      # Exemplo específico do XOR
│   └── classificacao.py    # Exemplo de classificação
├── docs/
│   ├── teoria.md           # Explicação teórica detalhada
│   ├── algoritmos.md       # Detalhes dos algoritmos
│   └── diagramas/          # Diagramas e visualizações
├── tests/
│   ├── test_rede_neural.py # Testes unitários
│   └── test_funcoes.py     # Testes das funções
├── requirements.txt        # Dependências do projeto
├── LICENSE                 # Licença MIT
└── README.md              # Este arquivo
```

## ✨ Funcionalidades

### Implementado ✅

- [x] Rede neural com múltiplas camadas totalmente conectadas
- [x] Forward propagation completa
- [x] Backpropagation com gradiente descendente
- [x] Funções de ativação: Sigmoid, ReLU, Tanh
- [x] Inicialização inteligente de pesos (Xavier/He)
- [x] Métricas de avaliação (erro, acurácia)
- [x] Visualização de treinamento
- [x] Exemplo XOR funcional
- [x] Suporte a datasets personalizados
- [x] Testes unitários
- [x] Documentação completa

### Próximas Funcionalidades 🚧

- [ ] Regularização (L1, L2)
- [ ] Diferentes otimizadores (Adam, RMSprop)
- [ ] Batch normalization
- [ ] Dropout
- [ ] Validação cruzada
- [ ] Suporte a MNIST completo

## 🧪 Testes

Execute os testes para verificar se tudo está funcionando:

```bash
python -m pytest tests/ -v
```

## 🤝 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📖 Referências

- [Neural Networks and Deep Learning - Michael Nielsen](http://neuralnetworksanddeeplearning.com/)
- [Deep Learning - Ian Goodfellow](https://www.deeplearningbook.org/)
- [3Blue1Brown - Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Backpropagation Algorithm](https://en.wikipedia.org/wiki/Backpropagation)

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👨‍💻 Créditos

Desenvolvido por [Sávio](https://github.com/SavioCodes)

---

⭐ Se este projeto te ajudou, considere dar uma estrela no GitHub!
