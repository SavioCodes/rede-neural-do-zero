# Tutorial Guiado

Este tutorial mostra um caminho curto para estudar o projeto sem se perder.

## 1. Comece pelo caso XOR

Rode:

```bash
python examples/xor_exemplo.py
```

Esse e o melhor ponto de entrada porque:

- o dataset e pequeno
- a rede aprende um padrao nao linear simples
- fica facil relacionar entrada, saida e predicoes finais

## 2. Leia o fluxo do modelo

Arquivos mais importantes:

- `src/rede_neural_core.py`
- `src/funcoes_ativacao.py`
- `src/utils.py`

Ordem recomendada:

1. `_forward_pass()`
2. `_calcular_delta_saida()`
3. `_backward_propagation()`
4. `_atualizar_parametros()`
5. `treinar()`

## 3. Entenda as configuracoes

O projeto permite duas formas de uso:

### API direta

```python
rede = RedeNeural([2, 4, 1], ativacao="sigmoid", seed=42)
```

### API organizada

```python
modelo = RedeNeural.from_config(
    ModelConfig([2, 16, 12, 3], ativacao="relu", seed=42, funcao_custo="categorical_crossentropy")
)
```

`TrainingConfig` organiza hiperparametros como:

- `epochs`
- `taxa_aprendizado`
- `batch_size`
- `otimizador`
- `l2_lambda`
- `dropout`
- `gradient_clip`

## 4. Estude os callbacks

Callbacks deixam o treino mais limpo:

- `History`: guarda logs em memoria
- `CSVLogger`: salva um CSV
- `ModelCheckpoint`: salva os melhores pesos
- `EarlyStopping`: interrompe treino sem progresso

Exemplo mental:

```text
treino produz logs -> callbacks observam logs -> callbacks salvam, param ou restauram pesos
```

## 5. Va para classificacao binaria

Rode:

```bash
python examples/classificacao.py
python examples/exemplo.py --no-plots
```

Aqui voce observa:

- normalizacao dos dados
- split treino/teste
- uso de `Adam`
- historico de treino
- metricas como precisao, recall e F1

## 6. Va para multiclasse

Rode:

```bash
python examples/multiclasse.py --no-plots
```

Esse exemplo mostra:

- `softmax`
- `categorical_crossentropy`
- one-hot encoding
- matriz de confusao multiclasse
- fronteira de decisao multiclasse

## 7. Compare configuracoes

Rode:

```bash
python -m src benchmark --mode binario
python -m src benchmark --config configs/benchmark/suite.yaml
```

Isso ajuda a responder perguntas como:

- `Adam` esta melhor do que `SGD` aqui?
- `batch_size` menor ajuda?
- `L2` e `dropout` melhoraram generalizacao?

## 8. Verifique a qualidade do projeto

Comandos uteis:

```bash
python -m src verify
python -m src evaluate --config configs/evaluate/diabetes.toml
```

## 9. Melhor jeito de continuar estudando

Se voce quiser aprofundar sem se perder, siga esta ordem:

1. XOR
2. classificacao binaria
3. callbacks
4. multiclasse
5. benchmark
6. testes

Essa progressao vai do menor caso possivel ate uma base de projeto mais parecida com bibliotecas reais.
