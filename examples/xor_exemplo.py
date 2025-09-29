
#!/usr/bin/env python3
"""
Exemplo Específico: Dataset XOR
Autor: Sávio (https://github.com/SavioCodes)

Demonstra o treinamento detalhado da rede neural
no problema clássico do XOR (OR Exclusivo).
"""

import sys
import os
import numpy as np

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rede_neural import RedeNeural
from src.utils import DataUtils


def imprimir_tabela_verdade():
    """Imprime a tabela verdade do XOR."""
    print("\n📋 TABELA VERDADE DO XOR:")
    print("-" * 25)
    print("| A | B | A XOR B |")
    print("|---|---|---------|")
    print("| 0 | 0 |    0    |")
    print("| 0 | 1 |    1    |")
    print("| 1 | 0 |    1    |") 
    print("| 1 | 1 |    0    |")
    print("-" * 25)


def testar_diferentes_arquiteturas():
    """Testa diferentes arquiteturas de rede neural no XOR."""
    
    print("\n🧪 TESTANDO DIFERENTES ARQUITETURAS")
    print("=" * 40)
    
    # Preparar dados
    X, y = DataUtils.gerar_xor_dataset()
    
    # Diferentes arquiteturas para testar
    arquiteturas = [
        ([2, 2, 1], "Mínima"),
        ([2, 3, 1], "Pequena"), 
        ([2, 4, 1], "Padrão"),
        ([2, 6, 1], "Maior"),
        ([2, 4, 3, 1], "Duas camadas"),
        ([2, 8, 4, 1], "Profunda")
    ]
    
    resultados = []
    
    for arquitetura, nome in arquiteturas:
        print(f"\n🔧 Testando arquitetura {nome}: {arquitetura}")
        
        # Criar e treinar rede
        rede = RedeNeural(arquitetura, ativacao='sigmoid', inicializacao='xavier')
        
        # Treinar com menos épocas para comparação rápida
        rede.treinar(X, y, epochs=1000, taxa_aprendizado=0.5, verbose=False)
        
        # Avaliar
        resultado = rede.avaliar(X, y)
        resultados.append((nome, arquitetura, resultado['acuracia'], resultado['erro']))
        
        print(f"   Acurácia: {resultado['acuracia']:.2f}%")
        print(f"   Erro: {resultado['erro']:.6f}")
    
    # Mostrar resumo
    print("\n📊 RESUMO COMPARATIVO:")
    print("-" * 50)
    print(f"{'Nome':<12} {'Arquitetura':<15} {'Acurácia':<10} {'Erro':<10}")
    print("-" * 50)
    
    for nome, arq, acc, erro in resultados:
        arq_str = str(arq).replace('[', '').replace(']', '').replace(' ', '')
        print(f"{nome:<12} {arq_str:<15} {acc:<10.2f}% {erro:<10.6f}")
    
    # Encontrar melhor
    melhor = max(resultados, key=lambda x: x[2])
    print(f"\n🏆 Melhor arquitetura: {melhor[0]} com {melhor[2]:.2f}% de acurácia")


def treinar_detalhado():
    """Treina uma rede com logging detalhado."""
    
    print("\n🎯 TREINAMENTO DETALHADO")
    print("=" * 30)
    
    # Preparar dados
    X, y = DataUtils.gerar_xor_dataset()
    
    print("Dados de treinamento:")
    for i, (entrada, saida) in enumerate(zip(X, y)):
        print(f"  Amostra {i+1}: [{entrada[0]}, {entrada[1]}] → {saida[0]}")
    
    # Criar rede
    rede = RedeNeural([2, 4, 1], ativacao='sigmoid', inicializacao='xavier')
    
    print(f"\nArquitetura: {rede.arquitetura}")
    print(f"Número total de parâmetros: {sum(w.size for w in rede.pesos) + sum(b.size for b in rede.biases)}")
    
    # Mostrar pesos iniciais
    print(f"\nPesos iniciais (camada 1): shape {rede.pesos[0].shape}")
    print(f"Pesos iniciais (camada 2): shape {rede.pesos[1].shape}")
    
    # Treinar com checkpoints
    epochs_total = 2000
    checkpoints = [0, 100, 500, 1000, 1500, epochs_total]
    
    print(f"\n🚀 Iniciando treinamento por {epochs_total} épocas...")
    
    for i, checkpoint in enumerate(checkpoints[:-1]):
        epochs_treinar = checkpoints[i+1] - checkpoint
        
        if epochs_treinar > 0:
            rede.treinar(X, y, epochs=epochs_treinar, taxa_aprendizado=0.5, verbose=False)
        
        # Avaliar no checkpoint
        predicoes = rede.prever(X)
        erro = np.mean((y - predicoes) ** 2)
        acuracia = np.mean((predicoes >= 0.5) == (y >= 0.5)) * 100
        
        print(f"\nÉpoca {checkpoints[i+1]:4d}: Erro = {erro:.6f}, Acurácia = {acuracia:.2f}%")
        
        # Mostrar predições atuais
        print("  Predições atuais:")
        for j, (entrada, esperado, predito) in enumerate(zip(X, y, predicoes)):
            decisao = "✅" if abs(predito[0] - esperado[0]) < 0.1 else "❌"
            print(f"    [{entrada[0]}, {entrada[1]}] → {esperado[0]} | {predito[0]:.4f} {decisao}")
    
    print("\n🎉 Treinamento concluído!")
    
    # Teste de robustez
    print(f"\n🔍 TESTE DE ROBUSTEZ")
    print("-" * 20)
    
    # Testar com ruído
    print("Testando com ruído nos dados:")
    X_ruido = X + np.random.normal(0, 0.1, X.shape)
    predicoes_ruido = rede.prever(X_ruido)
    
    for i, (entrada_orig, entrada_ruido, esperado, predito) in enumerate(zip(X, X_ruido, y, predicoes_ruido)):
        print(f"  Original: [{entrada_orig[0]:.1f}, {entrada_orig[1]:.1f}] | "
              f"Com ruído: [{entrada_ruido[0]:.2f}, {entrada_ruido[1]:.2f}] → {predito[0]:.4f}")


def main():
    """Função principal do exemplo XOR."""
    
    print("🧠 REDE NEURAL - EXEMPLO XOR DETALHADO")
    print("=" * 45)
    print("Autor: Sávio (https://github.com/SavioCodes)")
    print("=" * 45)
    
    # 1. Explicar o problema
    print("\n❓ O QUE É O XOR?")
    print("O XOR (OR Exclusivo) é um problema clássico em redes neurais.")
    print("É um problema não-linearmente separável, ou seja, não pode")
    print("ser resolvido com uma única camada (perceptron simples).")
    
    imprimir_tabela_verdade()
    
    print("\n🧠 POR QUE É IMPORTANTE?")
    print("- Demonstra a necessidade de camadas ocultas")
    print("- É um benchmark clássico para redes neurais")
    print("- Simples de entender, mas requer não-linearidade")
    
    # 2. Treinar e mostrar detalhes
    treinar_detalhado()
    
    # 3. Comparar arquiteturas
    testar_diferentes_arquiteturas()
    
    # 4. Dicas finais
    print(f"\n💡 DICAS PARA O XOR:")
    print("1. Mínimo de 2 neurônios na camada oculta")
    print("2. Função sigmoid funciona bem para este problema")
    print("3. Taxa de aprendizado entre 0.1 e 1.0")
    print("4. Inicialização Xavier é adequada")
    print("5. Convergência rápida (< 2000 épocas)")
    
    print(f"\n📚 REFERÊNCIAS:")
    print("- Minsky & Papert (1969) - Perceptrons")
    print("- Rumelhart et al. (1986) - Backpropagation")
    
    print(f"\n👨‍💻 Criado por Sávio - https://github.com/SavioCodes")


if __name__ == "__main__":
    main()
