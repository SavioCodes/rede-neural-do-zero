#!/usr/bin/env python3
"""
Exemplo: Classificação com Dataset Sintético
Autor: Sávio (https://github.com/SavioCodes)

Demonstra uso da rede neural em um problema de
classificação binária mais realista.
"""

import sys
import os
import numpy as np

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rede_neural import RedeNeural
from src.utils import DataUtils, VisualizationUtils, MetricUtils


def experimento_funcoes_ativacao():
    """Compara diferentes funções de ativação."""
    
    print("🧪 COMPARANDO FUNÇÕES DE ATIVAÇÃO")
    print("=" * 35)
    
    # Gerar dados
    X, y = DataUtils.gerar_dataset_classificacao(n_samples=800, noise=0.15)
    X_norm, _ = DataUtils.normalizar_dados(X)
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X_norm, y, test_size=0.25)
    
    # Funções para testar
    funcoes = ['sigmoid', 'relu', 'tanh']
    resultados = {}
    
    for funcao in funcoes:
        print(f"\n🔧 Testando função: {funcao.upper()}")
        
        # Escolher inicialização adequada
        init = 'he' if funcao == 'relu' else 'xavier'
        
        # Criar rede
        rede = RedeNeural(
            arquitetura=[2, 10, 6, 1],
            ativacao=funcao,
            inicializacao=init
        )
        
        # Treinar
        rede.treinar(X_train, y_train, epochs=1000, taxa_aprendizado=0.01, verbose=False)
        
        # Avaliar
        resultado = rede.avaliar(X_test, y_test)
        metricas = MetricUtils.precisao_recall_f1(y_test, resultado['predicoes'])
        
        resultados[funcao] = {
            'acuracia': resultado['acuracia'],
            'erro': resultado['erro'],
            'f1': metricas['f1_score'],
            'precisao': metricas['precisao'],
            'recall': metricas['recall']
        }
        
        print(f"  Acurácia: {resultado['acuracia']:.2f}%")
        print(f"  F1-Score: {metricas['f1_score']:.4f}")
    
    # Resumo
    print(f"\n📊 RESUMO COMPARATIVO:")
    print("-" * 60)
    print(f"{'Função':<10} {'Acurácia':<10} {'F1-Score':<10} {'Precisão':<10} {'Recall':<10}")
    print("-" * 60)
    
    for funcao, metricas in resultados.items():
        print(f"{funcao.capitalize():<10} "
              f"{metricas['acuracia']:<10.2f} "
              f"{metricas['f1']:<10.4f} "
              f"{metricas['precisao']:<10.4f} "
              f"{metricas['recall']:<10.4f}")
    
    # Melhor função
    melhor = max(resultados.items(), key=lambda x: x[1]['f1'])
    print(f"\n🏆 Melhor função: {melhor[0].upper()} (F1: {melhor[1]['f1']:.4f})")


def experimento_normalizacao():
    """Testa diferentes métodos de normalização."""
    
    print("\n🧪 COMPARANDO MÉTODOS DE NORMALIZAÇÃO") 
    print("=" * 40)
    
    # Gerar dados com escala diferente
    X, y = DataUtils.gerar_dataset_classificacao(n_samples=600, noise=0.1)
    
    # Artificialmente criar escalas diferentes
    X[:, 0] *= 100  # Primeira feature entre -100 e 100
    X[:, 1] *= 0.01  # Segunda feature entre -0.01 e 0.01
    
    print(f"Dados originais - Feature 1: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"Dados originais - Feature 2: [{X[:, 1].min():.4f}, {X[:, 1].max():.4f}]")
    
    # Métodos de normalização
    metodos = ['padrao', 'minmax', 'robusto']
    resultados = {}
    
    for metodo in metodos:
        print(f"\n🔧 Testando normalização: {metodo.upper()}")
        
        # Normalizar
        X_norm, _ = DataUtils.normalizar_dados(X, metodo=metodo)
        
        print(f"  Após {metodo} - Feature 1: [{X_norm[:, 0].min():.2f}, {X_norm[:, 0].max():.2f}]")
        print(f"  Após {metodo} - Feature 2: [{X_norm[:, 1].min():.2f}, {X_norm[:, 1].max():.2f}]")
        
        # Dividir dados
        X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X_norm, y, test_size=0.2)
        
        # Treinar rede
        rede = RedeNeural([2, 8, 1], ativacao='relu', inicializacao='he')
        rede.treinar(X_train, y_train, epochs=800, taxa_aprendizado=0.01, verbose=False)
        
        # Avaliar
        resultado = rede.avaliar(X_test, y_test)
        resultados[metodo] = resultado['acuracia']
        
        print(f"  Acurácia: {resultado['acuracia']:.2f}%")
    
    # Comparar com dados não normalizados
    print(f"\n🔧 Testando SEM normalização:")
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X, y, test_size=0.2)
    rede = RedeNeural([2, 8, 1], ativacao='relu', inicializacao='he')
    rede.treinar(X_train, y_train, epochs=800, taxa_aprendizado=0.001, verbose=False)  # Taxa menor
    resultado = rede.avaliar(X_test, y_test)
    resultados['sem_norm'] = resultado['acuracia']
    print(f"  Acurácia: {resultado['acuracia']:.2f}%")
    
    # Resumo
    print(f"\n📊 IMPACTO DA NORMALIZAÇÃO:")
    print("-" * 30)
    for metodo, acuracia in resultados.items():
        print(f"{metodo.capitalize():<12}: {acuracia:.2f}%")
    
    melhor = max(resultados.items(), key=lambda x: x[1])
    print(f"\n🏆 Melhor método: {melhor[0].capitalize()} ({melhor[1]:.2f}%)")


def experimento_hiperparametros():
    """Testa diferentes combinações de hiperparâmetros."""
    
    print("\n🧪 OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print("=" * 35)
    
    # Dados
    X, y = DataUtils.gerar_dataset_classificacao(n_samples=1000, noise=0.1)
    X_norm, _ = DataUtils.normalizar_dados(X)
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(X_norm, y, test_size=0.2)
    
    # Grid de hiperparâmetros
    taxas_aprendizado = [0.001, 0.01, 0.1, 0.5]
    arquiteturas = [
        [2, 4, 1],
        [2, 8, 1], 
        [2, 12, 1],
        [2, 8, 4, 1],
        [2, 12, 6, 1]
    ]
    
    melhor_resultado = 0
    melhor_config = None
    
    print("Testando combinações...")
    print(f"{'Taxa':<8} {'Arquitetura':<15} {'Acurácia':<10} {'F1':<8}")
    print("-" * 45)
    
    for taxa in taxas_aprendizado:
        for arq in arquiteturas:
            # Treinar rede
            rede = RedeNeural(arq, ativacao='relu', inicializacao='he')
            rede.treinar(X_train, y_train, epochs=800, taxa_aprendizado=taxa, verbose=False)
            
            # Avaliar
            resultado = rede.avaliar(X_test, y_test)
            metricas = MetricUtils.precisao_recall_f1(y_test, resultado['predicoes'])
            
            # Exibir resultado
            arq_str = str(arq).replace('[', '').replace(']', '').replace(' ', '')
            print(f"{taxa:<8.3f} {arq_str:<15} {resultado['acuracia']:<10.2f} {metricas['f1_score']:<8.4f}")
            
            # Atualizar melhor
            if metricas['f1_score'] > melhor_resultado:
                melhor_resultado = metricas['f1_score']
                melhor_config = (taxa, arq, resultado['acuracia'])
    
    print(f"\n🏆 MELHOR CONFIGURAÇÃO:")
    print(f"Taxa de aprendizado: {melhor_config[0]}")
    print(f"Arquitetura: {melhor_config[1]}")
    print(f"Acurácia: {melhor_config[2]:.2f}%")
    print(f"F1-Score: {melhor_resultado:.4f}")


def main():
    """Função principal."""
    
    print("🧠 REDE NEURAL - EXEMPLO DE CLASSIFICAÇÃO")
    print("=" * 45)
    print("Autor: Sávio (https://github.com/SavioCodes)")
    print("=" * 45)
    
    # Experimento 1: Funções de ativação
    experimento_funcoes_ativacao()
    
    # Experimento 2: Normalização
    experimento_normalizacao()
    
    # Experimento 3: Hiperparâmetros
    experimento_hiperparametros()
    
    # Dicas finais
    print(f"\n💡 LIÇÕES APRENDIDAS:")
    print("1. ReLU geralmente funciona bem para problemas complexos")
    print("2. Normalização é CRUCIAL para convergência")
    print("3. Arquiteturas mais profundas podem ajudar")
    print("4. Taxa de aprendizado deve ser ajustada cuidadosamente")
    print("5. F1-score é melhor que acurácia para datasets desbalanceados")
    
    print(f"\n📚 PRÓXIMOS PASSOS:")
    print("- Implementar validação cruzada")
    print("- Adicionar regularização (dropout, L1/L2)")
    print("- Testar otimizadores avançados (Adam, RMSprop)")
    print("- Implementar early stopping")
    
    print(f"\n👨‍💻 Criado por Sávio - https://github.com/SavioCodes")


if __name__ == "__main__":
    main()
