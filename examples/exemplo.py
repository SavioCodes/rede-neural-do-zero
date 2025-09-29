#!/usr/bin/env python3
"""
Exemplo Principal - Demonstração da Rede Neural do Zero
Autor: Sávio (https://github.com/SavioCodes)

Este exemplo demonstra:
1. Treinamento no dataset XOR
2. Avaliação de performance  
3. Visualização de resultados
4. Métricas detalhadas
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rede_neural import RedeNeural
from src.utils import DataUtils, VisualizationUtils, MetricUtils


def main():
    print("🧠 REDE NEURAL DO ZERO - EXEMPLO PRINCIPAL")
    print("=" * 50)
    print("Autor: Sávio (https://github.com/SavioCodes)")
    print("=" * 50)
    
    # ===== 1. PREPARAÇÃO DOS DADOS =====
    print("\n📊 1. PREPARANDO DADOS...")
    
    # Dataset XOR clássico
    X_xor, y_xor = DataUtils.gerar_xor_dataset()
    print(f"Dataset XOR carregado: {X_xor.shape[0]} amostras, {X_xor.shape[1]} features")
    
    # Dataset de classificação mais complexo
    X_complex, y_complex = DataUtils.gerar_dataset_classificacao(n_samples=1000, noise=0.2)
    print(f"Dataset complexo gerado: {X_complex.shape[0]} amostras, {X_complex.shape[1]} features")
    
    # Normalizar dataset complexo
    X_complex_norm, norm_params = DataUtils.normalizar_dados(X_complex, metodo='padrao')
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = DataUtils.dividir_treino_teste(
        X_complex_norm, y_complex, test_size=0.2, random_state=42
    )
    
    print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
    
    # ===== 2. EXPERIMENTO 1: XOR =====
    print("\n🎯 2. EXPERIMENTO 1: DATASET XOR")
    print("-" * 30)
    
    # Criar rede neural para XOR
    rede_xor = RedeNeural(
        arquitetura=[2, 4, 1],  # 2 entradas, 4 neurônios ocultos, 1 saída
        ativacao='sigmoid',
        inicializacao='xavier'
    )
    
    print("Arquitetura da rede XOR:", rede_xor.arquitetura)
    print("Função de ativação:", rede_xor.ativacao)
    
    # Treinar
    print("\nIniciando treinamento XOR...")
    rede_xor.treinar(
        X_xor, y_xor,
        epochs=2000,
        taxa_aprendizado=0.5,
        verbose=True
    )
    
    # Avaliar XOR
    print("\n📈 RESULTADOS XOR:")
    resultados_xor = rede_xor.avaliar(X_xor, y_xor)
    
    print(f"Erro final: {resultados_xor['erro']:.6f}")
    print(f"Acurácia: {resultados_xor['acuracia']:.2f}%")
    
    # Mostrar predições individuais
    print("\nPredições XOR individuais:")
    for i, (entrada, esperado) in enumerate(zip(X_xor, y_xor)):
        predicao = rede_xor.prever(entrada.reshape(1, -1))[0, 0]
        print(f"[{entrada[0]}, {entrada[1]}] → Esperado: {esperado[0]}, Predito: {predicao:.4f}")
    
    # ===== 3. EXPERIMENTO 2: CLASSIFICAÇÃO COMPLEXA =====
    print("\n🎯 3. EXPERIMENTO 2: CLASSIFICAÇÃO COMPLEXA")
    print("-" * 40)
    
    # Criar rede neural mais robusta
    rede_complex = RedeNeural(
        arquitetura=[2, 8, 4, 1],  # Rede mais profunda
        ativacao='relu',
        inicializacao='he'
    )
    
    print("Arquitetura da rede complexa:", rede_complex.arquitetura)
    print("Função de ativação:", rede_complex.ativacao)
    
    # Treinar com validação
    print("\nIniciando treinamento com validação...")
    rede_complex.treinar(
        X_train, y_train,
        epochs=1500,
        taxa_aprendizado=0.01,
        verbose=True,
        validacao_X=X_test,
        validacao_y=y_test
    )
    
    # Avaliar no conjunto de teste
    print("\n📈 RESULTADOS NO CONJUNTO DE TESTE:")
    resultados_test = rede_complex.avaliar(X_test, y_test)
    
    print(f"Erro no teste: {resultados_test['erro']:.6f}")
    print(f"Acurácia no teste: {resultados_test['acuracia']:.2f}%")
    
    # ===== 4. MÉTRICAS DETALHADAS =====
    print("\n📊 4. MÉTRICAS DETALHADAS")
    print("-" * 25)
    
    # Calcular métricas avançadas
    metricas = MetricUtils.precisao_recall_f1(y_test, resultados_test['predicoes'])
    
    print(f"Precisão: {metricas['precisao']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"F1-Score: {metricas['f1_score']:.4f}")
    
    print("\nMatriz de Confusão:")
    print(metricas['matriz_confusao'])
    print("         Pred: 0  1")
    print("Real: 0      TN  FP")
    print("      1      FN  TP")
    
    # ===== 5. VISUALIZAÇÕES =====
    print("\n🎨 5. GERANDO VISUALIZAÇÕES...")
    
    try:
        # Histórico de treinamento XOR
        print("Plotando histórico XOR...")
        VisualizationUtils.plotar_historico_treinamento(
            rede_xor.historico_erro,
            rede_xor.historico_acuracia,
            salvar='results/historico_xor.png'
        )
        
        # Histórico de treinamento complexo
        print("Plotando histórico classificação complexa...")
        VisualizationUtils.plotar_historico_treinamento(
            rede_complex.historico_erro,
            rede_complex.historico_acuracia,
            salvar='results/historico_complex.png'
        )
        
        # Dados e fronteira de decisão
        print("Plotando dados de classificação...")
        VisualizationUtils.plotar_dados_classificacao(
            X_test, y_test,
            titulo="Dataset de Teste",
            salvar='results/dados_teste.png'
        )
        
        # Fronteira de decisão
        print("Plotando fronteira de decisão...")
        VisualizationUtils.plotar_fronteira_decisao(
            rede_complex, X_test, y_test,
            titulo="Fronteira de Decisão - Rede Neural",
            salvar='results/fronteira_decisao.png'
        )
        
    except ImportError:
        print("⚠️  Matplotlib não disponível. Visualizações puladas.")
        print("   Instale com: pip install matplotlib")
    
    # ===== 6. SALVAR MODELO =====
    print("\n💾 6. SALVANDO MODELOS...")
    
    # Criar diretório de resultados
    os.makedirs('results', exist_ok=True)
    
    # Salvar parâmetros das redes
    rede_xor.salvar_parametros('results/modelo_xor.npz')
    rede_complex.salvar_parametros('results/modelo_complex.npz')
    
    # ===== 7. TESTE DE CARREGAMENTO =====
    print("\n🔄 7. TESTANDO CARREGAMENTO DE MODELO...")
    
    # Criar nova rede e carregar parâmetros
    rede_carregada = RedeNeural([2, 4, 1])
    rede_carregada.carregar_parametros('results/modelo_xor.npz')
    
    # Testar se funciona igual
    pred_original = rede_xor.prever(X_xor)
    pred_carregada = rede_carregada.prever(X_xor)
    
    diferenca = np.mean(np.abs(pred_original - pred_carregada))
    print(f"Diferença média entre modelos: {diferenca:.10f}")
    
    if diferenca < 1e-10:
        print("✅ Modelo carregado corretamente!")
    else:
        print("❌ Problema no carregamento do modelo.")
    
    # ===== 8. RESUMO FINAL =====
    print("\n" + "=" * 50)
    print("🎉 EXPERIMENTOS CONCLUÍDOS!")
    print("=" * 50)
    
    print(f"\n📊 RESUMO DOS RESULTADOS:")
    print(f"XOR - Acurácia: {resultados_xor['acuracia']:.2f}%")
    print(f"Classificação Complexa - Acurácia: {resultados_test['acuracia']:.2f}%")
    print(f"F1-Score: {metricas['f1_score']:.4f}")
    
    print(f"\n📁 ARQUIVOS GERADOS:")
    print("- results/modelo_xor.npz")
    print("- results/modelo_complex.npz")
    print("- results/historico_xor.png")
    print("- results/historico_complex.png")
    print("- results/dados_teste.png")
    print("- results/fronteira_decisao.png")
    
    print("\n💡 PRÓXIMOS PASSOS:")
    print("1. Experimente diferentes arquiteturas")
    print("2. Teste outros datasets")
    print("3. Ajuste hiperparâmetros")
    print("4. Implemente regularização")
    
    print(f"\n👨‍💻 Criado por Sávio - https://github.com/SavioCodes")


if __name__ == "__main__":
    main()
