#!/usr/bin/env python3
"""
YAKE Performance Profiler
=========================

Análise detalhada de performance do YAKE para identificar gargalos
e oportunidades de otimização.
"""

import cProfile
import pstats
import io
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from yake.core.yake import KeywordExtractor


def profile_yake_extraction(text, n_gram=3, iterations=10):
    """Profile YAKE extraction com análise detalhada."""
    
    print(f"🔍 PROFILING YAKE EXTRACTION")
    print(f"📏 Texto: {len(text)} chars, {len(text.split())} palavras")
    print(f"🔄 N-gram: {n_gram}, Iterações: {iterations}")
    print("=" * 60)
    
    # Configurar extrator
    extractor = KeywordExtractor(n=n_gram, top=20)
    
    # Warm up
    extractor.extract_keywords(text)
    
    # Profile com cProfile
    pr = cProfile.Profile()
    
    start_time = time.time()
    
    pr.enable()
    for i in range(iterations):
        keywords = extractor.extract_keywords(text)
    pr.disable()
    
    end_time = time.time()
    
    print(f"⏱️  Tempo total: {(end_time - start_time)*1000:.1f}ms")
    print(f"📊 Tempo por iteração: {(end_time - start_time)*1000/iterations:.1f}ms")
    print(f"🔑 Keywords extraídas: {len(keywords)}")
    
    # Análise detalhada do profiling
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 funções
    
    profile_output = s.getvalue()
    print("\n🎯 TOP FUNÇÕES POR TEMPO CUMULATIVO:")
    print("-" * 60)
    
    lines = profile_output.split('\n')
    for line in lines[5:25]:  # Skip header, get top 20
        if line.strip() and 'function calls' not in line:
            print(line)
    
    return keywords, (end_time - start_time)


def analyze_scaling_performance():
    """Análise de escalabilidade com diferentes tamanhos de texto."""
    
    print("\n" + "="*80)
    print("📈 ANÁLISE DE ESCALABILIDADE")
    print("="*80)
    
    # Texto base
    base_text = """
    Google announced today that it is acquiring Kaggle, the popular data science competition platform. 
    The acquisition is expected to strengthen Google's position in the machine learning and artificial 
    intelligence space. Kaggle has over one million users who participate in predictive modeling and 
    analytics competitions. The platform hosts datasets and provides tools for data scientists to 
    collaborate and share insights.
    """
    
    sizes = [1, 2, 4, 8]  # Multiplicadores
    results = []
    
    for size in sizes:
        text = (base_text + " ") * size
        text_len = len(text)
        words = len(text.split())
        
        print(f"\n📏 Testando texto {size}x: {text_len} chars, {words} palavras")
        
        keywords, elapsed = profile_yake_extraction(text, n_gram=3, iterations=5)
        
        throughput = words / elapsed if elapsed > 0 else 0
        
        results.append({
            'size_multiplier': size,
            'chars': text_len,
            'words': words,
            'time_s': elapsed,
            'time_per_iter_ms': elapsed * 1000 / 5,
            'throughput_words_per_sec': throughput,
            'keywords_count': len(keywords)
        })
    
    print(f"\n📊 RESUMO DA ESCALABILIDADE:")
    print("-" * 80)
    print(f"{'Size':<6} {'Words':<8} {'Time/iter (ms)':<15} {'Throughput (w/s)':<18} {'Keywords':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['size_multiplier']}x    {r['words']:<8} {r['time_per_iter_ms']:<15.1f} {r['throughput_words_per_sec']:<18.0f} {r['keywords_count']:<10}")
    
    # Análise de complexidade
    if len(results) >= 2:
        print(f"\n🔍 ANÁLISE DE COMPLEXIDADE:")
        base_time = results[0]['time_per_iter_ms']
        base_words = results[0]['words']
        
        for i, r in enumerate(results[1:], 1):
            time_ratio = r['time_per_iter_ms'] / base_time
            words_ratio = r['words'] / base_words
            complexity = time_ratio / words_ratio if words_ratio > 0 else 0
            
            if complexity <= 1.2:
                complexity_desc = "Linear (O(n)) ✅"
            elif complexity <= 2.0:
                complexity_desc = "Quasi-linear (O(n log n)) ⚠️"
            else:
                complexity_desc = "Super-linear (O(n²) ou pior) ❌"
                
            print(f"  {r['size_multiplier']}x vs 1x: {time_ratio:.1f}x tempo, {words_ratio:.1f}x palavras → {complexity_desc}")


def profile_specific_components():
    """Profile componentes específicos do YAKE."""
    
    print("\n" + "="*80)
    print("🔧 ANÁLISE DE COMPONENTES ESPECÍFICOS")
    print("="*80)
    
    text = """
    Machine learning and artificial intelligence have emerged as transformative technologies across 
    various industries. These technologies enable computers to learn from data without being explicitly 
    programmed. Deep learning, a subset of machine learning, uses neural networks with multiple layers 
    to model and understand complex patterns in data. Natural language processing allows machines to 
    understand and generate human language, while computer vision enables the interpretation of visual 
    information from the world around us.
    """ * 2
    
    print(f"📏 Texto de teste: {len(text)} chars, {len(text.split())} palavras")
    
    extractor = KeywordExtractor(n=3, top=20)
    
    # Warm up
    extractor.extract_keywords(text)
    
    components = {
        'DataCore initialization': lambda: __import__('yake.data', fromlist=['DataCore']).DataCore(
            text=text, stopword_set=extractor.stopword_set, 
            config={'windows_size': 1, 'n': 3}
        ),
        'Single terms features': lambda: profile_component_single_terms(text, extractor),
        'Multi terms features': lambda: profile_component_multi_terms(text, extractor),
        'Deduplication': lambda: profile_component_deduplication(text, extractor),
    }
    
    for component_name, component_func in components.items():
        print(f"\n🎯 Profiling: {component_name}")
        
        pr = cProfile.Profile()
        
        start = time.time()
        pr.enable()
        
        for _ in range(10):
            component_func()
            
        pr.disable()
        end = time.time()
        
        print(f"⏱️  Tempo médio: {(end-start)*1000/10:.1f}ms")
        
        # Show top 5 functions for this component
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats('cumulative')
        ps.print_stats(5)
        
        lines = s.getvalue().split('\n')[5:10]
        for line in lines:
            if line.strip() and 'function calls' not in line:
                print(f"  {line}")


def profile_component_single_terms(text, extractor):
    """Profile só a parte de single terms."""
    from yake.data import DataCore
    
    dc = DataCore(text=text, stopword_set=extractor.stopword_set, 
                  config={'windows_size': 1, 'n': 3})
    dc.build_single_terms_features()
    return dc


def profile_component_multi_terms(text, extractor):
    """Profile só a parte de multi terms."""
    from yake.data import DataCore
    
    dc = DataCore(text=text, stopword_set=extractor.stopword_set, 
                  config={'windows_size': 1, 'n': 3})
    dc.build_single_terms_features()
    dc.build_mult_terms_features()
    return dc


def profile_component_deduplication(text, extractor):
    """Profile só a deduplicação."""
    keywords = extractor.extract_keywords(text)
    return keywords


def main():
    """Execução principal do profiler."""
    
    print("🎯 YAKE PERFORMANCE PROFILER")
    print("="*80)
    print("Análise detalhada de performance para identificar gargalos")
    print("="*80)
    
    # Análise básica
    medium_text = """
    In a major move to strengthen its artificial intelligence and machine learning capabilities, 
    Google announced today the acquisition of Kaggle, the world's largest community of data scientists 
    and machine learning practitioners. The deal, whose financial terms were not disclosed, brings 
    together Google's cloud computing infrastructure with Kaggle's platform that hosts more than 
    one million data scientists who compete in predictive modeling challenges.
    
    Founded in 2010 by Anthony Goldbloom and Ben Hamner, Kaggle has become the go-to platform for 
    data science competitions, hosting challenges for companies like Mercedes-Benz, Airbnb, and 
    the U.S. government. The platform allows organizations to post their data and problems, 
    attracting thousands of data scientists who compete to build the best predictive models.
    """
    
    profile_yake_extraction(medium_text, n_gram=3, iterations=20)
    
    # Análise de escalabilidade
    analyze_scaling_performance()
    
    # Análise de componentes
    profile_specific_components()
    
    print("\n" + "="*80)
    print("✅ PROFILING CONCLUÍDO")
    print("="*80)
    print("📋 Próximos passos:")
    print("  1. Analisar as funções que consomem mais tempo")
    print("  2. Identificar gargalos de escalabilidade")
    print("  3. Implementar otimizações específicas")
    print("  4. Executar benchmark para validar melhorias")


if __name__ == '__main__':
    main()