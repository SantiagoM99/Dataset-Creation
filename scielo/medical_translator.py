#!/usr/bin/env python3
"""
Self-Tuning Medical Abstract Translator
Automatically finds optimal batch size for your hardware
"""

import json
import os
import re
import time
import argparse
import psutil
from datetime import datetime
from tqdm import tqdm

def get_system_info():
    """Get system memory and processing capabilities"""
    memory_gb = psutil.virtual_memory().total / (1024**3)
    cpu_count = psutil.cpu_count()
    
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        if has_gpu:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_memory = 0
            gpu_name = "None"
    except:
        has_gpu = False
        gpu_memory = 0
        gpu_name = "None"
    
    return {
        'memory_gb': memory_gb,
        'cpu_count': cpu_count,
        'has_gpu': has_gpu,
        'gpu_memory': gpu_memory,
        'gpu_name': gpu_name
    }

def suggest_batch_sizes(system_info):
    """Suggest batch sizes to test based on hardware"""
    if system_info['has_gpu']:
        if system_info['gpu_memory'] > 12:
            return [4, 8, 16, 24, 32]  # High-end GPU
        elif system_info['gpu_memory'] > 6:
            return [2, 4, 8, 12, 16]   # Mid-range GPU
        else:
            return [1, 2, 4, 6, 8]     # Entry GPU
    else:
        if system_info['memory_gb'] > 16:
            return [1, 2, 4, 6, 8]     # High RAM
        elif system_info['memory_gb'] > 8:
            return [1, 2, 3, 4, 6]     # Medium RAM
        else:
            return [1, 2, 3, 4]        # Low RAM

def load_model_with_batch_size(model_name, batch_size):
    """Load model with specific batch size"""
    try:
        from transformers import pipeline
        import torch
        
        device = 0 if torch.cuda.is_available() else -1
        
        translator = pipeline(
            "translation_en_to_es",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            batch_size=batch_size
        )
        
        return translator
        
    except Exception as e:
        return None

def benchmark_batch_size(translator, test_abstracts, batch_size):
    """Benchmark translation speed and memory usage for a batch size"""
    print(f"   Testing batch size {batch_size}...")
    
    try:
        # Monitor memory before
        memory_before = psutil.virtual_memory().percent
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                gpu_memory_before = torch.cuda.memory_allocated() / (1024**3)
        except:
            gpu_memory_before = 0
        
        # Time the translation
        start_time = time.time()
        
        # Translate test abstracts
        results = []
        for abstract in test_abstracts:
            sentences = re.split(r'(?<=[.!?])\s+', abstract)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Process in batches
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                result = translator(batch)
                results.extend(result)
        
        end_time = time.time()
        translation_time = end_time - start_time
        
        # Monitor memory after
        memory_after = psutil.virtual_memory().percent
        memory_usage = memory_after - memory_before
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_peak = torch.cuda.max_memory_allocated() / (1024**3)
                gpu_memory_used = gpu_memory_peak - gpu_memory_before
            else:
                gpu_memory_used = 0
        except:
            gpu_memory_used = 0
        
        # Calculate performance metrics
        abstracts_per_second = len(test_abstracts) / translation_time
        
        return {
            'batch_size': batch_size,
            'time_seconds': translation_time,
            'abstracts_per_second': abstracts_per_second,
            'memory_usage_percent': memory_usage,
            'gpu_memory_gb': gpu_memory_used,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'batch_size': batch_size,
            'success': False,
            'error': str(e),
            'time_seconds': float('inf'),
            'abstracts_per_second': 0
        }

def find_optimal_batch_size(model_name, test_abstracts):
    """Find the optimal batch size through testing"""
    print("🔍 FINDING OPTIMAL BATCH SIZE")
    print("=" * 50)
    
    system_info = get_system_info()
    
    print(f"💻 System: {system_info['memory_gb']:.1f}GB RAM, {system_info['cpu_count']} CPU cores")
    if system_info['has_gpu']:
        print(f"🎮 GPU: {system_info['gpu_name']} ({system_info['gpu_memory']:.1f}GB)")
    else:
        print("🔧 Using CPU only")
    
    batch_sizes_to_test = suggest_batch_sizes(system_info)
    print(f"🧪 Testing batch sizes: {batch_sizes_to_test}")
    
    results = []
    
    for batch_size in batch_sizes_to_test:
        print(f"\n📊 Testing batch size {batch_size}:")
        
        # Load model with this batch size
        translator = load_model_with_batch_size(model_name, batch_size)
        if not translator:
            print(f"   ❌ Failed to load model with batch size {batch_size}")
            continue
        
        # Benchmark this batch size
        benchmark = benchmark_batch_size(translator, test_abstracts, batch_size)
        results.append(benchmark)
        
        if benchmark['success']:
            print(f"   ✅ Speed: {benchmark['abstracts_per_second']:.2f} abstracts/sec")
            print(f"   📊 Memory: +{benchmark['memory_usage_percent']:.1f}% RAM")
            if benchmark['gpu_memory_gb'] > 0:
                print(f"   🎮 GPU: +{benchmark['gpu_memory_gb']:.2f}GB")
        else:
            print(f"   ❌ Failed: {benchmark['error']}")
            # If we hit memory limits, stop testing larger batch sizes
            if 'memory' in benchmark['error'].lower() or 'out of memory' in benchmark['error'].lower():
                print("   ⚠️ Hit memory limit, stopping larger batch tests")
                break
        
        # Memory cleanup
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Small delay to let memory settle
        time.sleep(1)
    
    # Find optimal batch size
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("\n❌ No batch sizes worked! Using batch size 1")
        return 1
    
    # Find fastest batch size
    optimal = max(successful_results, key=lambda x: x['abstracts_per_second'])
    
    print(f"\n🏆 OPTIMAL BATCH SIZE: {optimal['batch_size']}")
    print(f"   ⚡ Speed: {optimal['abstracts_per_second']:.2f} abstracts/sec")
    print(f"   📊 Memory impact: +{optimal['memory_usage_percent']:.1f}% RAM")
    
    # Show comparison table
    print(f"\n📋 PERFORMANCE COMPARISON:")
    print(f"{'Batch':<8} {'Speed':<12} {'RAM':<8} {'Status'}")
    print("-" * 35)
    
    for result in results:
        if result['success']:
            speed_str = f"{result['abstracts_per_second']:.2f}/sec"
            memory_str = f"+{result['memory_usage_percent']:.1f}%"
            status = "✅"
            if result['batch_size'] == optimal['batch_size']:
                status = "🏆 BEST"
        else:
            speed_str = "FAILED"
            memory_str = "N/A"
            status = "❌"
        
        print(f"{result['batch_size']:<8} {speed_str:<12} {memory_str:<8} {status}")
    
    return optimal['batch_size']

def translate_dataset_tuned(input_file, output_file, model_name="Helsinki-NLP/opus-mt-en-es", 
                           batch_size=None, auto_tune=True):
    """Translate dataset with optimal batch size"""
    
    print(f"📖 Loading dataset: {input_file}")
    
    # Load dataset
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False
    
    articles = dataset.get('articles', [])
    print(f"📚 Total articles: {len(articles):,}")
    
    # Find English abstracts
    english_abstracts = []
    for i, article in enumerate(articles):
        if (((article.get('abstract_language') == 'english') or (article.get('abstract_language') == 'mixed_or_unknown')) and 
            article.get('abstract', {}).get('has_content', False)):
            english_abstracts.append((i, article['abstract']['full_text']))
    
    print(f"🔤 English abstracts to translate: {len(english_abstracts):,}")
    
    if not english_abstracts:
        print("⚠️ No English abstracts found")
        return False
    
    # Auto-tune batch size if requested
    if auto_tune and batch_size is None:
        # Use first 5 abstracts for testing
        test_abstracts = [item[1] for item in english_abstracts[:5]]
        optimal_batch_size = find_optimal_batch_size(model_name, test_abstracts)
    else:
        optimal_batch_size = batch_size or 4
        print(f"🔧 Using specified batch size: {optimal_batch_size}")
    
    # Load model with optimal batch size
    print(f"\n🤖 Loading model with batch size {optimal_batch_size}...")
    translator = load_model_with_batch_size(model_name, optimal_batch_size)
    if not translator:
        print("❌ Failed to load model")
        return False
    
    # Continue with translation using batch processing...
    # [Rest of translation logic using the optimal batch size]
    
    print(f"🚀 Starting optimized translation...")
    start_time = time.time()
    
    successful = 0
    failed = 0
    
    # Simple batch processing with optimal size
    for i in tqdm(range(0, len(english_abstracts), optimal_batch_size), 
                 desc="Translating"):
        
        batch_data = english_abstracts[i:i + optimal_batch_size]
        
        for article_idx, abstract_text in batch_data:
            try:
                # Simple sentence-by-sentence translation (reliable)
                sentences = re.split(r'(?<=[.!?])\s+', abstract_text)
                translated_sentences = []
                
                for j in range(0, len(sentences), optimal_batch_size):
                    sentence_batch = sentences[j:j + optimal_batch_size]
                    sentence_batch = [s.strip() for s in sentence_batch if s.strip()]
                    
                    if sentence_batch:
                        results = translator(sentence_batch)
                        
                        for result in results:
                            if isinstance(result, list) and result:
                                translation = result[0].get('translation_text', '')
                            else:
                                translation = result.get('translation_text', '')
                            translated_sentences.append(translation)
                
                final_translation = " ".join(translated_sentences)
                
                # Add translation
                articles[article_idx]['abstract']['spanish_translation'] = {
                    'text': final_translation,
                    'success': True,
                    'model_used': model_name,
                    'batch_size_used': optimal_batch_size,
                    'translation_date': datetime.now().isoformat()
                }
                
                successful += 1
                
            except Exception as e:
                articles[article_idx]['abstract']['spanish_translation'] = {
                    'text': '',
                    'success': False,
                    'error': str(e),
                    'model_used': model_name
                }
                failed += 1
    
    # Save results
    total_time = time.time() - start_time
    
    dataset['metadata']['translation_info'] = {
        'model_used': model_name,
        'optimal_batch_size': optimal_batch_size,
        'translation_stats': {
            'successful': successful,
            'failed': failed,
            'total_time_minutes': total_time / 60,
            'abstracts_per_minute': len(english_abstracts) / (total_time / 60)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Results
    print(f"\n✅ Optimized translation completed!")
    print(f"🏆 Used batch size: {optimal_batch_size}")
    print(f"⚡ Speed: {len(english_abstracts) / (total_time / 60):.1f} abstracts/minute")
    print(f"📊 Success rate: {successful/len(english_abstracts)*100:.1f}%")
    
    return True

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Self-Tuning Medical Translator')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-m', '--model', default='Helsinki-NLP/opus-mt-en-es', help='Model')
    parser.add_argument('-b', '--batch-size', type=int, help='Force specific batch size')
    parser.add_argument('--no-tune', action='store_true', help='Skip auto-tuning')
    
    args = parser.parse_args()
    
    output_file = args.output or args.input_file.replace('.json', '_translated.json')
    
    if not os.path.exists(args.input_file):
        print(f"❌ File not found: {args.input_file}")
        return
    
    auto_tune = not args.no_tune and args.batch_size is None
    
    translate_dataset_tuned(
        args.input_file, 
        output_file, 
        args.model, 
        args.batch_size,
        auto_tune
    )

if __name__ == "__main__":
    main()

# python medical_translator.py spanish_medical_articles.json -o translated_medical_abstracts.json --no-tune -b 4