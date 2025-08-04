#!/usr/bin/env python3
"""
Self-Tuning Medical Abstract and Title Translator
Automatically finds optimal batch size for your hardware
Now includes title translation functionality with smart cleaning
"""

import json
import os
import re
import time
import argparse
import psutil
from datetime import datetime
from tqdm import tqdm

def clean_title(title):
    """
    Clean title by removing brackets/parentheses only from the extremes
    Keeps important parentheses/brackets within the text
    """
    if not title:
        return title
    
    cleaned = title.strip()
    
    # First remove trailing periods/punctuation
    cleaned = re.sub(r'[\.]+$', '', cleaned).strip()
    
    # Only remove brackets/parentheses that wrap the ENTIRE title
    # Remove outermost brackets: [entire title] -> entire title
    while cleaned.startswith('[') and cleaned.endswith(']'):
        cleaned = cleaned[1:-1].strip()
    
    # Remove outermost parentheses: (entire title) -> entire title  
    while cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = cleaned[1:-1].strip()
    
    # Final cleanup of any remaining trailing punctuation
    cleaned = re.sub(r'[\.]+$', '', cleaned).strip()
    
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

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

def benchmark_batch_size(translator, test_texts, batch_size, text_type="abstract"):
    """Benchmark translation speed and memory usage for a batch size"""
    print(f"   Testing batch size {batch_size} for {text_type}s...")
    
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
        
        # Translate test texts
        results = []
        if text_type == "title":
            # Titles are simpler - translate directly in batches
            for i in range(0, len(test_texts), batch_size):
                batch = test_texts[i:i + batch_size]
                result = translator(batch)
                results.extend(result)
        else:
            # Abstracts need sentence splitting
            for text in test_texts:
                sentences = re.split(r'(?<=[.!?])\s+', text)
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
        texts_per_second = len(test_texts) / translation_time
        
        return {
            'batch_size': batch_size,
            'time_seconds': translation_time,
            'texts_per_second': texts_per_second,
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
            'texts_per_second': 0
        }

def find_optimal_batch_size(model_name, test_texts, text_type="abstract"):
    """Find the optimal batch size through testing"""
    print(f"🔍 FINDING OPTIMAL BATCH SIZE FOR {text_type.upper()}S")
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
        benchmark = benchmark_batch_size(translator, test_texts, batch_size, text_type)
        results.append(benchmark)
        
        if benchmark['success']:
            print(f"   ✅ Speed: {benchmark['texts_per_second']:.2f} {text_type}s/sec")
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
        print(f"\n❌ No batch sizes worked for {text_type}s! Using batch size 1")
        return 1
    
    # Find fastest batch size
    optimal = max(successful_results, key=lambda x: x['texts_per_second'])
    
    print(f"\n🏆 OPTIMAL BATCH SIZE FOR {text_type.upper()}S: {optimal['batch_size']}")
    print(f"   ⚡ Speed: {optimal['texts_per_second']:.2f} {text_type}s/sec")
    print(f"   📊 Memory impact: +{optimal['memory_usage_percent']:.1f}% RAM")
    
    return optimal['batch_size']

def translate_dataset_tuned(input_file, output_file, model_name="Helsinki-NLP/opus-mt-en-es", 
                           batch_size=None, auto_tune=True, translate_titles=True, titles_only=False):
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
    
    # Find abstracts and titles to translate
    english_abstracts = []
    titles_to_translate = []
    
    for i, article in enumerate(articles):
        # Check abstracts (skip if titles_only mode)
        if not titles_only and (((article.get('abstract_language') == 'english') or (article.get('abstract_language') == 'mixed_or_unknown')) and 
            article.get('abstract', {}).get('has_content', False)):
            # Only add if not already translated
            if not article.get('abstract', {}).get('spanish_translation', {}).get('success', False):
                english_abstracts.append((i, article['abstract']['full_text']))
        
        # Check titles - translate ALL titles that exist and aren't already translated
        if translate_titles and article.get('title'):
            title = article['title'].strip()
            if title and not article.get('title_spanish_translation', {}).get('success', False):
                # Clean the title before adding to translation queue
                cleaned_title = clean_title(title)
                if cleaned_title:  # Only add if something remains after cleaning
                    titles_to_translate.append({
                        'article_idx': i,
                        'original_title': title,
                        'cleaned_title': cleaned_title
                    })
    
    if titles_only:
        print(f"📝 TITLES-ONLY MODE: Titles to translate: {len(titles_to_translate):,}")
        # Check for existing abstract translations
        existing_abstract_translations = sum(1 for article in articles 
                                           if article.get('abstract', {}).get('spanish_translation', {}).get('success', False))
        if existing_abstract_translations > 0:
            print(f"✅ Found {existing_abstract_translations:,} existing abstract translations (will be preserved)")
    else:
        print(f"🔤 English abstracts to translate: {len(english_abstracts):,}")
        if translate_titles:
            print(f"📝 Titles to translate: {len(titles_to_translate):,}")
    
    if not english_abstracts and not titles_to_translate:
        if titles_only:
            print("⚠️ No titles found to translate")
        else:
            print("⚠️ No content found to translate")
        return False
    
    # Auto-tune batch sizes if requested
    abstract_batch_size = batch_size or 4
    title_batch_size = batch_size or 8  # Titles are usually shorter, can use larger batch
    
    if auto_tune and batch_size is None:
        if english_abstracts and not titles_only:
            # Use first 5 abstracts for testing
            test_abstracts = [item[1] for item in english_abstracts[:5]]
            abstract_batch_size = find_optimal_batch_size(model_name, test_abstracts, "abstract")
        
        if titles_to_translate:
            # Use first 10 titles for testing (titles are shorter)
            test_titles = [item['cleaned_title'] for item in titles_to_translate[:10]]
            title_batch_size = find_optimal_batch_size(model_name, test_titles, "title")
    else:
        if titles_only:
            print(f"🔧 Using specified batch size for titles: {title_batch_size}")
        else:
            print(f"🔧 Using specified batch sizes: {abstract_batch_size} (abstracts), {title_batch_size} (titles)")
    
    # Start translation process
    print(f"\n🚀 Starting optimized translation...")
    start_time = time.time()
    
    abstract_successful = 0
    abstract_failed = 0
    title_successful = 0
    title_failed = 0
    
    # Translate abstracts (skip if titles_only mode)
    if english_abstracts and not titles_only:
        print(f"\n📄 Translating {len(english_abstracts)} abstracts...")
        translator = load_model_with_batch_size(model_name, abstract_batch_size)
        
        if translator:
            for i in tqdm(range(0, len(english_abstracts), abstract_batch_size), 
                         desc="Translating abstracts"):
                
                batch_data = english_abstracts[i:i + abstract_batch_size]
                
                for article_idx, abstract_text in batch_data:
                    try:
                        # Simple sentence-by-sentence translation (reliable)
                        sentences = re.split(r'(?<=[.!?])\s+', abstract_text)
                        translated_sentences = []
                        
                        for j in range(0, len(sentences), abstract_batch_size):
                            sentence_batch = sentences[j:j + abstract_batch_size]
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
                            'batch_size_used': abstract_batch_size,
                            'translation_date': datetime.now().isoformat()
                        }
                        
                        abstract_successful += 1
                        
                    except Exception as e:
                        articles[article_idx]['abstract']['spanish_translation'] = {
                            'text': '',
                            'success': False,
                            'error': str(e),
                            'model_used': model_name
                        }
                        abstract_failed += 1
    
    # Translate titles
    if titles_to_translate and translate_titles:
        print(f"\n📝 Translating {len(titles_to_translate)} titles...")
        translator = load_model_with_batch_size(model_name, title_batch_size)
        
        if translator:
            for i in tqdm(range(0, len(titles_to_translate), title_batch_size), 
                         desc="Translating titles"):
                
                batch_data = titles_to_translate[i:i + title_batch_size]
                titles_to_translate_clean = [item['cleaned_title'] for item in batch_data]
                
                try:
                    # Translate clean titles in batch
                    results = translator(titles_to_translate_clean)
                    
                    for title_data, result in zip(batch_data, results):
                        try:
                            if isinstance(result, list) and result:
                                translation = result[0].get('translation_text', '')
                            else:
                                translation = result.get('translation_text', '')
                            
                            # Add title translation
                            articles[title_data['article_idx']]['title_spanish_translation'] = {
                                'original_title': title_data['original_title'],
                                'cleaned_title': title_data['cleaned_title'],
                                'translated_title': translation,
                                'success': True,
                                'model_used': model_name,
                                'batch_size_used': title_batch_size,
                                'translation_date': datetime.now().isoformat()
                            }
                            
                            title_successful += 1
                            
                        except Exception as e:
                            articles[title_data['article_idx']]['title_spanish_translation'] = {
                                'original_title': title_data['original_title'],
                                'cleaned_title': title_data['cleaned_title'],
                                'translated_title': '',
                                'success': False,
                                'error': str(e),
                                'model_used': model_name
                            }
                            title_failed += 1
                            
                except Exception as e:
                    # If batch fails, mark all titles in batch as failed
                    for title_data in batch_data:
                        articles[title_data['article_idx']]['title_spanish_translation'] = {
                            'original_title': title_data['original_title'],
                            'cleaned_title': title_data['cleaned_title'],
                            'translated_title': '',
                            'success': False,
                            'error': str(e),
                            'model_used': model_name
                        }
                        title_failed += 1
    
    # Save results
    total_time = time.time() - start_time
    
    dataset['metadata']['translation_info'] = {
        'model_used': model_name,
        'abstract_batch_size': abstract_batch_size if not titles_only else None,
        'title_batch_size': title_batch_size if translate_titles else None,
        'titles_only_mode': titles_only,
        'translation_stats': {
            'abstracts': {
                'successful': abstract_successful,
                'failed': abstract_failed,
                'total': len(english_abstracts)
            } if not titles_only else None,
            'titles': {
                'successful': title_successful,
                'failed': title_failed,
                'total': len(titles_to_translate) if translate_titles else 0
            },
            'total_time_minutes': total_time / 60,
            'items_per_minute': (len(titles_to_translate) / (total_time / 60)) if titles_only else 
                               ((len(english_abstracts) + len(titles_to_translate)) / (total_time / 60) if translate_titles else len(english_abstracts) / (total_time / 60))
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    # Results
    print(f"\n✅ Optimized translation completed!")
    
    if titles_only:
        print(f"🏆 Used batch size for titles: {title_batch_size}")
        print(f"⚡ Speed: {len(titles_to_translate) / (total_time / 60):.1f} titles/minute")
        if title_successful + title_failed > 0:
            print(f"📝 Title success rate: {title_successful/(title_successful+title_failed)*100:.1f}%")
    else:
        print(f"🏆 Used batch sizes: {abstract_batch_size} (abstracts)" + 
              (f", {title_batch_size} (titles)" if translate_titles else ""))
        print(f"⚡ Speed: {(len(english_abstracts) + len(titles_to_translate)) / (total_time / 60):.1f} items/minute")
        
        if english_abstracts:
            print(f"📄 Abstract success rate: {abstract_successful/(abstract_successful+abstract_failed)*100:.1f}%")
        if titles_to_translate and translate_titles and (title_successful + title_failed > 0):
            print(f"📝 Title success rate: {title_successful/(title_successful+title_failed)*100:.1f}%")
    
    return True

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Self-Tuning Medical Translator with Title Support')
    parser.add_argument('input_file', help='Input JSON file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-m', '--model', default='Helsinki-NLP/opus-mt-en-es', help='Model')
    parser.add_argument('-b', '--batch-size', type=int, help='Force specific batch size')
    parser.add_argument('--no-tune', action='store_true', help='Skip auto-tuning')
    parser.add_argument('--no-titles', action='store_true', help='Skip title translation')
    parser.add_argument('--titles-only', action='store_true', help='Only translate titles (skip abstracts)')
    
    args = parser.parse_args()
    
    # Validation: titles-only and no-titles are mutually exclusive
    if args.titles_only and args.no_titles:
        print("❌ Error: --titles-only and --no-titles cannot be used together")
        return
    
    output_file = args.output or args.input_file.replace('.json', '_translated.json')
    
    if not os.path.exists(args.input_file):
        print(f"❌ File not found: {args.input_file}")
        return
    
    auto_tune = not args.no_tune and args.batch_size is None
    translate_titles = not args.no_titles or args.titles_only  # Always translate titles if titles_only
    titles_only = args.titles_only
    
    translate_dataset_tuned(
        args.input_file, 
        output_file, 
        args.model, 
        args.batch_size,
        auto_tune,
        translate_titles,
        titles_only
    )

if __name__ == "__main__":
    main()

# Usage examples:
# python medical_translator.py spanish_medical_articles.json -o translated_medical_abstracts.json
# python medical_translator.py spanish_medical_articles.json --no-titles  # Skip title translation
# python medical_translator.py spanish_medical_articles.json --titles-only  # Only translate titles
# python medical_translator.py spanish_medical_articles.json --no-tune -b 4  # Force batch size 4