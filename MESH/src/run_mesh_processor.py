# mesh/src/run_mesh_processor.py
"""
Script para procesar artículos científicos y extraer categorías MeSH

Uso:
    python mesh/src/run_mesh_processor.py <archivo_entrada> <archivo_salida>
    python mesh/src/run_mesh_processor.py pubmed/sample/translated_dataset_small.json dataset/translated_dataset_small.csv
    python mesh/src/run_mesh_processor.py pubmed/data/translated_dataset.json dataset/translated_dataset.csv

Este script:
1. Carga artículos desde el archivo JSON especificado
2. Procesa MeSH terms y extrae categorías (solo major subjects)
3. Incluye abstracts en español en el resultado
4. Guarda el resultado en la ruta especificada

Requisitos:
- Ejecutar primero mesh/src/run_mesh_downloader.py para generar mesh_data.json
- El archivo JSON debe tener la estructura correcta con 'articles'
"""

import sys
import os
import time
import argparse
from datetime import datetime

# Agregar mesh/src al path para imports
sys.path.append(os.path.dirname(__file__))

from mesh_processor_local import MeshProcessorLocal

def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Procesa artículos científicos y extrae categorías MeSH",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python mesh/src/run_mesh_processor.py pubmed/sample/translated_dataset_small.json dataset/small_mesh.csv
  python mesh/src/run_mesh_processor.py pubmed/data/spanish_medical_articles.json dataset/spanish_mesh.csv
  python mesh/src/run_mesh_processor.py --debug pubmed/sample/test.json dataset/test_mesh.csv

Archivos de entrada típicos:
  - pubmed/sample/translated_dataset_small.json
  - pubmed/sample/spanish_medical_articles_small.json
  - pubmed/data/translated_dataset.json
  - pubmed/data/spanish_medical_articles.json
        """
    )
    
    parser.add_argument(
        'input_file',
        help='Ruta al archivo JSON con artículos científicos'
    )
    
    parser.add_argument(
        'output_file', 
        help='Ruta donde guardar el archivo CSV con categorías MeSH'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Mostrar información detallada de debugging (primeros 3 artículos)'
    )
    
    return parser.parse_args()

def ensure_output_directory(output_file: str) -> None:
    """
    Crea el directorio de salida si no existe
    
    Args:
        output_file (str): Ruta del archivo de salida
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Directorio creado: {output_dir}")

def validate_files(input_file: str, output_file: str) -> None:
    """
    Valida que el archivo de entrada existe y el de salida es válido
    
    Args:
        input_file (str): Archivo de entrada
        output_file (str): Archivo de salida
        
    Raises:
        FileNotFoundError: Si no se encuentra el archivo de entrada
        ValueError: Si la extensión del archivo de salida no es CSV
    """
    # Verificar archivo de entrada
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"❌ Archivo de entrada no encontrado: {input_file}")
    
    # Verificar extensión de salida
    if not output_file.lower().endswith('.csv'):
        raise ValueError(f"❌ El archivo de salida debe tener extensión .csv: {output_file}")

def main():
    """Función principal del script"""
    
    # Parsear argumentos
    args = parse_arguments()
    
    print("=" * 80)
    print("🧬 MESH PROCESSOR - Extracción de Categorías MeSH")
    print("=" * 80)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Archivo de entrada: {args.input_file}")
    print(f"💾 Archivo de salida: {args.output_file}")
    print(f"🐛 Modo debug: {'Activado' if args.debug else 'Desactivado'}")
    print()
    
    try:
        # Validar archivos
        validate_files(args.input_file, args.output_file)
        
        # Crear directorio de salida si no existe
        ensure_output_directory(args.output_file)
        
        # Inicializar procesador
        print("🚀 Inicializando MeshProcessorLocal...")
        processor = MeshProcessorLocal(mesh_data_dir="mesh/data")
        print()
        
        # Procesar archivo
        print("📊 Procesando artículos...")
        start_time = time.time()
        
        df = processor.process_json_file(args.input_file, debug=args.debug)
        
        processing_time = time.time() - start_time
        
        # Guardar resultados
        print("\n💾 Guardando resultados...")
        processor.save_results(df, args.output_file)
        
        # Mostrar estadísticas
        print("\n" + "=" * 80)
        print("📊 ESTADÍSTICAS DEL PROCESAMIENTO")
        print("=" * 80)
        
        stats = processor.get_processing_stats(df)
        
        print(f"✅ Total de artículos procesados: {stats['total_articles']:,}")
        print(f"✅ Artículos con categorías MeSH: {stats['articles_with_categories']:,}")
        print(f"✅ Artículos con abstract en español: {stats['articles_with_spanish_abstract']:,}")
        print(f"✅ Cobertura de categorías: {stats['coverage_percentage']:.1f}%")
        print(f"✅ Cobertura de abstracts en español: {stats['spanish_abstract_percentage']:.1f}%")
        print(f"✅ Categorías únicas encontradas: {stats['unique_level1_categories']}")
        articles_without_categories = stats['total_articles'] - stats['articles_with_categories']
        percentage_without_categories = (articles_without_categories / stats['total_articles']) * 100
        print(f"📊 Artículos SIN categorías MeSH: {articles_without_categories:,}")
        print(f"📊 Porcentaje sin major subjects: {percentage_without_categories:.1f}%")
        
        print(f"\n🏆 Top 5 categorías más frecuentes:")
        for i, (category, count) in enumerate(list(stats['most_common_categories'].items())[:5], 1):
            print(f"   {i}. {category}: {count} artículos")
        
        print(f"\n⏱️  Tiempo de procesamiento: {processing_time:.1f} segundos")
        if processing_time > 0:
            print(f"🚀 Velocidad: {stats['total_articles'] / processing_time:.1f} artículos/segundo")
        
        print(f"\n📁 Columnas en el resultado:")
        print(f"   - pmid, title, year, spanish_abstract")
        print(f"   - level1_codes, level1_names, level2_codes, level2_names")
        print(f"   - level1_categories_str, level2_categories_str")
        print(f"   - major_subjects_count")
        
        print("\n🎉 ¡Procesamiento completado exitosamente!")
        print(f"📊 Dataset listo para entrenamiento en: {args.output_file}")
        

    except KeyboardInterrupt:
        print("\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(1)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n🔧 Archivos disponibles en pubmed/:")
        print("   📁 pubmed/sample/")
        print("      - translated_dataset_small.json")
        print("      - spanish_medical_articles_small.json")
        print("   📁 pubmed/data/")
        print("      - translated_dataset.json")
        print("      - spanish_medical_articles.json")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 El archivo de salida debe terminar en .csv")
        print("   Ejemplo: dataset/mi_resultado.csv")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {e}")
        print("\n🔧 Posibles soluciones:")
        print("   - Ejecutar primero: python mesh/src/run_mesh_downloader.py")
        print("   - Verificar que el archivo JSON tiene la estructura correcta")
        print("   - Verificar permisos de escritura en el directorio de salida")
        print("   - Verificar que el archivo contiene artículos con mesh_terms")
        sys.exit(1)

if __name__ == "__main__":
    main()