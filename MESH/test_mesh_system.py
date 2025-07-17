# mesh/test_mesh_system.py
"""
Script de testing para verificar que el sistema MeSH funciona correctamente

Uso:
    python mesh/test_mesh_system.py

Este script verifica:
1. Que existan los archivos necesarios
2. Que las clases funcionen correctamente
3. Que se puedan procesar artículos de ejemplo
"""

import sys
import os

# Agregar mesh/src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_file_structure():
    """Verifica que exista la estructura de archivos correcta"""
    print("🔍 Verificando estructura de archivos...")
    
    required_files = [
        "mesh/data/mesh_data.json",
        "mesh/src/mesh_downloader.py", 
        "mesh/src/mesh_processor_local.py",
        "mesh/src/run_mesh_downloader.py",
        "mesh/src/run_mesh_processor.py"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NO ENCONTRADO")
            all_exist = False
    
    return all_exist

def test_mesh_processor():
    """Verifica que MeshProcessorLocal funcione correctamente"""
    print("\n🔍 Verificando MeshProcessorLocal...")
    
    try:
        from mesh_processor_local import MeshProcessorLocal
        
        # Inicializar procesador
        processor = MeshProcessorLocal(mesh_data_dir="mesh/data")
        
        # Test básico: obtener tree numbers
        test_ui = "D004285"  # Dogs
        tree_numbers = processor.get_mesh_tree_numbers(test_ui)
        
        if tree_numbers:
            print(f"✅ Tree numbers para {test_ui}: {tree_numbers}")
            
            # Test: obtener categorías
            categories = processor.get_mesh_categories(test_ui)
            print(f"✅ Categorías: {categories}")
            
            # Test: obtener nombres
            if categories['level1_codes']:
                level1_code = list(categories['level1_codes'])[0]
                name = processor.get_category_name(level1_code)
                print(f"✅ Nombre para {level1_code}: {name}")
            
            return True
        else:
            print(f"❌ No se encontraron tree numbers para {test_ui}")
            return False
            
    except Exception as e:
        print(f"❌ Error en MeshProcessorLocal: {e}")
        return False

def test_sample_processing():
    """Verifica que se pueda procesar un archivo de ejemplo"""
    print("\n🔍 Verificando procesamiento de archivos de ejemplo...")
    
    # Buscar archivos de ejemplo
    sample_files = [
        "pubmed/sample/translated_dataset_small.json",
        "pubmed/sample/spanish_medical_articles_small.json"
    ]
    
    available_file = None
    for file_path in sample_files:
        if os.path.exists(file_path):
            available_file = file_path
            break
    
    if not available_file:
        print("❌ No se encontraron archivos de ejemplo en pubmed/sample/")
        return False
    
    print(f"✅ Archivo de ejemplo encontrado: {available_file}")
    
    try:
        from mesh_processor_local import MeshProcessorLocal
        
        processor = MeshProcessorLocal(mesh_data_dir="mesh/data")
        df = processor.process_json_file(available_file, debug=False)
        
        print(f"✅ Procesamiento exitoso: {len(df)} artículos")
        print(f"✅ Columnas generadas: {list(df.columns)}")
        
        # Verificar que hay datos
        has_categories = df['level1_categories_str'].str.len() > 0
        articles_with_categories = has_categories.sum()
        
        print(f"✅ Artículos con categorías: {articles_with_categories}/{len(df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error procesando archivo de ejemplo: {e}")
        return False

def main():
    """Función principal de testing"""
    print("=" * 60)
    print("🧪 TESTING DEL SISTEMA MESH")
    print("=" * 60)
    
    # Test 1: Estructura de archivos
    files_ok = test_file_structure()
    
    if not files_ok:
        print("\n❌ FALLO EN ESTRUCTURA DE ARCHIVOS")
        print("💡 Ejecuta: python mesh/src/run_mesh_downloader.py")
        return False
    
    # Test 2: Funcionalidad del procesador
    processor_ok = test_mesh_processor()
    
    if not processor_ok:
        print("\n❌ FALLO EN MESH PROCESSOR")
        return False
    
    # Test 3: Procesamiento de archivos
    processing_ok = test_sample_processing()
    
    if not processing_ok:
        print("\n❌ FALLO EN PROCESAMIENTO DE ARCHIVOS")
        return False
    
    # Todos los tests pasaron
    print("\n" + "=" * 60)
    print("🎉 ¡TODOS LOS TESTS PASARON!")
    print("=" * 60)
    print("✅ El sistema MeSH está funcionando correctamente")
    print("\n💡 Comandos disponibles:")
    print("   python mesh/src/run_mesh_processor.py pubmed/sample/translated_dataset_small.json dataset/test.csv")
    print("   python mesh/src/run_mesh_processor.py pubmed/data/spanish_medical_articles.json dataset/production.csv")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)