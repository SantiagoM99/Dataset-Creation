# mesh/src/run_mesh_downloader.py
"""
Script para descargar y procesar datos oficiales de MeSH

Uso:
    python mesh/src/run_mesh_downloader.py

Este script:
1. Descarga el archivo desc2025.xml oficial de NCBI (~200MB)
2. Parsea el XML y extrae términos, tree numbers y nombres
3. Genera mesh_data.json optimizado para búsquedas rápidas
4. Guarda todo en la carpeta mesh/data/

Solo necesitas ejecutar esto UNA VEZ para configurar los datos MeSH.
"""

import sys
import os
import time
from datetime import datetime

# Agregar mesh/src al path para imports
sys.path.append(os.path.dirname(__file__))

from mesh_downloader import MeshDownloader

def main():
    """Función principal del script"""
    
    print("=" * 60)
    print("🧬 MESH DOWNLOADER - Configuración de Datos MeSH")
    print("=" * 60)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Inicializar downloader
        print("🚀 Inicializando MeshDownloader...")
        downloader = MeshDownloader(mesh_data_dir="mesh/data")
        
        # Configurar datos MeSH
        start_time = time.time()
        mesh_data = downloader.setup_mesh_data()
        end_time = time.time()
        
        # Mostrar estadísticas
        print("\n" + "=" * 60)
        print("📊 ESTADÍSTICAS FINALES")
        print("=" * 60)
        
        stats = downloader.get_stats()
        if 'error' not in stats:
            print(f"✅ Términos MeSH totales: {stats['total_mesh_terms']:,}")
            print(f"✅ Términos con tree numbers: {stats['terms_with_tree_numbers']:,}")
            print(f"✅ Categorías totales: {stats['total_categories']:,}")
            print(f"✅ Categorías principales: {stats['main_categories']}")
            print(f"✅ Categorías nivel 2: {stats['level2_categories']}")
        
        print(f"\n⏱️  Tiempo total: {end_time - start_time:.1f} segundos")
        print(f"📁 Archivos generados en: mesh/data/")
        print(f"   - desc2025.xml (archivo oficial)")
        print(f"   - mesh_data.json (datos procesados)")
        
        print("\n🎉 ¡Configuración completada exitosamente!")
        print("💡 Ahora puedes usar mesh/src/run_mesh_processor.py para procesar artículos")
        
    except KeyboardInterrupt:
        print("\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error durante la configuración: {e}")
        print("\n🔧 Posibles soluciones:")
        print("   - Verificar conexión a internet")
        print("   - Verificar permisos de escritura en la carpeta mesh/data/")
        print("   - Verificar espacio en disco (se necesitan ~300MB)")
        sys.exit(1)

if __name__ == "__main__":
    main()