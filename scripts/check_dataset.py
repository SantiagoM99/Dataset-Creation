#!/usr/bin/env python3
"""
Comparador de directorios - Encuentra archivos que faltan entre dos carpetas
Ignora automáticamente archivos que empiecen por ._

Uso:
python compare_dirs.py carpeta1 carpeta2
python compare_dirs.py --dir1 original --dir2 traducido
python compare_dirs.py --dir1 AnatEM-original --dir2 AnatEM-spanish --extension .nersuite
"""

import os
import argparse
from pathlib import Path
from typing import Set, List

def get_valid_files(directory: Path, extension: str = None) -> Set[str]:
    """
    Obtiene lista de archivos válidos en un directorio
    Ignora archivos que empiecen por ._ y otros archivos ocultos
    """
    if not directory.exists():
        print(f"❌ El directorio no existe: {directory}")
        return set()
    
    valid_files = set()
    
    # Buscar archivos
    if extension:
        files = directory.glob(f"*{extension}")
    else:
        files = directory.glob("*")
    
    for file_path in files:
        # Saltar si es directorio
        if file_path.is_dir():
            continue
            
        filename = file_path.name
        
        # Ignorar archivos que empiecen por ._
        if filename.startswith('._'):
            continue
            
        # Ignorar otros archivos ocultos/temporales
        if filename.startswith('.'):
            continue
            
        # Ignorar archivos de sistema comunes
        if filename in ['.DS_Store', 'Thumbs.db', 'desktop.ini']:
            continue
            
        valid_files.add(filename)
    
    return valid_files

def compare_directories(dir1: Path, dir2: Path, extension: str = None):
    """Compara dos directorios y muestra diferencias"""
    
    print(f"🔍 Comparando directorios:")
    print(f"   📁 Directorio 1: {dir1}")
    print(f"   📁 Directorio 2: {dir2}")
    
    if extension:
        print(f"   🔧 Filtro de extensión: {extension}")
    
    print()
    
    # Obtener archivos de cada directorio
    files1 = get_valid_files(dir1, extension)
    files2 = get_valid_files(dir2, extension)
    
    print(f"📊 Archivos encontrados:")
    print(f"   📁 {dir1.name}: {len(files1)} archivos")
    print(f"   📁 {dir2.name}: {len(files2)} archivos")
    print()
    
    # Encontrar diferencias
    only_in_dir1 = files1 - files2  # En dir1 pero no en dir2
    only_in_dir2 = files2 - files1  # En dir2 pero no en dir1
    common_files = files1 & files2  # En ambos directorios
    
    # Mostrar resultados
    print(f"📈 RESUMEN DE COMPARACIÓN:")
    print(f"   ✅ Archivos en ambos directorios: {len(common_files)}")
    print(f"   ➡️  Solo en {dir1.name}: {len(only_in_dir1)}")
    print(f"   ⬅️  Solo en {dir2.name}: {len(only_in_dir2)}")
    print()
    
    # Mostrar archivos que faltan en dir2
    if only_in_dir1:
        print(f"❌ ARCHIVOS QUE FALTAN EN {dir2.name} ({len(only_in_dir1)} archivos):")
        sorted_files = sorted(only_in_dir1)
        
        if len(sorted_files) <= 20:
            for filename in sorted_files:
                print(f"   - {filename}")
        else:
            # Mostrar primeros 10 y últimos 10
            for filename in sorted_files[:10]:
                print(f"   - {filename}")
            print(f"   ... ({len(sorted_files) - 20} archivos más) ...")
            for filename in sorted_files[-10:]:
                print(f"   - {filename}")
        print()
    
    # Mostrar archivos que faltan en dir1
    if only_in_dir2:
        print(f"❌ ARCHIVOS QUE FALTAN EN {dir1.name} ({len(only_in_dir2)} archivos):")
        sorted_files = sorted(only_in_dir2)
        
        if len(sorted_files) <= 20:
            for filename in sorted_files:
                print(f"   - {filename}")
        else:
            # Mostrar primeros 10 y últimos 10
            for filename in sorted_files[:10]:
                print(f"   - {filename}")
            print(f"   ... ({len(sorted_files) - 20} archivos más) ...")
            for filename in sorted_files[-10:]:
                print(f"   - {filename}")
        print()
    
    # Mostrar estado final
    if not only_in_dir1 and not only_in_dir2:
        print("🎉 ¡PERFECTO! Ambos directorios tienen exactamente los mismos archivos.")
    else:
        print(f"⚠️  Los directorios NO están sincronizados:")
        if only_in_dir1:
            print(f"   - {len(only_in_dir1)} archivos faltan en {dir2.name}")
        if only_in_dir2:
            print(f"   - {len(only_in_dir2)} archivos faltan en {dir1.name}")
    
    print()
    
    # Calcular porcentaje de completitud
    if files1:
        completion_rate = (len(common_files) / len(files1)) * 100
        print(f"📊 Porcentaje de completitud: {completion_rate:.1f}%")
        
        if completion_rate < 100:
            missing_count = len(only_in_dir1)
            print(f"📊 Archivos pendientes: {missing_count}")

def save_missing_files_list(dir1: Path, dir2: Path, extension: str = None, output_file: str = "archivos_faltantes.txt"):
    """Guarda la lista de archivos faltantes en un archivo"""
    
    files1 = get_valid_files(dir1, extension)
    files2 = get_valid_files(dir2, extension)
    
    only_in_dir1 = sorted(files1 - files2)
    only_in_dir2 = sorted(files2 - files1)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"COMPARACIÓN DE DIRECTORIOS\n")
        f.write(f"=========================\n")
        f.write(f"Directorio 1: {dir1}\n")
        f.write(f"Directorio 2: {dir2}\n")
        f.write(f"Fecha: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if only_in_dir1:
            f.write(f"ARCHIVOS QUE FALTAN EN {dir2.name} ({len(only_in_dir1)} archivos):\n")
            f.write("-" * 50 + "\n")
            for filename in only_in_dir1:
                f.write(f"{filename}\n")
            f.write("\n")
        
        if only_in_dir2:
            f.write(f"ARCHIVOS QUE FALTAN EN {dir1.name} ({len(only_in_dir2)} archivos):\n")
            f.write("-" * 50 + "\n")
            for filename in only_in_dir2:
                f.write(f"{filename}\n")
            f.write("\n")
        
        if not only_in_dir1 and not only_in_dir2:
            f.write("¡PERFECTO! Ambos directorios están sincronizados.\n")
    
    print(f"💾 Lista guardada en: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Compara dos directorios y encuentra archivos faltantes")
    
    # Argumentos posicionales (más simple)
    parser.add_argument("directories", nargs='*', help="Dos directorios a comparar")
    
    # Argumentos con nombre (más explícito)
    parser.add_argument("--dir1", help="Primer directorio")
    parser.add_argument("--dir2", help="Segundo directorio")
    parser.add_argument("--extension", "-e", help="Filtrar por extensión (ej: .nersuite, .txt)")
    parser.add_argument("--save", "-s", help="Guardar lista de archivos faltantes en archivo")
    parser.add_argument("--quiet", "-q", action="store_true", help="Modo silencioso (solo mostrar resumen)")
    
    args = parser.parse_args()
    
    # Determinar directorios
    if args.directories and len(args.directories) == 2:
        dir1, dir2 = args.directories[0], args.directories[1]
    elif args.dir1 and args.dir2:
        dir1, dir2 = args.dir1, args.dir2
    else:
        print("❌ Error: Necesitas especificar dos directorios")
        print("\nUso:")
        print("  python compare_dirs.py carpeta1 carpeta2")
        print("  python compare_dirs.py --dir1 original --dir2 traducido")
        return
    
    dir1_path = Path(dir1)
    dir2_path = Path(dir2)
    
    # Verificar que los directorios existen
    if not dir1_path.exists():
        print(f"❌ Error: El directorio '{dir1}' no existe")
        return
    
    if not dir2_path.exists():
        print(f"❌ Error: El directorio '{dir2}' no existe")
        return
    
    # Comparar directorios
    if not args.quiet:
        compare_directories(dir1_path, dir2_path, args.extension)
    
    # Guardar lista si se solicita
    if args.save:
        save_missing_files_list(dir1_path, dir2_path, args.extension, args.save)
    
    # Modo silencioso - solo mostrar números
    if args.quiet:
        files1 = get_valid_files(dir1_path, args.extension)
        files2 = get_valid_files(dir2_path, args.extension)
        only_in_dir1 = len(files1 - files2)
        only_in_dir2 = len(files2 - files1)
        common = len(files1 & files2)
        
        print(f"Común: {common}, Solo en {dir1_path.name}: {only_in_dir1}, Solo en {dir2_path.name}: {only_in_dir2}")

if __name__ == "__main__":
    main()